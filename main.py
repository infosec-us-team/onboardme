"""Generate a JSON file describing entry-point execution flows with state variable info."""

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from slither.core.declarations import Contract
from slither.slither import Slither

from analysis.slither_env import (
    DEFAULT_ADDRESS,
    _load_dotenv,
    _normalize_address,
    _resolve_chain_and_address,
)
from analysis.flow_walk import _iter_audited_contracts, build_entry_point_flows
from analysis.render import render_html

# Silence Slither logging to keep console output clean.
logging.disable(logging.CRITICAL)

ProgressCallback = Callable[[str, Dict[str, Any] | None], None]

# ----------- Helper utilities -------------------------------------------------


def _report_progress(callback: ProgressCallback | None, message: str, **meta: Any) -> None:
    """Best-effort progress reporting."""
    if not callback:
        return
    try:
        callback(message, meta or None)
    except Exception:
        pass


def _resolve_contract_name_from_export(address: str, chain: str) -> str | None:
    """Try to infer the verified contract name from crytic-export folder names."""
    export_root = Path("crytic-export") / "etherscan-contracts"
    if not export_root.exists():
        return None
    prefix = f"{_normalize_address(address)}{chain.lower()}-"
    for entry in export_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.lower().startswith(prefix):
            candidate = name[len(prefix):]
            return candidate or None
    return None


def _resolve_root_contracts(slither: Slither, address: str, chain: str) -> List[Contract]:
    """Return the deployed contract(s) for the address, if detectable."""
    target = _normalize_address(address)
    if not target:
        return []

    matches: List[Contract] = []
    for contract in slither.contracts:
        for attr in ("address", "contract_address", "deployed_address"):
            val = getattr(contract, attr, None)
            if val and _normalize_address(str(val)) == target:
                matches.append(contract)
                break

    if matches:
        return matches

    name_hint = _resolve_contract_name_from_export(address, chain)
    if not name_hint:
        return []

    return [c for c in slither.contracts if c.name == name_hint]


# ----------- Main ------------------------------------------------------------

def generate_html(
    address: str,
    chain: str | None = None,
    output_dir: Path | None = None,
    progress_cb: ProgressCallback | None = None,
):
    """Generate the entry-point HTML for the given contract and return metadata."""

    _report_progress(progress_cb, "Loading environment")
    _load_dotenv()
    _report_progress(progress_cb, "Normalizing target")
    address, chain = _resolve_chain_and_address(address, chain)
    target = f"{chain}:{address}"
    _report_progress(progress_cb, "Fetching verified source", target=target)
    export_root = Path("crytic-export") / "etherscan-contracts"
    if export_root.exists():
        prefix = f"{_normalize_address(address)}{chain.lower()}-"
        has_cache = any(
            entry.is_dir() and entry.name.lower().startswith(prefix)
            for entry in export_root.iterdir()
        )
        _report_progress(
            progress_cb,
            "Source cache check",
            cached=bool(has_cache),
        )

    stop_fetch = threading.Event()

    def _fetch_heartbeat() -> None:
        start = time.time()
        while not stop_fetch.wait(2.0):
            elapsed = int(time.time() - start)
            _report_progress(
                progress_cb,
                "Fetching verified source",
                target=target,
                elapsed_seconds=elapsed,
            )

    heartbeat_thread = None
    if progress_cb:
        heartbeat_thread = threading.Thread(
            target=_fetch_heartbeat, name="fetch-heartbeat", daemon=True
        )
        heartbeat_thread.start()

    slither = Slither(target, skip_analyze=False)
    if heartbeat_thread:
        stop_fetch.set()
        heartbeat_thread.join(timeout=1.0)
    _report_progress(progress_cb, "Compiler analysis completed")

    _report_progress(progress_cb, "Resolving deployed contract")
    root_contracts = _resolve_root_contracts(slither, address, chain)
    _report_progress(progress_cb, "Building entry point flows")
    data = build_entry_point_flows(slither, root_contracts, progress_cb=progress_cb)

    audited_contracts = list(_iter_audited_contracts(slither, root_contracts))
    if audited_contracts:
        title_contract = audited_contracts[0].name
    else:
        title_contract = f"{chain}:{address}"
    output_path = render_html(
        data,
        chain,
        address,
        title_contract,
        output_dir=output_dir,
        progress_cb=progress_cb,
    )

    contract_names = sorted({entry.get("contract", "")
                            for entry in data if entry.get("contract")})

    return {
        "output_path": output_path,
        "contracts": contract_names,
        "entry_count": len(data),
        "target": target,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for generating a single contract HTML file."""
    parser = argparse.ArgumentParser(
        description="Analyze a contract's entry-point execution flows.")
    parser.add_argument(
        "address",
        nargs="?",
        default=DEFAULT_ADDRESS,
        help="Contract address (default: built-in sample).",
    )
    parser.add_argument(
        "chain",
        nargs="?",
        default=None,
        help="Chain name or chain id (default: mainnet).",
    )
    args = parser.parse_args(argv)

    result = generate_html(args.address, args.chain)
    output_path = result["output_path"]
    file_url = output_path.resolve().as_uri()
    print(
        f"Wrote {result['entry_count']} entry point flows to {output_path} using target {result['target']}"
    )
    print(f"Open in browser: {file_url}")


if __name__ == "__main__":
    main()
