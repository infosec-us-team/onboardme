"""Generate a JSON file describing entry-point execution flows with state variable info."""

import argparse
import hashlib
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
from analysis.state_vars import (
    _event_record,
    _interface_record,
    _library_record,
    _state_var_record,
    _type_alias_record,
)

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


def _local_project_hash(path: Path) -> str:
    """Stable hash for a local Solidity project or file."""
    sol_files: List[Path]
    if path.is_dir():
        sol_files = [p for p in path.rglob("*.sol") if p.is_file()]
    else:
        sol_files = [path] if path.suffix.lower() == ".sol" and path.is_file() else []
    if not sol_files:
        raise ValueError("No Solidity files found in project")

    pieces: List[str] = []
    for sol_file in sol_files:
        rel_path = str(sol_file.relative_to(path)) if path.is_dir() else sol_file.name
        digest = hashlib.sha256(sol_file.read_bytes()).hexdigest()
        pieces.append(f"{rel_path}:{digest}")
    joined = "|".join(sorted(pieces))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


def _collect_render_metadata(
    slither: Slither,
    root_contracts: Sequence[Contract] | None = None,
) -> Dict[str, Any]:
    """Collect metadata lists used by render_html."""
    audited_contracts = list(_iter_audited_contracts(slither, root_contracts))

    extra_storage_vars: List[Dict[str, Any]] = []
    seen_vars: set[str] = set()
    for contract in audited_contracts:
        for var in getattr(contract, "state_variables", []) or []:
            record = _state_var_record(var)
            key = record.get("qualified_name") or record.get("name", "")
            if not key or key in seen_vars:
                continue
            seen_vars.add(key)
            extra_storage_vars.append(record)
    for compilation_unit in getattr(slither, "compilation_units", []) or []:
        for var in getattr(compilation_unit, "variables_top_level", []) or []:
            if not (getattr(var, "is_constant", False) or getattr(var, "is_immutable", False)):
                continue
            record = _state_var_record(var)
            key = record.get("qualified_name") or record.get("name", "")
            if not key or key in seen_vars:
                continue
            seen_vars.add(key)
            extra_storage_vars.append(record)

    type_aliases: List[Dict[str, Any]] = []
    seen_aliases: set[str] = set()
    for contract in getattr(slither, "contracts", []) or []:
        for alias in getattr(contract, "type_aliases_declared", []) or []:
            record = _type_alias_record(alias)
            key = record.get("qualified_name") or record.get("name", "")
            if not key or key in seen_aliases:
                continue
            seen_aliases.add(key)
            type_aliases.append(record)
    for compilation_unit in getattr(slither, "compilation_units", []) or []:
        alias_map = getattr(compilation_unit, "type_aliases", {}) or {}
        aliases = alias_map.values() if isinstance(alias_map, dict) else alias_map
        for alias in aliases or []:
            record = _type_alias_record(alias)
            key = record.get("qualified_name") or record.get("name", "")
            if not key or key in seen_aliases:
                continue
            seen_aliases.add(key)
            type_aliases.append(record)

    libraries: List[Dict[str, Any]] = []
    seen_libraries: set[str] = set()
    for contract in getattr(slither, "contracts", []) or []:
        if not getattr(contract, "is_library", False):
            continue
        record = _library_record(contract)
        key = record.get("qualified_name") or record.get("name", "")
        if not key or key in seen_libraries:
            continue
        seen_libraries.add(key)
        libraries.append(record)

    events: List[Dict[str, Any]] = []
    seen_events: set[str] = set()
    for contract in getattr(slither, "contracts", []) or []:
        for event in getattr(contract, "events_declared", []) or []:
            record = _event_record(event)
            key = record.get("qualified_name") or record.get("name", "")
            if not key or key in seen_events:
                continue
            seen_events.add(key)
            events.append(record)

    interfaces: List[Dict[str, Any]] = []
    seen_interfaces: set[str] = set()
    for contract in getattr(slither, "contracts", []) or []:
        if not getattr(contract, "is_interface", False):
            continue
        record = _interface_record(contract)
        key = record.get("qualified_name") or record.get("name", "")
        if not key or key in seen_interfaces:
            continue
        seen_interfaces.add(key)
        interfaces.append(record)

    return {
        "extra_storage_vars": extra_storage_vars,
        "type_aliases": type_aliases,
        "libraries": libraries,
        "events": events,
        "interfaces": interfaces,
        "audited_contracts": audited_contracts,
    }


def _select_local_root_contracts(slither: Slither) -> List[Contract]:
    """Return deployable root contracts in a local project."""
    audited_contracts = list(_iter_audited_contracts(slither, None))
    if not audited_contracts:
        return []
    audited_set = set(audited_contracts)
    inherited: set[Contract] = set()
    for contract in audited_contracts:
        for base in getattr(contract, "inheritance", []) or []:
            if base in audited_set:
                inherited.add(base)
    roots = [contract for contract in audited_contracts if contract not in inherited]
    return roots or audited_contracts

# ----------- Main ------------------------------------------------------------

def generate_html(
    address: str,
    chain: str | None = None,
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

    metadata = _collect_render_metadata(slither, root_contracts)
    audited_contracts = metadata["audited_contracts"]
    if audited_contracts:
        title_contract = audited_contracts[0].name
    else:
        title_contract = f"{chain}:{address}"
    output_path = render_html(
        data,
        chain,
        address,
        title_contract,
        extra_storage_vars=metadata["extra_storage_vars"],
        type_aliases=metadata["type_aliases"],
        libraries=metadata["libraries"],
        events=metadata["events"],
        interfaces=metadata["interfaces"],
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


def generate_from_local(
    source_path: str | Path,
    progress_cb: ProgressCallback | None = None,
) -> List[Dict[str, Any]]:
    """Generate entry-point HTML for each deployable contract in a local project."""
    path = Path(source_path).resolve()
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if path.is_dir():
        target_label = str(path)
    else:
        target_label = path.name

    _report_progress(progress_cb, "Compiling local project", path=target_label)
    try:
        slither = Slither(str(path), skip_analyze=False)
    except Exception as exc:
        raise ValueError(f"Compilation failed for {path}: {exc}") from exc
    _report_progress(progress_cb, "Compiler analysis completed")

    root_contracts = _select_local_root_contracts(slither)
    if not root_contracts:
        raise ValueError("No deployable contracts found in project")

    project_hash = _local_project_hash(path)
    results: List[Dict[str, Any]] = []
    for contract in root_contracts:
        _report_progress(progress_cb, "Building entry point flows", contract=contract.name)
        data = build_entry_point_flows(slither, [contract], progress_cb=progress_cb)
        metadata = _collect_render_metadata(slither, [contract])

        output_address = f"{project_hash}_{contract.name}"
        output_path = render_html(
            data,
            "local",
            output_address,
            contract.name,
            extra_storage_vars=metadata["extra_storage_vars"],
            type_aliases=metadata["type_aliases"],
            libraries=metadata["libraries"],
            events=metadata["events"],
            interfaces=metadata["interfaces"],
            progress_cb=progress_cb,
        )

        contract_names = sorted({entry.get("contract", "") for entry in data if entry.get("contract")})
        results.append(
            {
                "output_path": output_path,
                "contract": contract.name,
                "contracts": contract_names,
                "entry_count": len(data),
                "target": str(path),
            }
        )

    return results


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for generating a single contract HTML file."""
    parser = argparse.ArgumentParser(
        description="Analyze a contract's entry-point execution flows.")
    parser.add_argument(
        "address_or_path",
        nargs="?",
        default=DEFAULT_ADDRESS,
        help="Contract address or local project path (default: built-in sample).",
    )
    parser.add_argument(
        "chain",
        nargs="?",
        default=None,
        help="Chain name or chain id (default: mainnet).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Analyze a local Solidity project instead of an on-chain contract.",
    )
    args = parser.parse_args(argv)

    if args.local:
        results = generate_from_local(args.address_or_path)
        rows = []
        for result in results:
            output_path = result["output_path"]
            http_url = f"http://localhost:8000/{output_path.name}"
            rows.append((result["contract"], http_url))

        name_width = max(len("Contract"), *(len(name) for name, _ in rows))
        url_width = max(len("URL"), *(len(url) for _, url in rows))

        header = f"| {'Contract'.ljust(name_width)} | {'URL'.ljust(url_width)} |"
        divider = f"|{'-' * (name_width + 2)}|{'-' * (url_width + 2)}|"

        print(header)
        print(divider)
        for name, url in rows:
            print(f"| {name.ljust(name_width)} | {url.ljust(url_width)} |")
    else:
        result = generate_html(args.address_or_path, args.chain)
        output_path = result["output_path"]
        http_url = f"http://localhost:8000/{output_path.name}"
        print(
            f"Wrote {result['entry_count']} entry point flows to {output_path} using target {result['target']}"
        )
        print(f"Open in browser: {http_url}")


if __name__ == "__main__":
    main()
