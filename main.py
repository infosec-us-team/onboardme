"""Generate a JSON file describing entry-point execution flows with state variable info."""

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence
import json

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
from analysis.local_analyzer import LocalProjectAnalyzer, ProgressCallback

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

def _update_local_registry(output_path: Path, analysis_result: Dict[str, Any]):
    """Update registry with local project info"""
    registry_path = output_path.parent / "registry.json"
    
    try:
        current = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(current, list):
            current = []
    except (FileNotFoundError, json.JSONDecodeError):
        current = []
    
    # Remove existing entry for the same project
    current = [
        entry for entry in current
        if not (
            str(entry.get("address", "")).lower() == analysis_result["project_hash"].lower()
            and str(entry.get("chain", "")).lower() == "local"
        )
    ]
    
    entry = {
        "address": analysis_result["project_hash"],
        "chain": "local",
        "path": f"/{output_path.name}",
        "contracts": analysis_result["contracts"],
        "primary_contract": analysis_result["main_contract"],
        "source_dir": analysis_result["source_dir"],
        "type": "local"
    }
    
    current.append(entry)
    registry_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    

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
    output_path = render_html(
        data,
        chain,
        address,
        title_contract,
        extra_storage_vars=extra_storage_vars,
        type_aliases=type_aliases,
        libraries=libraries,
        events=events,
        interfaces=interfaces,
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

def generate_from_local(
    source_dir: str | Path,
    contract_name: str | None = None,
    output_dir: Path | None = None,
    progress_cb: ProgressCallback | None = None
) -> Dict[str, Any]:
    """
    Generate visualization HTML from local Solidity project
    
    Args:
        source_dir: Solidity project directory path
        contract_name: Main contract name (optional)
        output_dir: Output directory (default: src/)
        progress_cb: Progress callback function
        
    Returns:
        Dictionary containing generated file path and other information
    """
    source_path = Path(source_dir).resolve()
    
    # Analyze project
    analyzer = LocalProjectAnalyzer(source_path, progress_cb=progress_cb)
    analysis_result = analyzer.analyze(contract_name=contract_name)
    
    # Determine output directory
    if output_dir:
        final_output_dir = Path(output_dir).resolve()
        final_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use default src directory
        src_dir = Path(__file__).parent / "src"
        src_dir.mkdir(exist_ok=True)
        final_output_dir = src_dir
    
    # Generate HTML
    _report_progress(progress_cb, "Generating visualization page...")
    
    output_path = render_html(
        data=analysis_result["entry_points"],
        chain="local",
        address=analysis_result["project_hash"],
        title_contract=analysis_result["main_contract"],
        extra_storage_vars=analysis_result["state_variables"],
        output_dir=final_output_dir,
        progress_cb=progress_cb
    )
    
    # Update registry
    _update_local_registry(output_path, analysis_result)
    
    return {
        "output_path": output_path,
        "url": f"/{output_path.name}",
        "project_hash": analysis_result["project_hash"],
        "main_contract": analysis_result["main_contract"],
        "contracts": analysis_result["contracts"],
        "entry_count": len(analysis_result["entry_points"]),
        "source_dir": str(source_path)
    }

def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for generating a single contract HTML file."""
    parser = argparse.ArgumentParser(
        description="Analyze a contract's entry-point execution flows.")
    
    # Original arguments for on-chain analysis
    parser.add_argument(
        "address_or_path",
        nargs="?",
        default=DEFAULT_ADDRESS,
        help="Contract address or local project path (default: built-in sample).",
    )
    parser.add_argument(
        "chain_or_contract",
        nargs="?",
        default=None,
        help="Chain name or main contract name (for local projects).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Analyze local Solidity project instead of on-chain contract.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated HTML files (default: src/).",
    )
    
    args = parser.parse_args(argv)
    
    if args.local:
        # Local project analysis mode
        source_dir = Path(args.address_or_path)
        contract_name = args.chain_or_contract
        
        print(f"🔍 Analyzing local project: {source_dir}")
        if contract_name:
            print(f"📄 Specified main contract: {contract_name}")
        
        start_time = time.time()
        
        result = generate_from_local(
            source_dir=source_dir,
            contract_name=contract_name,
            output_dir=args.output_dir
        )
        
        elapsed_time = time.time() - start_time
        
        output_path = result["output_path"]
        
        # Generate both file URL and HTTP URL
        file_url = output_path.resolve().as_uri()
        http_url = f"http://127.0.0.1:8000/{output_path.name}"
        
        print(f"\n✅ Analysis completed in {elapsed_time:.1f}s")
        print(f"📋 Main contract: {result['main_contract']}")
        print(f"📦 Found {len(result['contracts'])} contracts")
        print(f"🔗 Entry points: {result['entry_count']}")
        print(f"💾 Generated file: {output_path}")
        print(f"🌐 HTTP URL: {http_url}")
        
        if not args.output_dir:
            print(f"\n💡 Tip: Run 'python server.py' and visit {http_url}")
            
    else:
        # Original on-chain analysis mode
        result = generate_html(args.address, args.chain)
        output_path = result["output_path"]
        file_url = output_path.resolve().as_uri()
        print(
            f"Wrote {result['entry_count']} entry point flows to {output_path} using target {result['target']}"
        )
        print(f"Open in browser: {file_url}")

if __name__ == "__main__":
    main()
