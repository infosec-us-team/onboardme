"""HTML rendering helpers for entry-point analysis output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

ProgressCallback = Callable[[str, Dict[str, Any] | None], None]


def _report_progress(callback: ProgressCallback | None, message: str, **meta: Any) -> None:
    """Best-effort progress reporting."""
    if not callback:
        return
    try:
        callback(message, meta or None)
    except Exception:
        pass


def _build_storage_payload(
    data: List[Dict[str, Any]],
    extra_storage_vars: List[Dict[str, Any]] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Compute storage variables list + writers index for a contract view."""
    storage_vars_map: Dict[str, Dict[str, Any]] = {}
    writers_index: Dict[str, List[str]] = {}
    for entry in data:
        for var in entry.get("state_variables_read", []) + entry.get("state_variables_written", []):
            qualified = var.get("qualified_name") or var.get("name", "")
            if not qualified:
                continue
            storage_vars_map.setdefault(qualified, var)
        for qualified in entry.get("state_variables_written_keys", []) or []:
            if not qualified:
                continue
            writers = writers_index.setdefault(qualified, [])
            entry_name = entry.get("entry_point", "")
            if entry_name and entry_name not in writers:
                writers.append(entry_name)
    for var in extra_storage_vars or []:
        qualified = var.get("qualified_name") or var.get("name", "")
        if not qualified:
            continue
        storage_vars_map.setdefault(qualified, var)

    storage_vars = sorted(
        storage_vars_map.values(),
        key=lambda v: ((v.get("qualified_name") or v.get("name") or "").lower()),
    )
    return storage_vars, writers_index


def render_html(
    contract_views: Dict[str, Dict[str, Any]],
    default_contract: str,
    chain: str,
    address: str,
    title_contract: str,
    solidity_version: str | None = None,
    contract_name: str | None = None,
    output_dir: Path | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Render and write the entry-point HTML, returning the output path."""
    _report_progress(progress_cb, "Indexing storage variables")
    rendered_views: Dict[str, Dict[str, Any]] = {}
    for name, view in contract_views.items():
        data = view.get("entries", [])
        storage_vars, writers_index = _build_storage_payload(
            data, view.get("extra_storage_vars")
        )
        rendered_views[name] = {
            "entries": data,
            "storage_variables": storage_vars,
            "type_aliases": view.get("type_aliases") or [],
            "libraries": view.get("libraries") or [],
            "events": view.get("events") or [],
            "interfaces": view.get("interfaces") or [],
            "storage_writers_index": writers_index,
        }

    contract_list = sorted(rendered_views.keys())

    template_path = Path("template.html")
    _report_progress(progress_cb, "Rendering template")
    template_html = template_path.read_text(encoding="utf-8")

    views_json = json.dumps(rendered_views, ensure_ascii=False, indent=2)
    contract_list_json = json.dumps(contract_list, ensure_ascii=False, indent=2)
    default_contract_json = json.dumps(default_contract, ensure_ascii=False)

    filled_html = template_html.replace(
        "REPLACE_THIS_WITH_TITLE", title_contract
    ).replace(
        "REPLACE_THIS_WITH_CONTRACT_VIEWS", views_json
    ).replace(
        "REPLACE_THIS_WITH_CONTRACT_LIST", contract_list_json
    ).replace(
        "REPLACE_THIS_WITH_DEFAULT_CONTRACT", default_contract_json
    ).replace(
        "REPLACE_THIS_WITH_CHAIN", chain
    ).replace(
        "REPLACE_THIS_WITH_ADDRESS", address
    ).replace(
        "REPLACE_THIS_WITH_SOLIDITY_VERSION",
        solidity_version or "SOLIDITY VERSION UNKNOWN",
    ).replace(
        "REPLACE_THIS_WITH_CONTRACT_NAME",
        contract_name or title_contract,
    )

    output_dir = output_dir or Path("src")
    output_dir.mkdir(parents=True, exist_ok=True)
    _report_progress(progress_cb, "Writing output")
    repo_root = Path(__file__).resolve().parent.parent
    hotkeys_src = repo_root / "src" / "hotkeys.json"
    hotkeys_dst = output_dir / "hotkeys.json"
    if hotkeys_src.exists() and not hotkeys_dst.exists():
        hotkeys_dst.write_text(hotkeys_src.read_text(encoding="utf-8"), encoding="utf-8")
    output_path = output_dir / f"{chain}_{address}.html"
    # Always rebuild the dashboard so template changes propagate immediately.
    # write_text already overwrites; avoid deleting first to preserve last-good file if generation fails mid-run.
    output_path.write_text(filled_html, encoding="utf-8")
    _report_progress(progress_cb, "Output file written", path=str(output_path))
    return output_path
