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


def render_html(
    data: List[Dict[str, Any]],
    chain: str,
    address: str,
    title_contract: str,
    output_dir: Path | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Render and write the entry-point HTML, returning the output path."""
    _report_progress(progress_cb, "Indexing storage variables")
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

    storage_vars = sorted(
        storage_vars_map.values(),
        key=lambda v: ((v.get("qualified_name") or v.get("name") or "").lower()),
    )

    template_path = Path("template.html")
    _report_progress(progress_cb, "Rendering template")
    template_html = template_path.read_text(encoding="utf-8")

    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    inner_json = data_json[1:-1].strip() if len(data_json) > 2 else ""
    storage_vars_json = json.dumps(storage_vars, ensure_ascii=False, indent=2)
    storage_vars_inner = (
        storage_vars_json[1:-1].strip() if len(storage_vars_json) > 2 else ""
    )
    writers_index_json = json.dumps(writers_index, ensure_ascii=False, indent=2)

    filled_html = template_html.replace(
        "REPLACE_THIS_WITH_TITLE", title_contract
    ).replace(
        "REPLACE_THIS_WITH_ENTRY_POINTS_DATA", inner_json
    ).replace(
        "REPLACE_THIS_WITH_STORAGE_VARIABLES", storage_vars_inner
    ).replace(
        "REPLACE_THIS_WITH_STORAGE_WRITERS_INDEX", writers_index_json
    ).replace(
        "REPLACE_THIS_WITH_CHAIN", chain
    ).replace(
        "REPLACE_THIS_WITH_ADDRESS", address
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
