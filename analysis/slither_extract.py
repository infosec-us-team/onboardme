"""Lightweight extraction helpers for Slither entities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from slither.core.declarations.function import Function
from slither.core.declarations.modifier import Modifier
from slither.utils.function import get_function_id


def _display_name(item: Union[Function, Modifier]) -> str:
    """Readable name with declarer contract prefix."""
    declarer = getattr(item, "contract_declarer", None)
    if declarer and hasattr(item, "full_name"):
        return f"{declarer.name}.{item.full_name}"
    if declarer and hasattr(item, "name"):
        return f"{declarer.name}.{item.name}"
    return getattr(item, "full_name", getattr(item, "name", "<unknown>"))


def _source_text(item: Any) -> str:
    """Extract Solidity source text for a Slither element, if available."""
    sm = getattr(item, "source_mapping", None)
    if not sm:
        return ""
    content = getattr(sm, "content", None)
    if content:
        return content
    # Fallback: slice by byte offsets if content is not populated.
    filename = getattr(sm.filename, "absolute", None) or getattr(sm, "filename", None)
    if not filename:
        return ""
    try:
        data = Path(filename).read_bytes()
        return data[sm.start: sm.start + sm.length].decode("utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return ""


def _function_selector(item: Any) -> str | None:
    """
    Return the 4-byte function selector as hex string (0x........) when available.
    Applies to functions (public/external/internal) that expose solidity_signature.
    """
    sig = getattr(item, "solidity_signature", None)
    if not sig:
        return None
    try:
        fid = get_function_id(sig)
        return f"{fid:#0{10}x}"  # 0x + 8 hex chars
    except Exception:
        return None


def _contract_key(contract: Any) -> tuple[str, str]:
    """Stable contract identity key (name + source file)."""
    sm = getattr(contract, "source_mapping", None)
    filename = ""
    if sm and getattr(sm, "filename", None):
        filename = getattr(sm.filename, "absolute", "") or str(sm.filename)
    return (contract.name, filename)
