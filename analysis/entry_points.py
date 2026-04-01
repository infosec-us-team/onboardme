"""Helpers for enumerating entry points without duplicate inherited callables."""

from __future__ import annotations

from typing import Any, Callable, List, Sequence, Set


def entry_point_identity(entry_point: Any) -> tuple[Any, ...]:
    """
    Build a stable per-run identity for an entry point.

    The same inherited callable can surface while iterating both a root contract and one of its
    concrete bases. Prefer a semantic key based on declarer + source mapping so we can drop the
    duplicate without losing distinct overloads or overrides.
    """
    source_mapping = getattr(entry_point, "source_mapping", None)
    filename = ""
    start = None
    length = None
    if source_mapping is not None:
        filename_obj = getattr(source_mapping, "filename", None)
        filename = (
            getattr(filename_obj, "absolute", "")
            or getattr(filename_obj, "relative", "")
            or getattr(filename_obj, "short", "")
            or str(filename_obj or "")
        )
        raw_start = getattr(source_mapping, "start", None)
        raw_length = getattr(source_mapping, "length", None)
        start = raw_start if isinstance(raw_start, int) else None
        length = raw_length if isinstance(raw_length, int) else None

    declarer = getattr(entry_point, "contract_declarer", None) or getattr(entry_point, "contract", None)
    declarer_name = getattr(declarer, "name", "") or ""
    signature = (
        getattr(entry_point, "solidity_signature", None)
        or getattr(entry_point, "full_name", None)
        or getattr(entry_point, "name", None)
        or "<unknown>"
    )

    if filename or start is not None or length is not None:
        return (
            declarer_name,
            signature,
            filename,
            start,
            length,
        )

    return (
        declarer_name,
        signature,
        id(entry_point),
    )


def collect_unique_entry_points(
    contracts: Sequence[Any],
    entry_points_fn: Callable[[Any], List[Any]],
) -> List[tuple[Any, List[Any]]]:
    """Collect entry points once even if inherited functions surface on multiple contracts."""
    seen: Set[tuple[Any, ...]] = set()
    result: List[tuple[Any, List[Any]]] = []
    for contract in contracts:
        unique_for_contract: List[Any] = []
        for entry_point in entry_points_fn(contract):
            key = entry_point_identity(entry_point)
            if key in seen:
                continue
            seen.add(key)
            unique_for_contract.append(entry_point)
        result.append((contract, unique_for_contract))
    return result
