"""Flow traversal and dependency extraction helpers."""

from __future__ import annotations

import re
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Union

from slither.analyses.data_dependency.data_dependency import is_dependent
from slither.core.declarations import Contract, FunctionContract
from slither.core.declarations.function import Function
from slither.core.declarations.modifier import Modifier
from slither.core.declarations.solidity_variables import SolidityFunction, SolidityVariableComposed
from slither.core.variables.local_variable import LocalVariable
from slither.core.variables.state_variable import StateVariable
from slither.slithir.operations.high_level_call import HighLevelCall
from slither.slithir.operations.internal_call import InternalCall
from slither.slithir.operations.library_call import LibraryCall
from slither.slither import Slither
from slither.utils.code_complexity import compute_cyclomatic_complexity
from slither.utils.tests_pattern import is_test_file

from analysis.slither_extract import (
    _contract_key,
    _display_name,
    _function_selector,
    _source_text,
)
from analysis.state_vars import (
    _collect_state_vars,
    _collect_state_vars_via_storage_params,
    _state_var_record,
)

ProgressCallback = Callable[[str, Dict[str, Any] | None], None]


def _report_progress(callback: ProgressCallback | None, message: str, **meta: Any) -> None:
    """Best-effort progress reporting."""
    if not callback:
        return
    try:
        callback(message, meta or None)
    except Exception:
        pass


def _var_display(var: Any) -> str:
    """Readable label for variables in dependency output."""
    if isinstance(var, SolidityVariableComposed):
        return str(var)
    contract = getattr(var, "contract", None) or getattr(var, "contract_declarer", None)
    prefix = f"{contract.name}." if contract and getattr(var, "name", None) else ""
    return f"{prefix}{getattr(var, 'name', str(var))}"


def _collect_data_dependencies(callable_item: Union[Function, Modifier]) -> List[Dict[str, str]]:
    """
    Return dependency edges (dependent -> depends_on) for variables reachable from the entry point.
    Includes state variables and locals as dependents; parameters/state/locals/Solidity vars as sources.
    """
    contract = getattr(callable_item, "contract", None)
    state_vars = list(getattr(contract, "state_variables", [])) if contract else []
    params = list(getattr(callable_item, "parameters", []))
    locals_vars = list(getattr(callable_item, "local_variables", []))
    solidity_sources = [
        SolidityVariableComposed("msg.sender"),
        SolidityVariableComposed("msg.value"),
        SolidityVariableComposed("tx.origin"),
    ]

    param_set = set(params)

    def _var_scope(var: Any) -> str:
        """Classify variable storage scope."""
        if isinstance(var, SolidityVariableComposed):
            return "solidity_builtin"
        if isinstance(var, StateVariable):
            return "state"
        if isinstance(var, LocalVariable):
            return "parameter" if var in param_set else "local"
        # Fallback using best-effort attributes
        if getattr(var, "is_state_variable", False):
            return "state"
        return "unknown"

    dependents = state_vars + locals_vars
    sources = state_vars + params + locals_vars + solidity_sources

    edges: List[Dict[str, str]] = []
    seen: Set[tuple] = set()

    for dest in dependents:
        for src in sources:
            if dest is src:
                continue
            try:
                if is_dependent(dest, src, callable_item):
                    key = (_var_display(dest), _var_display(src))
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append(
                        {
                            "dependent": _var_display(dest),
                            "dependent_type": str(getattr(dest, "type", "")),
                            "dependent_scope": _var_scope(dest),
                            "depends_on": _var_display(src),
                            "depends_on_type": str(getattr(src, "type", "")),
                            "depends_on_scope": _var_scope(src),
                        }
                    )
            except Exception:
                # Best-effort: skip problematic dependency checks without aborting whole analysis.
                continue

    return sorted(
        edges,
        key=lambda edge: (
            edge.get("dependent", ""),
            edge.get("depends_on", ""),
            edge.get("dependent_type", ""),
            edge.get("depends_on_type", ""),
            edge.get("dependent_scope", ""),
            edge.get("depends_on_scope", ""),
        ),
    )


def _resolve_unimplemented_call(
    caller: Union[Function, Modifier],
    target: Union[Function, Modifier, None],
    root_contract: Contract | None,
    all_contracts: Sequence[Contract] | None = None,
) -> Union[Function, Modifier, None]:
    """Resolve unimplemented interface/abstract calls to the first implemented override."""
    if target is None:
        return target
    if not hasattr(target, "is_implemented") or target.is_implemented:
        return target

    target_name = getattr(target, "name", None)
    target_signature = getattr(target, "solidity_signature", None)
    target_full_name = getattr(target, "full_name", None)
    signature = target_signature or target_full_name or target_name
    if not signature and not target_name:
        return target
    caller_contract = getattr(caller, "contract_declarer", None) or getattr(caller, "contract", None)
    target_contract = getattr(target, "contract_declarer", None) or getattr(target, "contract", None)

    def _param_types(fn: Any) -> List[str]:
        params = getattr(fn, "parameters", None) or []
        return [str(getattr(p, "type", "")) for p in params]

    target_param_types = _param_types(target)

    def _matches_signature(fn: Function) -> bool:
        fn_sig = getattr(fn, "solidity_signature", None)
        if target_signature and fn_sig:
            return fn_sig == target_signature
        fn_full_name = getattr(fn, "full_name", None)
        if target_full_name and fn_full_name:
            return fn_full_name == target_full_name
        fn_name = getattr(fn, "name", None)
        if target_name and fn_name and fn_name != target_name:
            return False
        if target_param_types:
            return _param_types(fn) == target_param_types
        return True

    def _find_implemented(contract: Contract | None) -> Function | None:
        if contract is None:
            return None
        candidates = list(getattr(contract, "functions_declared", []) or [])
        for fn in getattr(contract, "functions", []) or []:
            if fn not in candidates:
                candidates.append(fn)
        for fn in candidates:
            if not _matches_signature(fn):
                continue
            if getattr(fn, "is_implemented", True):
                return fn
        return None

    def _find_descendant_override() -> Function | None:
        if all_contracts is None or target_contract is None:
            return None
        for contract in all_contracts:
            if contract is target_contract:
                continue
            inheritance = getattr(contract, "inheritance", []) or []
            if target_contract not in inheritance:
                continue
            resolved = _find_implemented(contract)
            if resolved is not None:
                return resolved
        return None

    # First, resolve using the main (root) contract, then walk its parents.
    if root_contract is not None:
        search_contracts: List[Contract] = [root_contract]
        search_contracts.extend(getattr(root_contract, "inheritance", []))
        for base in search_contracts:
            resolved = _find_implemented(base)
            if resolved is not None:
                return resolved

    # If no root contract is provided, fall back to the caller's contract then parents.
    if root_contract is None and caller_contract is not None:
        resolved = _find_implemented(caller_contract)
        if resolved is not None:
            return resolved
        for base in getattr(caller_contract, "inheritance", []):
            resolved = _find_implemented(base)
            if resolved is not None:
                return resolved

    # Finally, try any derived contracts in the compilation unit.
    resolved = _find_descendant_override()
    if resolved is not None:
        return resolved

    return target


def _call_target(call_obj: Any) -> Union[Function, Modifier, None]:
    candidate = None
    if isinstance(call_obj, tuple) and len(call_obj) >= 2:
        candidate = call_obj[1]
    elif isinstance(call_obj, (InternalCall, HighLevelCall, LibraryCall)):
        candidate = getattr(call_obj, "function", None)
    elif isinstance(call_obj, (Function, Modifier)):
        candidate = call_obj
    else:
        candidate = getattr(call_obj, "function", None)
    if isinstance(candidate, (InternalCall, HighLevelCall, LibraryCall)):
        candidate = getattr(candidate, "function", None)
    if isinstance(candidate, (Function, Modifier)):
        return candidate
    return None


def _call_sort_key(call_obj: Any) -> tuple[str, str]:
    target = _call_target(call_obj)
    if target is not None:
        return ("target", _display_name(target))
    return ("raw", f"{call_obj.__class__.__name__}:{call_obj}")


def _walk_callable(
    item: Union[Function, Modifier],
    visited: Set[str],
    collected: List[Dict[str, Any]],
    callable_index: Dict[str, Union[Function, Modifier]],
    root_contract: Contract,
    all_contracts: Sequence[Contract] | None = None,
) -> None:
    """
    Record this callable and recursively visit everything it can execute:
    - Modifiers on the callable
    - Internal calls
    - External high-level calls
    Solidity native functions (e.g., mload, revert) are skipped.
    """
    if item is None:
        return

    name = _display_name(item)
    if name in visited:
        return
    visited.add(name)

    if isinstance(item, SolidityFunction):
        return

    collected.append(
        {
            "name": name,
            "kind": item.__class__.__name__.lower(),
            "source": _source_text(item),
            "selector": _function_selector(item),
            "cyclomatic_complexity": compute_cyclomatic_complexity(item)
            if hasattr(item, "nodes")
            else 1,
            "contract": getattr(getattr(item, "contract_declarer", None), "name", None)
            or getattr(getattr(item, "contract", None), "name", None)
            or "Unknown",
        }
    )
    callable_index[name] = item

    # Visit modifiers first (they execute before the function body).
    for modifier in getattr(item, "modifiers", []):
        if isinstance(modifier, Modifier):
            _walk_callable(
                modifier,
                visited,
                collected,
                callable_index,
                root_contract,
                all_contracts,
            )

    CALL_ATTRS = (
        "internal_calls",
        "high_level_calls",
        "library_calls",
        "solidity_calls",
    )

    all_calls = list(chain.from_iterable(getattr(item, attr, []) for attr in CALL_ATTRS))
    operations = getattr(item, "all_slithir_operations", None)
    if callable(operations):
        for ir in operations():
            if isinstance(ir, (InternalCall, HighLevelCall, LibraryCall)):
                all_calls.append(ir)

    # Fallback: scan source for internal/private calls Slither may miss.
    caller_contract = getattr(item, "contract_declarer", None) or getattr(item, "contract", None)
    if caller_contract is not None:
        candidates = [caller_contract]
        candidates.extend(getattr(caller_contract, "inheritance", []) or [])
        fn_by_name: Dict[str, Function] = {}
        for contract in candidates:
            for fn in getattr(contract, "functions_declared", []) or []:
                name = getattr(fn, "name", None)
                if name and name.startswith("_"):
                    fn_by_name.setdefault(name, fn)
        source_text = _source_text(item)
        if source_text:
            for match in re.finditer(r"\b(_[A-Za-z0-9_]*)\s*\(", source_text):
                fallback_fn = fn_by_name.get(match.group(1))
                if fallback_fn is not None:
                    all_calls.append(fallback_fn)

    def _is_interface_target(target_item: Union[Function, Modifier, None]) -> bool:
        if target_item is None:
            return False
        contract = getattr(target_item, "contract_declarer", None) or getattr(
            target_item, "contract", None
        )
        return bool(contract is not None and getattr(contract, "is_interface", False))

    for call in sorted(all_calls, key=_call_sort_key):
        target = _call_target(call)
        target = _resolve_unimplemented_call(item, target, root_contract, all_contracts)
        if _is_interface_target(target):
            continue
        _walk_callable(
            target,
            visited,
            collected,
            callable_index,
            root_contract,
            all_contracts,
        )


def _entry_points_for(contract: Contract) -> List[Function]:
    """Return state-modifying public/external entry points for a contract."""
    candidates = [
        function
        for function in contract.functions
        if function.visibility in ["public", "external"]
        and isinstance(function, FunctionContract)
        and not function.is_constructor
        and not function.view
        and not function.pure
        and not function.is_shadowed
        and not (hasattr(function, "is_implemented") and not function.is_implemented)
    ]
    return sorted(candidates, key=_display_name)


def _iter_audited_contracts(
    slither: Slither,
    root_contracts: Sequence[Contract] | None = None,
) -> Sequence[Contract]:
    """Yield concrete, non-test contracts to inspect."""
    allowed_keys: Set[tuple[str, str]] | None = None
    if root_contracts:
        allowed_keys = set()
        for root in root_contracts:
            allowed_keys.add(_contract_key(root))
            for base in getattr(root, "inheritance", []):
                allowed_keys.add(_contract_key(base))

    return sorted(
        (
            contract
            for contract in slither.contracts_derived
            if not contract.is_test
            and not contract.is_from_dependency()
            and not is_test_file(Path(contract.source_mapping.filename.absolute))
            and not contract.is_interface
            and not contract.is_library
            and not contract.is_abstract
            and (allowed_keys is None or _contract_key(contract) in allowed_keys)
        ),
        key=lambda contract: contract.name,
    )


def build_entry_point_flows(
    slither: Slither,
    root_contracts: Sequence[Contract] | None = None,
    progress_cb: ProgressCallback | None = None,
) -> List[Dict[str, Any]]:
    """Assemble structured data for every entry point and its execution flow."""
    entries: List[Dict[str, Any]] = []
    reported_unimplemented: Set[str] = set()
    all_contracts = list(getattr(slither, "contracts", []) or [])

    audited_contracts = list(_iter_audited_contracts(slither, root_contracts))
    _report_progress(
        progress_cb,
        "Collecting audited contracts",
        count=len(audited_contracts),
    )
    total_entries = sum(len(_entry_points_for(contract)) for contract in audited_contracts)
    entry_index = 0

    for contract_index, contract in enumerate(audited_contracts, start=1):
        _report_progress(
            progress_cb,
            "Scanning contract",
            contract=contract.name,
            index=contract_index,
            total=len(audited_contracts),
        )
        for entry_point in _entry_points_for(contract):
            entry_index += 1
            _report_progress(
                progress_cb,
                "Analyzing entry point",
                entry_point=_display_name(entry_point),
                contract=contract.name,
                index=entry_index,
                total=total_entries,
            )
            visited: Set[str] = set()
            flow: List[Dict[str, Any]] = []
            callable_index: Dict[str, Union[Function, Modifier]] = {}

            reads_msg_sender = any(
                v.name == "msg.sender" for v in entry_point.all_solidity_variables_read()
            )

            state_vars_read = _collect_state_vars(entry_point, "read")
            state_vars_written = _collect_state_vars(entry_point, "written")
            extra_written = _collect_state_vars_via_storage_params(entry_point)
            if extra_written:
                existing = {
                    v.get("qualified_name") or v.get("name", "")
                    for v in state_vars_written
                }
                for var in sorted(extra_written, key=lambda v: v.name or ""):
                    record = _state_var_record(var)
                    key = record.get("qualified_name") or record.get("name", "")
                    if key and key not in existing:
                        state_vars_written.append(record)
                        existing.add(key)
            state_vars_read_keys = [
                v.get("qualified_name") or v.get("name", "")
                for v in state_vars_read
            ]
            state_vars_written_keys = [
                v.get("qualified_name") or v.get("name", "")
                for v in state_vars_written
            ]

            _walk_callable(
                entry_point,
                visited,
                flow,
                callable_index,
                contract,
                all_contracts,
            )

            for item in flow:
                target_fn = callable_index.get(item.get("name"))
                if (
                    isinstance(target_fn, Function)
                    and hasattr(target_fn, "is_implemented")
                    and not target_fn.is_implemented
                ):
                    target_contract = getattr(target_fn, "contract_declarer", None) or getattr(
                        target_fn, "contract", None
                    )
                    if target_contract is not None and getattr(target_contract, "is_library", False):
                        continue
                    if item["name"] not in reported_unimplemented:
                        print(f"Function not implemented: {item['name']}")
                        reported_unimplemented.add(item["name"])
                # Capture data dependencies for every function in the flow.
                item["data_dependencies"] = _collect_data_dependencies(target_fn)

            _report_progress(
                progress_cb,
                "Entry point complete",
                entry_point=_display_name(entry_point),
                contract=contract.name,
                index=entry_index,
                total=total_entries,
            )

            entries.append(
                {
                    "contract": contract.name,
                    "entry_point": _display_name(entry_point),
                    "reads_msg_sender": reads_msg_sender,
                    "state_variables_read": state_vars_read,
                    "state_variables_written": state_vars_written,
                    "state_variables_read_keys": state_vars_read_keys,
                    "state_variables_written_keys": state_vars_written_keys,
                    "flow": flow,
                }
            )

    return entries
