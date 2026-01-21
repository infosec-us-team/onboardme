"""State variable extraction helpers for Slither callables."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Union

from slither.core.declarations.function import Function
from slither.core.declarations.modifier import Modifier
from slither.core.declarations.structure import Structure
from slither.core.expressions.identifier import Identifier
from slither.core.expressions.index_access import IndexAccess
from slither.core.expressions.member_access import MemberAccess
from slither.core.solidity_types.user_defined_type import UserDefinedType
from slither.core.variables.local_variable import LocalVariable
from slither.core.variables.state_variable import StateVariable
from slither.slithir.operations.assignment import Assignment
from slither.slithir.operations.delete import Delete
from slither.slithir.operations.high_level_call import HighLevelCall
from slither.slithir.operations.internal_call import InternalCall
from slither.slithir.operations.library_call import LibraryCall
from slither.slithir.variables.reference import ReferenceVariable

from analysis.slither_extract import _source_text


def _attach_struct_info(record: Dict[str, Any], vtype: Any) -> None:
    if isinstance(vtype, UserDefinedType) and isinstance(vtype.type, Structure):
        struct = vtype.type
        record["struct"] = {
            "name": getattr(struct, "name", ""),
            "source": _source_text(struct),
        }


def _state_var_record(var: Any) -> Dict[str, Any]:
    """Return JSON-serializable info for a state variable, including its source code."""
    contract = getattr(var, "contract", None) or getattr(var, "contract_declarer", None)
    contract_name = getattr(contract, "name", "") if contract else ""
    var_name = getattr(var, "name", "")
    qualified_name = f"{contract_name}.{var_name}" if contract_name and var_name else var_name
    record: Dict[str, Any] = {
        "name": var_name,
        "contract": contract_name,
        "qualified_name": qualified_name,
        "type": str(getattr(var, "type", "")),
        "source": _source_text(var),
        "is_constant": bool(getattr(var, "is_constant", False)),
        "is_immutable": bool(getattr(var, "is_immutable", False)),
    }

    vtype = getattr(var, "type", None)
    _attach_struct_info(record, vtype)
    return record


def _local_var_record(
    var: LocalVariable,
    scope: str,
    contract: Any | None = None,
    function_name: str | None = None,
) -> Dict[str, Any]:
    contract_name = getattr(contract, "name", "") if contract else ""
    var_name = getattr(var, "name", "") or ""
    type_str = str(getattr(var, "type", "")) if var is not None else ""
    qualified_parts = [p for p in (contract_name, function_name, var_name) if p]
    qualified_name = ".".join(qualified_parts) if qualified_parts else var_name
    source = _source_text(var)
    if not source:
        init_expr = getattr(var, "expression", None)
        init_str = f" = {init_expr}" if init_expr is not None else ""
        source = f"{type_str} {var_name}{init_str}".strip()
    record: Dict[str, Any] = {
        "name": var_name,
        "contract": contract_name,
        "qualified_name": qualified_name,
        "type": type_str,
        "source": source,
        "scope": scope,
        "is_storage": bool(getattr(var, "is_storage", False)),
    }
    _attach_struct_info(record, getattr(var, "type", None))
    return record


def _collect_local_and_param_vars(item: Union[Function, Modifier]) -> List[Dict[str, Any]]:
    """Collect local and parameter variables with type/source info for tooltips."""
    if item is None:
        return []
    contract = getattr(item, "contract_declarer", None) or getattr(item, "contract", None)
    function_name = getattr(item, "name", None) or getattr(item, "full_name", None)
    params = list(getattr(item, "parameters", []) or [])
    locals_vars = list(getattr(item, "local_variables", []) or [])
    state_vars = set(getattr(contract, "state_variables", []) or []) if contract else set()

    param_set = {p for p in params if isinstance(p, LocalVariable)}
    alias_map = _build_storage_aliases(item, param_set, state_vars) if param_set else {}

    records: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for param in params:
        if not isinstance(param, LocalVariable):
            continue
        record = _local_var_record(param, "parameter", contract, function_name)
        key = record.get("name") or ""
        if key and key not in seen:
            seen.add(key)
            records.append(record)

    for local in locals_vars:
        if not isinstance(local, LocalVariable):
            continue
        scope = "storage_local" if getattr(local, "is_storage", False) else "local"
        record = _local_var_record(local, scope, contract, function_name)
        base = alias_map.get(local)
        if isinstance(base, StateVariable):
            record["storage_base"] = _state_var_record(base)
        elif isinstance(base, LocalVariable):
            record["storage_base"] = {
                "name": getattr(base, "name", ""),
                "type": str(getattr(base, "type", "")),
                "scope": "parameter" if base in param_set else "local",
            }
        key = record.get("name") or ""
        if key and key not in seen:
            seen.add(key)
            records.append(record)

    return records


def _collect_state_vars(item: Union[Function, Modifier], kind: str) -> List[Dict[str, Any]]:
    """
    Collect state variables read or written by this callable.
    Uses the most comprehensive Slither attributes available.
    kind: "read" | "written"
    """
    attr_all = f"all_state_variables_{kind}"
    attr_recursive = f"state_variables_{kind}_recursive"
    attr_direct = f"state_variables_{kind}"

    vars_all = getattr(item, attr_all, None)
    vars_rec = getattr(item, attr_recursive, None)
    vars_direct = getattr(item, attr_direct, None)

    # Slither versions may expose these as properties or methods; normalize by calling if callable.
    vars_all = vars_all() if callable(vars_all) else vars_all
    vars_rec = vars_rec() if callable(vars_rec) else vars_rec
    vars_direct = vars_direct() if callable(vars_direct) else vars_direct

    candidates = (
        vars_all
        if vars_all is not None
        else vars_rec
        if vars_rec is not None
        else vars_direct
        if vars_direct is not None
        else []
    )

    unique: Set[StateVariable] = {
        v for v in candidates if isinstance(v, StateVariable)
    }

    # Expand with node-level info to catch struct member writes/reads.
    for node in getattr(item, "nodes", []) or []:
        for attr in (f"state_variables_{kind}_recursive", f"state_variables_{kind}"):
            node_vars = getattr(node, attr, None)
            node_vars = node_vars() if callable(node_vars) else node_vars
            for var in node_vars or []:
                if isinstance(var, StateVariable):
                    unique.add(var)

        node_vars = getattr(node, f"variables_{kind}", None)
        node_vars = node_vars() if callable(node_vars) else node_vars
        for var in node_vars or []:
            if isinstance(var, StateVariable):
                unique.add(var)
                continue
            if getattr(var, "is_state_variable", False):
                unique.add(var)
                continue
            try:
                state_var = getattr(var, "state_variable", None)
            except Exception:
                state_var = None
            if isinstance(state_var, StateVariable):
                unique.add(state_var)

    return [_state_var_record(v) for v in sorted(unique, key=lambda v: v.name or "")]


def _base_storage_source(
    expr: Any, params: Set[LocalVariable], state_vars: Set[StateVariable]
) -> Any:
    """Resolve a storage alias expression back to a parameter or state variable."""
    if isinstance(expr, Identifier):
        val = expr.value
        if isinstance(val, StateVariable):
            return val
        if isinstance(val, LocalVariable) and val in params:
            return val
        return None
    if isinstance(expr, IndexAccess):
        return _base_storage_source(expr.expression_left, params, state_vars)
    if isinstance(expr, MemberAccess):
        return _base_storage_source(expr.expression, params, state_vars)
    return None


def _build_storage_aliases(
    item: Union[Function, Modifier],
    params: Set[LocalVariable],
    state_vars: Set[StateVariable],
) -> Dict[LocalVariable, Any]:
    """Map storage local variables to their base parameter or state variable."""
    aliases: Dict[LocalVariable, Any] = {}
    for var in getattr(item, "local_variables", []) or []:
        if not isinstance(var, LocalVariable):
            continue
        if not getattr(var, "is_storage", False):
            continue
        expr = getattr(var, "expression", None)
        if not expr:
            continue
        base = _base_storage_source(expr, params, state_vars)
        if base is not None:
            aliases[var] = base
    return aliases


def _resolve_written_base(
    var: Any,
    params: Set[LocalVariable],
    alias_map: Dict[LocalVariable, Any],
    name_map: Dict[str, Any],
) -> Any:
    """Resolve a written variable to its base storage param/state var."""
    if isinstance(var, ReferenceVariable):
        var = var.points_to_origin
        if var is None:
            return None
    if isinstance(var, StateVariable):
        return var
    if isinstance(var, LocalVariable):
        if var in params:
            return var
        if var in alias_map:
            return alias_map[var]
    name = getattr(var, "name", "") or ""
    if name:
        base = name_map.get(name)
        if base is not None:
            return base
        if "." in name:
            prefix = name.split(".", 1)[0]
            base = name_map.get(prefix)
            if base is not None:
                return base
    return None


def _resolve_written_lvalue(
    lvalue: Any,
    params: Set[LocalVariable],
    alias_map: Dict[LocalVariable, Any],
    name_map: Dict[str, Any],
) -> Any:
    """Resolve an IR lvalue (possibly a ReferenceVariable) to a base variable."""
    if isinstance(lvalue, ReferenceVariable):
        base = lvalue.points_to_origin
    else:
        base = lvalue
    return _resolve_written_base(base, params, alias_map, name_map)


def _storage_params_written(
    item: Union[Function, Modifier],
    cache: Dict[Any, Set[int]],
    visiting: Set[Any],
) -> Set[int]:
    """Return indices of storage parameters written within a callable (propagated)."""
    if item in cache:
        return cache[item]
    if item in visiting:
        return set()
    visiting.add(item)

    params = list(getattr(item, "parameters", []) or [])
    param_set = {p for p in params if isinstance(p, LocalVariable)}
    param_indices = {p: idx for idx, p in enumerate(params)}
    state_vars = set(getattr(getattr(item, "contract", None), "state_variables", []) or [])
    alias_map = _build_storage_aliases(item, param_set, state_vars)
    name_map: Dict[str, Any] = {p.name: p for p in param_set if p.name}
    for local, base in alias_map.items():
        if local.name:
            name_map[local.name] = base

    written_params: Set[LocalVariable] = set()

    for node in getattr(item, "nodes", []) or []:
        node_vars = getattr(node, "variables_written", None)
        node_vars = node_vars() if callable(node_vars) else node_vars
        for var in node_vars or []:
            base = _resolve_written_base(var, param_set, alias_map, name_map)
            if (
                isinstance(base, LocalVariable)
                and base in param_set
                and getattr(base, "is_storage", False)
            ):
                written_params.add(base)

    operations = getattr(item, "all_slithir_operations", None)
    if callable(operations):
        for ir in operations():
            if isinstance(ir, (Assignment, Delete)):
                lvalue = getattr(ir, "lvalue", None)
                base = _resolve_written_lvalue(lvalue, param_set, alias_map, name_map)
                if (
                    isinstance(base, LocalVariable)
                    and base in param_set
                    and getattr(base, "is_storage", False)
                ):
                    written_params.add(base)
                continue
            if isinstance(ir, (InternalCall, HighLevelCall, LibraryCall)):
                callee = getattr(ir, "function", None)
                if not isinstance(callee, Function):
                    continue
                callee_written = _storage_params_written(callee, cache, visiting)
                if not callee_written:
                    continue
                for idx in callee_written:
                    if idx >= len(ir.arguments):
                        continue
                    arg = ir.arguments[idx]
                    base = _resolve_written_base(arg, param_set, alias_map, name_map)
                    if (
                        isinstance(base, LocalVariable)
                        and base in param_set
                        and getattr(base, "is_storage", False)
                    ):
                        written_params.add(base)

    visiting.remove(item)
    written_indices = {param_indices[p] for p in written_params if p in param_indices}
    cache[item] = written_indices
    return written_indices


def _collect_state_vars_via_storage_params(entry_point: Function) -> Set[StateVariable]:
    """Resolve state vars written through storage parameters in called functions."""
    params = set(getattr(entry_point, "parameters", []) or [])
    state_vars = set(getattr(getattr(entry_point, "contract", None), "state_variables", []) or [])
    alias_map = _build_storage_aliases(entry_point, params, state_vars)
    name_map: Dict[str, Any] = {p.name: p for p in params if getattr(p, "name", None)}
    for local, base in alias_map.items():
        if local.name:
            name_map[local.name] = base

    written_state_vars: Set[StateVariable] = set()
    cache: Dict[Any, Set[int]] = {}
    visiting: Set[Any] = set()

    operations = getattr(entry_point, "all_slithir_operations", None)
    if not callable(operations):
        return written_state_vars

    for ir in operations():
        if not isinstance(ir, (InternalCall, HighLevelCall, LibraryCall)):
            continue
        callee = getattr(ir, "function", None)
        if not isinstance(callee, Function):
            continue
        callee_written = _storage_params_written(callee, cache, visiting)
        if not callee_written:
            continue
        for idx in callee_written:
            if idx >= len(ir.arguments):
                continue
            arg = ir.arguments[idx]
            if isinstance(arg, StateVariable):
                written_state_vars.add(arg)
                continue
            base = _resolve_written_base(arg, params, alias_map, name_map)
            if isinstance(base, StateVariable):
                written_state_vars.add(base)

    return written_state_vars
