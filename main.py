"""Generate a JSON file describing entry-point execution flows with state variable info."""

import argparse
import json
import logging
import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Union

from slither.core.declarations import Contract, FunctionContract
from slither.core.declarations.function import Function
from slither.core.declarations.modifier import Modifier
from slither.core.declarations.solidity_variables import SolidityFunction
from slither.core.declarations.structure import Structure
from slither.core.solidity_types.user_defined_type import UserDefinedType
from slither.core.cfg.node import NodeType
from slither.slithir.operations import SolidityCall
from slither.slither import Slither
from slither.utils.tests_pattern import is_test_file
from slither.utils.code_complexity import compute_cyclomatic_complexity
from slither.utils.function import get_function_id
from slither.analyses.data_dependency.data_dependency import is_dependent
from slither.core.declarations.solidity_variables import SolidityVariableComposed
from slither.core.variables.state_variable import StateVariable
from slither.core.variables.local_variable import LocalVariable

# Silence Slither logging to keep console output clean.
logging.disable(logging.CRITICAL)

DEFAULT_CHAIN = "mainnet"
SUPPORTED_NETWORK_V2: Dict[str, str] = {
    "mainnet": "1",
    "sepolia": "11155111",
    "holesky": "17000",
    "hoodi": "560048",
    "bsc": "56",
    "testnet.bsc": "97",
    "poly": "137",
    "amoy.poly": "80002",
    "base": "8453",
    "sepolia.base": "84532",
    "arbi": "42161",
    "nova.arbi": "42170",
    "sepolia.arbi": "421614",
    "linea": "59144",
    "sepolia.linea": "59141",
    "blast": "81457",
    "sepolia.blast": "168587773",
    "optim": "10",
    "sepolia.optim": "11155420",
    "avax": "43114",
    "testnet.avax": "43113",
    "bttc": "199",
    "testnet.bttc": "1029",
    "celo": "42220",
    "sepolia.celo": "11142220",
    "frax": "252",
    "hoodi.frax": "2523",
    "gno": "100",
    "mantle": "5000",
    "sepolia.mantle": "5003",
    "memecore": "43521",
    "moonbeam": "1284",
    "moonriver": "1285",
    "moonbase": "1287",
    "opbnb": "204",
    "testnet.opbnb": "5611",
    "scroll": "534352",
    "sepolia.scroll": "534351",
    "taiko": "167000",
    "hoodi.taiko": "167013",
    "era.zksync": "324",
    "sepoliaera.zksync": "300",
    "xdc": "50",
    "testnet.xdc": "51",
    "apechain": "33139",
    "curtis.apechain": "33111",
    "world": "480",
    "sepolia.world": "4801",
    "sophon": "50104",
    "testnet.sophon": "531050104",
    "sonic": "146",
    "testnet.sonic": "14601",
    "unichain": "130",
    "sepolia.unichain": "1301",
    "abstract": "2741",
    "sepolia.abstract": "11124",
    "berachain": "80094",
    "testnet.berachain": "80069",
    "swellchain": "1923",
    "testnet.swellchain": "1924",
    "testnet.monad": "10143",
    "hyperevm": "999",
    "katana": "747474",
    "bokuto.katana": "737373",
    "sei": "1329",
    "testnet.sei": "1328",
}
CHAIN_ID_TO_NAME: Dict[str, str] = {
    chain_id: name for name, chain_id in SUPPORTED_NETWORK_V2.items()
}
DEFAULT_ADDRESS = "0x80ac24aA929eaF5013f6436cdA2a7ba190f5Cc0b"
_DOTENV_LOADED = False


def _load_dotenv(path: Path | None = None) -> None:
    """Load key=value pairs from .env into environment if not already set."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    env_path = path or Path(__file__).parent / ".env"
    if not env_path.exists():
        _DOTENV_LOADED = True
        return

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    finally:
        _DOTENV_LOADED = True


# Set at runtime once CLI args are parsed.
slither_object: Slither | None = None

# ----------- Helper utilities -------------------------------------------------


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
    filename = getattr(sm.filename, "absolute",
                       None) or getattr(sm, "filename", None)
    if not filename:
        return ""
    try:
        with open(filename, "rb") as f:
            data = f.read()
        return data[sm.start: sm.start + sm.length].decode("utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return ""


def _function_selector(item: Any) -> Union[str, None]:
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


def _state_var_record(var: Any) -> Dict[str, Any]:
    """Return JSON-serializable info for a state variable, including its source code."""
    contract = getattr(var, "contract", None) or getattr(
        var, "contract_declarer", None)
    contract_name = getattr(contract, "name", "") if contract else ""
    var_name = getattr(var, "name", "")
    qualified_name = f"{contract_name}.{
        var_name}" if contract_name and var_name else var_name
    record: Dict[str, Any] = {
        "name": var_name,
        "contract": contract_name,
        "qualified_name": qualified_name,
        "type": str(getattr(var, "type", "")),
        "source": _source_text(var),
    }

    vtype = getattr(var, "type", None)
    if isinstance(vtype, UserDefinedType) and isinstance(vtype.type, Structure):
        struct = vtype.type
        record["struct"] = {
            "name": getattr(struct, "name", ""),
            "source": _source_text(struct),
        }
    return record


def _var_display(var: Any) -> str:
    """Readable label for variables in dependency output."""
    if isinstance(var, SolidityVariableComposed):
        return str(var)
    contract = getattr(var, "contract", None) or getattr(
        var, "contract_declarer", None)
    prefix = f"{contract.name}." if contract and getattr(
        var, "name", None) else ""
    return f"{prefix}{getattr(var, 'name', str(var))}"


def _normalize_chain(chain: str) -> str:
    """Return canonical chain name from chain name or chain id."""
    raw = (chain or "").strip().lower()
    if not raw:
        raise ValueError("chain is required")
    if raw in SUPPORTED_NETWORK_V2:
        return raw
    if raw.isdigit():
        mapped = CHAIN_ID_TO_NAME.get(raw)
        if mapped:
            return mapped
    raise ValueError(f"Unsupported chain '{chain}'.")


def _resolve_chain_and_address(address: str, chain: str | None) -> tuple[str, str]:
    """Resolve chain name and address, supporting prefix 'chain:address'."""
    if not address:
        raise ValueError("address is required")
    addr = address.strip()
    resolved_chain: str | None = None

    if ":" in addr:
        prefix, addr_part = addr.split(":", 1)
        prefix = prefix.strip()
        addr = addr_part.strip()
        if not addr:
            raise ValueError("address is required")
        resolved_chain = _normalize_chain(prefix)
        if chain:
            chain_name = _normalize_chain(chain)
            if chain_name != resolved_chain:
                raise ValueError(
                    f"Conflicting chain values '{prefix}' and '{chain}'."
                )

    if resolved_chain is None:
        resolved_chain = _normalize_chain(chain) if chain else DEFAULT_CHAIN

    return addr, resolved_chain


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

    unique = {v for v in candidates if hasattr(v, "name")}
    return [_state_var_record(v) for v in sorted(unique, key=lambda v: v.name or "")]


def _branch_contains_revert(node) -> bool:
    """Return True if a CFG node represents or calls a revert/throw/require."""
    # Direct node kinds
    if node.type in (NodeType.THROW, getattr(NodeType, "REVERT", NodeType.THROW)):
        return True

    # IR operations
    for ir in getattr(node, "irs", []):
        if "revert" in ir.__class__.__name__.lower():
            return True
        fn = getattr(ir, "function", None)
        if fn and getattr(fn, "name", "") == "revert":
            return True

    # Internal call list (alternative representation)
    if any(
        getattr(ir, "function", None) and getattr(
            ir.function, "name", "") == "revert"
        for ir in getattr(node, "internal_calls", [])
    ):
        return True

    # Fallback: use Slither helper if present (captures require/assert)
    contains_req = getattr(node, "contains_require_or_assert", None)
    if callable(contains_req) and contains_req():
        return True

    # Text heuristic as last resort
    expr = getattr(node, "expression", None)
    if expr and "revert" in str(expr).lower():
        return True

    return False


def _branch_subgraph_reverts(start_node) -> bool:
    """
    Depth-first walk from the branch's first node until merge, checking for revert.
    Stops when reaching ENDIF/ENDLOOP nodes to keep traversal local to the branch.
    """
    stack = [start_node]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if _branch_contains_revert(node):
            return True
        if node.type in (NodeType.ENDIF, NodeType.ENDLOOP):
            continue
        stack.extend(getattr(node, "sons", []))
    return False


def _collect_reverting_branches(callable_item: Union[Function, Modifier]) -> List[Dict[str, Any]]:
    """
    Find IF nodes whose true/false branch reverts anywhere within that branch.
    """
    findings: List[Dict[str, Any]] = []
    for node in getattr(callable_item, "nodes", []):
        if node.type != NodeType.IF or not getattr(node, "sons", []):
            continue
        branches = list(node.sons[:2])  # true and maybe false
        branch_labels = ["true", "false"]
        for idx, branch in enumerate(branches):
            if _branch_subgraph_reverts(branch):
                findings.append(
                    {
                        "branch": branch_labels[idx],
                        "expression": str(getattr(node, "expression", "")).strip()
                    }
                )
    return findings


def _require_assert_kind(node: Any) -> str:
    """Classify a node containing a require/assert."""
    # Prefer explicit IR function names if available.
    for ir in getattr(node, "internal_calls", []):
        fn = getattr(ir, "function", None)
        name = getattr(fn, "name", "")
        low = name.lower()
        if low.startswith("require"):
            return "require"
        if low.startswith("assert"):
            return "assert"

    expr = str(getattr(node, "expression", "")).lower()
    if "require" in expr:
        return "require"
    if "assert" in expr:
        return "assert"
    return "require_or_assert"


def _is_require_assert_ir(ir: Any) -> bool:
    """Return True if a SlithIR op represents a require/assert call."""
    if not isinstance(ir, SolidityCall):
        return False
    func = getattr(ir, "function", None)
    if isinstance(func, SolidityFunction):
        sig = getattr(func, "signature", None) or getattr(
            func, "full_name", None) or getattr(func, "name", "")
        if sig.startswith("require") or sig.startswith("assert"):
            return True
    # Fallback by name substring
    name = getattr(func, "name", "") if func else ""
    return name.startswith("require") or name.startswith("assert")


def _collect_require_asserts(callable_item: Union[Function, Modifier]) -> List[Dict[str, Any]]:
    """
    Capture every require/assert encountered within a callable.
    Uses SlithIR (SolidityCall) when available, and falls back to node.contains_require_or_assert.
    """
    findings: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    # 1) IR-level detection (most precise).
    operations = getattr(callable_item, "all_slithir_operations", None)
    if callable(operations):
        for ir in operations():

            if not _is_require_assert_ir(ir):
                continue

            node = getattr(ir, "node", None)
            expr = str(getattr(node, "expression", "")).strip()
            key = f"ir::{expr}"
            if key in seen:
                continue
            seen.add(key)
            kind = _require_assert_kind(node) if node else "require_or_assert"
            findings.append({"kind": kind, "expression": expr})

    # 2) Node-level fallback (covers cases where IR isn't generated as expected).
    for node in getattr(callable_item, "nodes", []):
        # 2a) Inspect IRs attached to the node.
        for ir in getattr(node, "irs", []):
            if not _is_require_assert_ir(ir):
                continue
            expr = str(getattr(node, "expression", "")).strip()
            key = f"nodeir::{expr}"
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                {"kind": _require_assert_kind(node), "expression": expr})

        contains = getattr(node, "contains_require_or_assert", None)
        has_req = contains() if callable(contains) else bool(contains)
        if not has_req:
            continue
        expr = str(getattr(node, "expression", "")).strip()
        key = f"node::{expr}"
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            {"kind": _require_assert_kind(node), "expression": expr})

    return findings


def _collect_data_dependencies(callable_item: Union[Function, Modifier]) -> List[Dict[str, str]]:
    """
    Return dependency edges (dependent -> depends_on) for variables reachable from the entry point.
    Includes state variables and locals as dependents; parameters/state/locals/Solidity vars as sources.
    """
    contract = getattr(callable_item, "contract", None)
    state_vars = list(getattr(contract, "state_variables", [])
                      ) if contract else []
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

    return edges


def _called_functions(fn: Function) -> Set[Function]:
    """Return all functions a callable invokes (high-level, internal, library)."""
    called = set()
    # high_level_calls and library_calls may be tuples (node, function) or call objects.

    def _extract(call_obj):
        if isinstance(call_obj, tuple) and len(call_obj) >= 2:
            return call_obj[1]
        return getattr(call_obj, "function", None)

    called.update(filter(None, (_extract(c)
                  for c in getattr(fn, "high_level_calls", []))))
    called.update(filter(None, (_extract(c)
                  for c in getattr(fn, "library_calls", []))))
    called.update(getattr(fn, "internal_calls", []))
    return {c for c in called if c}


def _iter_audited_contracts() -> Sequence[Contract]:
    """Yield concrete, non-test contracts to inspect."""
    if slither_object is None:
        raise RuntimeError("Slither instance is not initialized.")
    return sorted(
        (
            contract
            for contract in slither_object.contracts_derived
            if not contract.is_test
            and not contract.is_from_dependency()
            and not is_test_file(Path(contract.source_mapping.filename.absolute))
            and not contract.is_interface
            and not contract.is_library
            and not contract.is_abstract
        ),
        key=lambda contract: contract.name,
    )


# ----------- Traversal --------------------------------------------------------

def _resolve_unimplemented_call(
    caller: Union[Function, Modifier],
    target: Union[Function, Modifier, None],
    root_contract: Contract | None,
) -> Union[Function, Modifier, None]:
    """Resolve unimplemented interface/abstract calls to the first implemented override."""
    if target is None or not isinstance(target, FunctionContract):
        return target
    if not hasattr(target, "is_implemented") or target.is_implemented:
        return target

    signature = target.full_name
    caller_contract = getattr(caller, "contract_declarer", None) or getattr(caller, "contract", None)

    # First, resolve using the caller's inheritance (super linearization).
    if caller_contract is not None:
        for base in getattr(caller_contract, "inheritance", []):
            for fn in getattr(base, "functions_declared", []):
                if fn.full_name == signature and getattr(fn, "is_implemented", True):
                    return fn

    # If still unresolved and the caller is not the entry contract, search the entry contract overrides.
    if root_contract is not None and root_contract is not caller_contract:
        search_contracts: List[Contract] = [root_contract]
        search_contracts.extend(getattr(root_contract, "inheritance", []))
        for base in search_contracts:
            for fn in getattr(base, "functions_declared", []):
                if fn.full_name == signature and getattr(fn, "is_implemented", True):
                    return fn

    return target


def _walk_callable(
    item: Union[Function, Modifier],
    visited: Set[str],
    collected: List[Dict[str, Any]],
    # reverting_branches: List[Dict[str, Any]],
    # require_asserts: List[Dict[str, Any]],
    callable_index: Dict[str, Union[Function, Modifier]],
    root_contract: Contract,
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
            "cyclomatic_complexity": compute_cyclomatic_complexity(item) if hasattr(item, "nodes") else 1,
            "contract": getattr(getattr(item, "contract_declarer", None), "name", None)
            or getattr(getattr(item, "contract", None), "name", None)
            or "Unknown",
        }
    )
    callable_index[name] = item

    # Record reverting branches in this callable (if any).
    # reverting_branches.extend(_collect_reverting_branches(item))
    # Record require/assert statements in this callable.
    # require_asserts.extend(_collect_require_asserts(item))

    # Visit modifiers first (they execute before the function body).
    for modifier in getattr(item, "modifiers", []):
        if isinstance(modifier, Modifier):
            _walk_callable(
                modifier,
                visited,
                collected,
                # reverting_branches,
                # require_asserts,
                callable_index,
                root_contract,
            )

    CALL_ATTRS = (
        "internal_calls",
        "high_level_calls",
        "library_calls",
        "solidity_calls",
    )

    all_calls = chain.from_iterable(
        getattr(item, attr, []) for attr in CALL_ATTRS
    )

    for call in all_calls:
        target = getattr(call, "function", None)
        target = _resolve_unimplemented_call(item, target, root_contract)
        _walk_callable(
            target,
            visited,
            collected,
            # reverting_branches,
            # require_asserts,
            callable_index,
            root_contract,
        )

# ----------- Entry-point processing ------------------------------------------


def _entry_points_for(contract: Contract) -> List[Function]:
    """Return state-modifying public/external entry points for a contract."""
    return [
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


def build_entry_point_flows() -> List[Dict[str, Any]]:
    """Assemble structured data for every entry point and its execution flow."""
    if slither_object is None:
        raise RuntimeError("Slither instance is not initialized.")

    entries: List[Dict[str, Any]] = []
    reported_unimplemented: Set[str] = set()
    all_functions: List[Function] = [
        fn
        for contract in slither_object.contracts
        for fn in getattr(contract, "all_functions_called", [])
        if isinstance(fn, Function)
    ]

    # This is how to create a graph of connections between functions
    # for fn in all_functions:
    #    if len(fn.all_reachable_from_functions) < 1:
    #        continue
    #    all_reachable_from_functions = fn.all_reachable_from_functions
    #    for reachable_from in all_reachable_from_functions:
    #        print(f"fn {_display_name(fn)} can be reached from {
    #              _display_name(reachable_from)}")

    for contract in _iter_audited_contracts():
        for entry_point in _entry_points_for(contract):
            visited: Set[str] = set()
            flow: List[Dict[str, Any]] = []
            # reverting_branches: List[Dict[str, Any]] = []
            # require_asserts: List[Dict[str, Any]] = []
            callable_index: Dict[str, Union[Function, Modifier]] = {}

            reads_msg_sender = any(
                v.name == "msg.sender" for v in entry_point.all_solidity_variables_read())

            state_vars_read = _collect_state_vars(entry_point, "read")
            state_vars_written = _collect_state_vars(entry_point, "written")
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
                # reverting_branches,
                # require_asserts,
                callable_index,
                contract,
            )

            # Deduplicate reverting branches by branch side + expression.
            # seen_rb = set()
            # unique_reverts = []
            # for rb in reverting_branches:
            #    key = (rb.get("branch"), rb.get("expression"))
            #    if key in seen_rb:
            #        continue
            #    seen_rb.add(key)
            #    unique_reverts.append(rb)

            # Deduplicate require/assert statements by kind + expression.
            # seen_req_global = set()
            # unique_require_asserts = []
            # for ra in require_asserts:
            #    key = (ra.get("kind"), ra.get("expression"))
            #    if key in seen_req_global:
            #        continue
            #    seen_req_global.add(key)
            #    unique_require_asserts.append(ra)

            for item in flow:
                target_fn = callable_index.get(item.get("name"))
                if (
                    isinstance(target_fn, Function)
                    and hasattr(target_fn, "is_implemented")
                    and not target_fn.is_implemented
                ):
                    if item["name"] not in reported_unimplemented:
                        print(f"Function not implemented: {item['name']}")
                        reported_unimplemented.add(item["name"])
                # Capture data dependencies for every function in the flow.
                item["data_dependencies"] = _collect_data_dependencies(
                    target_fn)

            # Modifiers can also contain data dependencies; include them when possible.
            for item in flow:
                target_mod = callable_index.get(item.get("name"))
                if isinstance(target_mod, Modifier):
                    item["data_dependencies"] = _collect_data_dependencies(
                        target_mod)

            entries.append(
                {
                    "contract": contract.name,
                    "entry_point": _display_name(entry_point),
                    "reads_msg_sender": reads_msg_sender,
                    "state_variables_read": state_vars_read,
                    "state_variables_written": state_vars_written,
                    "state_variables_read_keys": state_vars_read_keys,
                    "state_variables_written_keys": state_vars_written_keys,
                    # "reverting_if_branches": unique_reverts,
                    # "require_assert_statements": unique_require_asserts,
                    # "data_dependencies": _collect_data_dependencies(entry_point),
                    "flow": flow,
                }
            )

    return entries


# ----------- Main ------------------------------------------------------------

def generate_html(address: str, chain: str | None = None, output_dir: Path | None = None):
    """Generate the entry-point HTML for the given contract and return metadata."""

    _load_dotenv()
    address, chain = _resolve_chain_and_address(address, chain)
    target = f"{chain}:{address}"
    global slither_object
    slither_object = Slither(target, skip_analyze=False)

    data = build_entry_point_flows()

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
        key=lambda v: (
            (v.get("qualified_name") or v.get("name") or "").lower()),
    )

    template_path = Path("template.html")
    template_html = template_path.read_text(encoding="utf-8")

    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    inner_json = data_json[1:-1].strip() if len(data_json) > 2 else ""
    storage_vars_json = json.dumps(storage_vars, ensure_ascii=False, indent=2)
    storage_vars_inner = (
        storage_vars_json[1:-1].strip() if len(storage_vars_json) > 2 else ""
    )
    writers_index_json = json.dumps(
        writers_index, ensure_ascii=False, indent=2
    )
    audited_contracts = list(_iter_audited_contracts())
    if audited_contracts:
        title_contract = audited_contracts[0].name
    else:
        title_contract = f"{chain}:{address}"

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
    hotkeys_src = Path(__file__).parent / "src" / "hotkeys.json"
    hotkeys_dst = output_dir / "hotkeys.json"
    if hotkeys_src.exists() and not hotkeys_dst.exists():
        hotkeys_dst.write_text(hotkeys_src.read_text(encoding="utf-8"), encoding="utf-8")
    output_path = output_dir / f"{chain}_{address}.html"
    # Always rebuild the dashboard so template changes propagate immediately.
    # write_text already overwrites; avoid deleting first to preserve last-good file if generation fails mid-run.
    output_path.write_text(filled_html, encoding="utf-8")

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
        f"Wrote {result['entry_count']} entry point flows to {
            output_path} using target {result['target']}"
    )
    print(f"Open in browser: {file_url}")


if __name__ == "__main__":
    main()
