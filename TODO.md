# Refactor Plan for main.py

## Goals
- Reduce `main.py` size while preserving behavior and output.
- Remove dead code and redundant logic.
- Improve structure for testing and future changes.
- Keep runtime behavior stable (no output format changes unless explicitly called out).

## Ground Rules
- Fix syntax issues first (broken f-strings) so the file runs before refactoring.
- Make changes in small, reversible steps with clear checkpoints.
- Avoid any behavioral changes to Slither analysis or output JSON/HTML structure unless noted.

## Phase 0: Baseline & Safety
1) Snapshot current behavior (manual notes)
   - Identify CLI usage: `python main.py [address] [chain]`.
   - Note expected outputs: console messages, HTML file location, JSON embedded in template.
2) Quick sanity readthrough
   - Confirm flow: `main -> generate_html -> Slither -> build_entry_point_flows`.
   - Identify where global state is used (`slither_object`).
3) Fix syntax issues that prevent execution
   - `main.py:224` broken f-string for `qualified_name`.
   - `main.py:1391` broken f-string in `print(...)`.

## Phase 1: Dead Code & Redundancy Cleanup
4) Remove dead functions that are never referenced
   - `_branch_contains_revert` (main.py:542)
   - `_branch_subgraph_reverts` (main.py:577)
   - `_collect_reverting_branches` (main.py:597)
   - `_require_assert_kind` (main.py:618)
   - `_is_require_assert_ir` (main.py:638)
   - `_collect_require_asserts` (main.py:653)
   - `_called_functions` (main.py:771)
   - Rationale: only referenced in commented blocks; no runtime use.
5) Remove unused variables/blocks
   - `all_functions` list in `build_entry_point_flows` (main.py:1066)
   - Commented graph exploration block (around main.py:1076) if no longer needed.
6) Remove redundant post-processing loop
   - Merge the second `flow` loop that only handles modifiers into the main loop.
   - Ensure data dependencies are computed once per flow item.

## Phase 2: Structural Refactor (Module Split)
7) Extract constants and environment helpers
   - Create `src/analysis/slither_env.py`:
     - `DEFAULT_CHAIN`, `SUPPORTED_NETWORK_V2`, `CHAIN_ID_TO_NAME`, `DEFAULT_ADDRESS`
     - `_load_dotenv`, `_normalize_chain`, `_resolve_chain_and_address`, `_normalize_address`
   - Ensure module has no side effects; keep `_DOTENV_LOADED` scoped there.
8) Extract general Slither helpers
   - Create `src/analysis/slither_extract.py`:
     - `_display_name`, `_source_text`, `_function_selector`, `_contract_key`
     - Any small helpers used across modules.
9) Extract state-variable analysis helpers
   - Create `src/analysis/state_vars.py`:
     - `_state_var_record`, `_collect_state_vars`, `_collect_state_vars_via_storage_params`
     - `_storage_params_written`, `_build_storage_aliases`, `_resolve_written_base`, `_resolve_written_lvalue`, `_base_storage_source`
   - Keep dependencies and typing consistent.
10) Extract traversal/walk logic
   - Create `src/analysis/flow_walk.py`:
     - `_resolve_unimplemented_call`, `_walk_callable`, `_entry_points_for`, `build_entry_point_flows`
     - Ensure `build_entry_point_flows` accepts `slither_object` as an explicit parameter, not global.
11) Extract render/output logic
   - Create `src/render.py`:
     - HTML template reading and `filled_html` construction
     - Output writing (including `hotkeys.json` copy)
     - Keep I/O behavior identical.

## Phase 3: Global State & Dependency Injection
12) Replace global `slither_object` with a context object
   - Create a small `AnalyzerContext` dataclass or simple container:
     - `slither: Slither`
     - `progress_cb: ProgressCallback | None`
   - Pass context into functions that need Slither.
13) Update functions to accept explicit dependencies
   - `build_entry_point_flows(slither, root_contracts, progress_cb)`
   - `_resolve_root_contracts(slither, address, chain)`
   - `_iter_audited_contracts(slither, root_contracts)`
   - Ensure no residual references to global state.

## Phase 4: Behavior Checks & Cleanup
14) Ensure import paths are correct
   - Update `main.py` to import from new modules.
   - Ensure no circular imports (helpers should not import `main`).
15) Verify output consistency
   - Run a small sample (same address/chain) and compare:
     - Entry count
     - HTML output file name and location
     - Embedded JSON sizes/structure
16) Remove now-unused imports
   - Clean up unused imports in `main.py` and new modules.
17) Update module docstrings
   - Keep concise docstrings at module top-level for clarity.

## Phase 5: Optional Improvements (Non-breaking)
18) Convert `ProgressCallback` to a protocol or type alias in a shared module
   - Keep as callable signature; re-export for consistency.
19) Add small tests (if desired later)
   - Unit test `_normalize_chain`, `_resolve_chain_and_address`
   - Unit test `_state_var_record` using mocked Slither objects
   - Snapshot test for `generate_html` to ensure template replacements are stable

## Execution Order Checklist
- [x] Fix broken f-strings.
- [x] Remove dead code and redundant loops.
- [ ] Extract modules (env helpers -> extract helpers -> state vars -> flow walk -> render).
- [ ] Replace global `slither_object` with explicit context.
- [ ] Update imports & clean unused.
- [ ] Run a minimal sample to confirm output unchanged.
