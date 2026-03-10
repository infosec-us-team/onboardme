# Custom Templates

`OnboardMe` can render generated contract-analysis data into any HTML template you provide through `TEMPLATE_PATH`. This file documents the actual interface contract between the backend and your template.

Use this document as the source of truth when:

- building a custom UI by hand
- asking an AI coding agent to generate a custom template
- debugging why a custom template rendered a blank page or malformed output

## How templating works

The backend does not use Jinja, JSX, Handlebars, or any other template engine. It does this instead:

1. Reads the template file as UTF-8 text.
2. Performs literal global `str.replace(...)` calls on a fixed set of placeholder tokens.
3. By default, writes the final HTML to `src/{chain}_{address}.html`.

Important consequences:

- There is no validation that your template contains the right placeholders.
- Missing placeholders do not raise an error; your UI will just not receive that piece of data.
- Placeholder replacement is literal and unescaped.
- Every occurrence of a placeholder token is replaced, not just the first one.
- Editing the generated `src/*.html` file is not durable. The backend overwrites it on the next render. Edit the source template instead.

## Template file requirements

These are the hard runtime requirements:

- `TEMPLATE_PATH` must point to a real file. Relative paths are resolved from the repository root.
- The template file must be valid UTF-8.
- The rendered result must be valid HTML/JS/CSS after placeholder replacement.
- If your template references local assets with relative URLs, those URLs must resolve from the generated file inside `src/`, not from the original template location.

These are the practical requirements for a useful data-driven template:

- Include `REPLACE_THIS_WITH_CONTRACT_VIEWS`.
- Include `REPLACE_THIS_WITH_DEFAULT_CONTRACT` if you need an initial selection.
- Include `REPLACE_THIS_WITH_CONTRACT_LIST` if you want to support multiple contracts in one generated page.
- Do not assume there is only one contract, only write functions, or only on-chain output.

## Output location and asset behavior

By default, the backend writes generated pages into [`src/`](/home/c/infosec_us_team_repos/onboardme/src).

Behavior to account for:

- [`serve.py`](/home/c/infosec_us_team_repos/onboardme/serve.py) serves static files from `src/`.
- [`src/hotkeys.json`](/home/c/infosec_us_team_repos/onboardme/src/hotkeys.json) is copied into the output directory on every render.
- No other assets are copied automatically.
- [`src/favicon.svg`](/home/c/infosec_us_team_repos/onboardme/src/favicon.svg) already exists for the default `src/` flow, but if you render elsewhere or add more assets, you must place them where the final HTML can reach them.

If you reference `./app.css`, `./app.js`, `./logo.png`, or similar, those files need to exist under `src/` or whatever output directory you are using.

## Supported placeholders

These are the only placeholders the backend replaces today.

| Placeholder | Replacement type | What it contains | Notes |
| --- | --- | --- | --- |
| `REPLACE_THIS_WITH_TITLE` | raw string | page title, usually the default contract name | Best used in HTML text contexts. No escaping is applied. |
| `REPLACE_THIS_WITH_CONTRACT_VIEWS` | JSON literal | full analysis payload keyed by contract name | Use directly in JavaScript, not as a quoted string. |
| `REPLACE_THIS_WITH_CONTRACT_LIST` | JSON literal | sorted array of contract names | Use directly in JavaScript, not as a quoted string. |
| `REPLACE_THIS_WITH_DEFAULT_CONTRACT` | JSON string literal | default contract name | Already JSON-encoded. Do not wrap it in quotes again. |
| `REPLACE_THIS_WITH_CHAIN` | raw string | resolved chain name | In local-project mode this is `local`. |
| `REPLACE_THIS_WITH_ADDRESS` | raw string | target address or local synthetic id | In local-project mode this is not a `0x...` address. |
| `REPLACE_THIS_WITH_SOLIDITY_VERSION` | raw string | resolved pragma label or fallback text | Often used as a display label. |
| `REPLACE_THIS_WITH_CONTRACT_NAME` | raw string | explicit contract name if passed, else the title/default contract | In the current built-in generation flow this resolves to the title/default contract. |

## Safe embedding rules

Follow these rules exactly. Most custom-template bugs come from violating one of them.

- `REPLACE_THIS_WITH_CONTRACT_VIEWS`, `REPLACE_THIS_WITH_CONTRACT_LIST`, and `REPLACE_THIS_WITH_DEFAULT_CONTRACT` are intended for JavaScript expression positions.
- Do not quote those JSON placeholders.
- Do not call `JSON.parse(...)` on those JSON placeholders unless you first convert them back to strings yourself.
- `REPLACE_THIS_WITH_TITLE`, `REPLACE_THIS_WITH_CHAIN`, `REPLACE_THIS_WITH_ADDRESS`, `REPLACE_THIS_WITH_SOLIDITY_VERSION`, and `REPLACE_THIS_WITH_CONTRACT_NAME` are raw text replacements, not escaped HTML and not escaped JavaScript strings.
- If you place raw-string placeholders inside attributes, inline scripts, inline styles, or HTML built with `innerHTML`, you are responsible for making sure the final output is still syntactically valid and safe.
- Prefer using raw-string placeholders in text nodes like `<title>...`, headings, badges, and labels.
- Avoid leaving placeholder tokens in comments, sample snippets, or hidden elements unless you want them replaced there too.

Good:

```html
<title>REPLACE_THIS_WITH_TITLE</title>
<h1>REPLACE_THIS_WITH_TITLE</h1>
<script>
  const CONTRACT_VIEWS = REPLACE_THIS_WITH_CONTRACT_VIEWS;
  const CONTRACT_LIST = REPLACE_THIS_WITH_CONTRACT_LIST;
  const DEFAULT_CONTRACT = REPLACE_THIS_WITH_DEFAULT_CONTRACT;
</script>
```

Bad:

```html
<script>
  const CONTRACT_VIEWS = "REPLACE_THIS_WITH_CONTRACT_VIEWS";
  const DEFAULT_CONTRACT = "REPLACE_THIS_WITH_DEFAULT_CONTRACT";
</script>
```

## Data model

The backend injects a `Record<string, ContractView>` under `REPLACE_THIS_WITH_CONTRACT_VIEWS`.

This is the effective schema:

```ts
type ContractViews = Record<string, ContractView>;

interface ContractView {
  entries: EntryPoint[];
  read_only_entries: EntryPoint[];
  storage_variables: StateVariable[];
  type_aliases: DeclarationRecord[];
  libraries: DeclarationRecord[];
  events: DeclarationRecord[];
  interfaces: DeclarationRecord[];
  storage_writers_index: Record<string, string[]>;
}

interface EntryPoint {
  contract: string;
  entry_point: string;
  inputs: AbiField[];
  outputs: AbiField[];
  reads_msg_sender: boolean;
  msg_sender_constrained: boolean;
  msg_sender_checks: MsgSenderCheck[];
  msg_sender_restrictions: string[];
  state_variables_read: StateVariable[];
  state_variables_written: StateVariable[];
  state_variables_read_keys: string[];
  state_variables_written_keys: string[];
  flow: FlowStep[];
}

interface AbiField {
  name: string;
  type: string;
}

interface FlowStep {
  name: string;
  kind: string;
  source: string;
  selector: string;
  cyclomatic_complexity: number;
  contract: string;
  data_dependencies?: DataDependency[];
  variable_records?: VariableRecord[];
  msg_sender_checks?: MsgSenderCheck[];
  msg_sender_constrained?: boolean;
  restrictions?: string[];
}

interface DataDependency {
  dependent: string;
  dependent_type: string;
  dependent_scope: string;
  depends_on: string;
  depends_on_type: string;
  depends_on_scope: string;
}

interface VariableRecord {
  name: string;
  contract: string;
  qualified_name: string;
  type: string;
  source: string;
  scope: string;
  is_storage: boolean;
  struct?: {
    name: string;
    source: string;
  };
  storage_base?: StateVariable | {
    name: string;
    type: string;
    scope: string;
  };
}

interface StateVariable {
  name: string;
  contract: string;
  qualified_name: string;
  type: string;
  source: string;
  is_constant: boolean;
  is_immutable: boolean;
  struct?: {
    name: string;
    source: string;
  };
}

interface DeclarationRecord {
  name: string;
  contract: string;
  qualified_name: string;
  type: string;
  source: string;
  kind: string;
}

interface MsgSenderCheck {
  kind: string;
  callable: string;
  callable_kind: string;
  location: SourceLocation;
  source: string;
  detail?: Record<string, unknown>;
}

interface SourceLocation {
  filename: string;
  lines: number[];
  start_line: number | null;
  end_line: number | null;
}
```

A few semantic notes matter when designing the UI:

- `entries` are state-changing public/external entry points.
- `read_only_entries` are `view` or `pure` public/external entry points.
- `flow[0]` is the entry point itself. Later items are modifiers and downstream calls discovered by the analyzer.
- `storage_variables` is broader than just the vars touched by one function. It is a contract-level list assembled for the active contract view.
- `storage_writers_index` maps a storage variable `qualified_name` to entry-point signatures that write it.
- `msg_sender_checks` and `msg_sender_restrictions` may be empty.
- Unknown future fields should be ignored, not treated as errors.

## Multi-contract behavior

Do not assume a generated page represents only one contract.

On on-chain targets, the backend can include multiple contract views when the verified source bundle exposes more than one relevant deployable contract. The contract names become:

- the keys of `CONTRACT_VIEWS`
- the contents of `CONTRACT_LIST`
- valid values for `DEFAULT_CONTRACT`

Recommended behavior:

- If `CONTRACT_LIST.length > 1`, provide a contract switcher.
- If `DEFAULT_CONTRACT` is missing from the current list for any reason, fall back to the first key in `CONTRACT_LIST` or `Object.keys(CONTRACT_VIEWS)`.
- When switching contracts, reload `entries`, `read_only_entries`, `storage_variables`, and the metadata arrays from the selected `ContractView`.

## Local-project behavior

If the user generates pages from a local Solidity project instead of an on-chain address, the placeholders behave differently:

- `REPLACE_THIS_WITH_CHAIN` becomes `local`.
- `REPLACE_THIS_WITH_ADDRESS` becomes a synthetic identifier in the form `{project_hash}_{contract_name}`.
- A template that hard-codes explorer links or assumes `address.startsWith("0x")` will break in local mode.

If you only want an on-chain UI, make that an explicit product choice. Do not accidentally break local mode by assuming every target is an EVM address.

## Minimal starter template

This is a safe baseline for custom work. It supports multiple contracts, both write and read-only entry points, and avoids the most common placeholder mistakes.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>REPLACE_THIS_WITH_TITLE</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0;
      padding: 24px;
      font: 16px/1.5 ui-sans-serif, system-ui, sans-serif;
      background: #0b1020;
      color: #e8edf7;
    }
    .shell { max-width: 1100px; margin: 0 auto; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    .muted { color: #97a3b6; }
    .panel {
      margin-top: 20px;
      padding: 16px;
      border: 1px solid #27324a;
      border-radius: 12px;
      background: #121a2d;
    }
    .grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }
    code { font-family: ui-monospace, SFMono-Regular, monospace; }
    ul { padding-left: 20px; }
    select {
      background: #0f1729;
      color: inherit;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 8px 10px;
    }
  </style>
</head>
<body>
  <div class="shell">
    <h1>REPLACE_THIS_WITH_TITLE</h1>
    <div class="row muted">
      <span>Chain: REPLACE_THIS_WITH_CHAIN</span>
      <span>Target: REPLACE_THIS_WITH_ADDRESS</span>
      <span>Solidity: REPLACE_THIS_WITH_SOLIDITY_VERSION</span>
    </div>

    <div class="panel">
      <label>
        Contract:
        <select id="contract-select"></select>
      </label>
    </div>

    <div id="app"></div>
  </div>

  <script>
    const CONTRACT_VIEWS = REPLACE_THIS_WITH_CONTRACT_VIEWS;
    const CONTRACT_LIST = REPLACE_THIS_WITH_CONTRACT_LIST;
    const DEFAULT_CONTRACT = REPLACE_THIS_WITH_DEFAULT_CONTRACT;

    const contractSelect = document.getElementById('contract-select');
    const app = document.getElementById('app');

    const getContractNames = () => {
      if (Array.isArray(CONTRACT_LIST) && CONTRACT_LIST.length) return CONTRACT_LIST;
      return Object.keys(CONTRACT_VIEWS || {});
    };

    let activeContract = getContractNames().includes(DEFAULT_CONTRACT)
      ? DEFAULT_CONTRACT
      : (getContractNames()[0] || '');

    const el = (tag, text) => {
      const node = document.createElement(tag);
      if (text != null) node.textContent = text;
      return node;
    };

    const renderEntryList = (title, entries) => {
      const panel = el('section');
      panel.className = 'panel';

      panel.appendChild(el('h2', title));
      if (!Array.isArray(entries) || !entries.length) {
        const empty = el('p', 'No items.');
        empty.className = 'muted';
        panel.appendChild(empty);
        return panel;
      }

      const list = el('ul');
      entries.forEach((entry) => {
        const item = el('li');
        const sig = el('code', entry.entry_point || '(unknown)');
        item.appendChild(sig);

        const details = [];
        if (Array.isArray(entry.inputs) && entry.inputs.length) {
          details.push(`inputs: ${entry.inputs.map((x) => `${x.type} ${x.name || ''}`.trim()).join(', ')}`);
        }
        if (Array.isArray(entry.outputs) && entry.outputs.length) {
          details.push(`outputs: ${entry.outputs.map((x) => `${x.type} ${x.name || ''}`.trim()).join(', ')}`);
        }
        if (details.length) {
          item.appendChild(el('div', details.join(' | ')));
        }

        list.appendChild(item);
      });
      panel.appendChild(list);
      return panel;
    };

    const render = () => {
      const view = CONTRACT_VIEWS?.[activeContract] || {};
      app.replaceChildren();

      const summary = el('section');
      summary.className = 'panel';
      summary.appendChild(el('h2', activeContract || 'No contract selected'));
      summary.appendChild(el(
        'p',
        `${(view.entries || []).length} write entry points, ${(view.read_only_entries || []).length} read-only entry points, ${(view.storage_variables || []).length} storage variables`
      ));
      app.appendChild(summary);

      const grid = el('div');
      grid.className = 'grid';
      grid.appendChild(renderEntryList('Write Entry Points', view.entries || []));
      grid.appendChild(renderEntryList('Read-Only Entry Points', view.read_only_entries || []));
      app.appendChild(grid);
    };

    getContractNames().forEach((name) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      option.selected = name === activeContract;
      contractSelect.appendChild(option);
    });

    contractSelect.addEventListener('change', (event) => {
      activeContract = event.target.value;
      render();
    });

    render();
  </script>
</body>
</html>
```

## Design guidance for custom UIs

The backend payload is rich enough to support more than a simple function list. Good custom templates usually expose some combination of:

- entry-point signatures, inputs, and outputs
- internal call flow from `entry.flow`
- storage reads and writes
- `msg.sender` restrictions from `msg_sender_checks` or `msg_sender_restrictions`
- declaration browsers for `storage_variables`, `type_aliases`, `libraries`, `events`, and `interfaces`
- contract switching when `CONTRACT_LIST` has multiple items

Recommended robustness rules:

- Treat every array as potentially empty.
- Treat every string field as potentially empty.
- Prefer `textContent` and DOM APIs over `innerHTML` when rendering backend data.
- If you use `innerHTML`, escape dynamic strings from the payload first.
- Ignore fields you do not need; do not delete or rename placeholders.
- Tolerate future additive fields in the payload.

## AI agent checklist

If you are an AI coding agent generating a custom template for this repo, treat the items below as non-negotiable:

- Keep placeholder tokens exactly spelled as documented here.
- Use `REPLACE_THIS_WITH_CONTRACT_VIEWS`, `REPLACE_THIS_WITH_CONTRACT_LIST`, and `REPLACE_THIS_WITH_DEFAULT_CONTRACT` as live JavaScript values, not strings.
- Support multi-contract pages unless the user explicitly says a single-contract template is acceptable.
- Support both `entries` and `read_only_entries`.
- Do not assume `REPLACE_THIS_WITH_ADDRESS` is always a hex address; local-project generation uses a synthetic id.
- Do not assume local relative assets live next to the template source file; they must resolve from `src/`.
- If you want hotkey help, fetch `./hotkeys.json`. The backend copies that file into the output directory automatically.
- Avoid editing generated `src/*.html` as the source of truth. Edit the template file referenced by `TEMPLATE_PATH`.

## Troubleshooting

Blank page after generation usually means one of these:

- a JSON placeholder was wrapped in quotes
- a raw string placeholder was inserted into a JavaScript or HTML context that became invalid after replacement
- your template assumes one contract but the payload contains multiple
- your template assumes an on-chain address but the page was generated from a local project
- a relative CSS/JS/image path points to the wrong place because the generated page lives in `src/`

If a backend or template change should have fixed the issue but the old behavior is still being served, stop `serve.py` and start it again before debugging further.
