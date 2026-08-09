# Forge

Forge is a provider-neutral Python coding-agent CLI. It reads and edits a
project, runs explicitly requested local commands, streams model events, and
stores inspectable JSONL sessions under `~/.forge/`.

Forge is distributed as one wheel containing three Python packages:
`forge_agent` (runtime), `forge_coding` (coding domain), and `forge_cli`
(CLI, renderers, and Textual TUI). The package boundaries are dependency
boundaries: the runtime never imports the coding or presentation layers, and
the coding layer never imports the CLI layer.

Forge has two deliberate foundations:

- **Pi** is the main architectural source. Forge follows Pi's separation of a
  reusable agent harness, a coding-session environment, and UI/event consumers.
- **Tau** is the Python product implementation baseline used for the tools,
  session storage, provider configuration, and Textual TUI. Forge is a Python
  derivative, not a claim that Tau is a complete or line-by-line Pi replica.

The production agent runtime is **LangChain Python**: `AgentHarness` is a
small Forge event/concurrency facade around the official
`langchain.agents.create_agent` graph and its
`astream_events(version="v3")` stream. Providers are constructed as native
`BaseChatModel` instances and coding tools as native `StructuredTool`
instances. The legacy provider protocol layer has been removed; there is no
second tool-calling loop.

## Development

Forge requires Python 3.12 or newer and uses `uv`.

```bash
uv sync --dev
uv run forge --help
uv run pytest
uv run ruff check src tests
uv run mypy
```

The command is also available through the project environment:

```bash
uv run forge --version
uv run forge -p "summarize this repository"
```

Model requests require the provider credentials configured by Forge (for
example `OPENAI_API_KEY`). Default tests use deterministic fake models and do
not access the network. Do not put credentials in this repository.

Forge is not currently published on PyPI. The optional startup version check is
disabled by default; set `FORGE_ENABLE_UPDATE_CHECK=1` only after configuring a
real Forge package release channel.

## What is included

- `forge_agent`: LangChain-native runtime, Forge UI events, session primitives,
  and the harness facade.
- `forge_coding`: project context discovery, safe coding tools, JSONL sessions,
  and provider configuration.
- `forge_cli`: the Typer entry point, print/transcript renderers, pure shared
  formatting helpers, and the optional Textual TUI.

The CLI supports one-shot print mode, JSON event output, interactive sessions,
session resume/export, slash commands, project `AGENTS.md` discovery, and
workspace-bounded read/write/edit/shell tools. Shell execution is not a
sandbox; it runs with the operating-system user's permissions.

## Attribution

Forge is independently maintained in this repository and is released under
the MIT License. It is inspired by and derives architectural lessons from:

- [Pi](https://github.com/earendil-works/pi), the primary TypeScript
  architecture and the ongoing learning reference for Forge improvements.
- [Tau](https://github.com/huggingface/tau), whose MIT-licensed Python
  implementation is the product baseline for Forge's coding tools, sessions,
  provider configuration, and TUI. Tau's MIT license is retained in
  [`LICENSE`](LICENSE).

Neither upstream project is presented as a Forge release, and Forge does not
use the Tau package name or Tau configuration directory at runtime.
