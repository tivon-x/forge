# Forge

Forge is a provider-neutral Python coding-agent CLI. It reads and edits a
project, runs explicitly requested local commands, streams model events, and
stores inspectable JSONL sessions under `~/.forge/`.

Forge has two deliberate foundations:

- **Pi** is the main architectural source. Forge follows Pi's separation of a
  reusable agent harness, a coding-session environment, and UI/event consumers.
- **Tau** is the Python product implementation baseline used for the tools,
  session storage, provider configuration, and Textual TUI. Forge is a Python
  derivative, not a claim that Tau is a complete or line-by-line Pi replica.

The production agent runtime is **LangChain Python**: `AgentHarness` is a
small Forge event/concurrency facade around the official
`langchain.agents.create_agent` graph and its `astream` API. The old provider
loop remains only as a compatibility module for focused adapter tests; the CLI
and harness do not use it as a second tool-calling loop.

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

- `forge_agent`: provider-neutral messages, events, tools, session primitives,
  and the LangChain-backed harness.
- `forge_ai`: model-provider adapters and deterministic fake providers.
- `forge_coding`: project context discovery, safe coding tools, JSONL sessions,
  provider configuration, CLI renderers, and the optional Textual TUI.

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
