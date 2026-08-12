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
workspace-bounded read/write/edit/shell tools. Forge sessions include the
official LangChain Todo middleware: the model can maintain a `write_todos` plan,
the TUI shows it above the composer, and `/todos` prints the full active list.
Interactive TUI turns can also call `ask_user_question`; Forge pauses the same
LangChain execution, opens a structured questionnaire, and resumes it with a
paired `ToolMessage`. Print mode is deliberately non-interactive and does not
register that tool. Shell execution is not a sandbox; it runs with the
operating-system user's permissions.

Todo and questionnaire controls are keyboard-first:

- `Ctrl+Shift+T` collapses or expands the Todo panel.
- `/todos` opens the complete active Todo list, including rows hidden by the
  panel height limit.
- In a questionnaire, `Up` and `Down` move through options, `Space` toggles a
  multi-select option, and `Enter` confirms the current question.
- `Tab` and `Shift+Tab` move between questions without discarding draft text or
  selections. Choose `Type something.` to enter a custom answer, and press
  `Esc` to cancel the questionnaire without leaving an unmatched tool call.

Todo snapshots are restored from the active JSONL branch and survive session
resume and compaction. A pending questionnaire is intentionally process-local:
answering, cancelling, closing, or switching sessions resolves the pending tool
call and releases its temporary in-memory checkpoint.

Forge also exposes a built-in `task` tool so the model can delegate one
bounded job to a fresh subagent context. Describe the delegation naturally,
for example: “Use a scout to trace the authentication flow, then explain the
relevant files.” The built-in roles are:

- `scout`: inspect code and collect evidence without file-editing tools.
- `worker`: implement one clearly scoped change and run targeted checks.
- `reviewer`: independently review existing work without file-editing tools.

Projects can add or override roles with
`.forge/agents/<name>/AGENT.md`; user-wide roles use
`~/.forge/agents/<name>/AGENT.md`. The Markdown frontmatter declares a short
`description`, an optional comma-separated tool allowlist, and optional
per-role model-call/result-size limits. Project roles override user roles,
which override the built-ins. Invalid higher-priority files are reported and
leave the lower-priority role available. Use `/agents` to inspect the active
registry and `/reload` to apply file changes while the session is idle.
Custom tool lists can only reduce the tools already enabled for the session,
and `task` is never available to a child. A prompt that describes a role as
read-only is not a sandbox; in particular, `bash` still has the operating-system
user's permissions.

Each call runs synchronously and in process, reuses the session's current
provider, model, project context, and safe coding tools, and returns only the
final result to the parent agent. Calls are serialized per session, limited to
eight child model calls and a 50 KiB UTF-8 result, and cannot delegate again.
Forge does not create child sessions or persist raw child transcripts. It
stores a bounded display trace on the active parent-session branch: visible
human/assistant text, tool names and success/error status only. Raw tool
arguments, tool output, artifacts, thinking, and provider metadata are not
stored. When the provider reports standard token usage, Forge also records the
aggregate input/output/total counts. The TUI shows the task inline, updates its
current activity, and uses `Ctrl+O` to expand or collapse the final result and
trace; `Esc` cancels the whole current prompt.

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
