# Forge Agent Instructions

## Project scope

Forge is a single-distribution, Python 3.12+ provider-neutral coding-agent
CLI. The distribution contains three Python packages: `src/forge_agent`,
`src/forge_coding`, and `src/forge_cli`. Keep the project runnable with `uv`
and do not add a
second language runtime, monorepo, or heavyweight task framework without an
explicit decision.

Pi is the primary architecture and learning source. Tau is the Python product
baseline and attribution source. Keep those references in `README.md` and
`NOTICE`; do not use Tau as an active package, CLI, or configuration name.

## Architecture boundaries

- `forge_agent` uses LangChain Core messages, `BaseChatModel`, and `BaseTool` as
  the runtime facts. It owns the outer `AgentHarness`, session primitives, and
  Forge product events, but does not mirror LangChain message/model/tool state.
  It must not import `forge_coding`, `forge_cli`, UI libraries, filesystem
  helpers, shell commands, or provider SDKs.
- The production agent/tool loop is LangChain Python's official
  `langchain.agents.create_agent` plus `astream_events(version="v3")`.
  `AgentHarness` is only the outer transcript, queue, cancellation, and
  Forge-event facade. There is no second provider/tool loop; the legacy
  protocol layer has been removed entirely.
- Production provider construction lives in `forge_coding.providers.runtime`
  and returns `BaseChatModel` directly. `forge_ai` no longer exists.
- `forge_coding` owns project context, safe tools, provider configuration, and
  sessions. It may consume `forge_agent`, never `forge_cli` or the reverse.
- `forge_cli` owns the Typer command, print/transcript renderers, pure shared
  formatting, and Textual TUI. It may consume `forge_coding` and
  `forge_agent`; `forge_cli.__init__` stays lightweight and must not eagerly
  import Textual.
- Coding tools are native LangChain `StructuredTool` instances. Their async
  implementations receive `ToolRuntime` and return `(content, artifact)` so
  LangChain writes a `ToolMessage` with Forge metadata. Renderers consume
  `AgentEvent` product/UI events; slash commands do not enter the model
  transcript.
- Every model-produced tool-call batch is executed by the outermost
  `SequentialToolCallMiddleware` in AIMessage order. The first error stops
  later handlers, which still receive bounded paired error `ToolMessage`
  results; dependent calls must wait for a later model turn. Direct
  `read`/`write`/`edit` operations additionally share the process-local
  same-file `FileOperationQueue`.
- Keep streaming on async iterators/generators and cancellation on
  `AbortSignal`-equivalent tokens. Do not replace this with EventEmitter/RxJS
  style abstractions.
- Todo planning must use LangChain's official `TodoListMiddleware`. Forge may
  project validated full-list snapshots into `TodoUpdateEvent` and
  `forge.todo.v1` `CustomEntry` records, but those records are product/UI state
  and must not become a second model transcript or a competing planning engine.
- Human input must use the official `HumanInTheLoopMiddleware` `respond`
  decision and resume the same graph with `Command(resume=...)`. Interactive
  runs may use a unique in-memory checkpointer only while a questionnaire is
  pending. Dispose it after answer, cancellation, close, or session switch;
  never use it for durable replay, branching, compaction, or model memory.
- Non-interactive modes must not register `ask_user_question`. Every persisted
  assistant tool call must have a paired `ToolMessage`; cancellation, close,
  and session switching must synthesize a bounded error result when abandoning
  a pending questionnaire.

## Safety

- Never read, print, log, commit, or test against `.env`, API keys, OAuth
  tokens, credentials, `session-temp.jsonl`, or other local state.
- Runtime user configuration belongs under `~/.forge/`; no `~/.tau` paths or
  `tau_*` imports may remain in active code, tests, docs, or CLI help.
- File tools enforce both lexical and resolved-path workspace boundaries,
  reject symlink escapes and final symlinks for writes, and use conservative
  exact editing semantics. Do not bypass the safe write helpers.
- Shell execution only controls the child process cwd; it is not a sandbox.
  Preserve timeout, cancellation, tail diagnostics, and byte-count metadata.
- Project instructions and resources must stay inside the configured project
  boundary and reject symlink escapes.
- Do not run destructive commands, access paths outside the workspace, or
  overwrite user changes unless the request explicitly authorizes it.

## Implementation and tests

- Read the current code, tests, `pyproject.toml`, and CI before changing a
  boundary. Prefer the smallest reusable implementation and preserve existing
  JSONL/session data and public event fields.
- Keep Python code formatted and linted with Ruff, strictly typed with mypy,
  and tested with pytest. Default tests are deterministic and offline; use
  fake LangChain chat models or Forge fake providers.
- Before completion run:

  ```bash
  uv sync --dev --extra providers --locked
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy
  uv run forge --help
  uv run forge --version
  ```

  A real provider smoke test is opt-in and requires the user's local
  credentials; it is never part of default tests or CI.
- Add a regression test before fixing a reported bug. Cover success, failure,
  cancellation, timeout, truncation, path boundaries, and message/tool pairing
  at the relevant layer.
- Questionnaire state is indexed by question, not by submission order or list
  cursor position. `Tab` and `Shift+Tab` preserve text and selections; only an
  explicit option toggle changes multi-select answers. Render model-provided
  questionnaire text with markup disabled, and test the mounted Textual screen
  for navigation, custom answers, cancellation, and literal rendering.
- Keep the Todo panel bounded and conversation-first: it sits above the
  composer, hides completed rows before truncating active rows, retains newly
  completed items until the next user turn, and exposes the full snapshot via
  `/todos`.

## Documentation authority

- Current implementation facts come from the source tree, `pyproject.toml`,
  `uv.lock`, `README.md`, `docs/architecture.md`,
  `src/forge_coding/data/release-notes/releases.json`, and
  `.github/workflows/ci.yml`.
- `docs/architecture-refactor-plan.md` is an implementation record, not a
  promise about a future change. Its historical paths and baselines must not
  override the current source tree.
- `docs/dev/` is an ignored local archive of historical plans and decision
  records. It is not shipped and is never authoritative for current package
  names, paths, versions, release status, or completed work. When it conflicts
  with current code, use the sources above and mark the plan as historical in
  any new references.
- Any new plan must state its status, date, and commit baseline. Do not direct
  contributors to a path that no longer exists.

## Hotspot ownership

- `src/forge_coding/sessions/session.py` owns session lifecycle and run
  coordination; pair changes with `tests/test_coding_session.py`.
- `src/forge_coding/providers/config.py` owns provider/catalog configuration;
  pair changes with `tests/test_provider_config.py` and
  `tests/test_provider_runtime.py` when runtime construction is involved.
- `src/forge_agent/langchain_runtime.py` and `src/forge_agent/subagents.py`
  own runtime and nested-run projections; pair changes with
  `tests/test_langchain_runtime.py` and the focused subagent tests.
- `src/forge_cli/tui/app.py`, `src/forge_cli/tui/widgets.py`,
  `src/forge_cli/tui/screens.py`, and `src/forge_cli/tui/state.py` own the
  Textual session shell and presentation state; pair changes with
  `tests/test_tui_app.py`.
- `src/forge_coding/sessions/export.py`,
  `src/forge_coding/commands/default_registry.py`, and
  `src/forge_coding/features/goals.py` own export, slash-command, and Goal
  boundaries respectively; use their focused module tests before broad checks.
- `tests/test_cli.py` is the CLI integration verification surface; keep CLI
  behavior changes paired with its narrowest relevant cases.
- `.agents/skills/langgraph-fundamentals/SKILL.md` is a checked-in framework
  reference, not product runtime code; update it only when the installed
  LangGraph guidance is intentionally refreshed.
- Large test modules are verification surfaces for the owning source module,
  not independent product layers. Keep new behavior next to the owning module
  and extend the narrowest relevant test first.

## Git and docs

- Use the existing Conventional Commit style (`feat(scope): ...`,
  `fix(scope): ...`, `test: ...`, `docs: ...`, `chore: ...`). Keep each commit
  scoped and independently runnable; never rewrite history, push, or create a
  branch unless explicitly requested.
- Do not commit `.venv`, caches, generated site output, lockfile edits made by
  hand, `dist`, `node_modules`, secrets, or local session files.
- Update `README.md` when CLI usage, dependencies, or environment variables
  change. Update `NOTICE` when attribution or source-baseline facts change.
- Avoid broad refactors and unrelated cleanup. At the end report commands run,
  commit hashes, changed scope, and any unverified risk.
