# Forge Agent Instructions

## Project scope

Forge is a single-package, Python 3.12+ provider-neutral coding-agent CLI.
The current implementation uses `src/forge_agent`, `src/forge_ai`, and
`src/forge_coding`. Keep the project runnable with `uv` and do not add a
second language runtime, monorepo, or heavyweight task framework without an
explicit decision.

Pi is the primary architecture and learning source. Tau is the Python product
baseline and attribution source. Keep those references in `README.md` and
`NOTICE`; do not use Tau as an active package, CLI, or configuration name.

## Architecture boundaries

- `forge_agent` owns Forge messages, events, tool contracts, session primitives,
  and `AgentHarness`. It must not import CLI renderers, filesystem helpers,
  shell commands, or provider SDKs.
- The production agent/tool loop is LangChain Python's official
  `langchain.agents.create_agent` plus `astream`. `AgentHarness` is only the
  outer transcript, queue, cancellation, and Forge-event facade. Do not add a
  second provider/tool loop to the production path.
- `forge_ai` adapts model providers and fake models to Forge's provider
  protocol. Third-party SDK types stop at this boundary.
- `forge_coding` owns project context, safe tools, provider configuration,
  sessions, CLI/TUI, and renderers. It may consume `forge_agent` and
  `forge_ai`, never the reverse.
- Tools are ordinary async callables with structured `AgentToolResult`, an
  optional cancellation token, and explicit execution context. Renderers
  consume `AgentEvent`; slash commands do not enter the model transcript.
- Keep streaming on async iterators/generators and cancellation on
  `AbortSignal`-equivalent tokens. Do not replace this with EventEmitter/RxJS
  style abstractions.

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
  uv run ruff check src tests
  uv run mypy
  uv run pytest
  uv run forge --help
  uv run forge --version
  ```

  A real provider smoke test is opt-in and requires the user's local
  credentials; it is never part of default tests or CI.
- Add a regression test before fixing a reported bug. Cover success, failure,
  cancellation, timeout, truncation, path boundaries, and message/tool pairing
  at the relevant layer.

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
