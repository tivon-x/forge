# Forge

Forge is a provider-neutral Python coding-agent CLI. It reads and edits a
project, runs explicitly requested local commands, streams model events, and
stores inspectable JSONL sessions under `~/.forge/`.

The distribution name is `forge-code`; the CLI command remains `forge`.

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
uv sync --dev --extra providers --locked
uv run forge --help
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Install the optional commit hook once to run the locked Ruff formatter and
lint check automatically:

```bash
uvx pre-commit install
```

The command is also available through the project environment:

```bash
uv run forge --version
uv run forge -p "summarize this repository"
```

Model requests require the provider credentials configured by Forge (for
example `OPENAI_API_KEY`). Default tests use deterministic fake models and do
only use local fixture servers, without external services. Do not put credentials in this repository.

Forge is not currently published on PyPI. The optional startup version check is
disabled by default; set `FORGE_ENABLE_UPDATE_CHECK=1` only after configuring a
real Forge package release channel.

### Project resource trust

Forge performs a metadata-only trust preflight before reading project-level
`AGENTS.md`, skills, prompts, or subagent profiles. User resources under
`~/.forge/` and `~/.agents/` remain trusted. Interactive TUI startup can ask
for a decision; non-interactive print mode denies project resources by default
and writes a warning to stderr.

Use `--trust yes`, `--trust no`, or `--trust ask` for one run. The environment
variable `FORGE_TRUST=always|never|ask` is the next precedence layer. Persistent
decisions are stored in `~/.forge/trust.json`; `/trust status`, `/trust once`,
`/trust always`, `/trust parent`, and `/trust deny` manage them from a session.
Trust changes never replace the active prompt automatically: run `/reload`
explicitly while the session is idle. `--trust yes|no` does not write the trust
store, and denied project resources do not change workspace tool access.

### Session branch operations

Interactive sessions provide four local slash commands:

- `/clone` copies only the active branch into a new session and switches to it.
- `/fork` selects a previous user message, copies its parent branch into a new
  session, and prefills that message without sending it.
- `/import <path>` validates a session JSONL file, rechecks trust for its project,
  copies it under a new session id, and switches only after the copy succeeds.
- `/copy` sends the latest complete assistant response to the native clipboard
  command (`clip.exe`, `pbcopy`, or the first available Linux clipboard tool).

Clone, fork, and import never overwrite their source. Cancelling an import trust
prompt creates no destination session.

### Provider error guidance

Final authentication, missing-key, unknown-provider, and unknown-model errors
include a short next action after Forge redacts provider text. Use `/login` for
credentials, `/model` to switch the active model, or `forge models` to list
configured providers and their models. Other errors keep their sanitized message
without a speculative suggestion.

## What is included

- `forge_agent`: LangChain-native runtime, Forge UI events, session primitives,
  and the harness facade.
- `forge_coding`: project context discovery, safe coding tools, JSONL sessions,
  and provider configuration.
- `forge_cli`: the Typer entry point, print/transcript renderers, pure shared
  formatting helpers, and the optional Textual TUI. The TUI is split into
  focused modules: `app.py` owns the session shell and event routing,
  `prompt.py` the prompt editor (kill ring, external editor), `bindings.py`
  the keybinding-to-Binding builders, `screens.py` the modal pickers and
  login flows, `presentation.py` the pure rendering helpers, `css.py` the
  stylesheet, and `startup.py` the provider/session bootstrap.

The CLI supports one-shot print mode, JSON event output, interactive sessions,
session resume/export, slash commands, project `AGENTS.md` discovery, and
workspace-bounded read/write/edit/grep/find/ls/shell tools. `grep` and `find`
use managed `rg` and `fd` binaries, preferring `~/.forge/bin` and the current
process `PATH` before downloading a fixed official GitHub release. Downloads
require HTTPS, a release SHA-256 digest, safe archive extraction, and a
matching `--version` result. Set `FORGE_OFFLINE=1` (or `true`/`yes`) to fail
closed with manual installation guidance instead of using the network; Forge
never falls back to a shell search implementation. Forge sessions include the
official LangChain Todo middleware: the model can maintain a `write_todos` plan,
the TUI shows it above the composer, and `/todos` prints the full active list.
Interactive TUI turns can also call `ask_user_question`; Forge pauses the same
LangChain execution, opens a structured questionnaire, and resumes it with a
paired `ToolMessage`. Print mode is deliberately non-interactive and does not
register that tool. Shell execution is not a sandbox; it runs with the
operating-system user's permissions.

Custom native composition tools may use
`forge_agent.tool_execution.get_nested_tool_executor().call(name, arguments)`
inside an active tool invocation. Calls keep native argument and workspace
checks, return temporary `ToolMessage` results, and attach bounded metadata to
the parent result. Mark composition tools with `forge.exposure=model-only` to
prevent recursive nesting. MCP and Codemode use this same execution boundary.

### MCP and Codemode

Configure servers in `~/.forge/mcp.json` or trusted project `.forge/mcp.json`:

```json
{
  "mcpServers": {
    "local": {
      "command": "python",
      "args": ["local_mcp.py"],
      "exposure": "deferred"
    },
    "remote": {
      "url": "https://mcp.example.com/mcp",
      "headers": {"Authorization": {"env": "MCP_AUTHORIZATION"}}
    }
  }
}
```

Provide your own server executable or URL. `command/args/cwd/env` and
`url/headers` are mutually exclusive; `env` and `headers` accept environment
variable references, never inline credentials. A project server replaces the
whole user server of the same name. Configuration and project trust do not
connect or authorize servers. `enabled: false` prevents explicit activation.

```bash
uv run forge --mcp local --codemode on -p "Use the local tools"
uv run forge --codemode only -p "Read and edit this project"
```

`--mcp` is repeatable and authorizes only the selected servers for this run.
Interactive `/mcp` lists servers; `/mcp enable|disable|tools|logs <server>` manages
them. Idle `/reload` refreshes authorized directories and revokes changed
configurations. Default MCP exposure is `deferred`: `tool_search(query, names,
namespace, limit)` loads selected schemas for the next model request. Set
`exposure: "direct"` or per-tool `toolExposure` for immediate declarations.
Hidden tools cannot be searched or invoked. Tool names are server-prefixed.
HTTP response bodies, individual SSE events and stdio JSON-RPC lines are limited
to 8 MiB before parsing. Tool catalogs across servers and each paginated resource
catalog also have an 8 MiB budget. HTTP requests use identity encoding and reject
compressed responses. An unchanged server keeps its callable catalog when
discovery temporarily fails; cancellation publishes any completed refreshes.
Resource list/read failures also preserve callable tools. If the composed catalog
conflicts with existing names, Forge revokes all MCP authorizations and clears
both catalog views; correct the conflict and explicitly enable servers again.
SSE media types are case-insensitive, and split CRLF bytes count toward each
event's budget.
The names `tool_search` and `codemode` are reserved for Forge and caller-provided
tools using them are rejected explicitly.

`/mcp login <server>` starts SDK OAuth only on an explicit human command.
Login and execution authorization are separate. `/mcp logout <server>` removes
cached credentials and revokes execution. SDK tokens live in a private,
configuration-specific `~/.forge/mcp-auth/` directory. HTTP redirects are disabled
for authenticated transports. Resource tools list resources/templates and read
only listed or matching template URIs on authorized servers; HTML resources are
excluded. Server elicitation, sampling and roots are unsupported.

`--codemode on` adds a JavaScript orchestration tool; `only` moves callable direct
tools into its API while control tools remain directly declared. Each call uses
a fresh QuickJS/WASM worker launched by the current Python interpreter, with no
Node installation. JavaScript has `tools.<normalized_name>(args)`, `ALL_TOOLS`,
`text`, `image`, `console`, `exit`, `store/load`, `searchTools(query, options)` and
`describeTool(name, options)`. It supports top-level `await` and `return`.

```javascript
// @options: {"timeout_ms": 60000, "max_output_tokens": 1000}
const result = await tools.read({path: "README.md"});
text(result.content);
store("lastRead", "README.md");
return result.path;
```

Only explicit JSON store values persist, and only after successful execution.
Store and loaded-tool snapshots follow the active session branch; restoring
history never authorizes an MCP server. Tool failures reject promises;
`Promise.allSettled` can collect independent failures. Completed file/network
side effects survive later script errors or cancellation.

Forge bounds a script to 300 seconds, 256 tool calls, 32 active children, a
256 MiB JS heap and 8 MiB collected output. Images accept inline PNG/JPEG/WebP/GIF,
up to 512 KiB each and 4 MiB in total. Store values are at most 64 KiB each,
1 MiB total. Oversized display output is saved through the workspace-safe write
helper under `.forge/artifacts/`; output lost beyond the collection limit is
explicitly marked. These are Forge limits and differ from Pi's unbounded default
CLI deadline. JavaScript cannot access Python objects, filesystem or networking
APIs directly; authorized native tools retain their existing permissions, and
shell/MCP execution is not an operating-system sandbox.

See [architecture](docs/architecture.md) and the
[implementation record](docs/mcp-tool-discovery-codemode-plan.md) for boundaries
and repeatable offline acceptance commands.

Todo and questionnaire controls are keyboard-first:

- `Ctrl+Shift+T` collapses or expands the Todo panel.
- `/todos` opens the complete active Todo list, including rows hidden by the
  panel height limit.
- In a questionnaire, `Up` and `Down` move through options, `Space` toggles a
  multi-select option, and `Enter` confirms the current question.
- `Tab` and `Shift+Tab` move between questions without discarding draft text or
  selections. Choose `Type something.` to enter a custom answer, and press
  `Esc` to cancel the questionnaire without leaving an unmatched tool call.

### TUI keyboard shortcuts and customization

All interactive shortcuts are configurable in `~/.forge/tui.json` under
`keybindings`; each action accepts one key or an array of keys. `/hotkeys`
shows the shortcuts with your configured keys, and editing `tui.json` or a
custom theme file is applied live while the TUI is running (pi-style hot
reload). Defaults:

```text
Enter              submit prompt            Shift+Enter     insert newline
Alt+Enter          queue follow-up          Alt+Up          restore queued messages
Esc                cancel / abort           Ctrl+D          quit
Ctrl+K             slash-command completions
Ctrl+R             session picker (search, Ctrl+S sort, Ctrl+R rename,
                   Ctrl+D delete after confirmation)
Shift+Tab          cycle thinking level     Ctrl+T          toggle thinking tokens
Ctrl+O             collapse/expand tool output
Ctrl+Shift+T       collapse/expand todos    Ctrl+Shift+F    search the transcript
Ctrl+G             edit the prompt in $VISUAL/$EDITOR
Alt+Backspace      delete word backward     Alt+D           delete word forward
Ctrl+U             delete to line start     Ctrl+Y          paste killed text
Alt+Y              cycle killed text after yank
```

The prompt border is color-coded while the agent is running: the border shows
your current thinking level, and `!`/`!!` shell commands use a dedicated shell
color. The session picker supports live search, title sorting, renaming, and
deletion. Custom themes are plain JSON files in `~/.forge/themes/*.json` that
may override any `TuiTheme` field (colors, `role_styles`, `thinking_borders`,
`shell_border`); missing fields fall back to the dark theme.

Todo snapshots are restored from the active JSONL branch and survive session
resume and compaction. A pending questionnaire is intentionally process-local:
answering, cancelling, closing, or switching sessions resolves the pending tool
call and releases its temporary in-memory checkpoint.

Transient model failures are retried up to three times with bounded 2/4/8-second
backoff. Agent-loop retries emit UI events and store at most one bounded,
redacted `forge.turn_error.v1` audit entry per logical run. Compaction, branch
summary, and auto-name calls share the same retry policy without creating a
second event stream; their successful usage remains recorded. Authentication,
quota, parameter, overflow, and cancellation failures are not retried. Set
`RetryPolicy(enabled=False)` in `AgentHarnessConfig` or `CodingSessionConfig` to
retain the pre-0.1.7 single-call behavior.

Forge also supports a session-level Goal for work that should continue until a
clear objective is verified. Use `/goal <objective>` to start one, `/goal` to
open the manager in interactive mode (or print the current status in plain
mode), and these deterministic actions:

```text
/goal status
/goal pause
/goal resume
/goal edit <objective>
/goal clear
```

Goal commands only parse an immutable action intent; the session applies the
action asynchronously and persists a complete snapshot before emitting its
`GoalUpdateEvent`. A Goal is separate from the model's Todo plan: completing
all Todo items does not complete the Goal, and only an explicit
`goal_complete` action can produce the `complete` state. Objectives are limited
to 4,000 characters. Automatic work pauses after 25 coordinator runs, and
three consecutive tool-free outputs with no observable progress mark the Goal
blocked. Plain output renders a compact `Goal: <status>` line when there is no
final assistant response.

Forge also exposes a built-in `task` tool so the model can delegate one
bounded job to a subagent context (fresh by default). Describe the delegation naturally,
for example: “Use a scout to trace the authentication flow, then explain the
relevant files.” The built-in roles are:

- `scout`: inspect code and collect evidence without file-editing tools.
- `worker`: implement one clearly scoped change and run targeted checks.
- `reviewer`: independently review existing work without file-editing tools.

Projects can add or override roles with
`.forge/agents/<name>/AGENT.md`; user-wide roles use
`~/.forge/agents/<name>/AGENT.md`. The Markdown frontmatter declares a short
`description`, an optional comma-separated tool allowlist, and optional
per-role model-call/result-size limits and `mode: foreground|background`
(default `foreground`). Project roles override user roles,
which override the built-ins. Invalid higher-priority files are reported and
leave the lower-priority role available. Use `/agents` to inspect the active
registry and `/reload` to apply file changes while the session is idle.
Roles also accept `model: inherit` (default), `fallback-models` (up to two
configured `provider/model` references), and `context: fresh|fork|summary`.
For example:

```yaml
---
description: Review the requested change
model: my-provider/review-model
fallback-models: my-provider/backup-model
context: fork
---
Review the evidence and report actionable findings.
```

Model references must exist in the current provider catalog; role files cannot
supply credentials or provider URLs. `/agents` shows the model chain and context.
The first `/` separates the provider from the complete model name, including
any later `/` or `:` characters. Invocation-owned model clients are released
when the task finishes or is abandoned; borrowed parent clients stay open.
`task(..., context="fork")` can override the role's context for one invocation.
`fresh` receives only the instruction. `fork` copies the parent history before
the entire pending delegation batch, retaining complete tool-call/result pairs
and removing reasoning, provider metadata, artifacts and parent system instructions.
The snapshot does not change when the parent continues. `summary` uses the
existing summary helper to produce a task brief; failed or empty summaries fail
the task, and summary usage is recorded separately.

Capacity uses the smallest window in the frozen model chain, reserving system,
declared tools and output space. Its conservative UTF-8 byte estimate can reject
a context earlier than a provider tokenizer would. Only the oldest complete
turns may be removed; the latest user request and child instruction stay.
If they still do not fit, the task fails with a suggestion to use `summary`.
Custom tool lists can only reduce the tools already enabled for the session,
and `task` is never available to a child. A prompt that describes a role as
read-only is not a sandbox; in particular, `bash` still has the operating-system
user's permissions.

Child catalogs refresh after MCP authorization changes. Child catalogs copy native
tool metadata from the parent and directly declare their selected deferred tools;
parent Codemode mode does not restrict a child's normal `read`/`write` calls.
Parent search, Codemode, questionnaire and other model-only control tools are
excluded. Script-only and hidden tools are also excluded; children cannot
modify the parent's loadout or JSON script store.

Foreground calls run synchronously and in process, select the role's configured
models without switching the parent, reuse project resources and safe coding tools,
and return only the
final result to the parent agent. Each child is limited to
eight logical child model requests and a 50 KiB UTF-8 result, and cannot delegate again.
Fallback retries only the current model request after a transient failure;
completed tools are not repeated. Authentication, quota, invalid requests,
context overflow, unknown errors and cancellation do not trigger fallback.
Within the child graph, responses from the same provider/model retain their
native continuation metadata, including thought signatures. Parent history and
responses passed to another provider/model remain portable and stripped of
provider metadata; native continuation metadata is never a display trace.
Child SDK and Forge retries are disabled. Three candidates permit at most
24 actual attempts; usage records the actual provider/model, including reported
usage from failed or cancelled attempts. Unreported usage remains unknown.
Forge does not create child sessions or persist raw child transcripts. It
stores a bounded display trace on the active parent-session branch: visible
human/assistant text, tool names and success/error status only. Raw tool
arguments, tool output, artifacts, thinking, and provider metadata are not
stored. When the provider reports standard token usage, Forge also records the
aggregate input/output/total counts. The TUI shows the task inline, updates its
current activity, and uses `Ctrl+O` to expand or collapse its safe activity
summary; `Esc` cancels the whole current foreground prompt.

Interactive sessions also support `task(agent, instruction, mode="background")`.
The parent receives a durable receipt with a run ID and can continue talking
while the child runs. `task_control` lists runs, reads status/results, waits up
to 60 seconds without cancelling on timeout, or cancels one exact run ID.
`/fleet` opens a separate task inspector with progress, safe trace, result,
cancel and collect actions. Collect prepares a parent prompt to read the result;
it does not start an extra model turn. The composer status line shows Fleet
counts and completed tasks produce a notice.

Background runs use the same LangChain agent loop on an independent v3 stream,
with a frozen role/tool/model/context snapshot. Fleet displays the actual model,
context mode, inherited message count, truncation, attempt count and sanitized
fallback reasons. Provider and model jointly identify a switch, including the
same model name served by different providers. Defaults are four
running tasks, sixteen queued tasks and a fifteen-minute run timeout. Python
embedders can set `CodingSessionConfig.background_subagent_concurrency`,
`background_subagent_queue_limit` and `background_subagent_timeout_seconds`.
Print/JSON modes reject background submission. Parent `Esc` leaves accepted
background work running; Fleet cancellation, session/branch switching and exit
stop the affected children before releasing shared clients. Model/provider/MCP
updates, resource reload and manual compaction require waiting for or cancelling
active background work. Startup Git baseline is a hint, not a file snapshot.
Background reads can overlap parent edits; assign worker files explicitly.

Receipts have exactly one paired parent `ToolMessage`. Terminal results use
bounded `forge.subagent_run.v1` product entries and child usage enters the common
ledger. Results stay out of the parent transcript until explicitly collected.
Reloading a session marks unfinished runs `interrupted`; it never restarts them.
There is no detached-process execution in this release.

Parent sessions can also run saved project workflows through the
model-only `workflow(name, args)` tool. `/workflows` lists the safely loaded
`.forge/workflows/*.js` resources; it does not execute them. The JSON `args`
value is exposed to a script as read-only `workflowArgs`, so user data is never
spliced into JavaScript source. Workflow scripts receive the current authorized
native tool catalog and the scoped
`runs.run({step_key, agent, instruction, mode?, context?})` and
`runs.control({action, run_id, timeout_seconds})` APIs. Child agents still cannot
call `task`, `workflow`, or Codemode, so a workflow cannot recursively delegate.

Each workflow has a stable `workflow_id`; every run keeps its stable `step_key`
and Fleet groups those steps together. Repeating a step key with the same
assignment returns the existing run and preserves its result, so a later failed
step does not rerun an earlier side effect. A child execution with `status:
completed` can still return a task-report with `outcome: blocked`; that is a
valid business result which asks for input or records an unmet condition.
`status: failed` means execution itself failed or timed out; `status: cancelled`
records cancellation.
The two fields stay separate in workflow branching and in Fleet details.

Workflow execution uses the existing QuickJS worker and Python coordinator with
a 15-minute default deadline (30-minute maximum), at most four live children,
16 launches per workflow, and a 64 KiB final report. Background submissions
must be explicitly waited to a settled state; cancellation and timeout drain
workflow-owned children and retain completed steps in the partial report. The
complete examples in [`docs/workflows`](docs/workflows) cover sequential
scout/worker/reviewer handoff, `Promise.allSettled` parallel review, and a
review/fix loop capped at three rounds.

Model calls are recorded as allowlisted `forge.usage.v1` JSONL entries. The
ledger covers agent replies, compaction, branch summaries, automatic session
naming, and subagents, while retaining `null` for provider fields that were not
reported. Costs use resolved catalog rates and a content hash captured at write
time, so later catalog edits do not change historical totals. `/session` and
the compact TUI footer show active-branch token/cache/cost aggregates.

## Attribution

Forge is independently maintained in this repository and is released under
the MIT License. It is inspired by and derives architectural lessons from:

- [Pi](https://github.com/earendil-works/pi), the primary TypeScript
  architecture and the ongoing learning reference for Forge improvements.
- [pi-subagents](https://github.com/nicobailon/pi-subagents), a learning
  reference for background delegation and Fleet task inspection.
- [Tau](https://github.com/huggingface/tau), whose MIT-licensed Python
  implementation is the product baseline for Forge's coding tools, sessions,
  provider configuration, and TUI. Tau's MIT license is retained in
  [`LICENSE`](LICENSE).

Neither upstream project is presented as a Forge release, and Forge does not
use the Tau package name or Tau configuration directory at runtime.
