# Forge architecture

Forge is one Python distribution with three packages. The split keeps the
runtime facts separate from coding-domain behavior and presentation concerns;
it does not add another runtime, loop, or persistence system.

```text
forge_cli  ->  forge_coding  ->  forge_agent
CLI/renderers/TUI   coding domain       runtime
```

## Package ownership

### `forge_agent`

The runtime owns LangChain message/model/tool integration, `AgentHarness`,
Forge product events, and session primitives. Production execution is the
official `create_agent()` graph streamed with `astream_events(version="v3")`.
The package must not import `forge_coding`, `forge_cli`, UI libraries, provider
SDKs, filesystem helpers, or shell commands.

### `forge_coding`

The coding layer owns project context, workspace-safe coding tools, provider
configuration/runtime construction, session domain behavior, resources, and
JSONL storage. It consumes `forge_agent` and does not import the CLI layer or
presentation libraries.

Coding tools are assembled as an ordered `ToolSet` of `ToolDefinition`
catalog entries. Each entry stores only Forge product metadata (`label`,
`prompt_snippet`, and `prompt_guidelines`) plus the native LangChain
`BaseTool`; schema, description, and execution are always derived from that
native object. `create_coding_tools()` remains the compatibility factory and
returns the ordered native tools (`read`, `write`, `edit`, `bash`).

### `forge_cli`

The presentation layer owns the `forge` Typer entry point, print/transcript
renderers, Textual TUI, terminal-title behavior, and
`forge_cli.formatting`—pure formatting shared by renderers and TUI state. Its
`__init__` is intentionally lightweight and does not import Textual eagerly.
The console entry point is:

```toml
forge = "forge_cli.cli:app"
```

The wheel includes `forge_agent`, `forge_coding`, and `forge_cli`; users still
install and invoke one `forge-ai` distribution.

`forge_cli.tool_rendering.ToolViewRegistry` is a pure name-to-formatter
registry. Plain transcript output, live Textual state, and restored JSONL
tool rows call the same bounded formatter entry points; the registry never
stores widgets or execution state.

## Runtime and persistence

Model/tool execution and Forge event projection remain unchanged. The CLI and
TUI consume `AgentEvent` values, while JSONL sessions retain LangChain message
content, tool calls, artifacts, and metadata through the existing codec. Slash
commands remain presentation/session controls and do not enter the model
transcript. Todo state is projected from the official `TodoListMiddleware` and
stored as versioned `forge.todo.v1` snapshots on the active JSONL branch; the
snapshot is product/UI state, not a second model transcript.

Goal is a session-scoped managed-run coordinator, not another agent loop. The
`forge_coding.GoalController` owns the lifecycle state machine and writes full
`forge.goal.v1` snapshots (or a clear tombstone) to the same JSONL branch.
`GoalMiddleware` adds the current objective to each active model request and
exposes `goal_complete`/`goal_blocked` only while the Goal is active. After a
settled run, `CodingSession` may call the existing `AgentHarness.continue_()`
again under the session's run ownership and safety limits; it never appends a
synthetic user message or creates a second provider/tool loop. Goal changes are
projected as `GoalUpdateEvent` values for the plain renderer and Textual TUI,
while Goal data remains outside the LangChain transcript. Todo completion is
independent from Goal completion: only an explicit Goal tool can reach the
`complete` state.

Interactive sessions also register the official
`HumanInTheLoopMiddleware` for the `ask_user_question` placeholder tool. A
paused turn uses a unique, in-memory LangGraph checkpointer only until it is
answered, cancelled, or the session is closed. JSONL remains the only durable
session fact source: the temporary checkpoint is never used for replay,
branching, compaction, or cross-process recovery. Non-interactive print runs do
not expose the ask tool.

Every model-produced tool-call batch is guarded by
`forge_agent.SequentialToolCallMiddleware`. It is the outermost middleware in
root and child graphs, executes calls in AIMessage order, stops after the first
error, and returns bounded paired `ToolMessage` errors for skipped or invalid
calls. A process-local `FileOperationQueue` additionally serializes `read`,
`write`, and `edit` on the same resolved path for direct execution and for
multiple sessions, including sessions running in different event loops;
different paths remain independent. Shell commands are not mapped to file
keys and the queue is not a cross-process sandbox.

## Subagents

Subagents use the LangChain supervisor-as-tool pattern. The parent agent keeps
the only user conversation and calls a native `task(agent, instruction)` tool.
That tool creates a fresh `create_agent()` child for one synchronous run, with
the current session provider, model, runtime context, project resources, and a
role-specific subset of the configured tools. The child has no checkpointer or
`task` tool, so it has no independent session and cannot recursively delegate.
A session-level lock serializes multiple task calls.

`forge_agent` owns the generic runner, safe display-trace projection, usage
aggregation, and nested-event projection. `forge_coding` owns declarative role
discovery, the built-in `scout`, `worker`, and `reviewer` profiles, capability
intersection, and session assembly. Built-in, user-level
`~/.forge/agents/<name>/AGENT.md`, and project-level
`.forge/agents/<name>/AGENT.md` profiles pass through the same compiler;
project profiles have the highest precedence. `forge_cli` owns the inline TUI
task block. Tool allowlists can only reduce the current session's configured
capabilities and never include `task`. Prompt-level restrictions and access to
`bash` are not a security sandbox; operating-system permissions and the
existing workspace tool boundaries still apply.

LangChain v3 child events have a non-empty namespace. Forge never projects
child messages, thinking, raw tool arguments/results, artifacts, or child
tool-call identifiers into the parent transcript. Recognized lifecycle and
tool activity becomes `ToolExecutionUpdateEvent` values associated with the
parent task call. The latest nested `values` snapshot is separately reduced to
a bounded, JSON-safe display trace and optional aggregate standard token
usage. That trace is written as a `forge.subagent_trace` `CustomEntry` between
the parent AI task call and its paired `ToolMessage` on the active JSONL
branch; `SessionState.messages` ignores it, so it never enters model context or
compaction summaries. The root task end remains authoritative and carries a
JSON-safe v2 `subagent_run` artifact without trace items. V1 artifacts and
sessions require no migration, and malformed or unknown nested/custom data
degrades by hiding the trace rather than leaking child content.

## Attribution

Pi is the primary architecture and learning source. Tau is the Python product
baseline and attribution source for coding tools, sessions, provider
configuration, and the Textual TUI. Forge is independently maintained; Tau is
not an active package, CLI, or configuration name.
