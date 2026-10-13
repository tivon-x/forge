# Forge architecture

Forge is one Python distribution with three packages. The split keeps the
runtime facts separate from coding-domain behavior and presentation concerns;
it keeps one LangChain model loop and one JSONL session format. Codemode alone
uses an isolated QuickJS/WASM worker within the Python distribution.

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
returns the ordered native tools (`read`, `write`, `edit`, `find`, `grep`, `ls`,
`bash`).

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
install and invoke one `forge-code` distribution.

`forge_cli.tool_rendering.ToolViewRegistry` is a pure name-to-formatter
registry. Plain transcript output, live Textual state, and restored JSONL
tool rows call the same bounded formatter entry points; the registry never
stores widgets or execution state.

## Runtime and persistence

Root tool completion comes from the official graph values after native
ToolMessages join the transcript; early tool callbacks are progress only.
Goal/Todo updates and their paired results can therefore share one commit.
The existing Goal middleware uses public `before_model` / `jump_to="end"`
to stop a terminal Goal before the next provider call, including eager v3 streams.

Session disk I/O is coding-owned. `forge_agent.session.storage` exposes only
`SessionStorage`; `forge_coding.sessions.storage.JsonlSessionStorage` adapts
existing entries to `forge_coding.sessions.journal.Journal`. A durable directory
uses Pi's `main.jsonl` commit markers and task/document sidecars. Related
messages, usage, product snapshots, background receipts and active leaf changes
share a commit sequence; tree/replay continue to consume `SessionEntry` views.
Candidates are validated and serialized before I/O. Sidecars precede the main
marker, which precedes memory adoption. Uncertain publication invalidates both
storage and its session; the same instance cannot dispatch another run.

Reopen keeps only marker-confirmed state and truncates unconfirmed byte tails.
Complete corruption or missing required sidecars fails closed. Latest document
bases and terminal-task sidecars are reclaimed after publication; failures only
defer reclamation. Rewindable conversation document history remains available.
`fsync` defaults to false; when enabled, publication flushes sidecars first and
main is flushed only to authorize reclamation. Process-exit validation does not
establish power-loss durability.

Legacy JSONL remains readable and is never overwritten by migration. First
write imports the complete validated tree into a deterministic durable location
before the resume index is atomically relocated. Retry reuses confirmed import
history. Legacy exports intentionally omit execution recovery information;
complete storage copies include main and all retained sidecars. Export writers
share source-storage protection across CLI, Session/TUI and legacy exports,
including retained originals and resolved-path/hardlink aliases. Task reclamation
uses the final task write of a commit; document validation, adoption and replay
use the same merged document action. D1 supplies
storage recovery only: it neither resumes tasks nor replays model/tool effects.

D2 adds `features.tasks.TaskCoordinator` on the same session commit line.
Task definitions carry versions and named checkpoints. Only committed
reservations invoke handlers. Waiting records hold a task set and an
allSettled/failFast policy, without a polling model or retained handler.
Completing records hold an outcome until ordinary owned work settles.
Abort marks publish before signals; abort handlers settle children before
owners. Ordinary cancellation and idle traversal stop at background anchors.
Definition migration happens at reservation. Missing, newer, or unmigratable
definitions stay pending with an inspect reason. Handles expire when their
invocation ends or its implementation is handed over. A waiting or terminal
commit expires the handle immediately; the coordinator still tracks the handler
until it returns so close can cancel and join its remaining code. Abort handlers
cannot commit waiting states.

`features.inputs` admits conversation-scoped submissions together with the
`forge.inbox.v1` journal document and run task. Matching request IDs and types
return the existing receipt regardless of content; a type conflict rejects.
The public native before_model hook places steers after paired tool batches;
the answer boundary places follow-ups. Queue mode selects the first or all
items per input kind, then preserves their admission order. Failures keep
unplaced input. Observer timeout/cancellation leaves the work running.
Queued withdrawal and run abort are separate operations. Public continuation,
including Goal entry points, uses the same run task with no synthetic user
message. Internal Goal turns stay within that task; replacement waits for the
durable abort outcome before starting another run.

Loading a session never dispatches model/tool work. `resume()`, `submit()` and
`wait()` enable scheduling. Abort enables scheduling after publishing its mark,
including after reopen. Closing drains accepted publication, joins stopped
invocations and preserves
their checkpoints; explicit abort terminates work. Legacy D2 executions lacking
native recovery records settle interrupted. D4 uses a coding-owned
`JournalCheckpointSaver` on the same commit line: channel-version blobs,
checkpoint parents/namespaces and deduplicated pending writes live in journal
documents. Native `create_agent` / v3 streaming runs with `durability="sync"`;
recovery supplies `None` to the original thread. No separate database or model
loop exists. Persistent child conversations remain D5.
Stateless child graphs explicitly use `checkpointer=False` and public async
durability, so they inherit neither the root saver nor its sync policy. Their
tool calls still enter the shared execution boundary. This avoids the installed
LangGraph version's sync/no-saver child failure without private API access.

`features.execution` binds preparation to the run/conversation and a native
request index, advanced by public `after_model` only after response persistence.
It reuses committed native responses across the response/checkpoint gap, keeps
aborted partial attempts outside effective model history, and persists the
existing retry counter plus absolute deadline. Public request hooks still run
on resends. The shared innermost tool boundary records final argument intent
before effects, then native outcome and live cleanup. Stored outcomes are
reused; missing outcomes rerun only with stored/current `forge.replay=safe` and
current authorization. Defaults, native file/shell tools and arbitrary JS stay
unsafe. Root results enter the transcript through fully paired native values;
nested outcomes stay scoped to their parent. Checkpoint ancestry is retained;
automatic copy/prune is not enabled.
Rejected inputs and non-serializable tool results also have committed bounded
outcomes. JSON schemas use the installed validator with remote retrieval disabled;
Pydantic inputs retain their original validation rules and exclude injected
runtime objects. Committed message IDs distinguish subsequent requests even if
the provider repeats a raw tool call ID; legacy messages without IDs remain
compatible. Existing overflow compaction atomically resets request preparation
and uses a public native `Command` to replace the graph context.

D3 routes native progress through the existing session commit line. Controlled
`forge.live.v1` documents retain text append and tool append/trim deltas with
latest-only history. When no generation, tool or background execution runs,
the same commit writes a complete base; delta chains never cross idle boundaries. A 100 ms trailing writer allows one progress commit in flight;
message/tool/abort/close terminal paths drain admitted progress before settlement.
Display-only interrupted assistant messages retain aborted partial text in the
document and never enter model history. Paired tool results clear live activity
in their transcript commit. Final assistant messages clear partial text in the
same message/billing/leaf commit; background completion becomes visible only
with its receipt, billing and settlement commit. Native public stream scopes
are joined on cancellation and close before owned resources are released.

`features.observation.CommittedView` derives complete snapshots from confirmed
entries, tasks and documents. Late observers start at a snapshot; the 101st
pending commit frame replaces the backlog with the latest full snapshot. Events
already represented by that snapshot are suppressed, while committed active
questionnaire controls survive replacement. Slow
observers cannot delay commits. Task graphs show live tasks, owner, background,
waiting and completing; missing definitions and migration reasons remain in
`inspect()`. `features.documents.DocumentStore` provides explicit JSON base/delta
changes and current/as-of reads through the same journal, without proxy mutation
or a second state store. Retired rewindable incarnations remain addressable
before a latest-only replacement was created. Session replacement and close
end owned observers for both journal and protocol-only storage. Questionnaire
pairing clears the waiting projection in the same transcript commit.

Model/tool execution remains in the sole native graph. The CLI and TUI consume
committed `AgentEvent` values and `CommittedSnapshotEvent` replacements, while
JSONL sessions retain LangChain message
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
durable paused root turn uses the same journal checkpoint as ordinary execution.
Close/crash preserve it; explicit abort pairs abandoned calls and terminates the
run. Non-durable harnesses retain the temporary in-memory saver. JSONL remains
the only durable fact source, with allowlisted serialization and pickle disabled.
Non-interactive print runs do not expose the ask tool.

Every model-produced tool-call batch is guarded by
`forge_agent.ToolCallBatchMiddleware`. It is the outermost middleware in root
and child graphs and preserves LangChain's native `ToolNode` concurrency by
default. A native tool with metadata
`forge.execution_mode=sequential` makes the complete batch execute in
AIMessage order; ordinary tool errors remain paired to their own calls and do
not stop siblings. Malformed batches or broken `ToolMessage` pairing fail
closed with bounded errors. A process-local `FileOperationQueue` additionally
serializes `read`, `write`, and `edit` on the same resolved path for direct
execution and for multiple sessions, including sessions running in different
event loops; different paths remain independent. Shell commands are not mapped
to file keys and the queue is not a cross-process sandbox.

Native tools also share `execute_tool_call` for exposure, result pairing and
bounded failure handling. `ToolDefinition` stores `forge.exposure`, namespace
and advisory annotations on native metadata. `ToolSet.registered_tools` keeps
the non-hidden execution catalog; `ToolExposureMiddleware` selects the initial
model declarations without copying schemas. Control tools remain
model-only, and hidden tools do not enter session or subagent prompts.

A composition tool can obtain `get_nested_tool_executor()` during its native
execution. The parent-scoped dispatcher uses public `ToolNode` invocation to
inject trusted state/context and runs independently of the model batch lock.
It owns at most 256 calls, 32 concurrent executions and 64 KiB per call's input;
parent settlement cancels and drains outstanding children. Internal `Command`
updates and return-direct tools are rejected. Child ToolMessages are temporary
results, never appended to the model transcript. Bounded `forge.nested_calls.v1`
metadata on the parent result carries names, ids, statuses, timings and limited
operational arguments; child content and shell commands are omitted. Optional
`parent_tool_call_id` events drive in-place TUI activity, while restore, HTML
export and file-operation compaction read the parent's recorded metadata.

## MCP, discovery and Codemode

`forge_coding.mcp` validates configuration, binds explicit session authorization
to effective configuration fingerprints, and owns bounded SDK operations.
It uses FastMCP's public transports, OAuth and resource APIs and LangChain's
native `as_langchain_tool`. Connections are per operation; metadata discovery
runs outside model context. Configuration changes revoke old closures and tool
metadata. Cancellation/close drain owned clients; failed cleanup closes the
runtime and prevents successful replacement. Token storage is separate from
provider credentials and protected with owner permissions/Windows ACLs.

`ToolDiscovery` uses stable BM25 over the authorized native catalog, records
`forge.tool_loadout.v1` snapshots and projects valid loaded schemas only on the
next model request. Session-owned metadata copies prevent caller-shared tools
from leaking loadouts across sessions. The registered ToolNode catalog remains
authorized; the shared execution guard rejects undeclared direct calls,
including search-plus-call in the same model-produced batch.

`forge_coding.codemode` exposes one native model-only tool. Its one-shot Python
worker owns QuickJS/Wasmtime and exchanges bounded JSONL with version/run/request
ids. Tool execution remains in the parent's asyncio loop via the same nested
ToolNode boundary. Worker contexts and JS globals never survive a call. Public
schema pages and search results derive from the native tools; MCP results retain
standard content/structuredContent/isError plus the official artifact.

Successful, parent-validated JSON store snapshots become
`forge.codemode_store.v1` branch entries after persistence succeeds. Loadouts and
store are restored from all active-path CustomEntries, including after
compaction. Clients, credentials and execution authorization are never replayed.
`exit`, timeout and cancellation stop dispatch and terminate owned workers;
completed native effects are not transactional. The API and limits are documented
in README; JavaScript isolation does not turn shell or MCP tools into a sandbox.

## Subagents

Subagents use the LangChain supervisor-as-tool pattern. The parent agent keeps
the only user conversation and calls a native `task(agent, instruction)` tool.
That tool creates a `create_agent()` child for one bounded run, with
session-owned role models, runtime context, project resources, and a
role-specific subset of the configured tools. The child has no checkpointer or
`task` tool, so it has no independent session and cannot recursively delegate.
Independent calls retain LangChain ToolNode concurrency and share the existing
same-file operation queue for direct file tools.

Role profiles declare `model` (default `inherit`), at most two `fallback-models`
from the configured provider catalog, and `context` (`fresh`, `fork`, `summary`).
The coding session resolves and constructs the fixed candidate chain with SDK
retries disabled; the parent model remains unchanged. Official
`awrap_model_call` middleware retries only a transient failure of the current
model request. Eight logical requests allow at most 24 attempts. Actual-model
usage, including reported partial failure/cancellation usage, enters the common
ledger. A thin `BaseChatModel` adapter observes public stream usage before the
v3 bridge finishes a message; it does not run tools or own another model loop.
The coding layer owns candidate clients per invocation and closes them after
completion, cancellation, timeout, discarded submission or failed preparation;
it never closes a borrowed parent model. Construction failure also releases
earlier candidates. Runtime cleanup crosses the boundary as an async callback,
without importing provider SDKs into `forge_agent`.

The model-call middleware tracks the provider/model that produced each child
AI message by its native message ID. Only that candidate receives its original
continuation metadata. Other candidates and all inherited parent messages receive
portable copies. This preserves native thought signatures without exporting them
to product events or JSONL display traces. See the
[Google thought-signature contract](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).

`ToolRuntime.state` supplies parent history for a deep, portable snapshot.
Fork stops before the entire pending delegation batch, keeps complete paired
tool groups, and strips parent system instructions, reasoning, artifacts and
provider metadata. Summary uses the existing summary helper and separate
`subagent_summary` usage records; failure never silently becomes fresh context.
Capacity uses the smallest candidate window with system/tool/output reserves
and a conservative UTF-8 byte estimate. Only oldest complete turns are trimmed;
the latest user task and new instruction are protected. Later tool-result growth
fails explicitly rather than trimming the running graph. Fleet projects the
actual model, context, inherited count, truncation and bounded attempt facts.
Run snapshots also retain the actual provider, so a same-named model on another
provider remains a visible fallback after restore.

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

Session startup and MCP catalog changes rebuild the coding child specs from the
current authorized native catalog. Child tool metadata is copied from the parent:
selected deferred tools are directly declared, parent declaration flags are
cleared, and model-only controls, hidden and script-only tools are excluded.
The generic child loop applies the same exposure middleware as the parent.
Context accounting uses the declared view, while execution retains the full
registered catalog. MCP publication runs after success, error or cancellation
so runtime and session directories remain consistent. Discovery failure on an
unchanged server preserves its old callable tools.
Resource listing, template discovery and resource reads use the same read-only
failure policy. A rejected composed catalog revokes all MCP authorization and
removes MCP tools/resources from both views, including child catalogs, while
preserving the original cancellation exception. Re-enable after correcting the
catalog conflict.

MCP HTTP response hooks enforce an 8 MiB JSON/SSE-message read budget before SDK
parsing and reject compressed bodies. The bounded stdio transport owns only
process pipes, line framing and cancellation; the SDK owns JSON-RPC validation
and the ClientSession. Tool directories across servers and each paginated
resource directory have an 8 MiB aggregate budget.
SSE media types are normalized before selecting event framing; split CRLF
delimiters retain their full wire-byte cost.

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

Interactive background delegation is coordinated by coding-owned `SubagentRuns`.
Submission prepares a frozen `SubagentInvocation`; no child starts until the
session writer has committed the parent call, queued product entry and receipt.
The runner creates an independent `create_agent` / `astream_events(version="v3")`
stream with parent callbacks cleared, no checkpointer and no recursive controls.
It publishes only bounded activity, display trace and token facts to the manager.
The queue caps concurrent and pending children, and child timeout/cancellation
releases capacity without cancelling siblings.

`CodingSession` remains the sole JSONL writer. A terminal background snapshot
and its usage facts commit without waiting for the parent turn or automatic
naming to settle. Background partial output and activity use the same trailing
commit gate; Fleet and live activity read confirmed document values.
`forge.subagent_run.v1` entries
belong to the active branch and never become model messages. The next model
request gets a bounded list of newly durable terminal IDs through middleware;
`task_control` reads a full result on demand. Fleet consumes a status allowlist
and loads safe detail separately from the main `TuiState`/transcript.

Session close/adoption/branch change stops children before shared provider/MCP
resources close. Model/provider/MCP updates and catalog reload reject active or
unsaved runs. Generation checks discard late progress; restore marks queued or
running records interrupted rather than replaying side effects. Git baseline
records startup HEAD only and does not isolate workspace reads or writes.

Saved project workflows are parent model-only calls backed by the same
QuickJS/Codemode worker and Python run coordinator. The loader reads only
`.forge/workflows/<name>.js` files inside the project boundary and reloads the
catalog atomically; `/workflows` only lists the loaded names. A workflow gets a
JSON `workflowArgs` snapshot exposed as read-only data. It cannot add
filesystem, network, process, or module APIs, and child catalogs still exclude
`task`, `workflow`, and Codemode controls.

The workflow scope owns a `workflow_id` and a stable `step_key` for each run.
`runs.run` is idempotent for an existing key and assignment, while a conflicting
assignment is rejected. `runs.control` accepts only run IDs owned by that
scope. `await` expresses ordered dependencies and `Promise.allSettled` keeps
independent reviewer failures visible. A workflow that stops after a failed
step retains completed steps in its partial report and never reruns an earlier
side effect merely because a later step failed.

Workflow budgets are a 900-second default deadline, 1,800-second maximum,
four live children, 16 cumulative launches, and a 64 KiB report. The Python
coordinator enforces these limits even when a script omits a check. It waits
for workflow-owned background children during normal completion, cancellation,
timeout, and script exit; no `Promise.race` remainder is abandoned.

Execution status and business outcome are separate. `failed` means the child
execution failed, timed out, or was cancelled. `completed` with a structured
task-report `outcome: blocked` means the child ran and reported a valid unmet
condition or question. Fleet shows the workflow group and step key alongside
the exact run ID so either state can be inspected without confusing execution
failure with a legitimate blocked report. See the three complete scripts in
[`docs/workflows`](workflows) for the supported sequential, parallel, and
bounded review/fix compositions.

## Attribution

Pi is the primary architecture and learning source. Tau is the Python product
baseline and attribution source for coding tools, sessions, provider
configuration, and the Textual TUI. Forge is independently maintained; Tau is
not an active package, CLI, or configuration name.
