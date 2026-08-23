"""LangChain-native agent runtime used by Forge's production harness.

LangChain owns the model/tool-calling state machine.  The native production
path passes LangChain messages and tools directly; the small Forge projections
below exist only to preserve the public UI event surface.  This module does
not reimplement a second agent loop.

Projection contract (each model call owns one complete lifecycle):

    TurnStart -> MessageStart -> deltas -> MessageEnd -> TurnEnd

Tool executions stream between two model-call lifecycles.  A steering
``HumanMessage`` injected by ``SteeringMiddleware`` is projected as a user
message lifecycle between model calls.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
    ModelCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.stream.transformers import CustomTransformer
from langgraph.types import Command

from forge_agent.context import ForgeRuntimeContext
from forge_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    HumanDecisionType,
    HumanInputRequest,
    HumanInputRequestedEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    TodoItem,
    TodoUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from forge_agent.retry import (
    ForgeModelRetryMiddleware,
    RetryPolicy,
    classify_model_error,
    redact_model_error,
)
from forge_agent.steering import SteeringMiddleware
from forge_agent.subagents import project_subagent_trace
from forge_agent.tool_execution import SequentialToolCallMiddleware
from forge_agent.tools import AgentToolResult, ToolCall
from forge_agent.types import CancellationToken, JSONValue


@dataclass(slots=True)
class LangChainRuntimeState:
    """Ephemeral graph/checkpoint state for one in-progress HITL turn."""

    graph: Any | None = None
    checkpointer: InMemorySaver | None = None
    config: RunnableConfig | None = None
    thread_id: str = field(default_factory=lambda: f"forge-hitl-{uuid4().hex}")
    pending_requests: tuple[HumanInputRequest, ...] = ()
    waiting: bool = False

    def clear(self) -> None:
        """Drop all in-memory graph/checkpoint state."""

        self.graph = None
        self.checkpointer = None
        self.config = None
        self.pending_requests = ()
        self.waiting = False
        self.thread_id = f"forge-hitl-{uuid4().hex}"


def _agent_middleware(
    max_turns: int | None,
    steering: SteeringMiddleware | None,
    middleware: Sequence[Any] = (),
    retry_policy: RetryPolicy | None = None,
) -> tuple[Any, ...]:
    """Return the agent middleware for one run.

    One assistant reply is exactly one model call, so the LangChain
    ``ModelCallLimitMiddleware`` enforces the same contract as the historical
    Forge loop: after ``max_turns`` replies the agent stops instead of silently
    producing another turn or hitting ``GRAPH_RECURSION_LIMIT``.  The steering
    middleware is enabled alongside it and only appends messages.
    """

    # Tool execution is a Forge-wide runtime invariant.  Keep it first so
    # goal/todo/HITL and the native tool are all covered by one ordering gate.
    resolved: list[Any] = [SequentialToolCallMiddleware()]
    if retry_policy is not None and retry_policy.enabled:
        resolved.append(ForgeModelRetryMiddleware(retry_policy))
    resolved.extend(middleware)
    if steering is not None:
        resolved.append(steering)
    if max_turns is not None:
        resolved.append(ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="error"))
    return tuple(resolved)


def _message_text(message: BaseMessage) -> str:
    return "".join(text for kind, text in _content_deltas(message) if kind == "text")


def _content_deltas(message: BaseMessage) -> list[tuple[str, str]]:
    """Extract ordered text/reasoning deltas from native message content."""

    content = message.content
    if isinstance(content, str):
        return [("text", content)] if content else []
    deltas: list[tuple[str, str]] = []
    for block in content:
        if isinstance(block, str):
            if block:
                deltas.append(("text", block))
            continue
        if not isinstance(block, Mapping):
            continue
        block_type = str(block.get("type", "")).lower()
        if block_type in {"reasoning", "thinking"}:
            for key in ("reasoning", "thinking", "text", "content"):
                value = block.get(key)
                if isinstance(value, str) and value:
                    deltas.append(("reasoning", value))
                    break
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            deltas.append(("text", text))
    additional_reasoning = message.additional_kwargs.get("reasoning_content")
    if isinstance(additional_reasoning, str) and additional_reasoning:
        deltas.append(("reasoning", additional_reasoning))
    return deltas


def _mapping_content_delta(delta: Mapping[str, Any]) -> tuple[str, str] | None:
    """Extract one v3 content-block delta without depending on provider fields."""

    block_type = str(delta.get("type", "")).lower()
    if block_type == "text-delta":
        value = delta.get("text")
        if isinstance(value, str) and value:
            return ("text", value)
        return None
    if block_type == "reasoning-delta":
        value = delta.get("reasoning")
        if isinstance(value, str) and value:
            return ("reasoning", value)
        return None
    if block_type in {"reasoning", "thinking"}:
        for key in ("reasoning", "thinking", "text", "content"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return ("reasoning", value)
        return None
    value = delta.get("text")
    if isinstance(value, str) and value:
        return ("text", value)
    value = delta.get("reasoning_content")
    if isinstance(value, str) and value:
        return ("reasoning", value)
    return None


def _mapping_tool_call_chunk(delta: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract one v3 tool-call-chunk block delta, if present."""

    block_type = str(delta.get("type", "")).lower()
    if block_type not in {"block-delta", "legacy-block-delta"}:
        return None
    fields = delta.get("fields")
    if not isinstance(fields, Mapping):
        return None
    if str(fields.get("type", "")).lower() not in {"tool_call_chunk", "server_tool_call_chunk"}:
        return None
    return {
        "id": fields.get("id"),
        "name": fields.get("name"),
        "args": fields.get("args"),
        "index": fields.get("index"),
    }


def _native_tool_calls(message: AIMessage) -> list[ToolCall]:
    """Project a native AIMessage's dict-form tool calls into Forge UI rows."""

    calls: list[ToolCall] = []
    for index, raw in enumerate(message.tool_calls):
        arguments = raw.get("args", {})
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ToolCall(
                id=str(raw.get("id") or f"call-{index}"),
                name=str(raw.get("name") or "unknown"),
                arguments=cast(dict[str, JSONValue], arguments),
            )
        )
    return calls


def _tool_result_from_native_message(message: ToolMessage) -> AgentToolResult:
    """Project a native ToolMessage into the Forge structured result shape."""

    artifact = message.artifact
    if isinstance(artifact, Mapping):
        try:
            stored = AgentToolResult.model_validate(artifact)
        except ValueError:
            pass
        else:
            return AgentToolResult(
                tool_call_id=str(message.tool_call_id),
                name=stored.name,
                ok=stored.ok,
                content=stored.content,
                data=stored.data,
                details=stored.details,
                error=stored.error,
            )
    ok = getattr(message, "status", "success") != "error"
    content = _message_text(message)
    return AgentToolResult(
        tool_call_id=str(message.tool_call_id),
        name=str(getattr(message, "name", "tool")),
        ok=ok,
        content=content,
        error=None if ok else content,
    )


class _ProjectionState:
    """Per-run lifecycle state for the v3 event projection.

    Each model call owns one closed lifecycle: TurnStart -> MessageStart ->
    deltas -> MessageEnd -> TurnEnd.  Message deltas are tracked per message
    id (the id carried by ``message-start`` / whole ``AIMessage`` events and
    by the final ``values`` row), so a final message that never streamed
    chunks still emits its text exactly once.
    """

    def __init__(self, messages: list[AnyMessage]) -> None:
        self.messages = messages
        self.completed_ids: set[str] = {
            str(getattr(message, "id", ""))
            for message in messages
            if isinstance(message, BaseMessage) and getattr(message, "id", None)
        }
        self.current_turn = 0
        self.turn_open = False
        self.message_open = False
        self.current_message_id: str | None = None
        self.emitted_delta_ids: set[str] = set()
        self.deltas_since_open = False
        self.open_text: list[str] = []

    def open_message(self) -> list[AgentEvent]:
        """Open (or keep) the current model-call lifecycle."""
        events: list[AgentEvent] = []
        if not self.turn_open:
            self.current_turn += 1
            self.turn_open = True
            events.append(TurnStartEvent(turn=self.current_turn))
        if not self.message_open:
            self.message_open = True
            self.deltas_since_open = False
            self.open_text.clear()
            events.append(MessageStartEvent())
        return events

    def close_message(self, message: AIMessage | ToolMessage) -> list[AgentEvent]:
        """Close the current model-call lifecycle with a final message."""
        if self.message_open:
            self.message_open = False
            self.deltas_since_open = False
            self.open_text.clear()
        if self.turn_open:
            self.turn_open = False
        return [MessageEndEvent(message=message), TurnEndEvent(turn=self.current_turn)]

    def note_delta(self, message_id: str | None) -> None:
        """Record that a delta was emitted for ``message_id``."""
        self.deltas_since_open = True
        if message_id:
            self.emitted_delta_ids.add(message_id)

    def has_deltas(self, message_id: str | None) -> bool:
        """Return whether deltas were already emitted for this message."""
        if message_id:
            return message_id in self.emitted_delta_ids
        return self.deltas_since_open

    def close_open_lifecycle(self) -> list[AgentEvent]:
        """Close an interrupted lifecycle with the text streamed so far.

        Used on error/cancellation paths where no final ``AIMessage`` exists:
        the open message is closed with a synthesized ``AIMessage`` carrying
        the accumulated text (never added to the transcript), and the turn is
        closed, so consumers never see an unmatched ``MessageStart``.
        """
        events: list[AgentEvent] = []
        if self.message_open:
            self.message_open = False
            events.append(
                MessageEndEvent(
                    message=AIMessage(
                        content="".join(self.open_text),
                        id=self.current_message_id,
                    )
                )
            )
        if self.turn_open:
            self.turn_open = False
            events.append(TurnEndEvent(turn=self.current_turn))
        self.deltas_since_open = False
        self.open_text.clear()
        self.current_message_id = None
        return events


class _NestedTaskProjection:
    """Isolate LangChain v3 child events and reduce activity to parent updates.

    A child ``create_agent`` invocation is streamed under a non-empty
    ``tools:<task-id>`` namespace.  Child messages stay invisible; values are
    cached only as an in-memory snapshot and projected at terminal/root drain.
    Lifecycle and child-tool activity are projected as updates addressed to
    the parent ``task`` call.  Unknown namespace shapes are
    dropped rather than guessed, because falling back to the root projection
    would leak a child transcript into the parent session.
    """

    def __init__(self) -> None:
        # Parent task call id -> the small amount of metadata needed by UI
        # activity updates.  Values are kept as plain strings to ensure no
        # provider objects can cross the event boundary.
        self._tasks: dict[str, dict[str, str | None]] = {}
        # LangGraph child tool events may append a deeper segment to the
        # namespace.  Correlation is therefore keyed by the first
        # ``tools:<run>`` segment, not the complete tuple.
        self._namespaces: dict[str, str] = {}
        # Latest native child values snapshot per parent task.  These messages
        # are transient runtime inputs to the projector only; they are never
        # attached to a Forge event or parent transcript.
        self._snapshots: dict[str, tuple[BaseMessage, ...]] = {}
        # A child may report terminal lifecycle and root task end separately.
        # Keep one drain marker so either path can emit at most one trace.
        self._trace_drained: set[str] = set()

    def record_task_start(self, tool_call: ToolCall) -> None:
        """Remember a root ``task`` call before nested events arrive."""

        if tool_call.name != "task":
            return
        raw_agent = tool_call.arguments.get("agent")
        raw_instruction = tool_call.arguments.get("instruction")
        self._tasks[tool_call.id] = {
            "agent": raw_agent if isinstance(raw_agent, str) else None,
            "instruction": raw_instruction if isinstance(raw_instruction, str) else None,
        }
        self._snapshots.pop(tool_call.id, None)
        self._trace_drained.discard(tool_call.id)

    def record_task_end(self, tool_call_id: str) -> None:
        """Forget a completed root task and all namespaces owned by it."""

        self._tasks.pop(tool_call_id, None)
        self._snapshots.pop(tool_call_id, None)
        self._trace_drained.discard(tool_call_id)
        self._namespaces = {
            namespace_key: owner
            for namespace_key, owner in self._namespaces.items()
            if owner != tool_call_id
        }

    def _trace_update(self, parent_task_id: str) -> ToolExecutionUpdateEvent | None:
        """Drain one cached child snapshot into a safe parent update."""

        if parent_task_id in self._trace_drained:
            return None
        snapshot = self._snapshots.get(parent_task_id)
        if snapshot is None:
            return None
        task = self._tasks.get(parent_task_id)
        if task is None:
            return None
        self._trace_drained.add(parent_task_id)
        trace = project_subagent_trace(snapshot, agent=task.get("agent") or "subagent")
        return ToolExecutionUpdateEvent(
            tool_call_id=parent_task_id,
            message="Subagent trace",
            data={
                "kind": "subagent_trace",
                "version": 1,
                **trace.to_dict(),
            },
        )

    def drain_trace(self, parent_task_id: str) -> ToolExecutionUpdateEvent | None:
        """Publicly named exactly-once trace drain used by the root projector."""

        return self._trace_update(parent_task_id)

    def project(self, method: str, params: Mapping[str, Any]) -> list[AgentEvent]:
        """Project one v3 event, returning only safe parent-facing updates."""

        payload = params.get("data")
        namespace = _event_namespace(params)
        if not namespace:
            return []

        if method == "lifecycle":
            if not isinstance(payload, Mapping):
                return []
            return self._project_lifecycle(namespace, payload)
        if method == "values":
            if not isinstance(payload, Mapping):
                return []
            parent_task_id = self._resolve_parent(namespace, payload)
            if parent_task_id is None or parent_task_id not in self._tasks:
                return []
            raw_messages = payload.get("messages")
            if not isinstance(raw_messages, Sequence) or isinstance(
                raw_messages, (str, bytes, bytearray)
            ):
                return []
            snapshot = tuple(
                message for message in raw_messages if isinstance(message, BaseMessage)
            )
            self._snapshots[parent_task_id] = snapshot
            return []
        if method != "tools" or not isinstance(payload, Mapping):
            # ``messages`` are deliberately suppressed here.
            return []

        parent_task_id = self._resolve_parent(namespace, payload)
        if parent_task_id is None:
            return []
        task = self._tasks.get(parent_task_id)
        if task is None:
            return []
        event_name = payload.get("event")
        raw_name = payload.get("tool_name")
        if isinstance(raw_name, str) and raw_name:
            tool_name = raw_name
        else:
            output = payload.get("output")
            output_name = getattr(output, "name", None)
            tool_name = output_name if isinstance(output_name, str) and output_name else "tool"
        if event_name == "tool-started":
            summary = _nested_tool_summary(tool_name, payload.get("input"))
            return [
                self._activity_update(
                    parent_task_id,
                    task,
                    status="running",
                    message=f"{tool_name} started",
                    activity={
                        "phase": "tool_started",
                        "tool": tool_name,
                        "summary": summary,
                    },
                )
            ]
        if event_name == "tool-finished":
            return [
                self._activity_update(
                    parent_task_id,
                    task,
                    status="running",
                    message=f"{tool_name} finished",
                    activity={
                        "phase": "tool_finished",
                        "tool": tool_name,
                        "summary": f"Finished {tool_name}",
                    },
                )
            ]
        return []

    def _project_lifecycle(
        self,
        namespace: tuple[str, ...],
        payload: Mapping[str, Any],
    ) -> list[AgentEvent]:
        parent_task_id = self._resolve_parent(namespace, payload)
        if parent_task_id is None:
            return []
        task = self._tasks.get(parent_task_id)
        if task is None:
            return []
        event_name = payload.get("event")
        if event_name == "started":
            raw_graph_name = payload.get("graph_name")
            if isinstance(raw_graph_name, str) and raw_graph_name:
                task["agent"] = raw_graph_name
            agent = task.get("agent") or "subagent"
            return [
                self._activity_update(
                    parent_task_id,
                    task,
                    status="running",
                    message=f"{agent} started",
                    activity={"phase": "started", "summary": f"{agent} started"},
                )
            ]
        # The root task's ToolExecutionEndEvent carries the durable artifact
        # and remains the sole authoritative completion.  A failed/interrupted
        # lifecycle is useful activity, but a completed lifecycle is not a
        # second end event.
        if event_name == "failed":
            error = payload.get("error")
            summary = error if isinstance(error, str) and error else "Subagent failed"
            events: list[AgentEvent] = []
            trace = self._trace_update(parent_task_id)
            if trace is not None:
                events.append(trace)
            events.extend(
                [
                    self._activity_update(
                        parent_task_id,
                        task,
                        status="failed",
                        message=summary,
                        activity={"phase": "failed", "summary": summary},
                    )
                ]
            )
            return events
        if event_name in {"interrupted", "drained"}:
            events = []
            trace = self._trace_update(parent_task_id)
            if trace is not None:
                events.append(trace)
            events.append(
                self._activity_update(
                    parent_task_id,
                    task,
                    status="cancelled",
                    message="Subagent cancelled",
                    activity={"phase": "interrupted", "summary": "Subagent cancelled"},
                )
            )
            return events
        if event_name == "completed":
            trace = self._trace_update(parent_task_id)
            return [trace] if trace is not None else []
        return []

    def _resolve_parent(
        self,
        namespace: tuple[str, ...],
        payload: Mapping[str, Any],
    ) -> str | None:
        """Resolve a nested namespace to a known root task call id."""

        if not namespace:
            return None
        first = namespace[0]
        name, separator, namespace_call_id = first.partition(":")
        if name != "tools" or not separator or not namespace_call_id:
            return None

        cause_call_id = _nested_cause_tool_call_id(payload)
        known_cause = cause_call_id if cause_call_id in self._tasks else None
        namespace_key = namespace[0]
        owner = self._namespaces.get(namespace_key)
        if owner is not None:
            # A lifecycle cause is a useful consistency check when present.
            if known_cause is not None and known_cause != owner:
                return None
            return owner

        # Some LangChain versions use the parent tool-call id directly in the
        # namespace; current versions use a generated task id and include the
        # parent id in lifecycle ``cause``.  Support both shapes without
        # guessing for unknown identifiers.
        direct_owner = namespace_call_id if namespace_call_id in self._tasks else None
        if direct_owner is not None and known_cause is not None and direct_owner != known_cause:
            return None
        owner = direct_owner or known_cause
        if owner is None:
            return None
        self._namespaces[namespace_key] = owner
        return owner

    def _activity_update(
        self,
        parent_task_id: str,
        task: Mapping[str, str | None],
        *,
        status: str,
        message: str,
        activity: dict[str, JSONValue],
    ) -> ToolExecutionUpdateEvent:
        agent = task.get("agent") or "subagent"
        return ToolExecutionUpdateEvent(
            tool_call_id=parent_task_id,
            message=message,
            data={
                "kind": "subagent_activity",
                "agent": agent,
                "status": status,
                "activity": activity,
            },
        )


def _coerce_namespace(value: Any) -> tuple[str, ...] | None:
    """Return a validated namespace tuple, or ``None`` for malformed input."""

    if value is None:
        return None
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(value)
        if all(isinstance(item, str) and item for item in values):
            return cast(tuple[str, ...], values)
    return None


def _event_namespace(params: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Resolve the nested namespace carried by either v3 params location.

    LangChain versions have emitted the namespace on ``params`` and inside
    ``params.data``.  A non-empty namespace in either location must keep the
    event on the nested path; conflicting or malformed nested values fail
    closed instead of allowing the root projector to see child state.
    """

    param_value = params.get("namespace")
    payload = params.get("data")
    data_value = payload.get("namespace") if isinstance(payload, Mapping) else None
    param_namespace = _coerce_namespace(param_value)
    data_namespace = _coerce_namespace(data_value)

    # A non-empty raw value that cannot be validated is still a nested hint;
    # route it to the nested projector, which will drop it without guessing.
    if (bool(param_value) and param_namespace is None) or (
        bool(data_value) and data_namespace is None
    ):
        return ()
    if param_namespace and data_namespace:
        if param_namespace != data_namespace:
            return ()
        return param_namespace
    if param_namespace:
        return param_namespace
    if data_namespace:
        return data_namespace
    return None


def _nested_cause_tool_call_id(payload: Mapping[str, Any]) -> str | None:
    cause = payload.get("cause")
    if not isinstance(cause, Mapping):
        return None
    for key in ("tool_call_id", "toolCallId"):
        value = cause.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _nested_tool_summary(tool_name: str, raw_input: Any) -> str:
    """Build a short activity summary without exposing child tool arguments."""

    del raw_input
    return f"Calling {tool_name}"


def _json_safe(value: Any, *, depth: int = 0) -> JSONValue | None:
    """Project untrusted interrupt arguments into bounded JSON values."""

    if depth > 8:
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            safe = _json_safe(item, depth=depth + 1)
            if safe is not None:
                result[key] = safe
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        result_list: list[JSONValue] = []
        for item in value:
            safe = _json_safe(item, depth=depth + 1)
            if safe is not None:
                result_list.append(safe)
        return result_list
    return None


def _project_retry_event(payload: Any) -> RetryEvent | None:
    """Project only the allowlisted retry custom event into the public stream."""

    if not isinstance(payload, Mapping) or payload.get("type") != "forge.model_retry.v1":
        return None
    attempt = payload.get("attempt")
    max_attempts = payload.get("max_attempts")
    delay = payload.get("delay_seconds")
    if (
        type(attempt) is not int
        or type(max_attempts) is not int
        or not 1 <= attempt <= max_attempts
        or not isinstance(delay, (int, float))
        or isinstance(delay, bool)
        or delay < 0
    ):
        return None
    raw_message = payload.get("message")
    message = redact_model_error(
        raw_message if isinstance(raw_message, str) else "Model call failed"
    )
    data: dict[str, JSONValue] = {}
    kind = payload.get("kind")
    if isinstance(kind, str) and kind in {
        "transient",
        "abort",
        "overflow",
        "auth",
        "quota",
        "invalid_request",
        "unknown",
    }:
        data["kind"] = kind
    status = payload.get("status_code")
    if type(status) is int:
        data["status_code"] = status
    return RetryEvent(
        attempt=attempt,
        max_attempts=max_attempts,
        delay_seconds=float(delay),
        message=message,
        data=data or None,
    )


def _todo_snapshot(value: Any) -> tuple[TodoItem, ...] | None:
    """Validate a LangChain planning state snapshot without coding imports."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    items: list[TodoItem] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            return None
        try:
            item = TodoItem.model_validate(raw)
        except ValueError:
            return None
        if not item.content.strip():
            return None
        items.append(item)
    return tuple(items)


def _interrupt_parts(raw_interrupt: Any) -> tuple[str, Mapping[str, Any]] | None:
    if isinstance(raw_interrupt, Mapping):
        raw_id = raw_interrupt.get("id")
        value = raw_interrupt.get("value")
    else:
        raw_id = getattr(raw_interrupt, "id", None)
        value = getattr(raw_interrupt, "value", None)
    if not isinstance(raw_id, str) or not raw_id or not isinstance(value, Mapping):
        return None
    return raw_id, value


def _project_human_requests(
    raw_interrupts: Any,
    *,
    pending_tool_calls: Mapping[str, ToolCall],
    state_messages: Sequence[AnyMessage],
) -> tuple[HumanInputRequest, ...]:
    """Project HITL action requests and correlate them with tool-call ids."""

    if not isinstance(raw_interrupts, Sequence) or isinstance(
        raw_interrupts, (str, bytes, bytearray)
    ):
        return ()
    requests: list[HumanInputRequest] = []
    used_ids: set[str] = set()
    fallback_calls = [call for call in pending_tool_calls.values() if call.id not in used_ids]
    for raw_interrupt in raw_interrupts:
        parts = _interrupt_parts(raw_interrupt)
        if parts is None:
            continue
        interrupt_id, value = parts
        actions = value.get("action_requests")
        reviews = value.get("review_configs")
        if not isinstance(actions, Sequence) or not isinstance(reviews, Sequence):
            continue
        for index, raw_action in enumerate(actions):
            if not isinstance(raw_action, Mapping):
                continue
            name = raw_action.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = raw_action.get("args")
            safe_args = _json_safe(args)
            if not isinstance(safe_args, dict):
                safe_args = {}
            allowed: tuple[HumanDecisionType, ...] = ("respond",)
            if index < len(reviews) and isinstance(reviews[index], Mapping):
                raw_allowed = reviews[index].get("allowed_decisions")
                if isinstance(raw_allowed, Sequence) and not isinstance(
                    raw_allowed, (str, bytes, bytearray)
                ):
                    allowed = cast(
                        tuple[HumanDecisionType, ...],
                        tuple(
                            item
                            for item in raw_allowed
                            if item in {"respond", "approve", "edit", "reject"}
                        )
                        or ("respond",),
                    )
            call = next(
                (
                    candidate
                    for candidate in (*fallback_calls, *pending_tool_calls.values())
                    if candidate.name == name and candidate.id not in used_ids
                ),
                None,
            )
            if call is None:
                for message in reversed(state_messages):
                    if not isinstance(message, AIMessage):
                        continue
                    for candidate in _native_tool_calls(message):
                        if candidate.name == name and candidate.id not in used_ids:
                            call = candidate
                            break
                    if call is not None:
                        break
            tool_call_id = call.id if call is not None else f"interrupt-{index}"
            used_ids.add(tool_call_id)
            description = raw_action.get("description")
            requests.append(
                HumanInputRequest(
                    interrupt_id=interrupt_id,
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    arguments=safe_args,
                    allowed_decisions=allowed,
                    description=description if isinstance(description, str) else None,
                )
            )
    return tuple(requests)


async def run_langchain_agent(
    *,
    provider: BaseChatModel,
    model: str,
    system: str,
    messages: list[AnyMessage],
    tools: Sequence[BaseTool] = (),
    max_turns: int | None = None,
    signal: CancellationToken | None = None,
    runtime_context: ForgeRuntimeContext | None = None,
    steering: SteeringMiddleware | None = None,
    queue_update: Callable[[], QueueUpdateEvent] | None = None,
    middleware: Sequence[Any] = (),
    retry_policy: RetryPolicy | None = None,
    runtime_state: LangChainRuntimeState | None = None,
    resume_decisions: Sequence[Mapping[str, JSONValue]] | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run one native LangChain agent and project its v3 events for Forge UI.

    Live text deltas are streamed as they arrive; a final model message that
    produced no chunked deltas emits its text once at completion.  Errors are
    always surfaced as ``ErrorEvent`` rows.
    """

    yield AgentStartEvent()
    if max_turns is not None and max_turns < 1:
        yield ErrorEvent(message="max_turns must be at least 1", recoverable=False)
        yield AgentEndEvent()
        return

    if runtime_state is not None and runtime_state.graph is not None:
        graph = runtime_state.graph
    else:
        checkpointer = None
        if runtime_state is not None:
            checkpointer = runtime_state.checkpointer or InMemorySaver()
            runtime_state.checkpointer = checkpointer
        graph = create_agent(
            provider,
            tools=list(tools),
            system_prompt=system,
            middleware=cast(
                Any,
                _agent_middleware(max_turns, steering, middleware, retry_policy),
            ),
            context_schema=ForgeRuntimeContext,
            checkpointer=checkpointer,
        )
        if runtime_state is not None:
            runtime_state.graph = graph
    input_message_count = len(messages)
    config: RunnableConfig = (
        runtime_state.config.copy() if runtime_state is not None and runtime_state.config else {}
    )
    if runtime_state is not None and "configurable" not in config:
        config["configurable"] = {"thread_id": runtime_state.thread_id}
    if max_turns is not None:
        # The recursion limit only guards against runaway graph super-steps. The
        # authoritative turn limit is the ModelCallLimitMiddleware, so each
        # assistant reply counts as exactly one model call.  Keep the graph
        # bound comfortably above the worst legal round (one model call + one
        # tool batch per turn) so the middleware error fires before the graph
        # ever trips its own limit.
        config["recursion_limit"] = max(25, max_turns * 2 + 2)
    if runtime_state is not None:
        runtime_state.config = config

    state = _ProjectionState(messages)
    pending_tool_calls: dict[str, ToolCall] = {}
    completed_tool_call_ids: set[str] = set()
    partial_arguments: dict[str, str] = {}
    partial_tool_names: dict[str, str] = {}
    nested_projection = _NestedTaskProjection()
    previous_todos: tuple[TodoItem, ...] | None = None
    if runtime_state is not None and resume_decisions is not None:
        runtime_state.waiting = False
        runtime_state.pending_requests = ()

    try:
        # The first turn opens eagerly so harness listeners (prompt projection,
        # auto-naming, persistence) run before the first model call streams.
        state.current_turn = 1
        state.turn_open = True
        yield TurnStartEvent(turn=state.current_turn)
        event_kwargs: dict[str, Any] = {
            "version": "v3",
            "config": config or None,
            "transformers": (CustomTransformer,),
        }
        if runtime_context is not None:
            event_kwargs["context"] = runtime_context
        stream_input: Any = {"messages": messages}
        if resume_decisions is not None:
            stream_input = Command(
                resume={"decisions": [dict(decision) for decision in resume_decisions]}
            )
        event_stream = cast(Any, graph).astream_events(stream_input, **event_kwargs)
        if inspect.isawaitable(event_stream):
            event_stream = await event_stream
        async for event in event_stream:
            if signal is not None and signal.is_cancelled():
                yield ErrorEvent(message="Agent run cancelled", recoverable=True)
                break

            if not isinstance(event, Mapping):
                continue
            method = event.get("method")
            params = event.get("params")
            if not isinstance(params, Mapping):
                continue
            payload = params.get("data")
            nested_namespace = _event_namespace(params)
            if method == "messages":
                if nested_namespace is not None:
                    # Child messages are intentionally invisible to the
                    # parent projection and transcript.
                    continue
                for item in _project_v3_message_event(
                    payload,
                    state=state,
                    partial_arguments=partial_arguments,
                    partial_tool_names=partial_tool_names,
                    queue_update=queue_update,
                ):
                    yield item
                continue
            if method == "tools":
                if nested_namespace is not None:
                    for item in nested_projection.project(method, params):
                        yield item
                    continue
                projected = _project_v3_tool_event(payload)
                for item in projected:
                    if isinstance(item, ToolExecutionStartEvent):
                        if item.tool_call.id in pending_tool_calls:
                            continue
                        pending_tool_calls[item.tool_call.id] = item.tool_call
                        nested_projection.record_task_start(item.tool_call)
                    elif isinstance(item, ToolExecutionEndEvent):
                        if item.result.tool_call_id in completed_tool_call_ids:
                            continue
                        completed_tool_call_ids.add(item.result.tool_call_id)
                        pending_tool_calls.pop(item.result.tool_call_id, None)
                        if item.result.name == "task":
                            trace = nested_projection.drain_trace(item.result.tool_call_id)
                            if trace is not None:
                                yield trace
                            nested_projection.record_task_end(item.result.tool_call_id)
                    yield item
                continue
            if method == "lifecycle":
                for item in nested_projection.project(method, params):
                    yield item
                continue
            if method == "custom":
                retry_event = _project_retry_event(payload)
                if retry_event is not None:
                    yield retry_event
                continue
            if method != "values" or not isinstance(payload, Mapping):
                continue
            if nested_namespace is not None:
                # Child values are cached only as a transient snapshot for the
                # nested trace projector; they never enter root state.
                for item in nested_projection.project(method, params):
                    yield item
                continue
            raw_messages = payload.get("messages")
            if not isinstance(raw_messages, Sequence):
                continue
            todo_event: TodoUpdateEvent | None = None
            if "todos" in payload:
                snapshot = _todo_snapshot(payload.get("todos"))
                if snapshot is not None and snapshot != previous_todos:
                    previous_todos = snapshot
                    todo_event = TodoUpdateEvent(todos=snapshot)
            human_event: HumanInputRequestedEvent | None = None
            raw_interrupts = params.get("interrupts")
            if raw_interrupts is None:
                raw_interrupts = payload.get("interrupts")
            if (
                isinstance(raw_interrupts, Sequence)
                and not isinstance(raw_interrupts, (str, bytes, bytearray))
                and raw_interrupts
            ):
                requests = _project_human_requests(
                    raw_interrupts,
                    pending_tool_calls=pending_tool_calls,
                    state_messages=(
                        *state.messages,
                        *[
                            item
                            for item in raw_messages
                            if isinstance(item, (AIMessage, ToolMessage, HumanMessage))
                        ],
                    ),
                )
                if requests:
                    if runtime_state is not None:
                        runtime_state.pending_requests = requests
                        runtime_state.waiting = True
                    first = requests[0]
                    human_event = HumanInputRequestedEvent(
                        interrupt_id=first.interrupt_id,
                        tool_call_id=first.tool_call_id,
                        tool_name=first.tool_name,
                        arguments=first.arguments,
                        allowed_decisions=first.allowed_decisions,
                        description=first.description,
                        requests=requests,
                    )
            for message_index, raw_message in enumerate(raw_messages):
                if message_index < input_message_count:
                    continue
                if not isinstance(raw_message, (AIMessage, ToolMessage, HumanMessage)):
                    continue
                raw_id = str(getattr(raw_message, "id", "") or "")
                if raw_id and raw_id in state.completed_ids:
                    continue
                if isinstance(raw_message, HumanMessage):
                    # A steering HumanMessage injected by the middleware.  The
                    # user lifecycle was already projected from its messages
                    # event; here it only joins the transcript.
                    state.messages.append(raw_message)
                    if raw_id:
                        state.completed_ids.add(raw_id)
                    continue
                if isinstance(raw_message, AIMessage):
                    calls = _native_tool_calls(raw_message)
                    for item in state.open_message():
                        yield item
                    if not state.has_deltas(raw_id):
                        for kind, delta_text in _content_deltas(raw_message):
                            state.note_delta(raw_id)
                            if kind == "reasoning":
                                yield ThinkingDeltaEvent(delta=delta_text)
                            else:
                                state.open_text.append(delta_text)
                                yield MessageDeltaEvent(delta=delta_text)
                    state.messages.append(raw_message)
                    if raw_id:
                        state.completed_ids.add(raw_id)
                    for item in state.close_message(raw_message):
                        yield item
                    already_started = set(pending_tool_calls)
                    pending_tool_calls.update({call.id: call for call in calls})
                    for call in calls:
                        if call.id not in already_started:
                            nested_projection.record_task_start(call)
                            yield ToolExecutionStartEvent(tool_call=call)
                else:
                    result = _tool_result_from_native_message(raw_message)
                    if result.name == "task":
                        trace = nested_projection.drain_trace(result.tool_call_id)
                        if trace is not None:
                            # Root values can append the ToolMessage before a
                            # later lifecycle event, so drain first to keep
                            # the trace between task call and task result.
                            yield trace
                    state.messages.append(raw_message)
                    if raw_id:
                        state.completed_ids.add(raw_id)
                    if result.tool_call_id not in completed_tool_call_ids:
                        completed_tool_call_ids.add(result.tool_call_id)
                        if result.name == "task":
                            nested_projection.record_task_end(result.tool_call_id)
                        yield ToolExecutionEndEvent(result=result)
                    pending_tool_calls.pop(result.tool_call_id, None)
                    partial_arguments.pop(result.tool_call_id, None)
                    partial_tool_names.pop(result.tool_call_id, None)
            if todo_event is not None:
                yield todo_event
            if human_event is not None:
                yield human_event
    except ModelCallLimitExceededError:
        yield ErrorEvent(
            message=f"Agent loop stopped after reaching max_turns={max_turns or 0}",
            recoverable=True,
        )
    except asyncio.CancelledError:
        # Cancellation must still close whatever lifecycle was started, then
        # propagate so the harness/session interruption handling runs.
        for item in state.close_open_lifecycle():
            yield item
        yield AgentEndEvent()
        raise
    except Exception as exc:  # noqa: BLE001 - surface model/tool failures as Forge events
        classification = classify_model_error(exc)
        data: dict[str, JSONValue] | None = None
        if classification.kind != "unknown":
            data = {"kind": classification.kind}
            if classification.status_code is not None:
                data["status_code"] = classification.status_code
        yield ErrorEvent(message=redact_model_error(exc), recoverable=False, data=data)
    for item in state.close_open_lifecycle():
        yield item
    yield AgentEndEvent()
    if runtime_state is not None and not runtime_state.waiting:
        runtime_state.clear()


def _project_v3_message_event(
    payload: Any,
    *,
    state: _ProjectionState,
    partial_arguments: dict[str, str],
    partial_tool_names: dict[str, str],
    queue_update: Callable[[], QueueUpdateEvent] | None,
) -> list[AgentEvent]:
    if not isinstance(payload, tuple) or not payload:
        return []
    item = payload[0]
    events: list[AgentEvent] = []

    if isinstance(item, HumanMessage):
        # A steering message injected by the middleware: project its user
        # lifecycle immediately and announce the drained queue so the TUI
        # stops showing it as pending.
        state.messages.append(item)
        message_id = str(getattr(item, "id", "") or "")
        if message_id:
            state.completed_ids.add(message_id)
        events.append(MessageStartEvent(message_role="user"))
        events.append(MessageEndEvent(message=item))
        if queue_update is not None:
            events.append(queue_update())
        return events

    if isinstance(item, AIMessageChunk):
        events.extend(state.open_message())
        deltas = _content_deltas(item)
        for kind, text in deltas:
            state.note_delta(str(getattr(item, "id", "") or "") or state.current_message_id)
            if kind == "reasoning":
                events.append(ThinkingDeltaEvent(delta=text))
            else:
                state.open_text.append(text)
                events.append(MessageDeltaEvent(delta=text))
        for chunk in item.tool_call_chunks:
            projected = _project_tool_call_chunk(
                chunk,
                partial_arguments=partial_arguments,
                partial_tool_names=partial_tool_names,
            )
            if projected is not None:
                events.append(projected)
        if item.id:
            state.current_message_id = str(item.id)
        return events

    if isinstance(item, AIMessage):
        events.extend(state.open_message())
        if item.id:
            state.current_message_id = str(item.id)
        if not state.has_deltas(state.current_message_id):
            for kind, delta_text in _content_deltas(item):
                state.note_delta(state.current_message_id)
                if kind == "reasoning":
                    events.append(ThinkingDeltaEvent(delta=delta_text))
                else:
                    state.open_text.append(delta_text)
                    events.append(MessageDeltaEvent(delta=delta_text))
        return events

    if isinstance(item, Mapping):
        event_name = item.get("event")
        if event_name == "message-start":
            start_message_id = item.get("id") or item.get("message_id")
            if start_message_id:
                state.current_message_id = str(start_message_id)
            events.extend(state.open_message())
            return events
        if event_name == "content-block-delta":
            delta = item.get("delta")
            if isinstance(delta, Mapping):
                content_delta = _mapping_content_delta(delta)
                if content_delta is not None:
                    kind, delta_text = content_delta
                    events.extend(state.open_message())
                    state.note_delta(state.current_message_id)
                    if kind == "reasoning":
                        events.append(ThinkingDeltaEvent(delta=delta_text))
                    else:
                        state.open_text.append(delta_text)
                        events.append(MessageDeltaEvent(delta=delta_text))
                    return events
                chunk_data = _mapping_tool_call_chunk(delta)
                if chunk_data is not None:
                    events.extend(state.open_message())
                    projected = _project_tool_call_chunk(
                        chunk_data,
                        partial_arguments=partial_arguments,
                        partial_tool_names=partial_tool_names,
                    )
                    if projected is not None:
                        events.append(projected)
                    return events
        if event_name == "content-block-start":
            events.extend(state.open_message())
    return events


def _project_tool_call_chunk(
    chunk: Mapping[str, Any],
    *,
    partial_arguments: dict[str, str],
    partial_tool_names: dict[str, str],
) -> ToolExecutionUpdateEvent | None:
    """Project one partial tool-call argument chunk into an update event.

    v3 ``tool_call_chunk.args`` carries the cumulative arguments-so-far, not
    an incremental fragment, so each chunk *replaces* the previous value;
    concatenating cumulative values would produce duplicated JSON.
    """

    raw_id = chunk.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        return None
    raw_name = chunk.get("name")
    args = chunk.get("args")
    if not isinstance(args, str):
        return None
    partial_arguments[raw_id] = args
    if isinstance(raw_name, str) and raw_name:
        partial_tool_names[raw_id] = raw_name
    return ToolExecutionUpdateEvent(
        tool_call_id=raw_id,
        message="streaming tool arguments",
        data={
            "arguments_delta": partial_arguments[raw_id],
            "tool_name": partial_tool_names.get(raw_id),
        },
    )


def _project_v3_tool_event(
    payload: Any,
) -> list[AgentEvent]:
    if not isinstance(payload, Mapping):
        return []
    event = payload.get("event")
    if event == "tool-started":
        raw_id = str(payload.get("tool_call_id") or "")
        raw_name = str(payload.get("tool_name") or "tool")
        raw_input = payload.get("input")
        arguments = (
            {
                str(key): cast(JSONValue, value)
                for key, value in raw_input.items()
                if key != "runtime"
            }
            if isinstance(raw_input, dict)
            else {}
        )
        return [
            ToolExecutionStartEvent(
                tool_call=ToolCall(id=raw_id, name=raw_name, arguments=arguments)
            )
        ]
    if event == "tool-finished":
        output = payload.get("output")
        if isinstance(output, ToolMessage):
            result = _tool_result_from_native_message(output)
        else:
            content = str(output or "")
            result = AgentToolResult(
                tool_call_id=str(payload.get("tool_call_id") or ""),
                name=str(payload.get("tool_name") or "tool"),
                ok=True,
                content=content,
            )
        return [ToolExecutionEndEvent(result=result)]
    return []
