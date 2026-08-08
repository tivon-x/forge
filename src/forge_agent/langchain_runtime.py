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
from typing import Any, cast

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

from forge_agent.context import ForgeRuntimeContext
from forge_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
    ThinkingDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from forge_agent.steering import SteeringMiddleware
from forge_agent.tools import AgentToolResult, ToolCall
from forge_agent.types import CancellationToken, JSONValue


def _agent_middleware(
    max_turns: int | None,
    steering: SteeringMiddleware | None,
) -> tuple[SteeringMiddleware | ModelCallLimitMiddleware, ...]:
    """Return the agent middleware for one run.

    One assistant reply is exactly one model call, so the LangChain
    ``ModelCallLimitMiddleware`` enforces the same contract as the historical
    Forge loop: after ``max_turns`` replies the agent stops instead of silently
    producing another turn or hitting ``GRAPH_RECURSION_LIMIT``.  The steering
    middleware is enabled alongside it and only appends messages.
    """

    middleware: list[SteeringMiddleware | ModelCallLimitMiddleware] = []
    if steering is not None:
        middleware.append(steering)
    if max_turns is not None:
        middleware.append(ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="error"))
    return tuple(middleware)


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
            events.append(MessageStartEvent())
        return events

    def close_message(self, message: AIMessage | ToolMessage) -> list[AgentEvent]:
        """Close the current model-call lifecycle with a final message."""
        if self.message_open:
            self.message_open = False
            self.deltas_since_open = False
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

    graph = create_agent(
        provider,
        tools=list(tools),
        system_prompt=system,
        middleware=cast(Any, _agent_middleware(max_turns, steering)),
    )
    input_message_count = len(messages)
    config: RunnableConfig = {}
    if max_turns is not None:
        # The recursion limit only guards against runaway graph super-steps. The
        # authoritative turn limit is the ModelCallLimitMiddleware, so each
        # assistant reply counts as exactly one model call.  Keep the graph
        # bound comfortably above the worst legal round (one model call + one
        # tool batch per turn) so the middleware error fires before the graph
        # ever trips its own limit.
        config["recursion_limit"] = max(25, max_turns * 2 + 2)

    state = _ProjectionState(messages)
    pending_tool_calls: dict[str, ToolCall] = {}
    completed_tool_call_ids: set[str] = set()
    partial_arguments: dict[str, str] = {}
    partial_tool_names: dict[str, str] = {}

    try:
        # The first turn opens eagerly so harness listeners (prompt projection,
        # auto-naming, persistence) run before the first model call streams.
        state.current_turn = 1
        state.turn_open = True
        yield TurnStartEvent(turn=state.current_turn)
        event_kwargs: dict[str, Any] = {
            "version": "v3",
            "config": config or None,
        }
        if runtime_context is not None:
            event_kwargs["context"] = runtime_context
        event_stream = cast(Any, graph).astream_events({"messages": messages}, **event_kwargs)
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
            if method == "messages":
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
                projected = _project_v3_tool_event(payload)
                for item in projected:
                    if isinstance(item, ToolExecutionStartEvent):
                        if item.tool_call.id in pending_tool_calls:
                            continue
                        pending_tool_calls[item.tool_call.id] = item.tool_call
                    elif isinstance(item, ToolExecutionEndEvent):
                        if item.result.tool_call_id in completed_tool_call_ids:
                            continue
                        completed_tool_call_ids.add(item.result.tool_call_id)
                        pending_tool_calls.pop(item.result.tool_call_id, None)
                    yield item
                continue
            if method != "values" or not isinstance(payload, Mapping):
                continue
            raw_messages = payload.get("messages")
            if not isinstance(raw_messages, Sequence):
                continue
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
                            yield ToolExecutionStartEvent(tool_call=call)
                else:
                    result = _tool_result_from_native_message(raw_message)
                    state.messages.append(raw_message)
                    if raw_id:
                        state.completed_ids.add(raw_id)
                    if result.tool_call_id not in completed_tool_call_ids:
                        completed_tool_call_ids.add(result.tool_call_id)
                        yield ToolExecutionEndEvent(result=result)
                    pending_tool_calls.pop(result.tool_call_id, None)
                    partial_arguments.pop(result.tool_call_id, None)
                    partial_tool_names.pop(result.tool_call_id, None)
    except ModelCallLimitExceededError:
        yield ErrorEvent(
            message=f"Agent loop stopped after reaching max_turns={max_turns or 0}",
            recoverable=True,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface model/tool failures as Forge events
        yield ErrorEvent(message=str(exc), recoverable=False)
    if state.turn_open:
        yield TurnEndEvent(turn=state.current_turn)
    yield AgentEndEvent()


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
    """Accumulate one partial tool-call argument chunk into an update event."""

    raw_id = chunk.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        return None
    raw_name = chunk.get("name")
    args = chunk.get("args")
    if not isinstance(args, str):
        return None
    partial_arguments[raw_id] = partial_arguments.get(raw_id, "") + args
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
