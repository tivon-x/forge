"""LangChain-native agent runtime used by Forge's production harness.

LangChain owns the model/tool-calling state machine.  The native production
path passes LangChain messages and tools directly; the small Forge projections
below exist only to preserve the public UI event surface.  This module does
not reimplement a second agent loop.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Mapping, Sequence
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
    ThinkingDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from forge_agent.message_codec import to_langchain_message
from forge_agent.tools import AgentToolResult, ToolCall
from forge_agent.types import CancellationToken, JSONValue


def _agent_middleware(max_turns: int | None) -> tuple[ModelCallLimitMiddleware, ...]:
    """Return the agent middleware enforcing Forge's ``max_turns`` semantics.

    One assistant reply is exactly one model call, so the LangChain
    ``ModelCallLimitMiddleware`` enforces the same contract as the historical
    Forge loop: after ``max_turns`` replies the agent stops instead of silently
    producing another turn or hitting ``GRAPH_RECURSION_LIMIT``.
    """

    if max_turns is None:
        return ()
    return (ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="error"),)


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
        middleware=_agent_middleware(max_turns),
    )
    input_messages = [to_langchain_message(message) for message in messages]
    input_message_count = len(input_messages)
    config: RunnableConfig = {}
    if max_turns is not None:
        # The recursion limit only guards against runaway graph super-steps. The
        # authoritative turn limit is the ModelCallLimitMiddleware, so each
        # assistant reply counts as exactly one model call.  Keep the graph
        # bound comfortably above the worst legal round (one model call + one
        # tool batch per turn) so the middleware error fires before the graph
        # ever trips its own limit.
        config["recursion_limit"] = max(25, max_turns * 2 + 2)

    current_turn = 1
    turn_open = True
    message_started = False
    streamed_ids: set[str] = set()
    completed_ids: set[str] = {
        str(getattr(message, "id", ""))
        for message in messages
        if isinstance(message, BaseMessage) and getattr(message, "id", None)
    }
    pending_tool_calls: dict[str, ToolCall] = {}
    completed_tool_call_ids: set[str] = set()
    delta_emitted: list[bool] = [False]

    def ensure_turn() -> list[AgentEvent]:
        nonlocal current_turn, turn_open, message_started
        if not turn_open:
            current_turn += 1
            turn_open = True
            message_started = False
        if not message_started:
            message_started = True
            delta_emitted[0] = False
            return [MessageStartEvent()]
        return []

    try:
        yield TurnStartEvent(turn=current_turn)
        event_kwargs: dict[str, Any] = {
            "version": "v3",
            "config": config or None,
        }
        if runtime_context is not None:
            event_kwargs["context"] = runtime_context
        event_stream = cast(Any, graph).astream_events({"messages": input_messages}, **event_kwargs)
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
                    ensure_turn=ensure_turn,
                    streamed_ids=streamed_ids,
                    delta_emitted=delta_emitted,
                ):
                    yield item
                continue
            if method == "tools":
                projected = _project_v3_tool_event(
                    payload,
                    current_turn=current_turn,
                    pending_tool_calls=pending_tool_calls,
                )
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
                if not isinstance(raw_message, (AIMessage, ToolMessage)):
                    continue
                raw_id = str(getattr(raw_message, "id", "") or "")
                if raw_id and raw_id in completed_ids:
                    continue
                if isinstance(raw_message, AIMessage):
                    calls = _native_tool_calls(raw_message)
                    if not calls:
                        for item in ensure_turn():
                            yield item
                        text = _message_text(raw_message)
                        if text and not streamed_ids and not delta_emitted[0]:
                            delta_emitted[0] = True
                            yield MessageDeltaEvent(delta=text)
                    else:
                        for item in ensure_turn():
                            yield item
                    messages.append(raw_message)
                    if raw_id:
                        completed_ids.add(raw_id)
                    yield MessageEndEvent(message=raw_message)
                    already_started = set(pending_tool_calls)
                    pending_tool_calls.update({call.id: call for call in calls})
                    for call in calls:
                        if call.id not in already_started:
                            yield ToolExecutionStartEvent(tool_call=call)
                    if not calls:
                        yield TurnEndEvent(turn=current_turn)
                        turn_open = False
                        message_started = False
                else:
                    result = _tool_result_from_native_message(raw_message)
                    messages.append(raw_message)
                    if raw_id:
                        completed_ids.add(raw_id)
                    if result.tool_call_id not in completed_tool_call_ids:
                        completed_tool_call_ids.add(result.tool_call_id)
                        yield ToolExecutionEndEvent(result=result)
                    pending_tool_calls.pop(result.tool_call_id, None)
    except ModelCallLimitExceededError:
        yield ErrorEvent(
            message=f"Agent loop stopped after reaching max_turns={max_turns or 0}",
            recoverable=True,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface model/tool failures as Forge events
        yield ErrorEvent(message=str(exc), recoverable=False)
    if turn_open:
        yield TurnEndEvent(turn=current_turn)
    yield AgentEndEvent()


def _project_v3_message_event(
    payload: Any,
    *,
    ensure_turn: Any,
    streamed_ids: set[str],
    delta_emitted: list[bool],
) -> list[AgentEvent]:
    if not isinstance(payload, tuple) or not payload:
        return []
    item = payload[0]
    events: list[AgentEvent] = []
    if isinstance(item, AIMessageChunk):
        deltas = _content_deltas(item)
        if not item.additional_kwargs.get("_forge_synthetic_final"):
            if deltas:
                events.extend(ensure_turn())
            for kind, text in deltas:
                if kind == "reasoning":
                    events.append(ThinkingDeltaEvent(delta=text))
                else:
                    delta_emitted[0] = True
                    events.append(MessageDeltaEvent(delta=text))
        if item.id:
            streamed_ids.add(str(item.id))
    elif isinstance(item, AIMessage):
        events.extend(ensure_turn())
        if item.id:
            streamed_ids.add(str(item.id))
    elif isinstance(item, Mapping):
        if item.get("event") == "content-block-delta":
            delta = item.get("delta")
            content_delta = _mapping_content_delta(delta) if isinstance(delta, Mapping) else None
            if content_delta is not None:
                kind, delta_text = content_delta
                events.extend(ensure_turn())
                if kind == "reasoning":
                    events.append(ThinkingDeltaEvent(delta=delta_text))
                else:
                    delta_emitted[0] = True
                    events.append(MessageDeltaEvent(delta=delta_text))
        elif item.get("event") == "message-start":
            events.extend(ensure_turn())
    return events


def _project_v3_tool_event(
    payload: Any,
    *,
    current_turn: int,
    pending_tool_calls: Mapping[str, ToolCall],
) -> list[AgentEvent]:
    del current_turn, pending_tool_calls
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
