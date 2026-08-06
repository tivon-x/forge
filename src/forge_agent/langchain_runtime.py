"""LangChain-backed agent runtime used by Forge's production harness.

LangChain owns the model/tool-calling state machine.  The native production
path passes LangChain messages and tools directly; the small Forge projections
below exist only to preserve the public UI event surface and legacy fixtures.
This module deliberately does not reimplement a second agent loop.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

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
from forge_agent.message_codec import is_langchain_message
from forge_agent.messages import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage
from forge_agent.provider import (
    CancellationToken,
    ModelProvider,
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderTextDeltaEvent,
)
from forge_agent.tools import AgentTool, AgentToolResult, ToolCall
from forge_agent.types import JSONValue


class ForgeProviderRuntimeError(RuntimeError):
    """Provider error surfaced while adapting a Forge provider to LangChain."""

    def __init__(self, message: str, data: dict[str, JSONValue] | None = None) -> None:
        super().__init__(message)
        self.data = data


class ForgeProviderChatModel(BaseChatModel):
    """Expose an existing Forge provider as a LangChain chat model.

    The provider performs one model response only.  LangChain's ``create_agent``
    graph is therefore the sole owner of tool-call turns and tool execution.
    """

    provider: Any = Field(exclude=True)
    model_name: str
    # ``AgentTool`` contains a Protocol-typed callable; keeping it as an
    # excluded ``Any`` field avoids asking Pydantic to build a runtime schema
    # for that opaque executor.
    forge_tools: Any = Field(default_factory=tuple, exclude=True)
    default_system: str = Field(default="", exclude=True)
    cancellation_token: Any = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "forge-provider"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ForgeProviderChatModel:
        """Accept LangChain's tool binding step.

        The Forge ``AgentTool`` objects are already attached by the runtime, so
        the generated schema is metadata for LangChain and does not need to be
        reconstructed here.  Returning ``self`` keeps the adapter lightweight
        while satisfying the official ``create_agent`` protocol.
        """

        del tools, tool_choice, kwargs
        return self

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        system, forge_messages = _from_langchain_messages(messages, self.default_system)
        text_emitted = False
        generation_emitted = False
        async for event in self.provider.stream_response(
            model=self.model_name,
            system=system,
            messages=forge_messages,
            tools=list(self.forge_tools),
            signal=self.cancellation_token,
        ):
            if isinstance(event, ProviderTextDeltaEvent):
                text_emitted = True
                generation_emitted = True
                yield ChatGenerationChunk(message=AIMessageChunk(content=event.delta))
            elif isinstance(event, ProviderResponseEndEvent):
                if event.message.content and not text_emitted:
                    generation_emitted = True
                    # LangChain needs a generation even when the upstream
                    # provider only emits a final response event.  Mark the
                    # synthetic chunk so Forge does not report it as a second
                    # visible text delta.
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=event.message.content,
                            additional_kwargs={"_forge_synthetic_final": True},
                        )
                    )
                elif not text_emitted and not event.message.tool_calls:
                    generation_emitted = True
                    yield ChatGenerationChunk(message=AIMessageChunk(content=""))
                if event.message.tool_calls:
                    generation_emitted = True
                    chunks: list[Any] = [
                        {
                            "name": call.name,
                            "args": json.dumps(call.arguments),
                            "id": call.id,
                            "index": index,
                        }
                        for index, call in enumerate(event.message.tool_calls)
                    ]
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content="", tool_call_chunks=chunks)
                    )
            elif isinstance(event, ProviderErrorEvent):
                raise ForgeProviderRuntimeError(event.message, event.data)
        if not generation_emitted:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs

        async def collect() -> AIMessage:
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            system, forge_messages = _from_langchain_messages(messages, self.default_system)
            async for event in self.provider.stream_response(
                model=self.model_name,
                system=system,
                messages=forge_messages,
                tools=list(self.forge_tools),
                signal=self.cancellation_token,
            ):
                if isinstance(event, ProviderTextDeltaEvent):
                    text_parts.append(event.delta)
                elif isinstance(event, ProviderResponseEndEvent):
                    tool_calls = list(event.message.tool_calls)
                elif isinstance(event, ProviderErrorEvent):
                    raise ForgeProviderRuntimeError(event.message, event.data)
            return AIMessage(
                content="".join(text_parts),
                tool_calls=[call.model_dump() for call in tool_calls],
            )

        return ChatResult(generations=[ChatGeneration(message=asyncio.run(collect()))])


def _langchain_tool(tool: AgentTool, signal: CancellationToken | None = None) -> StructuredTool:
    """Create a LangChain tool that delegates execution to ``AgentTool``."""

    schema = tool.input_schema
    properties_value = schema.get("properties")
    properties = properties_value if isinstance(properties_value, Mapping) else {}
    required_value = schema.get("required")
    required = (
        {str(value) for value in required_value} if isinstance(required_value, list) else set()
    )
    fields: dict[str, tuple[Any, Any]] = {
        str(name): (Any, ... if name in required else None) for name in properties
    }
    args_schema: type[BaseModel] = create_model(
        f"{tool.name.title()}Input", **cast(dict[str, Any], fields)
    )

    async def invoke(**arguments: Any) -> tuple[str, dict[str, JSONValue]]:
        result = await tool.execute(cast(Mapping[str, JSONValue], arguments), signal=signal)
        return _tool_result_text(result), cast(dict[str, JSONValue], result.model_dump(mode="json"))

    return StructuredTool.from_function(
        coroutine=invoke,
        name=tool.name,
        description=tool.description,
        args_schema=args_schema,
        response_format="content_and_artifact",
    )


def _native_tools(
    tools: Sequence[BaseTool | AgentTool], signal: CancellationToken | None = None
) -> list[BaseTool]:
    """Keep LangChain tools native and adapt only legacy Forge tools."""

    return [tool if isinstance(tool, BaseTool) else _langchain_tool(tool, signal) for tool in tools]


def _tool_result_text(result: AgentToolResult) -> str:
    content = result.content
    if result.data is not None and not content:
        content = json.dumps(result.data, ensure_ascii=False)
    if not result.ok and result.error and result.error not in content:
        content = f"{content}\n\nError: {result.error}"
    return content


def _from_langchain_messages(
    messages: Sequence[BaseMessage], default_system: str
) -> tuple[str, list[AgentMessage]]:
    system_parts: list[str] = []
    result: list[AgentMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(_message_text(message))
        elif isinstance(message, HumanMessage):
            result.append(UserMessage(content=_message_text(message)))
        elif isinstance(message, AIMessage):
            result.append(_assistant_from_langchain(message))
        elif isinstance(message, ToolMessage):
            result.append(_tool_result_from_message(message))
    return ("\n\n".join(system_parts) or default_system, result)


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


def _assistant_from_langchain(message: AIMessage) -> AssistantMessage:
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
    return AssistantMessage(content=_message_text(message), tool_calls=calls)


def _assistant_from_chunk(message: AIMessageChunk) -> AssistantMessage:
    calls: list[ToolCall] = []
    for index, raw in enumerate(message.tool_call_chunks):
        arguments: dict[str, JSONValue] = {}
        raw_args = raw.get("args")
        if isinstance(raw_args, str):
            try:
                decoded = json.loads(raw_args)
                if isinstance(decoded, dict):
                    arguments = cast(dict[str, JSONValue], decoded)
            except json.JSONDecodeError:
                pass
        calls.append(
            ToolCall(
                id=str(raw.get("id") or f"call-{index}"),
                name=str(raw.get("name") or "unknown"),
                arguments=arguments,
            )
        )
    return AssistantMessage(content=_message_text(message), tool_calls=calls)


def _tool_result_from_message(message: ToolMessage) -> ToolResultMessage:
    content = _message_text(message)
    artifact = message.artifact
    if isinstance(artifact, Mapping):
        try:
            stored = AgentToolResult.model_validate(artifact)
        except ValueError:
            pass
        else:
            return ToolResultMessage(
                tool_call_id=str(message.tool_call_id),
                name=stored.name,
                content=stored.content,
                ok=stored.ok,
                data=stored.data,
                details=stored.details,
                error=stored.error,
            )
    ok = getattr(message, "status", "success") != "error"
    return ToolResultMessage(
        tool_call_id=str(message.tool_call_id),
        name=str(getattr(message, "name", "tool")),
        content=content,
        ok=ok,
        error=None if ok else content,
    )


async def run_langchain_agent(
    *,
    provider: ModelProvider | BaseChatModel,
    model: str,
    system: str,
    messages: list[AgentMessage | AnyMessage],
    tools: Sequence[BaseTool | AgentTool] = (),
    max_turns: int | None = None,
    signal: CancellationToken | None = None,
    runtime_context: ForgeRuntimeContext | None = None,
    native_transcript: bool = False,
) -> AsyncIterator[AgentEvent]:
    """Run one native LangChain agent and project its v3 events for Forge UI.

    LangChain owns the model/tool loop.  Forge only keeps the transcript and
    projects the typed v3 lifecycle into the existing UI event vocabulary.
    """

    yield AgentStartEvent()
    if max_turns is not None and max_turns < 1:
        yield ErrorEvent(message="max_turns must be at least 1", recoverable=False)
        yield AgentEndEvent()
        return

    chat_model: BaseChatModel
    if isinstance(provider, BaseChatModel):
        native_model = True
        chat_model = provider
    else:
        native_model = False
        chat_model = ForgeProviderChatModel(
            provider=provider,
            model_name=model,
            forge_tools=tuple(tools),
            default_system=system,
            cancellation_token=signal,
        )
    graph = create_agent(
        chat_model,
        tools=_native_tools(tools, signal),
        system_prompt=system,
    )
    input_messages = [_to_langchain_message(message) for message in messages]
    input_message_count = len(input_messages)
    config: RunnableConfig = {}
    if max_turns is not None:
        config["recursion_limit"] = max(3, max_turns * 3)

    current_turn = 1
    turn_open = True
    message_started = False
    streamed_ids: set[str] = set()
    completed_ids: set[str] = {
        str(getattr(message, "id", ""))
        for message in messages
        if is_langchain_message(message) and getattr(message, "id", None)
    }
    pending_tool_calls: dict[str, ToolCall] = {}
    completed_tool_call_ids: set[str] = set()
    legacy_text_buffer: list[str] = []

    def ensure_turn() -> list[AgentEvent]:
        nonlocal current_turn, turn_open, message_started
        if not turn_open:
            current_turn += 1
            turn_open = True
            message_started = False
        if not message_started:
            message_started = True
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
                    stream_deltas=native_model,
                    legacy_text_buffer=legacy_text_buffer,
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
                    assistant = _assistant_from_langchain(raw_message)
                    if not assistant.tool_calls:
                        for item in ensure_turn():
                            yield item
                        if assistant.content and native_model and not streamed_ids:
                            yield MessageDeltaEvent(delta=assistant.content)
                    elif assistant.tool_calls:
                        for item in ensure_turn():
                            yield item
                    messages.append(raw_message if native_transcript else assistant)
                    if raw_id:
                        completed_ids.add(raw_id)
                    yield MessageEndEvent(message=raw_message if native_transcript else assistant)
                    already_started = set(pending_tool_calls)
                    pending_tool_calls.update({call.id: call for call in assistant.tool_calls})
                    for call in assistant.tool_calls:
                        if call.id not in already_started:
                            yield ToolExecutionStartEvent(tool_call=call)
                    if not assistant.tool_calls:
                        yield TurnEndEvent(turn=current_turn)
                        turn_open = False
                        message_started = False
                else:
                    result = _tool_result_from_message(raw_message)
                    messages.append(raw_message if native_transcript else result)
                    if raw_id:
                        completed_ids.add(raw_id)
                    if result.tool_call_id not in completed_tool_call_ids:
                        completed_tool_call_ids.add(result.tool_call_id)
                        yield ToolExecutionEndEvent(
                            result=AgentToolResult(
                                tool_call_id=result.tool_call_id,
                                name=result.name,
                                ok=result.ok,
                                content=result.content,
                                data=result.data,
                                details=result.details,
                                error=result.error,
                            )
                        )
                    pending_tool_calls.pop(result.tool_call_id, None)
    except ForgeProviderRuntimeError as exc:
        yield ErrorEvent(message=str(exc), recoverable=False, data=exc.data)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface model/tool failures as Forge events
        if not native_model:
            raise
        yield ErrorEvent(message=str(exc), recoverable=False)
    if turn_open:
        yield TurnEndEvent(turn=current_turn)
    yield AgentEndEvent()


def _project_v3_message_event(
    payload: Any,
    *,
    ensure_turn: Any,
    streamed_ids: set[str],
    stream_deltas: bool,
    legacy_text_buffer: list[str],
) -> list[AgentEvent]:
    if not isinstance(payload, tuple) or not payload:
        return []
    item = payload[0]
    events: list[AgentEvent] = []
    if isinstance(item, AIMessageChunk):
        deltas = _content_deltas(item)
        if stream_deltas and not item.additional_kwargs.get("_forge_synthetic_final"):
            if deltas:
                events.extend(ensure_turn())
            for kind, text in deltas:
                if kind == "reasoning":
                    events.append(ThinkingDeltaEvent(delta=text))
                else:
                    events.append(MessageDeltaEvent(delta=text))
        elif not stream_deltas:
            legacy_text_buffer.append("".join(text for kind, text in deltas if kind == "text"))
        if item.id:
            streamed_ids.add(str(item.id))
    elif isinstance(item, AIMessage):
        if stream_deltas:
            events.extend(ensure_turn())
        if item.id:
            streamed_ids.add(str(item.id))
    elif isinstance(item, Mapping):
        if item.get("event") == "content-block-delta":
            delta = item.get("delta")
            content_delta = _mapping_content_delta(delta) if isinstance(delta, Mapping) else None
            if content_delta is not None:
                kind, delta_text = content_delta
                if stream_deltas:
                    events.extend(ensure_turn())
                    events.append(
                        ThinkingDeltaEvent(delta=delta_text)
                        if kind == "reasoning"
                        else MessageDeltaEvent(delta=delta_text)
                    )
                else:
                    if kind == "text":
                        legacy_text_buffer.append(delta_text)
        elif stream_deltas and item.get("event") == "message-start":
            events.extend(ensure_turn())
        elif not stream_deltas and item.get("event") == "message-finish":
            additional_kwargs = item.get("additional_kwargs")
            synthetic = (
                isinstance(additional_kwargs, Mapping)
                and additional_kwargs.get("_forge_synthetic_final") is True
            )
            if not synthetic and legacy_text_buffer:
                events.extend(ensure_turn())
                events.append(MessageDeltaEvent(delta="".join(legacy_text_buffer)))
            legacy_text_buffer.clear()
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
            result = _tool_result_from_message(output)
        else:
            content = str(output or "")
            result = ToolResultMessage(
                tool_call_id=str(payload.get("tool_call_id") or ""),
                name=str(payload.get("tool_name") or "tool"),
                content=content,
                ok=True,
            )
        return [
            ToolExecutionEndEvent(
                result=AgentToolResult(
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                    ok=result.ok,
                    content=result.content,
                    data=result.data,
                    details=result.details,
                    error=result.error,
                )
            )
        ]
    return []


def _to_langchain_message(message: AgentMessage | AnyMessage) -> BaseMessage:
    if isinstance(message, BaseMessage):
        return message
    if isinstance(message, UserMessage):
        return HumanMessage(content=message.content)
    if isinstance(message, AssistantMessage):
        return AIMessage(
            content=message.content,
            tool_calls=[
                {"id": call.id, "name": call.name, "args": call.arguments, "type": "tool_call"}
                for call in message.tool_calls
            ],
        )
    return ToolMessage(
        content=message.content,
        tool_call_id=message.tool_call_id,
        name=message.name,
        status="success" if message.ok else "error",
        additional_kwargs={"_forge_error": message.error} if message.error else {},
    )
