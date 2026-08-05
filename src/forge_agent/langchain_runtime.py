"""LangChain-backed agent runtime used by Forge's production harness.

Forge keeps its own messages, tools, and event models at the application
boundary.  LangChain owns the actual model/tool-calling state machine.  This
module only translates between those two contracts; it deliberately does not
reimplement a second agent loop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from forge_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
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
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
            parts.append(str(block["text"]))
    return "".join(parts)


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
    messages: list[AgentMessage],
    tools: list[AgentTool],
    max_turns: int | None = None,
    signal: CancellationToken | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run one LangChain ``create_agent`` invocation and adapt its events."""

    yield AgentStartEvent()
    if max_turns is not None and max_turns < 1:
        yield ErrorEvent(message="max_turns must be at least 1", recoverable=False)
        yield AgentEndEvent()
        return

    if isinstance(provider, BaseChatModel):
        chat_model = provider
    else:
        chat_model = ForgeProviderChatModel(
            provider=provider,
            model_name=model,
            forge_tools=tuple(tools),
            default_system=system,
            cancellation_token=signal,
        )
    graph = create_agent(
        chat_model,
        tools=[_langchain_tool(tool, signal) for tool in tools],
        system_prompt=system,
    )
    input_messages = [_to_langchain_message(message) for message in messages]
    config: RunnableConfig = {}
    if max_turns is not None:
        config["recursion_limit"] = max(3, max_turns * 3)

    current_turn = 0
    turn_open = True
    message_started = False
    streamed_ids: set[str] = set()
    pending_tool_calls: dict[str, ToolCall] = {}
    stream_modes: list[Literal["messages", "updates"]] = ["messages", "updates"]
    current_turn = 1
    try:
        yield TurnStartEvent(turn=current_turn)
        async for mode, payload in graph.astream(  # type: ignore[call-overload]
            {"messages": input_messages},
            stream_mode=stream_modes,
            config=config or None,
        ):
            if signal is not None and signal.is_cancelled():
                yield ErrorEvent(message="Agent run cancelled", recoverable=True)
                break
            if mode == "messages":
                raw_message, _metadata = payload
                if isinstance(raw_message, AIMessageChunk):
                    if not turn_open:
                        current_turn += 1
                        turn_open = True
                        message_started = False
                        yield TurnStartEvent(turn=current_turn)
                    if not message_started:
                        yield MessageStartEvent()
                        message_started = True
                    text = _message_text(raw_message)
                    if text and not raw_message.additional_kwargs.get("_forge_synthetic_final"):
                        yield MessageDeltaEvent(delta=text)
                    if raw_message.id:
                        streamed_ids.add(str(raw_message.id))
                continue

            if mode != "updates" or not isinstance(payload, Mapping):
                continue
            for update in payload.values():
                if not isinstance(update, Mapping):
                    continue
                for raw_message in update.get("messages", ()):
                    if isinstance(raw_message, AIMessage):
                        if not turn_open:
                            current_turn += 1
                            turn_open = True
                            message_started = False
                            yield TurnStartEvent(turn=current_turn)
                        if not message_started:
                            yield MessageStartEvent()
                            message_started = True
                        assistant = _assistant_from_langchain(raw_message)
                        if (
                            not raw_message.tool_calls
                            and raw_message.id not in streamed_ids
                            and assistant.content
                        ):
                            yield MessageDeltaEvent(delta=assistant.content)
                        messages.append(assistant)
                        yield MessageEndEvent(message=assistant)
                        pending_tool_calls = {call.id: call for call in assistant.tool_calls}
                        for call in assistant.tool_calls:
                            yield ToolExecutionStartEvent(tool_call=call)
                        if not assistant.tool_calls:
                            yield TurnEndEvent(turn=current_turn)
                            turn_open = False
                            message_started = False
                    elif isinstance(raw_message, ToolMessage):
                        result = _tool_result_from_message(raw_message)
                        messages.append(result)
                        yield ToolExecutionEndEvent(
                            result=AgentToolResult(
                                tool_call_id=result.tool_call_id,
                                name=result.name,
                                ok=result.ok,
                                content=result.content,
                                error=result.error,
                            )
                        )
                        pending_tool_calls.pop(result.tool_call_id, None)
                        if turn_open and not pending_tool_calls:
                            yield TurnEndEvent(turn=current_turn)
                            turn_open = False
                            message_started = False
    except ForgeProviderRuntimeError as exc:
        yield ErrorEvent(message=str(exc), recoverable=False, data=exc.data)
    except Exception:
        raise
    if turn_open:
        yield TurnEndEvent(turn=current_turn)
    yield AgentEndEvent()


def _to_langchain_message(message: AgentMessage) -> BaseMessage:
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
