"""Legacy Forge provider/tool/message adapters (offline-fixture boundary).

The production agent runtime is LangChain-native: it accepts ``BaseChatModel``,
``BaseTool`` and LangChain messages only.  This module adapts historical
callers that still pass a Forge ``ModelProvider``, ``AgentTool`` or legacy
``AgentMessage`` rows into the harness.  The default CLI path never touches
this module; it exists so offline fixtures and old JSONL consumers keep
working without embedding legacy protocol conversion in the runtime.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

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
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from forge_agent.context import ForgeRuntimeContext
from forge_agent.events import AgentEvent, ErrorEvent
from forge_agent.langchain_runtime import run_langchain_agent
from forge_agent.message_codec import message_text
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
        system, forge_messages = from_langchain_messages(messages, self.default_system)
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
                            "args": json_dumps(call.arguments),
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
            system, forge_messages = from_langchain_messages(messages, self.default_system)
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


def langchain_tool(tool: AgentTool, signal: CancellationToken | None = None) -> StructuredTool:
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


def from_langchain_messages(
    messages: Sequence[BaseMessage],
    default_system: str,
) -> tuple[str, list[AgentMessage]]:
    """Convert LangChain messages back to historical Forge rows."""

    system_parts: list[str] = []
    result: list[AgentMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(message_text(message))
        elif isinstance(message, HumanMessage):
            result.append(UserMessage(content=message_text(message)))
        elif isinstance(message, AIMessage):
            result.append(assistant_from_langchain(message))
        elif isinstance(message, ToolMessage):
            result.append(tool_result_from_message(message))
    return ("\n\n".join(system_parts) or default_system, result)


def assistant_from_langchain(message: AIMessage) -> AssistantMessage:
    """Convert a native AIMessage to the historical assistant row."""

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
    return AssistantMessage(content=message_text(message), tool_calls=calls)


def tool_result_from_message(message: ToolMessage) -> ToolResultMessage:
    """Convert a native ToolMessage to the historical tool-result row."""

    content = message_text(message)
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


def to_legacy_message(message: BaseMessage) -> AgentMessage:
    """Convert a native message to the historical Forge row shape.

    Used by the compatibility wrapper to mirror the native runtime's transcript
    back into a legacy caller's message list.
    """

    if isinstance(message, HumanMessage):
        return UserMessage(content=message_text(message))
    if isinstance(message, ToolMessage):
        return tool_result_from_message(message)
    return assistant_from_langchain(cast(AIMessage, message))


def _tool_result_text(result: AgentToolResult) -> str:
    content = result.content
    if result.data is not None and not content:
        content = json_dumps(result.data)
    if not result.ok and result.error and result.error not in content:
        content = f"{content}\n\nError: {result.error}"
    return content


def json_dumps(value: object) -> str:
    """Serialize a JSON-compatible value without non-ASCII escaping."""

    import json

    return json.dumps(value, ensure_ascii=False)


async def run_compat_agent(
    *,
    provider: ModelProvider,
    model: str,
    system: str,
    messages: list[AgentMessage | AnyMessage],
    tools: Sequence[BaseTool | AgentTool] = (),
    max_turns: int | None = None,
    signal: CancellationToken | None = None,
    runtime_context: ForgeRuntimeContext | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run the native agent loop for a legacy ``ModelProvider`` caller.

    The conversion happens at this compatibility boundary: the Forge provider
    is wrapped in ``ForgeProviderChatModel``, ``AgentTool`` entries become
    ``StructuredTool`` objects, and the runtime's transcript adapter keeps the
    caller's message list in the historical row format.  Error semantics match
    the historical runtime: Forge provider errors become ``ErrorEvent``, every
    other exception propagates to the caller.
    """

    chat_model = ForgeProviderChatModel(
        provider=provider,
        model_name=model,
        forge_tools=tuple(tools),
        default_system=system,
        cancellation_token=signal,
    )
    native_tools = [
        tool if isinstance(tool, BaseTool) else langchain_tool(tool, signal) for tool in tools
    ]
    try:
        async for event in run_langchain_agent(
            provider=chat_model,
            model=model,
            system=system,
            messages=messages,
            tools=native_tools,
            max_turns=max_turns,
            signal=signal,
            runtime_context=runtime_context,
            stream_deltas=False,
            error_policy="raise",
            transcript_adapter=to_legacy_message,
        ):
            yield event
    except ForgeProviderRuntimeError as exc:
        yield ErrorEvent(message=str(exc), recoverable=False, data=exc.data)
