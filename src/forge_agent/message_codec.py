"""Lossless conversion between Forge's historical rows and LangChain messages.

The compatibility conversion is deliberately isolated at the persistence
boundary.  Runtime code should pass ``AnyMessage`` objects directly to
LangChain; the legacy branch only exists so old JSONL files and downstream
callers can be opened without an in-place rewrite.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import TypeAdapter

from forge_agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from forge_agent.tools import ToolCall
from forge_agent.types import JSONValue

_ANY_MESSAGE_ADAPTER: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)


def is_langchain_message(message: object) -> bool:
    """Return whether ``message`` is a LangChain Core message."""

    return isinstance(message, BaseMessage)


def to_langchain_message(message: AnyMessage | AgentMessage) -> AnyMessage:
    """Convert a historical Forge message to its LangChain equivalent."""

    if is_langchain_message(message):
        return cast(AnyMessage, message)
    if isinstance(message, UserMessage):
        return HumanMessage(content=message.content)
    if isinstance(message, AssistantMessage):
        return AIMessage(
            content=message.content,
            tool_calls=[
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.arguments,
                    "type": "tool_call",
                }
                for call in message.tool_calls
            ],
        )
    if isinstance(message, ToolResultMessage):
        artifact = message.model_dump(mode="json")
        return ToolMessage(
            content=message.content,
            tool_call_id=message.tool_call_id,
            name=message.name,
            status="success" if message.ok else "error",
            artifact=artifact,
        )
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def message_to_json(message: AnyMessage | AgentMessage) -> dict[str, Any]:
    """Serialize one message while retaining native content blocks/metadata."""

    if is_langchain_message(message):
        return cast(
            dict[str, Any],
            _ANY_MESSAGE_ADAPTER.dump_python(cast(AnyMessage, message), mode="json"),
        )
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [call.model_dump(mode="json") for call in message.tool_calls],
        }
    if isinstance(message, ToolResultMessage):
        return message.model_dump(mode="json")
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def message_from_json(value: object) -> AnyMessage | AgentMessage:
    """Decode native LangChain rows or historical Forge role rows."""

    if isinstance(value, Mapping) and isinstance(value.get("type"), str):
        try:
            return _ANY_MESSAGE_ADAPTER.validate_python(value)
        except ValueError:
            pass
    if not isinstance(value, Mapping):
        raise ValueError("Session message must be an object")
    role = value.get("role")
    if role == "user":
        content = value.get("content")
        if not isinstance(content, str):
            raise ValueError("Legacy user message content must be a string")
        return UserMessage(content=content)
    if role == "assistant":
        content = value.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        raw_calls = value.get("tool_calls", [])
        calls: list[ToolCall] = []
        if isinstance(raw_calls, list):
            for raw in raw_calls:
                if isinstance(raw, Mapping):
                    calls.append(
                        ToolCall(
                            id=str(raw.get("id", "")),
                            name=str(raw.get("name", "")),
                            arguments=cast(
                                dict[str, JSONValue],
                                raw.get("arguments", raw.get("args", {})),
                            ),
                            thought_signature=(
                                str(raw["thought_signature"])
                                if raw.get("thought_signature") is not None
                                else None
                            ),
                        )
                    )
        return AssistantMessage(content=content, tool_calls=calls)
    if role == "tool":
        return ToolResultMessage(
            tool_call_id=str(value.get("tool_call_id", "")),
            name=str(value.get("name", "tool")),
            content=str(value.get("content", "")),
            ok=bool(value.get("ok", True)),
            data=cast(dict[str, JSONValue] | None, value.get("data")),
            details=cast(dict[str, JSONValue] | None, value.get("details")),
            error=(str(value["error"]) if value.get("error") is not None else None),
        )
    raise ValueError(f"Unsupported session message role/type: {role!r}")


def message_text(message: AnyMessage | AgentMessage) -> str:
    """Return display text without dropping structured content blocks."""

    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
