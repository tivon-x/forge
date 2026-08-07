"""Serialisation helpers for LangChain messages at the persistence boundary.

Forge sessions store LangChain ``AnyMessage`` rows losslessly.  This module
provides JSON round-tripping and display-text extraction without mirroring a
second message model.  The historical Forge role-row format is no longer
produced or read; runtime code passes native ``AnyMessage`` objects directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
)
from pydantic import TypeAdapter

_ANY_MESSAGE_ADAPTER: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)


def is_langchain_message(message: object) -> bool:
    """Return whether ``message`` is a LangChain Core message."""

    return isinstance(message, BaseMessage)


def to_langchain_message(message: AnyMessage) -> AnyMessage:
    """Return the native LangChain message unchanged.

    Kept as a stable projection for call sites that historically normalised
    legacy rows; with native-only messages it is effectively identity.
    """

    return message


def message_to_json(message: AnyMessage) -> dict[str, Any]:
    """Serialize one message while retaining native content blocks/metadata."""

    return cast(dict[str, Any], _ANY_MESSAGE_ADAPTER.dump_python(message, mode="json"))


def message_from_json(value: object) -> AnyMessage:
    """Decode a native LangChain message row."""

    if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
        raise ValueError("Session message must be a native LangChain message")
    try:
        return _ANY_MESSAGE_ADAPTER.validate_python(value)
    except ValueError as exc:
        raise ValueError("Unsupported native session message row") from exc


def message_text(message: AnyMessage) -> str:
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
