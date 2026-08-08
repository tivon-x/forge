"""Serialisation helpers for LangChain messages at the persistence boundary.

Forge sessions store LangChain ``AnyMessage`` rows losslessly.  This module
provides JSON round-tripping and display-text extraction without mirroring a
second message model.  Runtime code passes native ``AnyMessage`` objects
directly.

The only projection at this boundary is the ``ToolMessage`` artifact: native
messages stay the in-memory truth and third-party ``BaseTool`` artifacts are
unrestricted in memory, but persistence replaces values pydantic cannot
serialize to JSON (arbitrary objects, bytes) with a stable placeholder that
never contains ``repr()`` output or raw bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.messages import (
    AnyMessage,
    ToolMessage,
)
from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError

from forge_agent.types import JSONValue

_ANY_MESSAGE_ADAPTER: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)
_JSON_VALUE_ADAPTER: TypeAdapter[JSONValue] = TypeAdapter(JSONValue)


class ForgeSerializationOmitted:
    """Marker for values that cannot be persisted as JSON.

    The persisted shape is ``{"forge_serialization": {"status": "omitted",
    "python_type": ...}}``; it never carries ``repr()`` output, raw bytes, or
    object fields.
    """


def _omitted_artifact(artifact: object) -> dict[str, JSONValue]:
    python_type = f"{type(artifact).__module__}.{type(artifact).__qualname__}"
    return {
        "forge_serialization": {
            "status": "omitted",
            "python_type": python_type,
        }
    }


def _contains_bytes(value: object) -> bool:
    """Return whether any ``bytes`` value is nested inside JSON-like data."""
    if isinstance(value, bytes):
        return True
    if isinstance(value, Mapping):
        return any(_contains_bytes(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(_contains_bytes(item) for item in value)
    return False


def _json_safe_artifact(artifact: object) -> JSONValue:
    """Project one tool artifact into a JSON-safe persistence value.

    JSON-compatible values (primitives, dicts/lists of primitives, pydantic
    models) are kept; ``bytes`` anywhere inside is rejected before pydantic
    gets a chance to silently decode it, and values pydantic cannot serialize
    (arbitrary objects) become the stable omission placeholder.
    """
    if artifact is None or isinstance(artifact, str | int | float | bool):
        return cast(JSONValue, artifact)
    if _contains_bytes(artifact):
        return _omitted_artifact(artifact)
    try:
        projected = _JSON_VALUE_ADAPTER.dump_python(cast(JSONValue, artifact), mode="json")
        return cast(JSONValue, projected)
    except (PydanticSerializationError, TypeError, ValueError, UnicodeDecodeError):
        return _omitted_artifact(artifact)


def message_to_json(message: AnyMessage) -> dict[str, Any]:
    """Serialize one message while retaining native content blocks/metadata.

    Only the ``ToolMessage`` artifact is projected; content, ``tool_call_id``,
    ``name``, ``status``, response metadata and usage metadata are preserved
    unchanged.
    """

    return cast(
        dict[str, Any],
        _ANY_MESSAGE_ADAPTER.dump_python(project_message_artifact(message), mode="json"),
    )


def project_message_artifact(message: AnyMessage) -> AnyMessage:
    """Return a copy of ``message`` whose tool artifact is JSON-safe.

    In-memory messages keep their unrestricted artifact; only persistence
    paths (entry JSONL rows) call this before serializing.
    """

    if isinstance(message, ToolMessage) and message.artifact is not None:
        return message.model_copy(update={"artifact": _json_safe_artifact(message.artifact)})
    return message


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
