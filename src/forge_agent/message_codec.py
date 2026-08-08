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
from pydantic import BaseModel, TypeAdapter
from pydantic_core import PydanticSerializationError

from forge_agent.types import JSONValue

_ANY_MESSAGE_ADAPTER: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)
# ``Any``-typed dump: a JSONValue-typed adapter would emit
# PydanticSerializationWarning with the artifact's ``input_value`` (a repr
# leak) for values outside the union; the Any adapter only raises for truly
# unserializable values.
_ARTIFACT_ADAPTER: TypeAdapter[Any] = TypeAdapter(Any)


def _omitted_artifact(artifact: object) -> dict[str, JSONValue]:
    python_type = f"{type(artifact).__module__}.{type(artifact).__qualname__}"
    return {
        "forge_serialization": {
            "status": "omitted",
            "python_type": python_type,
        }
    }


def _contains_bytes(value: object, seen: set[int] | None = None) -> bool:
    """Return whether any ``bytes`` value is nested inside JSON-like data.

    ``seen`` tracks visited containers so self-referencing dicts/lists cannot
    recurse forever.  Pydantic models are inspected through their python-mode
    dump so a bytes-typed field is rejected instead of being silently decoded
    to a string by the JSON serializer.
    """
    if isinstance(value, bytes):
        return True
    if seen is None:
        seen = set()
    if isinstance(value, BaseModel):
        return _contains_bytes(value.model_dump(), seen)
    if isinstance(value, Mapping):
        return _contains_mapping_values(value, seen)
    if isinstance(value, list | tuple | set):
        return _contains_iterable_values(value, seen)
    return False


def _contains_mapping_values(value: Mapping[Any, Any], seen: set[int]) -> bool:
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    try:
        return any(_contains_bytes(item, seen) for item in value.values())
    finally:
        seen.discard(identity)


def _contains_iterable_values(
    value: list[Any] | tuple[Any, ...] | set[Any], seen: set[int]
) -> bool:
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    try:
        return any(_contains_bytes(item, seen) for item in value)
    finally:
        seen.discard(identity)


def _json_safe_artifact(artifact: object) -> JSONValue:
    """Project one tool artifact into a JSON-safe persistence value.

    JSON-compatible values (primitives, dicts/lists of primitives, pydantic
    models) are kept; ``bytes`` anywhere inside is rejected before pydantic
    gets a chance to silently decode it, and values pydantic cannot serialize
    (arbitrary objects, cyclic containers) become the stable omission
    placeholder.
    """
    if artifact is None or isinstance(artifact, str | int | float | bool):
        return cast(JSONValue, artifact)
    if _contains_bytes(artifact):
        return _omitted_artifact(artifact)
    try:
        projected = _ARTIFACT_ADAPTER.dump_python(artifact, mode="json", warnings="error")
        return cast(JSONValue, projected)
    except (PydanticSerializationError, TypeError, ValueError, UnicodeDecodeError, RecursionError):
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
