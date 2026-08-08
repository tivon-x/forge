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


def _json_container_ok(value: object, seen: set[int]) -> bool:
    """Return whether ``value`` is JSON base types or containers of them.

    Only explicit JSON base types (``None``/``str``/``int``/``float``/``bool``)
    and dict/list/tuple/set containers pass; everything else -- bytes,
    dataclasses, pydantic models, arbitrary objects -- is conservatively
    omitted, so ``bytes`` can never be silently decoded to a string by the
    serializer.  ``seen`` tracks visited containers so self-referencing
    structures cannot recurse forever.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return True
    identity = id(value)
    if identity in seen:
        return True  # cycle: still containers; the dump below will fail and omit
    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            return all(_json_container_ok(item, seen) for item in value.values())
        finally:
            seen.discard(identity)
    if isinstance(value, list | tuple | set):
        seen.add(identity)
        try:
            return all(_json_container_ok(item, seen) for item in value)
        finally:
            seen.discard(identity)
    return False


def _json_safe_artifact(artifact: object) -> JSONValue:
    """Project one tool artifact into a JSON-safe persistence value.

    Explicit JSON base types and containers of them round-trip unchanged;
    everything else (bytes, dataclasses, pydantic models, arbitrary objects,
    cyclic containers) becomes the stable omission placeholder.  The pydantic
    conversion and every recursion-capable step sit inside one exception
    boundary, so no artifact shape can crash persistence or leak a repr.
    """
    if artifact is None or isinstance(artifact, str | int | float | bool):
        return cast(JSONValue, artifact)
    try:
        if not _json_container_ok(artifact, set()):
            return _omitted_artifact(artifact)
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
