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
from math import isfinite
from typing import Any, cast

from langchain_core.messages import (
    AnyMessage,
    ToolMessage,
)
from pydantic import BaseModel, TypeAdapter

from forge_agent.types import JSONValue

_ANY_MESSAGE_ADAPTER: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)
_INVALID_JSON_VALUE = object()


def _omitted_artifact(artifact: object) -> dict[str, JSONValue]:
    python_type = f"{type(artifact).__module__}.{type(artifact).__qualname__}"
    return {
        "forge_serialization": {
            "status": "omitted",
            "python_type": python_type,
        }
    }


def _project_json_value(value: object, seen: set[int]) -> JSONValue | object:
    """Return a conservative JSON projection or ``_INVALID_JSON_VALUE``.

    Built-in containers are walked directly so custom mapping hooks cannot run
    during persistence. Mapping keys must already be strings. Pydantic models
    get one explicit ``model_dump(mode="python")`` pass, then the same rules
    validate their output before any bytes can be decoded by a JSON serializer.
    """
    if value is None or isinstance(value, str | bool | int):
        return cast(JSONValue, value)
    if isinstance(value, float):
        return value if isfinite(value) else _INVALID_JSON_VALUE

    identity = id(value)
    if identity in seen:
        return _INVALID_JSON_VALUE

    if type(value) is dict:
        seen.add(identity)
        try:
            projected: dict[str, JSONValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    return _INVALID_JSON_VALUE
                projected_item = _project_json_value(item, seen)
                if projected_item is _INVALID_JSON_VALUE:
                    return _INVALID_JSON_VALUE
                projected[key] = cast(JSONValue, projected_item)
            return projected
        finally:
            seen.discard(identity)

    if type(value) in (list, tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        seen.add(identity)
        try:
            projected_items: list[JSONValue] = []
            for item in sequence:
                projected_item = _project_json_value(item, seen)
                if projected_item is _INVALID_JSON_VALUE:
                    return _INVALID_JSON_VALUE
                projected_items.append(cast(JSONValue, projected_item))
            return projected_items
        finally:
            seen.discard(identity)

    if isinstance(value, BaseModel):
        seen.add(identity)
        try:
            dumped = value.model_dump(mode="python", warnings="error")
            return _project_json_value(dumped, seen)
        finally:
            seen.discard(identity)

    return _INVALID_JSON_VALUE


def _json_safe_artifact(artifact: object) -> JSONValue:
    """Project one tool artifact into a JSON-safe persistence value.

    Explicit JSON values and JSON-safe pydantic data round-trip unchanged;
    everything else (bytes, dataclasses, custom mappings, arbitrary objects,
    cyclic containers) becomes the stable omission placeholder. Every
    conversion and recursion-capable step sits inside one exception boundary,
    so no artifact shape can crash persistence or leak a repr.
    """
    try:
        projected = _project_json_value(artifact, set())
        if projected is _INVALID_JSON_VALUE:
            return _omitted_artifact(artifact)
        return cast(JSONValue, projected)
    except Exception:  # noqa: BLE001 - artifact code must never break persistence
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
