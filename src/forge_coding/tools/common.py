"""Shared argument and workspace helpers for coding tools."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from forge_agent.context import ForgeRuntimeContext
from forge_agent.types import JSONValue
from forge_coding.tools.base import ToolInputError

UTF8_BOM = "\ufeff"


def _workspace_root(context: ForgeRuntimeContext | None, fallback: Path) -> Path:
    """Return the effective workspace root, preferring the injected context."""

    if context is not None and context.workspace_root:
        return Path(context.workspace_root)
    return fallback


def _shell_command_prefix(context: ForgeRuntimeContext | None, fallback: str | None) -> str | None:
    """Return the effective shell prefix, preferring the injected context."""

    if context is not None and context.shell_command_prefix:
        return context.shell_command_prefix
    return fallback


def _str_arg(arguments: Mapping[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be a string")
    return value


def _path_arg(
    arguments: Mapping[str, JSONValue],
    name: str,
    *,
    cwd: Path,
    for_write: bool = False,
) -> Path:
    value = _str_arg(arguments, name)
    root = cwd.resolve()
    supplied = Path(value).expanduser()
    candidate = supplied if supplied.is_absolute() else root / supplied
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ToolInputError(f"Path is outside the project workspace: {value}") from exc
    if for_write and lexical.is_symlink():
        raise ToolInputError(f"Refusing to write through a symlink: {value}")
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ToolInputError(f"Path resolves outside the project workspace: {value}") from exc
    return resolved


def _optional_int_arg(arguments: Mapping[str, JSONValue], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ToolInputError(f"{name} must be an integer")
    return value


def _optional_float_arg(arguments: Mapping[str, JSONValue], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ToolInputError(f"{name} must be a number")
    return float(value)


def _prepare_edit_arguments(arguments: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    prepared = dict(arguments)
    edits_value = prepared.get("edits")
    if isinstance(edits_value, str):
        try:
            parsed = json.loads(edits_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            prepared["edits"] = parsed

    old_text = prepared.get("oldText")
    new_text = prepared.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = prepared.get("edits")
        edit_list = edits if isinstance(edits, list) else []
        prepared["edits"] = [*edit_list, {"oldText": old_text, "newText": new_text}]
        prepared.pop("oldText", None)
        prepared.pop("newText", None)
    return prepared


def _edits_arg(arguments: Mapping[str, JSONValue]) -> list[dict[str, str]]:
    value = arguments.get("edits")
    if not isinstance(value, list) or not value:
        raise ToolInputError(
            "Edit tool input is invalid. edits must contain at least one replacement."
        )

    edits: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ToolInputError(f"edits[{index}] must be an object")
        old_text = item.get("oldText")
        new_text = item.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolInputError(
                f"edits[{index}].oldText and edits[{index}].newText must be strings"
            )
        edits.append({"oldText": old_text, "newText": new_text})
    return edits
