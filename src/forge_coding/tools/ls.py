"""LangChain-native workspace directory listing."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken
from forge_agent.types import JSONValue
from forge_coding.tools.base import ForgeStructuredTool, _create_native_tool
from forge_coding.tools.common import _path_arg, _workspace_root
from forge_coding.tools.definition import ToolDefinition
from forge_coding.tools.truncation import (
    DEFAULT_MAX_OUTPUT_LINES,
    format_size,
    truncate_head,
)

_MAX_RETAINED_CHILDREN = DEFAULT_MAX_OUTPUT_LINES + 1


class LsToolInput(BaseModel):
    """Model-facing arguments for the native ls tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: StrictStr = Field(default=".", description="Workspace-relative directory")


def create_ls_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        if signal is not None and signal.is_cancelled():
            return _cancelled_result()
        raw_path = arguments.get("path", ".")
        if not isinstance(raw_path, str):
            raise ValueError("path must be a string")
        workspace = _workspace_root(context, root)
        path = _path_arg({"path": raw_path}, "path", cwd=workspace)
        if not path.exists():
            raise ValueError(f"Path does not exist: {raw_path}")
        if path.is_file():
            entries = [_entry(path, path.name)]
            entry_count = 1
        elif path.is_dir():
            children: list[Path] = []
            entry_count = 0
            for child in path.iterdir():
                if signal is not None and signal.is_cancelled():
                    return _cancelled_result()
                entry_count += 1
                children.append(child)
                if len(children) >= _MAX_RETAINED_CHILDREN * 2:
                    children.sort(key=_listing_key)
                    del children[_MAX_RETAINED_CHILDREN:]
            children.sort(key=_listing_key)
            del children[_MAX_RETAINED_CHILDREN:]
            entries = [
                _entry(child, child.name)
                for child in children
            ]
        else:
            raise ValueError(f"Path is not a file or directory: {raw_path}")

        lines = [_format_entry(item) for item in entries]
        truncation = truncate_head("\n".join(lines))
        relative_path = path.relative_to(workspace.resolve()).as_posix() or "."
        return AgentToolResult(
            tool_call_id="",
            name="ls",
            ok=True,
            content=truncation.content or "(empty directory)",
            data={
                "path": relative_path,
                "entry_count": entry_count,
                "truncation": truncation.to_json(),
            },
        )

    tool = _create_native_tool(
        name="ls",
        description=(
            "List files and directories in the workspace using a stable name order. "
            "Entries include type and byte size."
        ),
        args_schema=LsToolInput,
        executor=execute,
    )
    return ToolDefinition(
        tool=tool,
        label="ls",
        prompt_snippet="List files and directories",
        prompt_guidelines=("Use ls to inspect a directory before reading or editing files.",),
    )


def create_ls_tool(*, cwd: str | Path | None = None) -> ForgeStructuredTool:
    return create_ls_tool_definition(cwd=cwd).tool  # type: ignore[return-value]


def _cancelled_result() -> AgentToolResult:
    return AgentToolResult(
        tool_call_id="",
        name="ls",
        ok=False,
        content="ls cancelled",
        error="ls cancelled",
        data={"cancelled": True},
    )


def _entry(path: Path, name: str) -> dict[str, JSONValue]:
    if path.is_symlink():
        kind = "symlink"
    elif path.is_dir():
        kind = "directory"
    else:
        kind = "file"
    try:
        size = path.lstat().st_size
    except OSError:
        size = 0
    return {"name": name, "type": kind, "size": size}


def _listing_key(path: Path) -> tuple[str, str]:
    return path.name.casefold(), path.name


def _format_entry(entry: Mapping[str, JSONValue]) -> str:
    name = entry.get("name")
    kind = entry.get("type")
    size = entry.get("size")
    if kind == "directory":
        return f"{name}/ (directory, {format_size(size if isinstance(size, int) else 0)})"
    if kind == "symlink":
        return f"{name} (symlink, {format_size(size if isinstance(size, int) else 0)})"
    return f"{name} ({format_size(size if isinstance(size, int) else 0)})"


__all__ = ["LsToolInput", "create_ls_tool", "create_ls_tool_definition"]
