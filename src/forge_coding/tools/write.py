"""LangChain-native write tool."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken
from forge_agent.types import JSONValue
from forge_coding.tools.base import ForgeStructuredTool, _create_native_tool
from forge_coding.tools.common import _path_arg, _str_arg, _workspace_root
from forge_coding.tools.definition import ToolDefinition
from forge_coding.tools.file_operation_queue import file_operation_queue


class WriteToolInput(BaseModel):
    """Validated model-facing arguments for the ``write`` tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: StrictStr = Field(description="Path to the file to write")
    content: StrictStr = Field(description="Content to write to the file")


def create_write_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create a definition for the `write` tool.

    The tool writes the supplied string `content` to `path`, resolving relative
    paths against `cwd`. Parent directories are created automatically and any
    existing file is overwritten. Writes use UTF-8 text encoding and are guarded
    by a per-path async lock so multiple writes/edits to the same resolved file
    are serialized within this process.

    The executor raises `ToolInputError` when `path` or `content` has the wrong
    type. Successful results include the resolved path and number of characters
    written in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        del signal
        workspace = _workspace_root(context, root)
        path = _path_arg(arguments, "path", cwd=workspace, for_write=True)
        content = _str_arg(arguments, "content")

        async with file_operation_queue.operation(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        return AgentToolResult(
            tool_call_id="",
            name="write",
            ok=True,
            content=f"Successfully wrote to {path}.",
            data={"path": str(path), "characters": len(content)},
        )

    tool = _create_native_tool(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        executor=execute,
        args_schema=WriteToolInput,
    )
    return ToolDefinition(
        tool=tool,
        label="write",
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=("Use write only for new files or complete rewrites.",),
    )


def create_write_tool(*, cwd: str | Path | None = None) -> ForgeStructuredTool:
    """Create a native LangChain tool for creating or overwriting UTF-8 files."""
    return create_write_tool_definition(cwd=cwd).tool  # type: ignore[return-value]
