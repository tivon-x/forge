"""LangChain-native ``find`` backed by managed fd."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken
from forge_agent.types import JSONValue
from forge_coding.tools.base import ForgeStructuredTool, _create_native_tool
from forge_coding.tools.definition import ToolDefinition
from forge_coding.tools.search_common import (
    bounded_result,
    display_path,
    run_search,
    search_path,
    strip_output_marker,
    timeout_arg,
)
from forge_coding.tools.tool_manager import DEFAULT_TOOL_MANAGER, ToolManager


class FindToolInput(BaseModel):
    """Model-facing arguments for the native find tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: StrictStr = Field(min_length=1, description="Glob pattern for file names")
    path: StrictStr = Field(default=".", description="Workspace-relative directory")
    limit: StrictInt = Field(default=1000, ge=1, description="Maximum matching paths")


def create_find_tool_definition(
    *,
    cwd: str | Path | None = None,
    tool_manager: ToolManager | None = None,
) -> ToolDefinition:
    root = Path.cwd() if cwd is None else Path(cwd)
    manager = DEFAULT_TOOL_MANAGER if tool_manager is None else tool_manager

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a non-empty string")
        workspace, relative = search_path(arguments, root=root, context=context)
        limit = _positive_int(arguments.get("limit"), default=1000)
        timeout = timeout_arg(arguments)
        command = [
            "--glob",
            "--color=never",
            "--hidden",
            "--print0",
            "--max-results",
            str(limit),
            "--",
            pattern,
            relative.as_posix() or ".",
        ]
        early, output, returncode, timed_out, cancelled = await run_search(
            manager=manager,
            tool_name="fd",
            arguments=command,
            workspace=workspace,
            timeout=timeout,
            signal=signal,
        )
        if early is not None:
            return early.model_copy(update={"name": "find"})
        output, output_truncated = strip_output_marker(output)
        paths = _format_fd_output(output, workspace=workspace, limit=limit)
        return bounded_result(
            name="find",
            content="\n".join(paths),
            data={
                "pattern": pattern,
                "path": relative.as_posix() or ".",
                "matches": len(paths),
                "limit": limit,
                "output_truncated": output_truncated,
            },
            returncode=returncode,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    tool = _create_native_tool(
        name="find",
        description=(
            "Find files and directories by glob name with fd. Results are workspace-relative, "
            "include hidden files, and respect .gitignore."
        ),
        args_schema=FindToolInput,
        executor=execute,
    )
    return ToolDefinition(
        tool=tool,
        label="find",
        prompt_snippet="Find files by name",
        prompt_guidelines=(
            "Use find for file-name search; pass a glob such as '*.py' or 'test_*'.",
        ),
    )


def create_find_tool(
    *,
    cwd: str | Path | None = None,
    tool_manager: ToolManager | None = None,
) -> ForgeStructuredTool:
    return create_find_tool_definition(cwd=cwd, tool_manager=tool_manager).tool  # type: ignore[return-value]


def _format_fd_output(data: bytes, *, workspace: Path, limit: int) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in data.split(b"\0"):
        if len(paths) >= limit:
            break
        value = raw_path.decode("utf-8", errors="replace")
        if not value:
            continue
        rendered = display_path(value, workspace=workspace)
        if rendered is None:
            continue
        if rendered in seen:
            continue
        seen.add(rendered)
        paths.append(rendered)
    return sorted(paths, key=lambda item: (item.casefold(), item))


def _positive_int(value: JSONValue, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("limit must be a positive integer")
    return value


__all__ = ["FindToolInput", "create_find_tool", "create_find_tool_definition"]
