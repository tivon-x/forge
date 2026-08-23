"""LangChain-native ``grep`` backed by managed ripgrep."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

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


class GrepToolInput(BaseModel):
    """Model-facing arguments for the native grep tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: StrictStr = Field(min_length=1, description="Text or regular expression to search for")
    path: StrictStr = Field(default=".", description="Workspace-relative file or directory")
    glob: StrictStr | None = Field(default=None, description="Optional file glob filter")
    ignore_case: StrictBool = Field(default=False, description="Match without case sensitivity")
    literal: StrictBool = Field(default=False, description="Treat pattern as literal text")
    context: StrictInt = Field(default=0, ge=0, description="Context lines around matches")
    limit: StrictInt = Field(default=100, ge=1, description="Maximum matching lines")


def create_grep_tool_definition(
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
        glob = arguments.get("glob")
        if glob is not None and (not isinstance(glob, str) or not glob):
            raise ValueError("glob must be a non-empty string")
        ignore_case = _bool_arg(arguments, "ignore_case")
        literal = _bool_arg(arguments, "literal")
        context_lines = _nonnegative_int(arguments, "context", default=0)
        limit = _positive_int(arguments, "limit", default=100)
        timeout = timeout_arg(arguments)

        command = ["--json", "--line-number", "--color=never", "--hidden"]
        if ignore_case:
            command.append("--ignore-case")
        if literal:
            command.append("--fixed-strings")
        if context_lines:
            command.extend(["--context", str(context_lines)])
        if isinstance(glob, str):
            command.extend(["--glob", glob])
        command.extend(["--", pattern, relative.as_posix() or "."])
        early, output, returncode, timed_out, cancelled = await run_search(
            manager=manager,
            tool_name="rg",
            arguments=command,
            workspace=workspace,
            timeout=timeout,
            signal=signal,
        )
        if early is not None:
            return early.model_copy(update={"name": "grep"})
        output, output_truncated = strip_output_marker(output)
        content, matches = _format_rg_output(output, workspace=workspace, limit=limit)
        return bounded_result(
            name="grep",
            content=content,
            data={
                "pattern": pattern,
                "path": str(relative.as_posix() or "."),
                "matches": matches,
                "limit": limit,
                "output_truncated": output_truncated,
            },
            returncode=returncode,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    tool = _create_native_tool(
        name="grep",
        description=(
            "Search workspace files with ripgrep. Supports regular expressions, literal text, "
            "case-insensitive matching, globs, context lines, and a match limit."
        ),
        args_schema=GrepToolInput,
        executor=execute,
    )
    return ToolDefinition(
        tool=tool,
        label="grep",
        prompt_snippet="Search files with grep",
        prompt_guidelines=(
            "Use grep for content search; it respects .gitignore while including hidden files.",
        ),
    )


def create_grep_tool(
    *,
    cwd: str | Path | None = None,
    tool_manager: ToolManager | None = None,
) -> ForgeStructuredTool:
    return create_grep_tool_definition(cwd=cwd, tool_manager=tool_manager).tool  # type: ignore[return-value]


def _format_rg_output(data: bytes, *, workspace: Path, limit: int) -> tuple[str, int]:
    lines: list[str] = []
    matches = 0
    pending_context: list[str] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            if raw_line and len(lines) < limit:
                lines.append(raw_line)
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        payload = event.get("data")
        if not isinstance(payload, dict) or event_type not in {"match", "context"}:
            continue
        path_data = payload.get("path")
        line_number = payload.get("line_number")
        line_data = payload.get("lines")
        path_text = path_data.get("text") if isinstance(path_data, dict) else None
        text = line_data.get("text") if isinstance(line_data, dict) else None
        if (
            not isinstance(path_text, str)
            or not isinstance(line_number, int)
            or not isinstance(text, str)
        ):
            continue
        path_display = display_path(path_text, workspace=workspace)
        if path_display is None:
            continue
        rendered = f"{path_display}:{line_number}: {text.rstrip()}"
        if event_type == "match":
            if matches >= limit:
                break
            lines.extend(pending_context)
            pending_context = []
            lines.append(rendered)
            matches += 1
        elif matches < limit:
            pending_context.append(rendered)
        elif matches == limit:
            lines.append(rendered)
    if matches < limit:
        lines.extend(pending_context)
    return "\n".join(lines), matches


def _bool_arg(arguments: Mapping[str, JSONValue], name: str) -> bool:
    value = arguments.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _nonnegative_int(arguments: Mapping[str, JSONValue], name: str, *, default: int) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(arguments: Mapping[str, JSONValue], name: str, *, default: int) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = ["GrepToolInput", "create_grep_tool", "create_grep_tool_definition"]
