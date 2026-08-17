"""LangChain-native read tool."""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken
from forge_agent.types import JSONValue
from forge_coding.tools.base import ForgeStructuredTool, ToolInputError, _create_native_tool
from forge_coding.tools.common import (
    _optional_int_arg,
    _path_arg,
    _str_arg,
    _workspace_root,
)
from forge_coding.tools.definition import ToolDefinition
from forge_coding.tools.file_operation_queue import file_operation_queue
from forge_coding.tools.truncation import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    format_size,
    truncate_head,
)

SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class ReadToolInput(BaseModel):
    """Validated model-facing arguments for the ``read`` tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: StrictStr = Field(description="Path to the file to read")
    offset: StrictInt | None = Field(
        default=None,
        ge=0,
        description="Line number to start reading from",
    )
    limit: StrictInt | None = Field(
        default=None,
        ge=1,
        description="Maximum number of lines to read",
    )


def create_read_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create a definition for the `read` tool.

    The tool reads a file resolved relative to `cwd` unless an absolute path is
    supplied. Text files are decoded as UTF-8 and may be sliced with optional
    1-indexed `offset` and positive integer `limit` arguments. Returned text is
    truncated to `DEFAULT_MAX_OUTPUT_LINES` lines or `DEFAULT_MAX_OUTPUT_BYTES`
    bytes, whichever comes first, and continuation hints are appended when more
    lines remain. Supported image paths (`jpg`, `png`, `gif`, and `webp`) are
    detected by MIME type and returned as base64 metadata instead of text.

    The executor raises `ToolInputError` for invalid arguments, missing files,
    directories, and offsets beyond the end of the file. Successful results
    include the resolved path and truncation metadata in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        del signal
        raw_path = _str_arg(arguments, "path")
        workspace = _workspace_root(context, root)
        path = _path_arg(arguments, "path", cwd=workspace)
        offset = _optional_int_arg(arguments, "offset")
        limit = _optional_int_arg(arguments, "limit")

        if offset is not None and offset < 0:
            raise ToolInputError("offset must be at least 0")
        if limit is not None and limit < 1:
            raise ToolInputError("limit must be at least 1")
        async with file_operation_queue.operation(path):
            if not path.exists():
                raise ToolInputError(f"File not found: {path}")
            if path.is_dir():
                raise ToolInputError(f"Path is a directory: {path}")

            mime_type = _detect_supported_image_mime_type(path)
            if mime_type is not None:
                image_data = path.read_bytes()
            else:
                image_data = None
                text = path.read_text(encoding="utf-8")

        if mime_type is not None:
            return AgentToolResult(
                tool_call_id="",
                name="read",
                ok=True,
                content=f"Read image file [{mime_type}]",
                data={
                    "path": str(path),
                    "mime_type": mime_type,
                    "bytes": len(image_data or b""),
                    "image_base64": _base64_text(image_data or b""),
                },
            )

        all_lines = text.split("\n")
        start_line = 0 if offset is None or offset == 0 else offset - 1
        if start_line >= len(all_lines):
            raise ToolInputError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )

        user_limited_lines: int | None = None
        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected)
        start_display = start_line + 1
        details: dict[str, JSONValue] = {"path": str(path), "truncation": truncation.to_json()}

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode()))
            output = (
                f"[Line {start_display} is {first_line_size}, exceeds "
                f"{format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit. Use bash: sed -n "
                f"'{start_display}p' {raw_path} | head -c {DEFAULT_MAX_OUTPUT_BYTES}]"
            )
        elif truncation.truncated:
            end_display = start_display + truncation.output_lines - 1
            next_offset = end_display + 1
            output = truncation.content
            if truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output = (
                f"{truncation.content}\n\n[{remaining} more lines in file. "
                f"Use offset={next_offset} to continue.]"
            )
        else:
            output = truncation.content

        return AgentToolResult(
            tool_call_id="",
            name="read",
            ok=True,
            content=output,
            data=details,
        )

    tool = _create_native_tool(
        name="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
            "Images are returned as base64 metadata. For text files, output is truncated to "
            f"{DEFAULT_MAX_OUTPUT_LINES} lines or {DEFAULT_MAX_OUTPUT_BYTES // 1024}KB "
            "(whichever is hit first). Use offset/limit for large files. When you need the "
            "full file, continue with offset until complete."
        ),
        executor=execute,
        args_schema=ReadToolInput,
    )
    return ToolDefinition(
        tool=tool,
        label="read",
        prompt_snippet="Read file contents",
        prompt_guidelines=("Use read to examine files instead of cat or sed.",),
    )


def create_read_tool(*, cwd: str | Path | None = None) -> ForgeStructuredTool:
    """Create a native LangChain tool for reading UTF-8 files and images."""
    return create_read_tool_definition(cwd=cwd).tool  # type: ignore[return-value]


def _detect_supported_image_mime_type(path: Path) -> str | None:
    mime_type, _encoding = mimetypes.guess_type(path)
    return mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else None


def _base64_text(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")
