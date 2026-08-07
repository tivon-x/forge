"""Built-in filesystem and shell tools for Forge coding sessions.

The module exposes native LangChain `StructuredTool` factories plus richer
`ToolDefinition` objects for callers that need prompt metadata and JSON
schemas. The tools operate relative to a configurable working directory and
return `(content, artifact)` pairs so LangChain records a structured
`ToolMessage`. The historical `AgentTool` conversion remains only for offline
compatibility callers.
"""

from __future__ import annotations

import asyncio
import difflib
import inspect
import json
import mimetypes
import os
import signal
import subprocess
import tempfile
from collections.abc import Awaitable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any, cast

from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentTool, AgentToolResult, ToolCancellationToken, ToolExecutor
from forge_agent.types import JSONValue

DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2_000
SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
UTF8_BOM = "\ufeff"


class ToolInputError(ValueError):
    """Raised when a tool receives invalid structured arguments."""


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """Metadata describing how a tool output was shortened.

    `content` contains the returned slice. The remaining fields record whether
    truncation happened, whether the line or byte limit was responsible, the
    total size of the original output, the size of the returned output, and
    edge cases such as partial-line output or a first line that is too large to
    display safely.
    """

    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    def to_json(self) -> dict[str, JSONValue]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Complete definition for a coding tool before provider conversion.

    A definition contains the tool name, user-facing description, prompt
    metadata, JSON input schema, and async executor. `to_langchain_tool()` is
    the production conversion; `to_agent_tool()` is retained for old fixtures.
    """

    name: str
    description: str
    prompt_snippet: str
    prompt_guidelines: tuple[str, ...]
    input_schema: Mapping[str, JSONValue]
    executor: ToolExecutor

    def to_langchain_tool(self) -> ForgeStructuredTool:
        """Build the native LangChain ``StructuredTool`` for this definition."""

        args_schema = _args_schema_for_tool(self)

        async def invoke(
            *,
            runtime: ToolRuntime,
            **arguments: Any,
        ) -> tuple[str, dict[str, JSONValue]]:
            context = runtime.context if isinstance(runtime.context, ForgeRuntimeContext) else None
            try:
                result = await _call_executor(self.executor, arguments, context=context)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - tool boundary is model-facing
                # A raised executor failure is a real execution error, not a
                # completed tool run.  Raising ToolException (with
                # handle_tool_error=True) lets LangChain record a
                # ToolMessage(status="error") so traces/middleware agree with
                # Forge's failure view instead of a success ToolMessage whose
                # artifact merely carries ok=False.
                result = AgentToolResult(
                    tool_call_id="",
                    name=self.name,
                    ok=False,
                    content=str(exc),
                    error=str(exc),
                )
                runtime_tool_call_id = getattr(runtime, "tool_call_id", None)
                if not result.tool_call_id and isinstance(runtime_tool_call_id, str):
                    result = result.model_copy(update={"tool_call_id": runtime_tool_call_id})
                raise ToolException(_tool_result_text(result)) from exc
            runtime_tool_call_id = getattr(runtime, "tool_call_id", None)
            if not result.tool_call_id and isinstance(runtime_tool_call_id, str):
                result = result.model_copy(update={"tool_call_id": runtime_tool_call_id})
            result = _apply_runtime_context(result, getattr(runtime, "context", None))
            artifact = result.model_dump(mode="json")
            return _tool_result_text(result), cast(dict[str, JSONValue], artifact)

        # ``from __future__ import annotations`` leaves the annotation as a
        # string, while StructuredTool's injected-argument cache intentionally
        # inspects the raw signature.  Restore the runtime class explicitly so
        # LangGraph forwards ToolRuntime to the coroutine.
        invoke.__annotations__["runtime"] = ToolRuntime
        tool = ForgeStructuredTool.from_function(
            coroutine=invoke,
            name=self.name,
            description=self.description,
            args_schema=args_schema,
            response_format="content_and_artifact",
            handle_tool_error=True,
            forge_input_schema=self.input_schema,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
            definition=self,
        )
        return cast(ForgeStructuredTool, tool)

    def to_agent_tool(self) -> AgentTool:
        return AgentTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor=self.executor,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
        )


class ForgeStructuredTool(StructuredTool):
    """A native StructuredTool with Forge metadata and a test-only execute seam."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    forge_input_schema: Mapping[str, JSONValue] = Field(default_factory=dict, exclude=True)
    prompt_snippet: str | None = Field(default=None, exclude=True)
    prompt_guidelines: tuple[str, ...] = Field(default_factory=tuple, exclude=True)
    _definition: ToolDefinition | None = PrivateAttr(default=None)

    def __init__(self, *args: Any, definition: ToolDefinition | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._definition = definition

    async def execute(
        self,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        """Compatibility helper; agent execution uses ``ainvoke``/ToolRuntime."""

        if self._definition is None:
            raise RuntimeError(f"Tool {self.name} has no Forge definition")
        result = await self._definition.executor(arguments, signal=signal)
        if result.tool_call_id:
            return result
        return result.model_copy(update={"tool_call_id": ""})


def _args_schema_for_tool(definition: ToolDefinition) -> type[BaseModel]:
    """Return a precise enough Pydantic schema for each built-in tool.

    Unrecognized tool names fall back to the definition's JSON ``input_schema``
    so every Forge tool keeps a non-empty args schema.  An empty schema
    breaks LangChain's ``ToolRuntime`` injection for the native agent loop
    (the tool node then calls the coroutine without its injected ``runtime``
    keyword), which would make custom-named Forge tools unusable.
    """

    if definition.name == "read":
        return create_model(
            "ReadToolInput",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            path=(str, Field(description="Path to the file to read")),
            offset=(int | None, Field(default=None, description="1-indexed line offset")),
            limit=(int | None, Field(default=None, description="Maximum lines to read")),
        )
    if definition.name == "write":
        return create_model(
            "WriteToolInput",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            path=(str, Field(description="Path to the file to write")),
            content=(str, Field(description="UTF-8 file content")),
        )
    if definition.name == "edit":
        return create_model(
            "EditToolInput",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            path=(str, Field(description="Path to the file to edit")),
            edits=(list[Any], Field(description="Exact replacements")),
        )
    if definition.name == "bash":
        return create_model(
            "BashToolInput",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            command=(str, Field(description="Shell command to execute")),
            timeout=(float | None, Field(default=None, description="Timeout in seconds")),
        )
    return create_model(
        f"{definition.name.title()}ToolInput",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **_args_fields_from_json_schema(definition.input_schema),
    )


_JSON_TYPE_TO_PYTHON: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict[str, Any],
    "array": list[Any],
}


def _args_fields_from_json_schema(input_schema: Mapping[str, JSONValue]) -> dict[str, Any]:
    """Translate a JSON ``input_schema`` into Pydantic field definitions.

    Optional properties get a ``None`` default so providers may omit them; the
    required list is honored for required arguments.  Types outside the JSON
    primitives are treated as ``Any``.
    """

    fields: dict[str, Any] = {}
    properties = input_schema.get("properties", {})
    raw_required = input_schema.get("required")
    required = set(raw_required) if isinstance(raw_required, list) else set()
    if not isinstance(properties, Mapping):
        return fields
    for name, prop in properties.items():
        if not isinstance(name, str) or not isinstance(prop, Mapping):
            continue
        field_type: Any = Any
        if isinstance(prop.get("type"), str):
            field_type = _JSON_TYPE_TO_PYTHON.get(str(prop.get("type")), Any)
        description = prop.get("description")
        if name in required:
            fields[name] = (
                field_type,
                Field(description=description if isinstance(description, str) else None),
            )
        else:
            fields[name] = (
                field_type | None,
                Field(
                    default=None,
                    description=description if isinstance(description, str) else None,
                ),
            )
    return fields


def _call_executor(
    executor: ToolExecutor,
    arguments: Mapping[str, JSONValue],
    *,
    context: ForgeRuntimeContext | None,
) -> Awaitable[AgentToolResult]:
    """Invoke a tool executor with the injected runtime context.

    Executors that declare a ``context`` parameter (the built-in Forge tools)
    receive the session-owned ``ForgeRuntimeContext``; legacy-style executors
    that only accept ``(arguments, signal)`` are invoked unchanged so offline
    fixtures keep working.
    """

    try:
        parameters = inspect.signature(executor).parameters
    except (TypeError, ValueError):
        parameters = ()  # type: ignore[assignment]  # unsignable legacy callable
    if "context" in parameters:
        return executor(arguments, signal=None, context=context)
    return executor(arguments, signal=None)


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


def _tool_result_text(result: AgentToolResult) -> str:
    content = result.content
    if result.data is not None and not content:
        content = json.dumps(result.data, ensure_ascii=False)
    if not result.ok and result.error and result.error not in content:
        content = f"{content}\n\nError: {result.error}"
    return content


def _apply_runtime_context(result: AgentToolResult, context: object) -> AgentToolResult:
    """Merge the injected ``ToolRuntime`` context into the structured result.

    The context is the session-owned ``ForgeRuntimeContext`` (workspace root,
    session id, shell prefix).  It is recorded on the ``details`` metadata field
    so the artifact stays a valid ``AgentToolResult`` JSON payload while the
    execution function actually consumes the context LangChain injected.
    """

    if not isinstance(context, ForgeRuntimeContext):
        return result
    details = dict(result.details or {})
    details["workspace_root"] = context.workspace_root
    details["session_id"] = context.session_id
    details["shell_command_prefix"] = context.shell_command_prefix
    return result.model_copy(update={"details": details})


_file_locks: dict[Path, asyncio.Lock] = {}


def create_coding_tools(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
) -> list[ForgeStructuredTool]:
    """Create the default coding-tool set for a local project.

    The returned tools are ordered as `read`, `write`, `edit`, and `bash`.
    Relative paths used with those tools are resolved against `cwd`; when `cwd`
    is omitted, the process current working directory at factory-call time is
    used. The tools share per-path write/edit locks within this process so
    concurrent mutations of the same file do not interleave. When configured,
    `shell_command_prefix` is prepended to every bash tool command.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    return [
        create_read_tool(cwd=root),
        create_write_tool(cwd=root),
        create_edit_tool(cwd=root),
        create_bash_tool(cwd=root, shell_command_prefix=shell_command_prefix),
    ]


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
        if not path.exists():
            raise ToolInputError(f"File not found: {path}")
        if path.is_dir():
            raise ToolInputError(f"Path is a directory: {path}")

        mime_type = _detect_supported_image_mime_type(path)
        if mime_type is not None:
            data = path.read_bytes()
            return AgentToolResult(
                tool_call_id="",
                name="read",
                ok=True,
                content=f"Read image file [{mime_type}]",
                data={
                    "path": str(path),
                    "mime_type": mime_type,
                    "bytes": len(data),
                    "image_base64": _base64_text(data),
                },
            )

        text = path.read_text(encoding="utf-8")
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

    return ToolDefinition(
        name="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
            "Images are returned as base64 metadata. For text files, output is truncated to "
            f"{DEFAULT_MAX_OUTPUT_LINES} lines or {DEFAULT_MAX_OUTPUT_BYTES // 1024}KB "
            "(whichever is hit first). Use offset/limit for large files. When you need the "
            "full file, continue with offset until complete."
        ),
        prompt_snippet="Read file contents",
        prompt_guidelines=("Use read to examine files instead of cat or sed.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "integer", "description": "Line number to start reading from"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
        executor=execute,
    )


def create_read_tool(*, cwd: str | Path | None = None) -> ForgeStructuredTool:
    """Create a native LangChain tool for reading UTF-8 files and images."""
    return create_read_tool_definition(cwd=cwd).to_langchain_tool()


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

        async with _file_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        return AgentToolResult(
            tool_call_id="",
            name="write",
            ok=True,
            content=f"Successfully wrote to {path}.",
            data={"path": str(path), "characters": len(content)},
        )

    return ToolDefinition(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=("Use write only for new files or complete rewrites.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        executor=execute,
    )


def create_write_tool(*, cwd: str | Path | None = None) -> ForgeStructuredTool:
    """Create a native LangChain tool for creating or overwriting UTF-8 files."""
    return create_write_tool_definition(cwd=cwd).to_langchain_tool()


def create_edit_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create a definition for the `edit` tool.

    The tool applies one or more exact text replacements to a single UTF-8 file
    resolved relative to `cwd`. Each edit item contains `oldText` and `newText`.
    Every `oldText` must be non-empty, must occur exactly once in the original
    file, and must not overlap another edit span. All replacements are validated
    before writing, so the file is left unchanged if any edit fails.

    File content and edit text are normalized to LF for matching, then the
    original file's dominant line ending is restored after replacement. UTF-8
    byte-order marks are preserved. The executor also accepts legacy top-level
    `oldText`/`newText` arguments and JSON-string `edits` values by normalizing
    them into the canonical edits list.

    Successful results include the resolved path, edit count, an ndiff-style
    diff, a unified patch, and the first changed line in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        del signal
        prepared = _prepare_edit_arguments(arguments)
        workspace = _workspace_root(context, root)
        path = _path_arg(prepared, "path", cwd=workspace, for_write=True)
        edits = _edits_arg(prepared)

        if not path.exists():
            raise ToolInputError(f"Could not edit file: {path}. File not found.")
        if path.is_dir():
            raise ToolInputError(f"Could not edit file: {path}. Path is a directory.")

        async with _file_lock(path):
            raw_content = path.read_text(encoding="utf-8")
            bom, content = _strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized = normalize_to_lf(content)
            base_content, new_content = apply_edits_to_normalized_content(
                normalized, edits, str(path)
            )
            final_content = bom + restore_line_endings(new_content, original_ending)
            path.write_text(final_content, encoding="utf-8")

        diff_text, first_changed_line = generate_diff_string(base_content, new_content)
        patch = generate_unified_patch(str(path), base_content, new_content)
        return AgentToolResult(
            tool_call_id="",
            name="edit",
            ok=True,
            content=f"Successfully replaced {len(edits)} block(s) in {path}.",
            data={
                "path": str(path),
                "edits": len(edits),
                "diff": diff_text,
                "patch": patch,
                "first_changed_line": first_changed_line,
            },
        )

    return ToolDefinition(
        name="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match "
            "a unique, non-overlapping region of the original file. If two changes affect the "
            "same block or nearby lines, merge them into one edit instead of emitting overlapping "
            "edits. Do not include large unchanged regions just to connect distant changes."
        ),
        prompt_snippet=(
            "Make precise file edits with exact text replacement, including multiple disjoint "
            "edits in one call"
        ),
        prompt_guidelines=(
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with "
            "multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier "
            "edits are applied. Do not emit overlapping or nested edits. Merge nearby "
            "changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. "
            "Do not pad with large unchanged regions.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                        "required": ["oldText", "newText"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        },
        executor=execute,
    )


def create_edit_tool(*, cwd: str | Path | None = None) -> ForgeStructuredTool:
    """Create a native LangChain tool for exact validated text replacements."""
    return create_edit_tool_definition(cwd=cwd).to_langchain_tool()


def create_bash_tool_definition(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
) -> ToolDefinition:
    """Create a definition for the `bash` tool.

    The tool runs a shell command with `cwd` as the subprocess working
    directory and combines stdout and stderr into one UTF-8 decoded output
    stream. The optional `timeout` argument must be positive when supplied. On
    timeout, POSIX commands are started in a new session and the entire process
    group is killed so shell children from pipelines or compound commands do
    not continue running; non-POSIX platforms fall back to killing the direct
    subprocess.

    Output is tail-truncated to `DEFAULT_MAX_OUTPUT_LINES` lines or
    `DEFAULT_MAX_OUTPUT_BYTES` bytes. When truncation occurs, the full output is
    written to a temporary log file and that path is reported in `data`.
    Successful and failed command results both include exit code, timeout state,
    duration, truncation metadata, and full-output path metadata.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    prefix = shell_command_prefix.strip() if shell_command_prefix else None

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        command = _str_arg(arguments, "command")
        workspace = _workspace_root(context, root)
        effective_prefix = _shell_command_prefix(context, prefix)
        shell_command = _prefixed_shell_command(command, effective_prefix)
        timeout = _optional_float_arg(arguments, "timeout")
        if timeout is not None and timeout <= 0:
            raise ToolInputError("timeout must be greater than 0")
        if signal is not None and signal.is_cancelled():
            raise ToolInputError("Command cancelled")

        start = monotonic()
        if os.name == "posix":
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                executable="bash" if effective_prefix else None,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        output_bytes, _stderr, timed_out, cancelled = await _communicate_with_cancellation(
            process,
            timeout=timeout,
            signal=signal,
        )

        output = output_bytes.decode(errors="replace")
        truncation = truncate_tail(output)
        full_output_path: str | None = None
        output_text = truncation.content or "(no output)"
        if truncation.truncated:
            full_output_path = _write_temp_output(output)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. "
                    f"Full output: {full_output_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {full_output_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Full output: {full_output_path}]"
                )

        exit_code = process.returncode
        status: str | None = None
        if timed_out:
            status = (
                f"Command timed out after {timeout:g} seconds" if timeout else "Command timed out"
            )
        elif cancelled:
            status = "Command cancelled"
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        if status:
            output_text = append_status_block(output_text, status)

        ok = exit_code == 0 and not timed_out and not cancelled
        return AgentToolResult(
            tool_call_id="",
            name="bash",
            ok=ok,
            content=output_text,
            error=None if ok else status,
            data={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "cancelled": cancelled,
                "duration_seconds": round(monotonic() - start, 3),
                "truncation": truncation.to_json(),
                "full_output_path": full_output_path,
                "shell_command_prefix_applied": effective_prefix is not None,
            },
        )

    return ToolDefinition(
        name="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_OUTPUT_LINES} lines or "
            f"{DEFAULT_MAX_OUTPUT_BYTES // 1024}KB (whichever is hit first). If truncated, "
            "full output is saved to a temp file. Optionally provide a timeout in seconds."
        ),
        prompt_snippet="Execute bash commands (ls, grep, find, etc.)",
        prompt_guidelines=(),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional, no default timeout)",
                },
            },
            "required": ["command"],
        },
        executor=execute,
    )


def create_bash_tool(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
) -> ForgeStructuredTool:
    """Create a native LangChain tool for executing bounded shell commands."""
    return create_bash_tool_definition(
        cwd=cwd,
        shell_command_prefix=shell_command_prefix,
    ).to_langchain_tool()


def _prefixed_shell_command(command: str, prefix: str | None) -> str:
    """Return a shell command with an opt-in setup prefix applied."""
    if prefix is None:
        return command
    return f"{prefix}\n{command}"


def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def append_status_block(text: str, status: str) -> str:
    """Append command status text after a blank line when output already exists."""
    return f"{text}\n\n{status}" if text else status


async def _communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    *,
    timeout: float | None,
    signal: ToolCancellationToken | None,
) -> tuple[bytes, bytes | None, bool, bool]:
    communicate = asyncio.create_task(process.communicate())
    cancel_watch: asyncio.Task[None] | None = None
    try:
        wait_for: set[asyncio.Task[Any]] = {communicate}
        if signal is not None:
            cancel_watch = asyncio.create_task(_wait_for_cancel(signal))
            wait_for.add(cancel_watch)

        done, _pending = await asyncio.wait(
            wait_for,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if communicate in done:
            output_bytes, stderr = communicate.result()
            return output_bytes, stderr, False, False

        cancelled = cancel_watch is not None and cancel_watch in done
        _kill_process_tree(process)
        try:
            output_bytes, stderr = await communicate
        except asyncio.CancelledError:
            output_bytes = b""
            stderr_result: bytes | None = None
        else:
            stderr_result = stderr
        return output_bytes, stderr_result, not cancelled, cancelled
    except asyncio.CancelledError:
        _kill_process_tree(process)
        if not communicate.done():
            communicate.cancel()
        raise
    finally:
        if cancel_watch is not None:
            cancel_watch.cancel()


async def _wait_for_cancel(signal: ToolCancellationToken) -> None:
    while not signal.is_cancelled():
        await asyncio.sleep(0.05)


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _truncation_result(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )

    first_line_bytes = len(lines[0].encode()) if lines else 0
    if first_line_bytes > max_bytes:
        return _truncation_result(
            "", True, "bytes", total_lines, total_bytes, 0, 0, first_line=True
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    for index, line in enumerate(lines[:max_lines]):
        line_bytes = len(line.encode()) + (1 if index > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes += line_bytes

    output = "\n".join(output_lines)
    return _truncation_result(
        output,
        True,
        truncated_by,
        total_lines,
        total_bytes,
        len(output_lines),
        len(output.encode()),
    )


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _truncation_result(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    last_line_partial = False
    for line in reversed(lines):
        line_bytes = len(line.encode()) + (1 if output_lines else 0)
        if len(output_lines) >= max_lines:
            truncated_by = "lines"
            break
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                clipped = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, clipped)
                output_bytes = len(clipped.encode())
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes

    output = "\n".join(output_lines)
    return _truncation_result(
        output,
        True,
        truncated_by,
        total_lines,
        total_bytes,
        len(output_lines),
        len(output.encode()),
        last_line_partial=last_line_partial,
    )


def detect_line_ending(content: str) -> str:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[dict[str, str]],
    path: str,
) -> tuple[str, str]:
    normalized_edits = [
        {"oldText": normalize_to_lf(edit["oldText"]), "newText": normalize_to_lf(edit["newText"])}
        for edit in edits
    ]
    for index, edit in enumerate(normalized_edits):
        if not edit["oldText"]:
            raise ToolInputError(_empty_old_text_error(path, index, len(normalized_edits)))

    matches: list[tuple[int, int, str]] = []
    for index, edit in enumerate(normalized_edits):
        old_text = edit["oldText"]
        occurrences = _count_occurrences(normalized_content, old_text)
        if occurrences == 0:
            raise ToolInputError(_not_found_error(path, index, len(normalized_edits)))
        if occurrences > 1:
            raise ToolInputError(_duplicate_error(path, index, len(normalized_edits), occurrences))
        start = normalized_content.index(old_text)
        matches.append((start, start + len(old_text), edit["newText"]))

    _validate_non_overlapping(matches)
    new_content = normalized_content
    for start, end, new_text in sorted(matches, reverse=True):
        new_content = f"{new_content[:start]}{new_text}{new_content[end:]}"
    if new_content == normalized_content:
        raise ToolInputError(_no_change_error(path, len(normalized_edits)))
    return normalized_content, new_content


def generate_diff_string(old: str, new: str) -> tuple[str, int | None]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = "\n".join(difflib.ndiff(old_lines, new_lines))
    first_changed_line: int | None = None
    new_line_number = 0
    for line in difflib.ndiff(old_lines, new_lines):
        if line.startswith("  "):
            new_line_number += 1
        elif line.startswith("+"):
            new_line_number += 1
            if first_changed_line is None:
                first_changed_line = new_line_number
        elif line.startswith("-") and first_changed_line is None:
            first_changed_line = max(new_line_number + 1, 1)
    return diff, first_changed_line


def generate_unified_patch(path: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )


def _truncation_result(
    content: str,
    truncated: bool,
    truncated_by: str | None,
    total_lines: int,
    total_bytes: int,
    output_lines: int,
    output_bytes: int,
    *,
    last_line_partial: bool = False,
    first_line: bool = False,
) -> TruncationResult:
    return TruncationResult(
        content=content,
        truncated=truncated,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=output_lines,
        output_bytes=output_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=first_line,
        max_lines=DEFAULT_MAX_OUTPUT_LINES,
        max_bytes=DEFAULT_MAX_OUTPUT_BYTES,
    )


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[-max_bytes:]
    return clipped.decode(errors="ignore")


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


def _validate_non_overlapping(spans: list[tuple[int, int, str]]) -> None:
    previous_end = -1
    for start, end, _new_text in sorted(spans):
        if start < previous_end:
            raise ToolInputError("Edits must not overlap")
        previous_end = end


def _count_occurrences(content: str, text: str) -> int:
    count = 0
    start = 0
    while True:
        index = content.find(text, start)
        if index == -1:
            return count
        count += 1
        start = index + len(text)


def _strip_bom(content: str) -> tuple[str, str]:
    return (UTF8_BOM, content[1:]) if content.startswith(UTF8_BOM) else ("", content)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"Could not find the exact text in {path}. The old text must match exactly "
            "including all whitespace and newlines."
        )
    return (
        f"Could not find edits[{edit_index}] in {path}. The oldText must match exactly "
        "including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> str:
    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            "Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return f"oldText must not be empty in {path}."
    return f"edits[{edit_index}].oldText must not be empty in {path}."


def _no_change_error(path: str, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing "
            "as expected."
        )
    return f"No changes made to {path}. The replacements produced identical content."


def _detect_supported_image_mime_type(path: Path) -> str | None:
    mime_type, _encoding = mimetypes.guess_type(path)
    return mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else None


def _base64_text(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except ProcessLookupError:
            return
    else:
        # ``create_subprocess_shell`` starts cmd.exe on Windows. Killing only
        # that direct process leaves grandchildren holding the captured output
        # pipe open, so ``communicate()`` cannot return promptly on cancellation.
        # taskkill receives a validated integer PID as a distinct argument.
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return
        try:
            process.kill()
        except ProcessLookupError:
            return


def _write_temp_output(output: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="forge-bash-",
        suffix=".log",
        delete=False,
    ) as handle:
        handle.write(output)
        return handle.name


class _FileLockContext:
    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        lock = _file_locks.setdefault(self._path, asyncio.Lock())
        self._lock = lock
        await lock.acquire()

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self._lock is not None:
            self._lock.release()


def _file_lock(path: Path) -> _FileLockContext:
    return _FileLockContext(path)
