"""LangChain-native shell execution tool."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from time import monotonic

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken
from forge_agent.types import JSONValue
from forge_coding.tools.base import ForgeStructuredTool, ToolInputError, _create_native_tool
from forge_coding.tools.common import (
    _optional_float_arg,
    _shell_command_prefix,
    _str_arg,
    _workspace_root,
)
from forge_coding.tools.definition import ToolDefinition
from forge_coding.tools.shell import (
    _communicate_with_cancellation,
    _decode_shell_output,
    _prefixed_shell_command,
    _windows_bash_path,
    _write_temp_output,
)
from forge_coding.tools.truncation import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    append_status_block,
    format_size,
    truncate_tail,
)


class BashToolInput(BaseModel):
    """Validated model-facing arguments for the ``bash`` tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: StrictStr = Field(description="Bash command to execute")
    timeout: float | None = Field(
        default=None,
        gt=0,
        description="Timeout in seconds (optional, no default timeout)",
    )


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
    subprocess tree.

    On Windows, `create_subprocess_shell` would run `cmd.exe`, which cannot
    execute bash commands such as `ls`. The tool therefore runs a real bash
    (Git for Windows) when one is installed and falls back to the system shell
    otherwise; the effective shell is reported in `data["shell"]`.

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
        shell_name = "bash"
        if os.name == "posix":
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                executable="bash" if effective_prefix else None,
            )
        elif (bash_path := _windows_bash_path()) is not None:
            # On Windows, ``create_subprocess_shell`` runs cmd.exe, which
            # cannot execute bash commands such as ``ls`` unless Git's tools
            # happen to be on PATH. Run a real bash when one is installed.
            process = await asyncio.create_subprocess_exec(
                bash_path,
                "-c",
                shell_command,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            shell_name = "cmd"
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

        output = _decode_shell_output(output_bytes)
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
                "shell": shell_name,
            },
        )

    tool = _create_native_tool(
        name="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. "
            "On Windows, runs Git Bash when installed and falls back to cmd.exe otherwise; "
            "quote Windows paths (backslashes are escape characters outside quotes). "
            f"Output is truncated to last {DEFAULT_MAX_OUTPUT_LINES} lines or "
            f"{DEFAULT_MAX_OUTPUT_BYTES // 1024}KB (whichever is hit first). If truncated, "
            "full output is saved to a temp file. Optionally provide a timeout in seconds."
        ),
        executor=execute,
        args_schema=BashToolInput,
    )
    return ToolDefinition(
        tool=tool,
        label="bash",
        prompt_snippet="Execute bash commands (ls, grep, find, etc.)",
        prompt_guidelines=(),
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
    ).tool  # type: ignore[return-value]
