"""Pure formatting compatibility facade shared by CLI renderers and TUI."""

from __future__ import annotations

from forge_cli.tool_rendering import (
    TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES,
    TOOL_PATCH_PREVIEW_LINES,
    TOOL_RESULT_PREVIEW_CHARS,
    TOOL_RESULT_PREVIEW_LINES,
    TOOL_VIEW_REGISTRY,
    ToolFormatter,
    ToolViewRegistry,
    _int_argument,
    _number_argument,
    _read_line_suffix,
    _string_argument,
    format_generic_tool_result_block,
    format_task_agent,
    format_terminal_command_result_block,
    format_tool_call_block,
    format_tool_call_invocation,
    format_tool_result_block,
    format_tool_result_summary,
)

__all__ = [
    "TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES",
    "TOOL_PATCH_PREVIEW_LINES",
    "TOOL_RESULT_PREVIEW_CHARS",
    "TOOL_RESULT_PREVIEW_LINES",
    "TOOL_VIEW_REGISTRY",
    "ToolFormatter",
    "ToolViewRegistry",
    "_int_argument",
    "_number_argument",
    "_read_line_suffix",
    "_string_argument",
    "format_terminal_command_result_block",
    "format_generic_tool_result_block",
    "format_task_agent",
    "format_tool_call_block",
    "format_tool_call_invocation",
    "format_tool_result_block",
    "format_tool_result_summary",
]
