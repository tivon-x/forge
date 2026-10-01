"""Pure tool-call and tool-result presentation registry.

The registry deliberately knows nothing about Rich, Textual, persistence, or
agent execution. It accepts Forge's small tool projections and returns bounded
plain text that every presentation surface can reuse.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from forge_agent.tools import ToolCall
from forge_agent.types import JSONValue

TOOL_RESULT_PREVIEW_LINES = 8
TOOL_PATCH_PREVIEW_LINES = 32
TOOL_RESULT_PREVIEW_CHARS = 2_000
TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES = 120
TASK_AGENT_MAX_CHARS = 32
TASK_CONTROL_MAX_RUNS = 16
TASK_CONTROL_MAX_ID_CHARS = 64
TASK_CONTROL_MAX_INTEGER = 2**63 - 1

_TASK_CONTROL_STATUSES = frozenset(
    {"queued", "running", "completed", "failed", "cancelled", "interrupted", "error"}
)

ToolResultData = dict[str, JSONValue] | None
CallFormatter = Callable[[ToolCall], str]
ResultFormatter = Callable[[str, bool, str, ToolResultData], str]


@dataclass(frozen=True, slots=True)
class ToolFormatter:
    """Pure formatter pair registered for one tool name."""

    call: CallFormatter | None = None
    result: ResultFormatter | None = None


@dataclass(slots=True)
class ToolViewRegistry:
    """Name-to-formatter registry used by plain and interactive renderers."""

    _formatters: dict[str, ToolFormatter] = field(default_factory=dict)

    def register(
        self,
        name: str,
        *,
        call: CallFormatter | None = None,
        result: ResultFormatter | None = None,
    ) -> None:
        """Register or replace pure formatters for ``name``."""
        if not name or not name.strip():
            raise ValueError("tool formatter name must not be empty")
        if call is None and result is None:
            raise ValueError("tool formatter must define call or result")
        self._formatters[name] = ToolFormatter(call=call, result=result)

    def formatter(self, name: str) -> ToolFormatter | None:
        """Return the registered formatter, if any."""
        return self._formatters.get(name)

    def format_call(self, tool_call: ToolCall) -> str:
        """Format a tool call, using the bounded generic fallback."""
        formatter = self._formatters.get(tool_call.name)
        if formatter is not None and formatter.call is not None:
            return formatter.call(tool_call)
        return _fallback_tool_call_invocation(tool_call)

    def format_result(
        self,
        *,
        name: str,
        ok: bool,
        content: str,
        data: ToolResultData = None,
    ) -> str:
        """Format a tool result, using the bounded generic fallback."""
        formatter = self._formatters.get(name)
        if formatter is not None and formatter.result is not None:
            return formatter.result(name, ok, content, data)
        return _generic_tool_result_block(name=name, ok=ok, content=content, data=data)

    def format_summary(self, *, name: str, ok: bool) -> str:
        """Format the collapsed/orphan result line."""
        return _format_tool_result_summary(name=name, ok=ok)


def _builtin_call(tool_call: ToolCall) -> str:
    arguments = tool_call.arguments
    if tool_call.name == "read":
        path = _string_argument(arguments, "path")
        if path is not None:
            return f"read {path}{_read_line_suffix(arguments)}"
    elif tool_call.name in {"edit", "write"}:
        path = _string_argument(arguments, "path")
        if path is not None:
            return f"{tool_call.name} {path}"
    elif tool_call.name == "bash":
        command = _string_argument(arguments, "command")
        if command is not None:
            timeout = _number_argument(arguments, "timeout")
            suffix = f" (timeout {timeout:g}s)" if timeout is not None else ""
            return f"$ {command}{suffix}"
    elif tool_call.name == "task":
        # Only the allowlisted role is shown. The child prompt is never a
        # transcript formatter input, even when the model supplied it.
        agent = _string_argument(arguments, "agent")
        if agent is not None:
            return f"task {format_task_agent(agent)}"
    return _fallback_tool_call_invocation(tool_call)


def _builtin_result(name: str, ok: bool, content: str, data: ToolResultData) -> str:
    if name == "task":
        # Child output and prompt are not part of the generic task projection.
        return _format_tool_result_summary(name=name, ok=ok)
    if name == "task_control":
        return _format_task_control_result(ok=ok, data=data)
    return _generic_tool_result_block(name=name, ok=ok, content=content, data=data)


def _make_default_registry() -> ToolViewRegistry:
    registry = ToolViewRegistry()
    for name in ("read", "write", "bash"):
        registry.register(name, call=_builtin_call, result=_builtin_result)
    registry.register("edit", call=_builtin_call, result=_builtin_result)
    registry.register("task", call=_builtin_call, result=_builtin_result)
    registry.register("task_control", result=_builtin_result)
    return registry


TOOL_VIEW_REGISTRY = _make_default_registry()


def format_tool_call_block(tool_call: ToolCall) -> str:
    """Format a collapsed tool call for live and restored transcript blocks."""
    invocation = TOOL_VIEW_REGISTRY.format_call(tool_call)
    return invocation if tool_call.name == "bash" else f"→ {invocation}"


def format_tool_call_invocation(tool_call: ToolCall) -> str:
    """Format a terse human-readable tool invocation."""
    return TOOL_VIEW_REGISTRY.format_call(tool_call)


def format_tool_result_summary(*, name: str, ok: bool) -> str:
    return TOOL_VIEW_REGISTRY.format_summary(name=name, ok=ok)


def format_task_agent(agent: str) -> str:
    """Return a bounded model-provided task role for presentation state."""
    if len(agent) <= TASK_AGENT_MAX_CHARS:
        return agent
    return f"{agent[:TASK_AGENT_MAX_CHARS]}…"


def format_tool_result_block(
    *,
    name: str,
    ok: bool,
    content: str,
    data: ToolResultData = None,
) -> str:
    return TOOL_VIEW_REGISTRY.format_result(name=name, ok=ok, content=content, data=data)


def format_generic_tool_result_block(
    *,
    name: str,
    ok: bool,
    content: str,
    data: ToolResultData = None,
) -> str:
    """Format a result without a name-specific projection.

    This is used for malformed/unknown task artifacts that intentionally fall
    back to the legacy ordinary-tool row.
    """
    return _generic_tool_result_block(name=name, ok=ok, content=content, data=data)


def format_terminal_command_result_block(*, ok: bool, added_to_context: bool, output: str) -> str:
    """Format an input-bar terminal command result for visible TUI display."""
    status = "✓" if ok else "✗"
    suffix = " · added to context" if added_to_context else " · not added to context"
    lines = [f"{status} bash{suffix}"]
    if output:
        lines.append(_preview_text(output, max_lines=TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES))
    return "\n".join(lines)


def _generic_tool_result_block(
    *,
    name: str,
    ok: bool,
    content: str,
    data: ToolResultData,
) -> str:
    status = "✓" if ok else "✗"
    lines = [f"{status} {name}"]
    if content:
        lines.append(_preview_text(content, max_lines=TOOL_RESULT_PREVIEW_LINES))
    patch = _result_patch(name=name, ok=ok, data=data)
    if patch:
        lines.extend(["", "Patch:", _preview_text(patch, max_lines=TOOL_PATCH_PREVIEW_LINES)])
    return "\n".join(lines)


def _format_tool_result_summary(*, name: str, ok: bool) -> str:
    status = "✓" if ok else "✗"
    return f"{status} {name}"


def _format_task_control_result(*, ok: bool, data: ToolResultData) -> str:
    """Render only bounded control metadata, never child payloads."""
    lines = [_format_tool_result_summary(name="task_control", ok=ok)]
    if not ok:
        # Tool errors may contain provider or child text. Keep this projection
        # fixed so the parent model remains the only full-result consumer.
        lines.append("Error: task_control failed")

    records: list[Mapping[str, JSONValue]] = []
    if isinstance(data, Mapping):
        raw_runs = data.get("runs")
        if isinstance(raw_runs, list):
            records.extend(
                value for value in raw_runs[:TASK_CONTROL_MAX_RUNS] if isinstance(value, Mapping)
            )
        else:
            records.append(data)
    for record in records:
        line = _format_task_control_run(record)
        if line is not None:
            lines.append(line)
    return "\n".join(lines)


def _format_task_control_run(record: Mapping[str, JSONValue]) -> str | None:
    """Return a single allowlisted run row from a control response."""
    run_id = _task_control_token(record.get("run_id"), TASK_CONTROL_MAX_ID_CHARS)
    agent = _task_control_token(record.get("agent"), TASK_AGENT_MAX_CHARS)
    raw_status = record.get("status")
    status = (
        raw_status if isinstance(raw_status, str) and raw_status in _TASK_CONTROL_STATUSES else None
    )
    if not run_id and not agent and status is None:
        return None

    fields = [value for value in (run_id, agent, status) if value]
    statistics: list[str] = []
    duration_ms = _task_control_integer(record.get("duration_ms"))
    if duration_ms is not None:
        statistics.append(f"{duration_ms}ms")
    tool_calls = _task_control_integer(record.get("tool_calls"))
    if tool_calls is not None:
        statistics.append(f"{tool_calls} tools")
    total_tokens = _task_control_integer(record.get("total_tokens"))
    if total_tokens is not None:
        statistics.append(f"{total_tokens} tokens")
    if statistics:
        fields.extend(statistics)
    return "  " + " · ".join(fields)


def _task_control_token(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if not value:
        return None
    return value[:limit]


def _task_control_integer(value: object) -> int | None:
    if type(value) is int and 0 <= value <= TASK_CONTROL_MAX_INTEGER:
        return value
    return None


def _result_patch(*, name: str, ok: bool, data: ToolResultData) -> str | None:
    if name != "edit" or not ok or data is None:
        return None
    patch = data.get("patch")
    return patch if isinstance(patch, str) and patch.strip() else None


def _preview_text(text: str, *, max_lines: int) -> str:
    lines = text.splitlines()
    if not lines:
        return text[:TOOL_RESULT_PREVIEW_CHARS]

    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    hidden_lines = max(0, len(lines) - len(preview_lines))
    truncated_by_chars = len(preview) > TOOL_RESULT_PREVIEW_CHARS
    if truncated_by_chars:
        preview = preview[:TOOL_RESULT_PREVIEW_CHARS].rstrip()

    if hidden_lines or truncated_by_chars:
        details: list[str] = []
        if hidden_lines:
            details.append(f"{hidden_lines} more line{'s' if hidden_lines != 1 else ''}")
        if truncated_by_chars:
            details.append("additional text")
        preview = f"{preview}\n\n[Preview only: {', '.join(details)} hidden from the TUI.]"
    return preview


def _fallback_tool_call_invocation(tool_call: ToolCall) -> str:
    if tool_call.name == "task":
        return "task"
    if tool_call.arguments:
        arguments = _preview_text(
            str(tool_call.arguments),
            max_lines=TOOL_RESULT_PREVIEW_LINES,
        )
        return f"{tool_call.name} {arguments}"
    return tool_call.name


def _read_line_suffix(arguments: Mapping[str, JSONValue]) -> str:
    offset = _int_argument(arguments, "offset")
    limit = _int_argument(arguments, "limit")
    if offset is None and limit is None:
        return ""
    start = 1 if offset is None else max(1, offset)
    if limit is None:
        return f":{start}-"
    return f":{start}-{start + max(1, limit) - 1}"


def _string_argument(arguments: Mapping[str, JSONValue], key: str) -> str | None:
    value = arguments.get(key)
    return value if isinstance(value, str) else None


def _int_argument(arguments: Mapping[str, JSONValue], key: str) -> int | None:
    value = arguments.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _number_argument(arguments: Mapping[str, JSONValue], key: str) -> int | float | None:
    value = arguments.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int | float) else None


__all__ = [
    "TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES",
    "TOOL_PATCH_PREVIEW_LINES",
    "TOOL_RESULT_PREVIEW_CHARS",
    "TOOL_RESULT_PREVIEW_LINES",
    "TOOL_VIEW_REGISTRY",
    "ToolFormatter",
    "ToolViewRegistry",
    "format_terminal_command_result_block",
    "format_task_agent",
    "format_tool_call_block",
    "format_tool_call_invocation",
    "format_generic_tool_result_block",
    "format_tool_result_block",
    "format_tool_result_summary",
]
