"""Bounded text output helpers shared by coding tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from forge_agent.types import JSONValue

DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2_000


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


def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def append_status_block(text: str, status: str) -> str:
    """Append command status text after a blank line when output already exists."""
    return f"{text}\n\n{status}" if text else status


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
