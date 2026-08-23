"""File-operation extraction shared by model-assisted summaries.

Compaction and branch summaries both append ``<read-files>``/``<modified-files>``
context so a later model knows which files the summarized region touched.
Extraction never executes tools: it only reads ``AIMessage.tool_calls``
arguments, so it is deterministic and offline-testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

from forge_agent import ErrorEvent
from forge_agent.retry import classify_model_error
from forge_agent.session import SessionState
from forge_coding.sessions.context_usage import estimate_message_tokens
from forge_coding.sessions.tree import _message_role

DETAIL_READ_FILES = "read_files"
DETAIL_MODIFIED_FILES = "modified_files"


@dataclass(slots=True)
class FileOperations:
    """Accumulated read/write/edit paths from summarized tool calls."""

    read: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)


def extract_file_operations(messages: Sequence[AnyMessage]) -> FileOperations:
    """Collect read/edit/write paths from tool calls in assistant messages."""
    operations = FileOperations()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if not isinstance(call, Mapping):
                continue
            raw_args = call.get("args")
            arguments = raw_args if isinstance(raw_args, Mapping) else {}
            path = arguments.get("path")
            if not isinstance(path, str) or not path:
                continue
            name = str(call.get("name") or "tool")
            if name == "read":
                operations.read.add(path)
            elif name == "edit":
                operations.edited.add(path)
            elif name == "write":
                operations.written.add(path)
    return operations


def merge_file_operations(*operations: FileOperations) -> FileOperations:
    """Merge accumulated operations, preserving write/edit history."""
    merged = FileOperations()
    for operations_set in operations:
        merged.read.update(operations_set.read)
        merged.edited.update(operations_set.edited)
        merged.written.update(operations_set.written)
    return merged


def file_operations_from_details(details: Mapping[str, object] | None) -> FileOperations:
    """Rebuild accumulated operations from a compaction entry's details."""
    operations = FileOperations()
    if not details:
        return operations
    operations.read.update(_string_list(details.get(DETAIL_READ_FILES)))
    # Files modified by an earlier compaction stay modified: re-read them as
    # edited so the final list keeps them under ``modified``, never read-only.
    operations.edited.update(_string_list(details.get(DETAIL_MODIFIED_FILES)))
    return operations


def compute_file_lists(operations: FileOperations) -> tuple[list[str], list[str]]:
    """Return sorted (read_only, modified) path lists for one summary."""
    modified = set(operations.edited) | set(operations.written)
    read_only = sorted(path for path in operations.read if path not in modified)
    return read_only, sorted(modified)


def details_from_file_operations(
    operations: FileOperations,
) -> dict[str, list[str]]:
    """Serialize accumulated operations into durable compaction details."""
    read_only, modified = compute_file_lists(operations)
    return {DETAIL_READ_FILES: read_only, DETAIL_MODIFIED_FILES: modified}


def format_file_operations(read_files: Sequence[str], modified_files: Sequence[str]) -> str:
    """Format file lists as the XML sections appended to summaries."""
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n{'\n'.join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n{'\n'.join(modified_files)}\n</modified-files>")
    if not sections:
        return ""
    return f"\n\n{'\n\n'.join(sections)}"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    """Prepared active-context entries for a compaction run."""

    replace_entry_ids: tuple[str, ...]
    messages_to_summarize: tuple[Any, ...]
    turn_prefix_messages: tuple[Any, ...] = ()


def _first_recent_context_index(
    rows: tuple[tuple[str, Any], ...],
    *,
    keep_recent_tokens: int,
) -> int:
    if keep_recent_tokens <= 0:
        return len(rows)

    accumulated_tokens = 0
    candidate_index: int | None = None
    for index in range(len(rows) - 1, -1, -1):
        _entry_id, message = rows[index]
        accumulated_tokens += estimate_message_tokens(message)
        if accumulated_tokens >= keep_recent_tokens:
            candidate_index = index
            break

    if candidate_index is None:
        return 0

    candidate_message = rows[candidate_index][1]
    if _message_role(candidate_message) == "user":
        if candidate_index > 0:
            return candidate_index
        next_user_index = _next_user_message_index(rows, start=1)
        return next_user_index if next_user_index is not None else 0

    next_user_index = _next_user_message_index(rows, start=candidate_index + 1)
    if next_user_index is not None:
        return next_user_index

    for index in range(candidate_index, len(rows)):
        if _message_role(rows[index][1]) != "tool":
            return index
    return len(rows)


def _next_user_message_index(
    rows: tuple[tuple[str, Any], ...],
    *,
    start: int,
) -> int | None:
    for index in range(start, len(rows)):
        if _message_role(rows[index][1]) == "user":
            return index
    return None


def _last_user_message_index(
    rows: tuple[tuple[str, Any], ...],
    *,
    end: int,
) -> int | None:
    """Return the newest user message index before ``end``, if any."""
    for index in range(end - 1, -1, -1):
        if _message_role(rows[index][1]) == "user":
            return index
    return None


def _last_compaction_details(state: SessionState) -> dict[str, list[str]] | None:
    """Return the latest compaction entry's durable file details, if any."""
    if not state.compaction_entries:
        return None
    return state.compaction_entries[-1].details


def _is_context_overflow_error(event: ErrorEvent) -> bool:
    text = event.message
    if event.data is not None:
        text = f"{text} {event.data}"
    return classify_model_error(RuntimeError(text)).kind == "overflow"
