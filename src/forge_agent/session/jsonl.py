"""JSONL serialization helpers for session entries."""

from __future__ import annotations

from json import JSONDecodeError, dumps, loads

from pydantic import TypeAdapter, ValidationError

from forge_agent.message_codec import (
    message_from_json,
    message_to_json,
    project_message_artifact,
)
from forge_agent.session.entries import MessageEntry, SessionEntry

_SESSION_ENTRY_ADAPTER: TypeAdapter[SessionEntry] = TypeAdapter(SessionEntry)


class SessionJsonlError(ValueError):
    """Raised when a session JSONL line cannot be decoded."""


def entry_to_json_line(entry: SessionEntry) -> str:
    """Serialize one session entry as a JSONL line.

    The message artifact is projected before the entry itself is dumped so an
    arbitrary third-party artifact can never abort persistence.
    """
    if isinstance(entry, MessageEntry):
        entry = entry.model_copy(update={"message": project_message_artifact(entry.message)})
    data = entry.model_dump(mode="json")
    if isinstance(entry, MessageEntry):
        data["message"] = message_to_json(entry.message)
    return dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"


def entry_from_json_line(line: str, *, line_number: int | None = None) -> SessionEntry:
    """Deserialize one JSONL line into a typed session entry."""
    try:
        raw = loads(line)
    except JSONDecodeError as exc:
        location = f" on line {line_number}" if line_number is not None else ""
        raise SessionJsonlError(f"Invalid session entry{location}: {exc}") from exc
    try:
        if isinstance(raw, dict) and raw.get("type") == "message":
            raw["message"] = message_from_json(raw.get("message"))
        return _SESSION_ENTRY_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        location = f" on line {line_number}" if line_number is not None else ""
        raise SessionJsonlError(f"Invalid session entry{location}: {exc}") from exc


def is_torn_tail_line(line: str) -> bool:
    """Return whether ``line`` is a non-empty JSON fragment.

    Only a fragment that does not parse as a complete JSON document counts as
    a torn tail.  A line that parses but fails entry validation is *not* torn;
    it stays an error so schema mistakes are never silently dropped.
    """

    if not line.strip():
        return False
    try:
        loads(line)
    except JSONDecodeError:
        return True
    return False


def entries_from_json_lines(
    lines: list[str],
    *,
    tolerate_torn_tail: bool = False,
) -> list[SessionEntry]:
    """Deserialize non-empty JSONL lines in order.

    When ``tolerate_torn_tail`` is set (the file does not end with a newline),
    a trailing line that is an incomplete JSON fragment is dropped; every
    earlier line, a complete-but-invalid last line, and any middle corruption
    still raise :class:`SessionJsonlError`.
    """
    parse_lines = list(lines)
    if tolerate_torn_tail and parse_lines and is_torn_tail_line(parse_lines[-1]):
        parse_lines.pop()
    entries: list[SessionEntry] = []
    for index, line in enumerate(parse_lines, start=1):
        if not line.strip():
            continue
        entries.append(entry_from_json_line(line, line_number=index))
    return entries
