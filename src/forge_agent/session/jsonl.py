"""JSONL serialization helpers for session entries."""

from __future__ import annotations

from json import dumps, loads

from pydantic import TypeAdapter, ValidationError

from forge_agent.message_codec import message_from_json, message_to_json
from forge_agent.session.entries import MessageEntry, SessionEntry

_SESSION_ENTRY_ADAPTER: TypeAdapter[SessionEntry] = TypeAdapter(SessionEntry)


class SessionJsonlError(ValueError):
    """Raised when a session JSONL line cannot be decoded."""


def entry_to_json_line(entry: SessionEntry) -> str:
    """Serialize one session entry as a JSONL line."""
    data = entry.model_dump(mode="json")
    if isinstance(entry, MessageEntry):
        data["message"] = message_to_json(entry.message)
    return dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"


def entry_from_json_line(line: str, *, line_number: int | None = None) -> SessionEntry:
    """Deserialize one JSONL line into a typed session entry."""
    try:
        raw = loads(line)
        if isinstance(raw, dict) and raw.get("type") == "message":
            raw["message"] = message_from_json(raw.get("message"))
        return _SESSION_ENTRY_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        location = f" on line {line_number}" if line_number is not None else ""
        raise SessionJsonlError(f"Invalid session entry{location}: {exc}") from exc


def entries_from_json_lines(lines: list[str]) -> list[SessionEntry]:
    """Deserialize non-empty JSONL lines in order."""
    entries: list[SessionEntry] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        entries.append(entry_from_json_line(line, line_number=index))
    return entries
