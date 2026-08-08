"""Session storage protocols and JSONL implementation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from forge_agent.session.entries import SessionEntry
from forge_agent.session.jsonl import (
    entries_from_json_lines,
    entry_to_json_line,
    is_torn_tail_line,
)


class SessionStorage(Protocol):
    """Append-only session storage interface."""

    async def append(self, entry: SessionEntry) -> None:
        """Append one entry to storage."""
        ...

    async def read_all(self) -> list[SessionEntry]:
        """Read all entries in storage order."""
        ...


class JsonlSessionStorage:
    """Local append-only JSONL session storage.

    A file whose last line is an incomplete JSON fragment (a torn tail, e.g.
    from a crash mid-write) is tolerated on read: the fragment is dropped and
    every complete record before it is returned.  The next append truncates
    the torn tail under the file lock before writing, so the damage never
    spreads.  Middle corruption and complete-but-invalid lines still raise
    :class:`forge_agent.session.jsonl.SessionJsonlError`.

    All tail inspection and truncation is byte-based: a crash can split a
    multi-byte UTF-8 character, and a character-indexed truncation would
    corrupt the previous record.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def append(self, entry: SessionEntry) -> None:
        """Append one entry, creating parent directories if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            repair_torn_tail(self.path)
            # Binary append: text-mode newline translation on Windows would
            # turn the record into \r\n and break byte-based tail math.
            with self.path.open("ab") as file:
                file.write(entry_to_json_line(entry).encode("utf-8"))

    async def read_all(self) -> list[SessionEntry]:
        """Read all entries in file order. Missing files are empty sessions."""
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw.endswith(b"\n"):
            return entries_from_json_lines(raw.decode("utf-8").splitlines())
        # No trailing newline: the final line may be a torn fragment whose
        # bytes do not even form valid UTF-8.  Drop it only when it is not a
        # complete JSON line; every earlier line stays strict.
        lines = raw.split(b"\n")
        tail = lines[-1]
        if tail and not _line_is_complete_utf8(tail):
            lines.pop()
        return entries_from_json_lines(b"\n".join(lines).decode("utf-8").splitlines())


def _line_is_complete_utf8(line: bytes) -> bool:
    """Return whether ``line`` is a complete JSON line in valid UTF-8."""
    try:
        decoded = line.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return not is_torn_tail_line(decoded)


def repair_torn_tail(path: Path) -> None:
    """Drop or separate a trailing incomplete JSON fragment in place.

    A file that ends with a newline is untouched.  A final line that parses as
    complete JSON but lacks the trailing newline is kept and a newline is
    appended so the next row cannot concatenate onto it.  A final line that is
    an incomplete JSON fragment (or invalid UTF-8) is truncated back to the
    previous record.  Truncation uses the byte offset of the last newline so a
    fragment that splits a multi-byte UTF-8 character never corrupts the
    previous record.
    """

    if not path.exists():
        return
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return
    last_newline = raw.rfind(b"\n")
    tail = raw if last_newline == -1 else raw[last_newline + 1 :]
    if not tail.strip():
        return
    if _line_is_complete_utf8(tail):
        with path.open("ab") as file:
            file.write(b"\n")
        return
    with path.open("r+b") as file:
        file.truncate(last_newline + 1)
