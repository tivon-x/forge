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
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def append(self, entry: SessionEntry) -> None:
        """Append one entry, creating parent directories if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            repair_torn_tail(self.path)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(entry_to_json_line(entry))

    async def read_all(self) -> list[SessionEntry]:
        """Read all entries in file order. Missing files are empty sessions."""
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8")
        return entries_from_json_lines(
            raw.splitlines(),
            tolerate_torn_tail=not raw.endswith("\n"),
        )


def repair_torn_tail(path: Path) -> None:
    """Drop or separate a trailing incomplete JSON fragment in place.

    A file that ends with a newline is untouched.  A final line that parses as
    complete JSON but lacks the trailing newline is kept and a newline is
    appended so the next row cannot concatenate onto it.  A final line that is
    an incomplete JSON fragment is truncated back to the previous record.
    """

    if not path.exists():
        return
    raw = path.read_text(encoding="utf-8")
    if raw.endswith("\n"):
        return
    lines = raw.splitlines()
    if not lines or not lines[-1].strip():
        return
    if not is_torn_tail_line(lines[-1]):
        with path.open("a", encoding="utf-8") as file:
            file.write("\n")
        return
    last_newline = raw.rfind("\n")
    with path.open("r+", encoding="utf-8") as file:
        file.truncate(last_newline + 1)
