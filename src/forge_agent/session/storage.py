"""Storage protocol for session entries.

Concrete persistence belongs to :mod:`forge_coding.sessions.storage`; the
agent package only describes the callback surface it consumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from forge_agent.session.entries import SessionEntry


class SessionStorage(Protocol):
    """Append-compatible session storage supplied by the coding layer."""

    async def commit(self, entries: Sequence[SessionEntry]) -> int | None:
        """Atomically commit one or more session entries."""
        ...

    async def append(self, entry: SessionEntry) -> int | None:
        """Commit one entry through the atomic path."""
        ...

    async def read_all(self) -> list[SessionEntry]:
        """Read all entries in storage order."""
        ...
