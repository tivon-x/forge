"""Process-local FIFO serialization for same-file operations."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class _Waiter:
    """A loop-independent lease hand-off signal."""

    event: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False
    granted: bool = False


@dataclass(slots=True)
class _PathState:
    owner: bool = False
    waiters: deque[_Waiter] = field(default_factory=deque)


class FileOperationQueue:
    """Serialize operations by resolved path across event loops in one process.

    The queue intentionally does not coordinate independent processes or shell
    commands.  A small thread lock and ``threading.Event`` hand-offs keep the
    same path exclusive when sessions run in different event loops/threads;
    the async side waits without blocking its event loop.
    """

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._states: dict[Path, _PathState] = {}

    @property
    def path_state_count(self) -> int:
        """Return the number of active path states across all event loops."""
        with self._state_lock:
            return len(self._states)

    @asynccontextmanager
    async def operation(self, path: Path) -> AsyncIterator[None]:
        """Acquire the FIFO lease for ``path`` until the operation completes."""
        resolved = path.resolve()
        with self._state_lock:
            state = self._states.setdefault(resolved, _PathState())
            if state.owner:
                waiter = _Waiter()
                state.waiters.append(waiter)
            else:
                waiter = None
                state.owner = True

        if waiter is not None:
            try:
                # ``threading.Event`` is deliberately loop-neutral.  The
                # blocking wait runs in the loop's executor instead of
                # blocking the event loop that owns this operation.
                await asyncio.to_thread(waiter.event.wait)
            except BaseException:  # noqa: BLE001 - every failed wait must release its queue state
                with self._state_lock:
                    if waiter.granted:
                        # Release a lease that was handed off just as the
                        # waiter was cancelled, otherwise the next waiter
                        # would remain blocked forever.
                        self._release_locked(resolved, state)
                    else:
                        waiter.cancelled = True
                        with suppress(ValueError):
                            state.waiters.remove(waiter)
                        # Wake the executor thread left behind by to_thread.
                        waiter.event.set()
                        self._cleanup_locked(resolved, state)
                raise

        try:
            yield
        finally:
            with self._state_lock:
                self._release_locked(resolved, state)

    def _release_locked(self, path: Path, state: _PathState) -> None:
        """Hand the lease to the next live waiter; caller holds the lock."""
        while state.waiters:
            waiter = state.waiters.popleft()
            if waiter.cancelled:
                waiter.event.set()
                continue
            # Keep owner=True while transferring the lease. The awakened task
            # enters its critical section without creating a second owner.
            waiter.granted = True
            waiter.event.set()
            return
        state.owner = False
        self._cleanup_locked(path, state)

    def _cleanup_locked(self, path: Path, state: _PathState) -> None:
        """Remove an idle path state; caller holds the lock."""
        if not state.owner and not state.waiters and self._states.get(path) is state:
            self._states.pop(path, None)


file_operation_queue = FileOperationQueue()


__all__ = ["FileOperationQueue", "file_operation_queue"]
