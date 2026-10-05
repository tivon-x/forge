"""Coding-owned session persistence and legacy JSONL compatibility."""

from __future__ import annotations

import asyncio
import json
from _thread import LockType
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

from forge_agent.session.entries import SessionEntry
from forge_agent.session.jsonl import (
    entries_from_json_lines,
    entry_to_json_line,
    is_torn_tail_line,
)
from forge_coding.sessions.journal import (
    Journal,
    JournalPoisonedError,
)

if TYPE_CHECKING:
    from forge_coding.sessions.manager import SessionManager


_PATH_LOCKS_GUARD = Lock()
# ponytail: process-local locks match Forge's one-process storage ownership;
# add OS-level locking if multi-process writers become supported.
_PATH_LOCKS: dict[Path, LockType] = {}


class SessionStorageError(RuntimeError):
    """Raised when a durable session cannot accept a write."""


class LegacyJsonlSessionStorage:
    """Reader/writer retained for legacy fixtures and explicit exports.

    Production sessions use :class:`JsonlSessionStorage`; this class never
    participates in the durable migration path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def commit(self, entries: Sequence[SessionEntry]) -> int:
        async with self._lock:
            write_task = asyncio.create_task(asyncio.to_thread(self._commit_sync, list(entries)))
            return await _await_thread(write_task)

    async def append(self, entry: SessionEntry) -> int:
        return await self.commit([entry])

    def _commit_sync(self, entries: Sequence[SessionEntry]) -> int:
        with _path_lock(self.path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            repair_torn_tail(self.path)
            with self.path.open("ab") as file:
                for entry in entries:
                    file.write(entry_to_json_line(entry).encode("utf-8"))
        return len(entries)

    async def read_all(self) -> list[SessionEntry]:
        async with self._lock:
            read_task = asyncio.create_task(asyncio.to_thread(self._read_all_sync))
            return await _await_thread(read_task)

    def _read_all_sync(self) -> list[SessionEntry]:
        with _path_lock(self.path):
            if not self.path.exists():
                return []
            raw = self.path.read_bytes()
            if raw.endswith(b"\n"):
                return entries_from_json_lines(raw.decode("utf-8").splitlines())
            lines = raw.split(b"\n")
            if lines[-1] and not _line_is_complete_utf8(lines[-1]):
                lines.pop()
            return entries_from_json_lines(b"\n".join(lines).decode("utf-8").splitlines())


class JsonlSessionStorage:
    """SessionStorage adapter backed by the coding-layer atomic journal."""

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        source = Path(path)
        self._source_path = source
        self._directory = _durable_directory(source)
        self.path = _main_path(source) if _is_durable_path(source) else source
        self._fsync = fsync
        self._journal: Journal | None = None
        self._journal_stamp: tuple[int, int] | None = None
        self._lock = asyncio.Lock()
        self._admission_guard = Lock()
        self._pending_commits = 0
        self._commits_drained = asyncio.Event()
        self._commits_drained.set()
        self._closed = False
        self._poisoned = False
        self._manager: SessionManager | None = None
        self._session_id: str | None = None
        self._migration_pending = False

    @property
    def source_path(self) -> Path:
        """Return the original legacy path, when this storage was migrated."""
        return self._source_path

    @property
    def export_path(self) -> Path:
        """Return a stable filename stem for user-facing exports."""
        if self._source_path.name == "main.jsonl":
            return self._source_path.parent.with_suffix(".jsonl")
        if self._source_path.suffix == ".jsonl":
            return self._source_path
        return self._source_path.with_suffix(".jsonl")

    @property
    def poisoned(self) -> bool:
        """Whether a write outcome is uncertain and this instance is unusable."""
        return self._poisoned or (self._journal is not None and self._journal.poisoned)

    def bind_manager(self, manager: SessionManager | None, session_id: str | None) -> None:
        """Bind the index switch callback used after first durable commit."""
        self._manager = manager
        self._session_id = session_id

    async def commit(self, entries: Sequence[SessionEntry]) -> int:
        return await self.commit_records((), entries)

    async def commit_records(
        self, writes: Sequence[dict[str, Any]], entries: Sequence[SessionEntry] = ()
    ) -> int:
        self._admit_commit()
        task = asyncio.create_task(self._commit_admitted(entries, writes))
        try:
            return await _await_thread(task)
        except asyncio.CancelledError:
            self._poisoned = True
            raise

    async def _commit_admitted(
        self, entries: Sequence[SessionEntry], records: Sequence[dict[str, Any]] = ()
    ) -> int:
        try:
            async with self._lock:
                self._assert_commit_usable()

                async def write() -> int:
                    await self._refresh_journal()
                    legacy = await self._legacy_entries_for_migration()
                    journal = await self._ensure_journal()
                    payload = [*legacy, *entries]
                    writes = [*[_entry_write(entry) for entry in payload], *records]
                    try:
                        sequence = await journal.commit(writes)
                    except JournalPoisonedError:
                        self._poisoned = True
                        raise
                    except asyncio.CancelledError:
                        # Journal.commit shields the worker thread until it settles;
                        # cancellation therefore leaves publication uncertain.
                        self._poisoned = True
                        raise
                    try:
                        self._journal_stamp = self._main_stamp()
                    except OSError:
                        self._poisoned = True
                        raise
                    self.path = _main_path(self._directory)
                    if self._migration_pending or legacy:
                        try:
                            await self._relocate_after_migration()
                        except BaseException:
                            self._poisoned = True
                            raise
                        self._migration_pending = False
                    return sequence

                return await _run_path_locked(_storage_lock_path(self._directory), write)
        finally:
            self._release_commit()

    async def append(self, entry: SessionEntry) -> int:
        return await self.commit([entry])

    async def read_records(self) -> dict[str, Any]:
        """Read scheduling tables through the same storage line as transcript writes."""
        async with self._lock:
            self._assert_usable()

            async def read() -> dict[str, Any]:
                await self._refresh_journal()
                if not self._durable_exists():
                    return {
                        "tasks": [],
                        "submissions": [],
                        "conversations": [],
                        "documents": [],
                        "ids": [],
                        "sequence": 0,
                    }
                journal = await self._ensure_journal()
                return await journal.read_records()

            return await _run_path_locked(_storage_lock_path(self._directory), read)

    async def document(self, document_id: int, at: int | str = "current") -> dict[str, Any] | None:
        """Read confirmed document history through the same storage boundary."""
        async with self._lock:
            self._assert_usable()

            async def read() -> dict[str, Any] | None:
                await self._refresh_journal()
                if not self._durable_exists():
                    return None
                journal = await self._ensure_journal()
                if at != "current" and type(at) is not int:
                    raise ValueError("document point must be current or an integer")
                return await journal.document(document_id, cast(Any, at))

            return await _run_path_locked(_storage_lock_path(self._directory), read)

    async def read_all(self) -> list[SessionEntry]:
        async with self._lock:
            self._assert_usable()

            async def read() -> list[SessionEntry]:
                await self._refresh_journal()
                if self._durable_exists():
                    journal = await self._ensure_journal()
                    durable = await journal.read_all()
                    if _is_durable_path(self._source_path) or not self._source_path.is_file():
                        return durable
                    legacy = await LegacyJsonlSessionStorage(self._source_path).read_all()
                    if not durable:
                        return legacy
                    if len(durable) <= len(legacy) and legacy[: len(durable)] == durable:
                        self._migration_pending = True
                        return legacy
                    if durable[: len(legacy)] == legacy:
                        return durable
                    raise SessionStorageError(
                        "Durable migration target does not match legacy history"
                    )
                if self._source_path.is_file():
                    return await LegacyJsonlSessionStorage(self._source_path).read_all()
                return []

            return await _run_path_locked(_storage_lock_path(self._directory), read)

    async def close(self) -> None:
        with self._admission_guard:
            self._closed = True
        await self._commits_drained.wait()
        async with self._lock:

            async def close_journal() -> None:
                if self._journal is not None:
                    await self._journal.close()

            await _run_path_locked(_storage_lock_path(self._directory), close_journal)

    def _admit_commit(self) -> None:
        with self._admission_guard:
            if self._closed:
                raise SessionStorageError("Session storage is closed")
            if self.poisoned:
                raise SessionStorageError("Session storage is poisoned and must be reopened")
            self._pending_commits += 1
            self._commits_drained.clear()

    def _release_commit(self) -> None:
        with self._admission_guard:
            self._pending_commits -= 1
            if self._pending_commits == 0:
                self._commits_drained.set()

    async def _copy_to(self, destination: Path) -> None:
        async with self._lock:
            self._assert_usable()

            async def copy() -> None:
                await self._refresh_journal()
                journal = await self._ensure_journal()
                await journal.copy_to(destination)

            await _run_path_locked(_storage_lock_path(self._directory), copy)

    async def _refresh_journal(self) -> None:
        if self._journal is not None and self._main_stamp() != self._journal_stamp:
            await self._journal.close()
            self._journal = None

    def _main_stamp(self) -> tuple[int, int] | None:
        try:
            stat = _main_path(self._directory).stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    async def _legacy_entries_for_migration(self) -> list[SessionEntry]:
        if _is_durable_path(self._source_path) or not self._source_path.is_file():
            return []
        entries = await LegacyJsonlSessionStorage(self._source_path).read_all()
        if self._durable_exists():
            existing = await self._read_durable_if_present()
            if not existing:
                self._migration_pending = True
                return entries
            if len(existing) <= len(entries) and entries[: len(existing)] == existing:
                self._migration_pending = True
                return entries[len(existing) :]
            if existing[: len(entries)] != entries:
                raise SessionStorageError("Durable migration target does not match legacy history")
            self._migration_pending = True
            return []
        self._migration_pending = True
        return entries

    async def _read_durable_if_present(self) -> list[SessionEntry]:
        journal = await self._ensure_journal()
        return await journal.read_all()

    async def _ensure_journal(self) -> Journal:
        if self._journal is not None:
            return self._journal
        self._journal = await Journal.open(self._directory, fsync=self._fsync)
        self._journal_stamp = self._main_stamp()
        return self._journal

    async def _relocate_after_migration(self) -> None:
        if self._manager is None or self._session_id is None:
            return
        self._manager.relocate_session(self._session_id, self.path)

    def _durable_exists(self) -> bool:
        return _main_path(self._directory).is_file()

    def _assert_open(self) -> None:
        if self._closed:
            raise SessionStorageError("Session storage is closed")

    def _assert_usable(self) -> None:
        self._assert_open()
        if self.poisoned:
            raise SessionStorageError("Session storage is poisoned and must be reopened")

    def _assert_commit_usable(self) -> None:
        if self.poisoned:
            raise SessionStorageError("Session storage is poisoned and must be reopened")


def _entry_write(entry: SessionEntry) -> dict[str, object]:
    value = json.loads(entry_to_json_line(entry))
    return {"type": "entry", "value": value}


async def import_legacy(source: str | Path, destination: str | Path) -> Path:
    """Import a legacy JSONL file into a new journal directory without replacing it."""
    source_path = Path(source)
    destination_path = Path(destination)
    entries = await LegacyJsonlSessionStorage(source_path).read_all()
    journal = JsonlSessionStorage(destination_path)
    try:
        if destination_path.exists() and not destination_path.is_dir():
            raise SessionStorageError("Migration destination is not a durable directory")
        existing = await journal.read_all()
        confirmed_main = _main_path(journal._directory).is_file()
        if existing:
            if len(existing) > len(entries) or entries[: len(existing)] != existing:
                raise SessionStorageError("Existing migration target differs from legacy source")
            missing = entries[len(existing) :]
        else:
            missing = entries
        if missing:
            await journal.commit(missing)
        elif not confirmed_main:
            await journal.commit([])
        else:
            destination_path.mkdir(parents=True, exist_ok=True)
        return _main_path(journal._directory)
    finally:
        await journal.close()


async def export_legacy(source: str | Path, destination: str | Path) -> Path:
    """Export only the legacy entry projection, without recovery metadata."""
    storage = JsonlSessionStorage(source)
    entries = await storage.read_all()
    output = Path(destination)
    try:
        validate_export_destination(output, storage.source_path, storage.path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(entry_to_json_line(entry) for entry in entries), encoding="utf-8")
        return output
    finally:
        await storage.close()


def validate_export_destination(destination: Path, *sources: Path) -> None:
    """Reject exports into source storage, including symlink/hardlink aliases."""
    output = destination.resolve(strict=False)
    protected: set[Path] = set()
    for source in sources:
        protected.add(source)
        directory = _durable_directory(source)
        if _is_durable_path(source) or _main_path(directory).is_file():
            if output.is_relative_to(directory.resolve(strict=False)):
                raise SessionStorageError("Export cannot overwrite source storage")
            protected.update(path for path in directory.iterdir() if path.is_file())
            legacy = directory.with_name(directory.name + ".jsonl")
            if legacy.is_file():
                protected.add(legacy)
    for path in protected:
        if output == path.resolve(strict=False) or (
            destination.exists() and path.is_file() and destination.samefile(path)
        ):
            raise SessionStorageError("Export cannot overwrite source storage")


async def copy_session_storage(
    source: str | Path | JsonlSessionStorage,
    destination: str | Path,
    *,
    fsync: bool = False,
) -> Path:
    """Copy every confirmed storage file, preserving the journal directory layout.

    A live storage object remains open and keeps its existing fsync and poison
    state.  Path sources use a temporary quiescent adapter and close it after
    the snapshot; ``fsync`` applies to that adapter.
    """
    destination_dir = Path(destination)
    if isinstance(source, JsonlSessionStorage):
        storage = source
        owns_storage = False
    else:
        source_path = Path(source)
        if not _is_durable_path(source_path):
            raise SessionStorageError("Full storage copy requires a durable session directory")
        storage = JsonlSessionStorage(source_path, fsync=fsync)
        owns_storage = True
    try:
        if not storage._durable_exists():
            raise SessionStorageError("Full storage copy requires confirmed durable storage")
        destination_dir.parent.mkdir(parents=True, exist_ok=True)
        await storage._copy_to(destination_dir)
    finally:
        if owns_storage:
            await storage.close()
    return destination_dir


def _durable_directory(path: Path) -> Path:
    if _is_durable_path(path):
        return path.parent if path.name == "main.jsonl" else path
    return path.with_suffix("") if path.suffix == ".jsonl" else path


def _is_durable_path(path: Path) -> bool:
    return path.is_dir() or path.name == "main.jsonl"


def _main_path(directory: Path) -> Path:
    return directory if directory.name == "main.jsonl" else directory / "main.jsonl"


def _storage_lock_path(directory: Path) -> Path:
    return directory.parent if directory.is_file() else directory


async def _await_thread[T](task: asyncio.Task[T]) -> T:
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _run_path_locked[T](path: Path, operation: Callable[[], Awaitable[T]]) -> T:
    """Serialize all owners of one durable path across event loops."""
    lock = _path_lock(path)
    # No detached lock-acquiring thread may outlive a cancelled reader or
    # event loop: it could acquire the lock after its release path is gone.
    while not lock.acquire(blocking=False):
        await asyncio.sleep(0.001)
    try:
        return await operation()
    finally:
        lock.release()


def _path_lock(path: Path) -> LockType:
    resolved = path.resolve(strict=False)
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, Lock())


def _line_is_complete_utf8(line: bytes) -> bool:
    try:
        decoded = line.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return not is_torn_tail_line(decoded)


def repair_torn_tail(path: Path) -> None:
    if not path.exists() or path.read_bytes().endswith(b"\n"):
        return
    raw = path.read_bytes()
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


__all__ = [
    "JsonlSessionStorage",
    "LegacyJsonlSessionStorage",
    "SessionStorageError",
    "copy_session_storage",
    "export_legacy",
    "import_legacy",
    "repair_torn_tail",
]
