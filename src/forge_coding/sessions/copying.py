"""Atomic copying of one active session branch into a new session."""

from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from tempfile import mkstemp

from forge_agent.session import (
    CustomEntry,
    JsonlSessionStorage,
    LeafEntry,
    SessionEntry,
    SessionInfoEntry,
    SessionJsonlError,
    SessionTreeError,
    entry_to_json_line,
    path_to_entry,
)
from forge_coding.sessions.manager import CodingSessionRecord

SESSION_ORIGIN_NAMESPACE = "forge.session_origin.v1"


class SessionCopyError(ValueError):
    """Raised when an active session branch cannot be copied safely."""


async def copy_active_branch(
    source: CodingSessionRecord,
    leaf_id: str,
    destination: CodingSessionRecord,
    *,
    validated_entries: Sequence[SessionEntry] | None = None,
) -> CodingSessionRecord:
    """Copy one root-to-leaf path into a new, atomically-created session file.

    The source is never opened for writing.  Existing entry ids and parent
    links are retained, except that the old session-info root is replaced by a
    new one and children of that root point at the replacement.
    """

    leaf_id = leaf_id.strip()
    if not leaf_id:
        raise SessionCopyError("Cannot copy a session without a leaf id")

    source_path = source.path.resolve(strict=False)
    destination_path = destination.path.resolve(strict=False)
    if source_path == destination_path:
        raise SessionCopyError("Source and destination session paths must differ")
    if destination.path.exists() or destination.path.is_symlink():
        raise SessionCopyError("Destination session already exists")
    if validated_entries is None and not source.path.exists():
        raise SessionCopyError("Source session does not exist")

    if validated_entries is None:
        try:
            entries = await JsonlSessionStorage(source.path).read_all()
        except (OSError, SessionJsonlError) as exc:
            raise SessionCopyError("Cannot read source session") from exc
    else:
        # Import validation already consumed the source.  Reusing that immutable
        # tuple avoids a second read and the resulting TOCTOU window.
        entries = list(validated_entries)

    try:
        active_path = path_to_entry(entries, leaf_id)
    except SessionTreeError as exc:
        raise SessionCopyError("Cannot copy the active session branch") from exc

    session_info_ids = {
        entry.id for entry in active_path if isinstance(entry, SessionInfoEntry)
    }
    if not session_info_ids:
        raise SessionCopyError("Source active branch has no session info entry")

    new_info = SessionInfoEntry(
        timestamp=destination.created_at,
        created_at=destination.created_at,
        cwd=str(destination.cwd),
        title=destination.title,
    )
    origin = CustomEntry(
        parent_id=new_info.id,
        namespace=SESSION_ORIGIN_NAMESPACE,
        data={"source_session_id": source.id, "source_leaf_id": leaf_id},
    )
    copied_entries: list[SessionEntry] = [new_info, origin]
    for entry in active_path:
        if isinstance(entry, (SessionInfoEntry, LeafEntry)):
            continue
        parent_id = origin.id if entry.parent_id in session_info_ids else entry.parent_id
        copied_entries.append(entry.model_copy(update={"parent_id": parent_id}))

    target_id = copied_entries[-1].id
    copied_entries.append(LeafEntry(parent_id=target_id, entry_id=target_id))

    payload = "".join(entry_to_json_line(entry) for entry in copied_entries).encode("utf-8")
    _atomic_create(destination.path, payload)
    return destination


def _atomic_create(path: Path, payload: bytes) -> None:
    """Create ``path`` without replacing an existing file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SessionCopyError("Cannot create the destination directory") from exc

    descriptor, temporary_name = mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            # A hard link is an atomic create primitive that refuses to replace
            # a concurrently-created destination on both POSIX and Windows.
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise SessionCopyError("Destination session already exists") from exc
        except OSError as exc:
            raise SessionCopyError("Cannot create the destination session") from exc
    except OSError as exc:
        raise SessionCopyError("Cannot write the destination session") from exc
    finally:
        with suppress(OSError):
            temporary_path.unlink()


__all__ = ["SESSION_ORIGIN_NAMESPACE", "SessionCopyError", "copy_active_branch"]
