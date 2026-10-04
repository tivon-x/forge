"""Read and validate session JSONL files before an import is committed."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from forge_agent.session import (
    LeafEntry,
    SessionEntry,
    SessionInfoEntry,
    SessionJsonlError,
    SessionTreeError,
    entries_by_id,
    entry_from_json_line,
    entry_to_json_line,
    path_to_entry,
)
from forge_agent.session.jsonl import is_torn_tail_line
from forge_coding.resources import ForgeResourcePaths
from forge_coding.resources.trust import TrustError, TrustResult, TrustStore, resolve_project_trust
from forge_coding.sessions.manager import CodingSessionRecord, SessionManager
from forge_coding.sessions.storage import JsonlSessionStorage, SessionStorageError


class SessionImportError(ValueError):
    """Raised when an imported session is incomplete or malformed."""


class SessionImportTrustRequired(SessionImportError):
    """Raised when an import needs an interactive trust decision."""


class SessionImportDenied(SessionImportError):
    """Raised when the user cancels or explicitly denies an import."""


@dataclass(frozen=True, slots=True)
class SessionImport:
    """Validated source data for a later atomic session import."""

    source_path: Path
    entries: tuple[SessionEntry, ...]
    session_info: SessionInfoEntry
    cwd: Path
    active_leaf: LeafEntry | None

    @property
    def source_leaf_id(self) -> str | None:
        """Return the source active leaf id, when the file records one."""
        return self.active_leaf.entry_id if self.active_leaf is not None else None


@dataclass(frozen=True, slots=True)
class PreparedSessionImport:
    """Validated import data and its target-project trust preflight."""

    parsed: SessionImport
    trust_result: TrustResult
    trust_store: TrustStore
    resource_paths: ForgeResourcePaths
    cli_override: str | None = None
    env: Mapping[str, str] | None = None

    @property
    def cwd(self) -> Path:
        """Return the validated imported working directory."""
        return self.parsed.cwd

    @property
    def entries(self) -> tuple[SessionEntry, ...]:
        """Return the immutable entries validated during preparation."""
        return self.parsed.entries

    @property
    def trust(self) -> TrustResult:
        """Compatibility alias for callers that name the preflight ``trust``."""
        return self.trust_result

    @property
    def trust_required(self) -> bool:
        """Return whether a caller must ask before writing a destination."""
        return (
            self.trust_result.project_resources_present
            and not self.trust_result.project_resources_allowed
        )


async def prepare_session_import(
    path: str | Path,
    *,
    paths: ForgeResourcePaths | None = None,
    store: TrustStore | None = None,
    cli_override: str | None = None,
    env: Mapping[str, str] | None = None,
) -> PreparedSessionImport:
    """Parse an import once and preflight trust for its target cwd.

    Parsing is offloaded so a large JSONL file cannot block the TUI event loop.
    No session destination is created here.
    """

    parsed = await asyncio.to_thread(parse_import_session, path)
    trust_store = store or TrustStore()
    resource_paths = paths or ForgeResourcePaths(cwd=parsed.cwd)
    try:
        trust_result = resolve_project_trust(
            parsed.cwd,
            paths=resource_paths,
            store=trust_store,
            cli_override=cli_override,
            env=env,
            interactive=False,
        )
    except TrustError as exc:
        raise SessionImportError("Could not resolve project trust for import") from exc

    # Explicit non-interactive overrides are a hard deny.  The default/store
    # deny is intentionally surfaced as a trust-required result for the TUI.
    if (
        trust_result.project_resources_present
        and not trust_result.project_resources_allowed
        and trust_result.source in {"cli", "environment"}
    ):
        raise SessionImportDenied("Import denied by the active trust policy")
    return PreparedSessionImport(
        parsed,
        trust_result,
        trust_store,
        resource_paths,
        cli_override,
        env,
    )


async def commit_session_import(
    prepared: PreparedSessionImport,
    trust_decision: str | None = None,
    *,
    manager: SessionManager | None = None,
    model: str = "imported",
    provider_name: str | None = None,
    title: str | None = None,
) -> CodingSessionRecord:
    """Atomically commit the complete validated tree into the session manager.

    ``prepared.entries`` is passed directly to the durable storage commit.  This
    is important for imports: validating and then re-reading the source would
    create a TOCTOU window.  Trust denial returns before any destination path
    or index is written.
    """

    decision = _normalize_import_trust_decision(trust_decision)
    if decision == "deny":
        raise SessionImportDenied("Import cancelled; project resources were not trusted")
    _revalidate_import_cwd(prepared.cwd)
    revalidated = _revalidate_import_trust(prepared, decision)
    trust_required = (
        revalidated.project_resources_present and not revalidated.project_resources_allowed
    )
    if trust_required:
        if decision is None:
            raise SessionImportTrustRequired("Import requires a project trust decision")
        if decision in {"always", "parent"}:
            try:
                prepared.trust_store.set(
                    prepared.cwd,
                    "allow",
                    scope="parent" if decision == "parent" else "folder",
                )
            except TrustError as exc:
                raise SessionImportError("Could not save project trust decision") from exc

    session_manager = manager or SessionManager()
    session_id = _import_session_id(prepared)
    existing = session_manager.get_session(session_id)
    if existing is not None:
        storage = JsonlSessionStorage(existing.path)
        try:
            existing_entries = await storage.read_all()
            existing_prefix = prepared.entries[: len(existing_entries)]
            if len(existing_entries) <= len(prepared.entries):
                if list(existing_prefix) != existing_entries:
                    raise SessionImportError(
                        "An import with the same identity has different history"
                    )
                missing = prepared.entries[len(existing_entries) :]
                if missing:
                    await storage.commit(missing)
            elif list(prepared.entries) != existing_entries[: len(prepared.entries)]:
                raise SessionImportError("An import with the same identity has different history")
            return existing
        finally:
            await storage.close()

    destination = session_manager.prepare_session(
        cwd=prepared.cwd,
        model=model,
        provider_name=provider_name,
        title=title if title is not None else prepared.parsed.session_info.title,
        session_id=session_id,
    )

    try:
        storage = JsonlSessionStorage(destination.path)
        try:
            existing_entries = await storage.read_all()
            existing_prefix = prepared.entries[: len(existing_entries)]
            if len(existing_entries) > len(prepared.entries) or list(existing_prefix) != (
                existing_entries
            ):
                raise SessionStorageError("Existing import target differs from validated source")
            missing = prepared.entries[len(existing_entries) :]
            if missing:
                await storage.commit(missing)
        finally:
            await storage.close()
        session_manager.index_session(destination)
        return destination
    except (SessionStorageError, OSError, ValueError) as exc:
        if isinstance(exc, FileExistsError):
            message = "Destination session already exists"
        else:
            message = str(exc) or "Could not commit imported session"
        raise SessionImportError(message) from exc


def _import_session_id(prepared: PreparedSessionImport) -> str:
    """Derive a stable destination identity from the validated import source."""
    digest = sha256()
    digest.update(str(prepared.parsed.source_path.resolve()).encode("utf-8"))
    digest.update(b"\0")
    for entry in prepared.entries:
        digest.update(entry_to_json_line(entry).encode("utf-8"))
    return f"import-{digest.hexdigest()[:16]}"


def _revalidate_import_trust(
    prepared: PreparedSessionImport,
    decision: str | None,
) -> TrustResult:
    """Recheck resource metadata immediately before creating a destination."""

    try:
        result = resolve_project_trust(
            prepared.cwd,
            paths=prepared.resource_paths,
            store=prepared.trust_store,
            cli_override=prepared.cli_override,
            env=prepared.env,
            session_decision="allow" if decision in {"once", "always", "parent"} else None,
            interactive=False,
        )
    except TrustError as exc:
        raise SessionImportError("Could not resolve project trust for import") from exc
    if (
        result.project_resources_present
        and not result.project_resources_allowed
        and result.source in {"cli", "environment"}
    ):
        raise SessionImportDenied("Import denied by the active trust policy")
    return result


def _revalidate_import_cwd(cwd: Path) -> None:
    """Reject a target directory that disappeared or changed after parsing."""
    try:
        current = cwd.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SessionImportError("Imported session cwd no longer exists") from exc
    if not current.is_dir() or current != cwd:
        raise SessionImportError("Imported session cwd changed after validation")


def _normalize_import_trust_decision(value: str | None) -> str | None:
    if value is None:
        return None
    decision = value.strip().casefold()
    if decision not in {"once", "always", "parent", "deny"}:
        raise SessionImportError("Trust decision must be once, always, parent, or deny")
    return decision


def parse_import_session(path: str | Path) -> SessionImport:
    """Stream, decode, and validate a session JSONL import source.

    Complete lines are strict.  A final line without a newline may be an
    incomplete JSON fragment from a torn write and is ignored; malformed lines
    in the middle of the file, and complete-but-invalid records, are rejected.
    This function never writes to the source or to session storage.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise SessionImportError("Session import source is not a file")

    entries: list[SessionEntry | None] = []
    try:
        with source_path.open("rb") as source:
            for line_number, raw_line in enumerate(source, start=1):
                if raw_line.endswith(b"\n"):
                    entries.append(_parse_complete_line(raw_line, line_number))
                    continue

                # Only a final line can lack its newline.  A JSON fragment from
                # an interrupted write is the one recoverable import tail.
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if is_torn_tail_line(line):
                    continue
                entries.append(_parse_line(line, line_number))
    except OSError as exc:
        raise SessionImportError("Cannot read session import source") from exc

    return _validate_import(source_path, entries)


def _parse_complete_line(raw_line: bytes, line_number: int) -> SessionEntry | None:
    try:
        line = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SessionImportError(f"Invalid UTF-8 in session import on line {line_number}") from exc
    if not line.strip():
        return None
    return _parse_line(line, line_number)


def _parse_line(line: str, line_number: int) -> SessionEntry:
    try:
        return entry_from_json_line(line, line_number=line_number)
    except SessionJsonlError as exc:
        raise SessionImportError(f"Invalid session record on line {line_number}") from exc


def _validate_import(source_path: Path, parsed_entries: list[SessionEntry | None]) -> SessionImport:
    entries = [entry for entry in parsed_entries if entry is not None]
    try:
        by_id = entries_by_id(entries)
    except SessionTreeError as exc:
        raise SessionImportError("Imported session contains duplicate entry ids") from exc

    info_entries = [entry for entry in entries if isinstance(entry, SessionInfoEntry)]
    if not info_entries:
        raise SessionImportError("Imported session is missing SessionInfo")
    if len(info_entries) != 1:
        raise SessionImportError("Imported session contains multiple SessionInfo entries")
    session_info = info_entries[0]
    if session_info.parent_id is not None:
        raise SessionImportError("Imported SessionInfo must be the root entry")
    cwd = _validate_cwd(session_info.cwd)

    for entry in entries:
        if entry.parent_id is not None and entry.parent_id not in by_id:
            raise SessionImportError("Imported session has a dangling parent")
        if (
            isinstance(entry, LeafEntry)
            and entry.entry_id is not None
            and entry.entry_id not in by_id
        ):
            raise SessionImportError("Imported session has a dangling leaf target")

    try:
        for entry in entries:
            path_to_entry(entries, entry.id)
    except SessionTreeError as exc:
        raise SessionImportError("Imported session contains a parent cycle") from exc

    active_leaf = next(
        (entry for entry in reversed(entries) if isinstance(entry, LeafEntry)),
        None,
    )
    target_id = active_leaf.entry_id if active_leaf is not None else None
    if target_id is None:
        target_id = next(
            (entry.id for entry in reversed(entries) if not isinstance(entry, LeafEntry)),
            None,
        )
    if target_id is None:
        raise SessionImportError("Imported session has no active entry")
    try:
        active_path = path_to_entry(entries, target_id)
    except SessionTreeError as exc:
        raise SessionImportError("Imported session has an invalid active path") from exc
    if active_path[0].id != session_info.id or any(
        isinstance(entry, LeafEntry) for entry in active_path
    ):
        raise SessionImportError("Imported session has an invalid active path")
    return SessionImport(
        source_path=source_path.resolve(),
        entries=tuple(entries),
        session_info=session_info,
        cwd=cwd,
        active_leaf=active_leaf,
    )


def _validate_cwd(raw_cwd: str | None) -> Path:
    if raw_cwd is None or not raw_cwd.strip():
        raise SessionImportError("Imported SessionInfo is missing cwd")
    candidate = Path(raw_cwd)
    if not candidate.is_absolute():
        raise SessionImportError("Imported SessionInfo cwd must be absolute")
    try:
        cwd = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SessionImportError("Imported SessionInfo cwd does not exist") from exc
    if not cwd.is_dir():
        raise SessionImportError("Imported SessionInfo cwd is not a directory")
    return cwd
