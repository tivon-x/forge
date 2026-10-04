"""Pi-compatible JSONL commit journal for coding sessions.

The journal keeps ``main.jsonl`` as the only commit authority.  Live task and
document payloads are appended to sidecars first and become visible only when
their main marker is durable enough for the configured filesystem contract.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from forge_agent.session import SessionEntry, entry_from_json_line, entry_to_json_line

_FORMAT_VERSION = 1
_MAIN_NAME = "main.jsonl"
_RECLAIM_SUFFIX = ".reclaim"
_SIDECAR_RE = re.compile(r"^(?P<kind>task|doc)-(?P<id>0|[1-9][0-9]*)\.jsonl$")
_RECLAIM_RE = re.compile(r"^(?:task|doc)-(?:0|[1-9][0-9]*)\.jsonl\.reclaim$")
_MAX_SAFE_INTEGER = 9_007_199_254_740_991


class JournalError(RuntimeError):
    """Base class for journal lifecycle and protocol failures."""


class JournalCorruptionError(JournalError, ValueError):
    """Raised when a complete journal record violates the commit protocol."""


class JournalPoisonedError(OSError, JournalError):
    """Raised after an uncertain publication result; reopen the journal."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__("JSONL journal is poisoned and must be reopened")
        self.__cause__ = cause


class JournalClosedError(JournalError):
    """Raised when a journal has stopped admitting operations."""


@dataclass(slots=True)
class _Revision:
    seq: int
    content: dict[str, Any]


@dataclass(slots=True)
class _Document:
    record: dict[str, Any]
    revisions: list[_Revision]


@dataclass(frozen=True, slots=True)
class _CandidateState:
    entries: tuple[dict[str, Any], ...]
    entry_ids: frozenset[str]
    tasks: dict[int, dict[str, Any]]
    documents: dict[int, _Document]
    id_kinds: dict[int, str]


@dataclass(frozen=True, slots=True)
class _PreparedCommit:
    seq: int
    writes: tuple[dict[str, Any], ...]
    marker: bytes
    sidecars: tuple[tuple[str, bytes], ...]
    reclaims: tuple[tuple[str, bytes | None], ...]
    candidate: _CandidateState


def _is_int(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_SAFE_INTEGER


def _require_int(value: object, description: str, *, positive: bool = False) -> int:
    if not _is_int(value) or (positive and cast(int, value) < 1):
        raise JournalCorruptionError(f"Invalid {description}")
    return cast(int, value)


def _json_copy(value: Any, description: str = "JSON value") -> Any:
    """Detach and validate the JSON subset used by durable records."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError(f"{description} contains an unsafe integer")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{description} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_copy(item, description) for item in value]
    if isinstance(value, tuple):
        return [_json_copy(item, description) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{description} contains a non-string object key")
            result[key] = _json_copy(item, description)
        return result
    raise TypeError(f"{description} contains unsupported value {type(value).__name__}")


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _line_json(raw: bytes, description: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JournalCorruptionError(f"Invalid UTF-8 in complete {description}") from exc
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (ValueError, json.JSONDecodeError) as exc:
        raise JournalCorruptionError(f"Malformed complete {description}") from exc
    if not isinstance(value, dict):
        raise JournalCorruptionError(f"Invalid JSON object in {description}")
    return cast(dict[str, Any], value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value} is not allowed")


def _sidecar_name(kind: Literal["task", "doc"], record_id: int) -> str:
    return f"{kind}-{record_id}.jsonl"


def _current_only(record: Mapping[str, Any]) -> bool:
    scope = record.get("scope")
    if not isinstance(scope, Mapping) or scope.get("kind") != "conversation":
        return True
    history = record.get("history", "latest")
    return isinstance(history, str) and history == "latest"


def _document_address_key(record: Mapping[str, Any]) -> str:
    return json.dumps(
        [record.get("kind"), record.get("key"), record.get("scope")],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_document_record(record: Mapping[str, Any], description: str) -> dict[str, Any]:
    value = cast(dict[str, Any], _json_copy(dict(record), description))
    _record_id(value, description)
    kind = value.get("kind")
    if not isinstance(kind, str) or not kind:
        raise JournalCorruptionError(f"Invalid {description} kind")
    scope = value.get("scope")
    if not isinstance(scope, Mapping) or not isinstance(scope.get("kind"), str):
        raise JournalCorruptionError(f"Invalid {description} scope")
    scope_kind = cast(str, scope["kind"])
    if scope_kind not in {"session", "conversation", "task"}:
        raise JournalCorruptionError(f"Invalid {description} scope")
    if scope_kind in {"conversation", "task"}:
        owner_name = "conversationId" if scope_kind == "conversation" else "taskId"
        _require_int(scope.get(owner_name), f"{description} {owner_name}")
    if "key" in value and not isinstance(value["key"], str):
        raise JournalCorruptionError(f"Invalid {description} key")
    if "createdAt" in value or "retiredAt" in value:
        raise JournalCorruptionError(f"{description} cannot contain storage timestamps")
    history = value.get("history")
    if scope_kind != "conversation" and history is not None:
        raise JournalCorruptionError(f"Invalid {description} history")
    if history is not None and history not in {"latest", "rewindable"}:
        raise JournalCorruptionError(f"Invalid {description} history")
    return value


def _record_id(record: Mapping[str, Any], description: str) -> int:
    return _require_int(record.get("id"), f"{description} id")


def _state_status(value: Mapping[str, Any], description: str) -> str:
    state = value.get("state")
    if not isinstance(state, Mapping) or not isinstance(state.get("status"), str):
        raise JournalCorruptionError(f"Invalid {description} state")
    return cast(str, state["status"])


def _validate_content(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JournalCorruptionError(f"Invalid {description}")
    content = cast(dict[str, Any], _json_copy(dict(value), description))
    version = _require_int(content.get("version"), f"{description} version", positive=True)
    content["version"] = version
    kind = content.get("kind")
    if kind == "base":
        if not isinstance(content.get("value"), dict):
            raise JournalCorruptionError(f"Invalid {description} base value")
    elif kind == "delta":
        if not isinstance(content.get("ops"), list):
            raise JournalCorruptionError(f"Invalid {description} delta operations")
    else:
        raise JournalCorruptionError(f"Invalid {description} kind")
    return content


def _path_value(root: Any, path: Sequence[Any]) -> tuple[Any, Any]:
    current = root
    for part in path[:-1]:
        if isinstance(current, dict):
            if not isinstance(part, str):
                raise ValueError("document operation path is invalid")
            current = current[part]
        elif isinstance(current, list):
            if not isinstance(part, int):
                raise ValueError("document operation path is invalid")
            current = current[part]
        else:
            raise ValueError("document operation path is invalid")
    if not path:
        return None, root
    return current, path[-1]


def _apply_local_ops(value: dict[str, Any], ops: Sequence[Any]) -> dict[str, Any]:
    """Apply only the small D1/D3 document operation vocabulary."""

    result = copy.deepcopy(value)
    for raw in ops:
        if isinstance(raw, Mapping):
            op = raw.get("op")
            path = raw.get("path", [])
            operand = raw.get("value")
            start = raw.get("start", 0)
            end = raw.get("end")
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
            op = raw[0]
            path = raw[1] if len(raw) > 1 else []
            operand = raw[2] if len(raw) > 2 else None
            start = raw[2] if op == "trim" and len(raw) > 2 else 0
            end = raw[3] if op == "trim" and len(raw) > 3 else None
        else:
            raise ValueError("document operation must be an object or sequence")
        if (
            not isinstance(op, str)
            or not isinstance(path, Sequence)
            or isinstance(path, (str, bytes))
        ):
            raise ValueError("document operation has invalid op/path")
        parent, key = _path_value(result, list(path))
        if op == "set":
            if parent is None:
                if not isinstance(operand, dict):
                    raise ValueError("root set requires an object")
                result = copy.deepcopy(operand)
            elif isinstance(parent, dict):
                if not isinstance(key, str):
                    raise ValueError("set path is invalid")
                parent[key] = copy.deepcopy(operand)
            elif isinstance(parent, list):
                if not isinstance(key, int):
                    raise ValueError("set path is invalid")
                parent[key] = copy.deepcopy(operand)
            else:
                raise ValueError("set path is invalid")
        elif op == "append":
            target = result if parent is None else parent[key]
            if isinstance(target, list):
                target.append(copy.deepcopy(operand))
            elif isinstance(target, str) and isinstance(operand, str):
                if parent is None:
                    raise ValueError("root string append is invalid")
                parent[key] = target + operand
            else:
                raise ValueError("append path is invalid")
        elif op == "trim":
            target = result if parent is None else parent[key]
            if not isinstance(target, (list, str)) or type(start) is not int:
                raise ValueError("trim path is invalid")
            trimmed = target[start:end]
            if parent is None:
                raise ValueError("root trim is invalid")
            parent[key] = trimmed
        else:
            raise ValueError(f"unsupported document operation: {op}")
    return result


class Journal:
    """One-owner, append-only Pi-style JSONL commit journal.

    A directory has one live Journal owner.  Concurrent commits and close are
    serialized by that instance; callers needing a consistent copy use
    ``copy_to`` on the owner rather than opening a second writer.
    """

    def __init__(self, directory: str | Path, *, fsync: bool = False) -> None:
        self.directory = Path(directory)
        self.main_path = self.directory / _MAIN_NAME
        self.fsync = fsync
        self._lock = asyncio.Lock()
        self._closed = False
        self._closing = False
        self._poison: JournalPoisonedError | None = None
        self._sequence = 0
        self._entries: list[dict[str, Any]] = []
        self._entry_ids: set[str] = set()
        self._tasks: dict[int, dict[str, Any]] = {}
        self._documents: dict[int, _Document] = {}
        self._id_kinds: dict[int, str] = {}
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(cls, directory: str | Path, *, fsync: bool = False) -> Journal:
        journal = cls(directory, fsync=fsync)
        journal.directory.mkdir(parents=True, exist_ok=True)
        task = asyncio.create_task(asyncio.to_thread(journal._recover_sync))
        await _await_thread(task)
        return journal

    async def commit(self, writes: Sequence[dict[str, Any]] | Iterable[Any]) -> int:
        self._assert_admitting()
        task = asyncio.create_task(self._commit_admitted(writes))
        return await _await_thread(task)

    async def _commit_admitted(self, writes: Sequence[dict[str, Any]] | Iterable[Any]) -> int:
        async with self._lock:
            self._assert_healthy()
            task = asyncio.create_task(asyncio.to_thread(self._commit_sync, list(writes)))
            return await _await_thread(task)

    async def append(self, entry: SessionEntry) -> None:
        await self.commit([{"type": "entry", "value": _entry_dict(entry)}])

    async def read_all(self) -> list[SessionEntry]:
        self._assert_usable()
        async with self._lock:
            self._assert_usable()
            return [
                entry_from_json_line(json.dumps(value, ensure_ascii=False))
                for value in self._entries
            ]

    async def entries(self) -> list[dict[str, Any]]:
        self._assert_usable()
        async with self._lock:
            self._assert_usable()
            return copy.deepcopy(self._entries)

    async def task(self, task_id: int) -> dict[str, Any] | None:
        self._assert_usable()
        async with self._lock:
            self._assert_usable()
            value = self._tasks.get(task_id)
            return copy.deepcopy(value) if value is not None else None

    async def document(
        self, document_id: int, at: int | Literal["current"] = "current"
    ) -> dict[str, Any] | None:
        self._assert_usable()
        async with self._lock:
            self._assert_usable()
            document = self._documents.get(document_id)
            if document is None:
                return None
            record = document.record
            created = _created_at(record)
            retired = _retired_at(record)
            if at == "current":
                if retired is not None:
                    return None
                revisions = document.revisions
            else:
                if not _is_int(at):
                    raise ValueError("document point must be current or a safe integer")
                if _current_only(record):
                    raise ValueError(f"Document {document_id} does not retain historical content")
                if created > at or (retired is not None and at >= retired):
                    return None
                revisions = [revision for revision in document.revisions if revision.seq <= at]
            if not revisions:
                return None
            base_index = max(
                (
                    index
                    for index, revision in enumerate(revisions)
                    if revision.content["kind"] == "base"
                ),
                default=-1,
            )
            if base_index < 0:
                raise JournalCorruptionError(f"Document {document_id} is missing a required base")
            base = revisions[base_index]
            value = copy.deepcopy(cast(dict[str, Any], base.content["value"]))
            version = _require_int(
                base.content["version"], f"document {document_id} version", positive=True
            )
            for revision in revisions[base_index + 1 :]:
                content = revision.content
                if content["kind"] != "delta":
                    if content["kind"] != "base":
                        raise JournalCorruptionError(f"Invalid document {document_id} revision")
                    value = copy.deepcopy(cast(dict[str, Any], content["value"]))
                    version = _require_int(
                        content["version"], f"document {document_id} version", positive=True
                    )
                    continue
                if content["version"] != version:
                    raise JournalCorruptionError(
                        f"Document {document_id} crosses a version boundary without a base"
                    )
                value = _apply_local_ops(value, cast(Sequence[Any], content["ops"]))
            return {
                "record": copy.deepcopy(record),
                "version": version,
                "value": value,
                "deltas_since_base": len(revisions) - base_index - 1,
            }

    @property
    def sequence(self) -> int:
        self._assert_usable()
        return self._sequence

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close_impl())
        await _await_task(self._close_task)

    async def copy_to(self, destination: str | Path) -> None:
        self._assert_usable()
        async with self._lock:
            self._assert_usable()
            destination_path = Path(destination)
            task = asyncio.create_task(asyncio.to_thread(self._copy_sync, destination_path))
            await _await_thread(task)

    async def _close_impl(self) -> None:
        async with self._lock:
            self._closed = True

    def _assert_usable(self) -> None:
        if self._closed or self._closing:
            raise JournalClosedError("journal is closed")
        if self._poison is not None:
            raise self._poison

    def _assert_admitting(self) -> None:
        self._assert_usable()

    def _assert_healthy(self) -> None:
        if self._closed:
            raise JournalClosedError("journal is closed")
        if self._poison is not None:
            raise self._poison

    @property
    def poisoned(self) -> bool:
        """Whether a write outcome is uncertain and this instance is unusable."""

        return self._poison is not None

    def _commit_sync(self, raw_writes: list[Any]) -> int:
        prepared = self._prepare_commit(raw_writes)
        try:
            for name, content in prepared.sidecars:
                self._append_file(self.directory / name, content)
            if self.fsync:
                for name, _content in prepared.sidecars:
                    self._flush_file(self.directory / name)
            self._append_file(self.main_path, prepared.marker)
            try:
                self._apply_commit(prepared)
            except Exception as exc:
                raise self._poison_storage(exc) from exc
        except JournalPoisonedError:
            raise
        except Exception as exc:
            raise self._poison_storage(exc) from exc

        if prepared.reclaims:
            if self.fsync:
                try:
                    self._flush_file(self.main_path)
                except Exception:
                    return prepared.seq
            for name, replacement in prepared.reclaims:
                try:
                    self._replace_sidecar(name, replacement)
                except Exception:
                    continue
        return prepared.seq

    def _prepare_commit(self, raw_writes: Sequence[Any]) -> _PreparedCommit:
        writes = tuple(self._normalize_write(raw) for raw in raw_writes)
        seq = self._sequence + 1
        actions = self._validate_writes(writes, seq=seq)
        candidate = self._candidate_state(writes, seq, actions)
        main_writes: list[dict[str, Any]] = []
        sidecar_records: dict[str, list[dict[str, Any]]] = {}
        ordinal = 0

        def add_sidecar(name: str, payload: dict[str, Any]) -> int:
            nonlocal ordinal
            sidecar_records.setdefault(name, []).append(
                {
                    "format": _FORMAT_VERSION,
                    "type": "record",
                    "seq": seq,
                    "ordinal": ordinal,
                    "payload": payload,
                }
            )
            result = ordinal
            ordinal += 1
            return result

        for write in writes:
            kind = cast(str, write["type"])
            if kind in {"entry", "conversation", "submission", "document.retire"}:
                main_writes.append(copy.deepcopy(write))
            elif kind == "task":
                value = cast(dict[str, Any], write["value"])
                task_id = _record_id(value, "task")
                if _state_status(value, "task") == "terminal":
                    main_writes.append({"type": "task", "value": copy.deepcopy(value)})
                else:
                    name = _sidecar_name("task", task_id)
                    marker_ordinal = add_sidecar(
                        name, {"type": "task", "value": copy.deepcopy(value)}
                    )
                    main_writes.append(
                        {"type": "task.sidecar", "id": task_id, "ordinal": marker_ordinal}
                    )
            elif kind == "document.create":
                record = cast(dict[str, Any], write["record"])
                document_id = _record_id(record, "document")
                name = _sidecar_name("doc", document_id)
                marker_ordinal = add_sidecar(
                    name,
                    {
                        "type": "document",
                        "id": document_id,
                        "content": copy.deepcopy(cast(dict[str, Any], write["content"])),
                    },
                )
                main_writes.append(
                    {
                        "type": "document.create",
                        "record": copy.deepcopy(record),
                        "ordinal": marker_ordinal,
                    }
                )
            elif kind == "document.change":
                document_id = _require_int(write.get("id"), "document id")
                name = _sidecar_name("doc", document_id)
                marker_ordinal = add_sidecar(
                    name,
                    {
                        "type": "document",
                        "id": document_id,
                        "content": copy.deepcopy(cast(dict[str, Any], write["content"])),
                    },
                )
                main_writes.append(
                    {"type": "document.change", "id": document_id, "ordinal": marker_ordinal}
                )
            else:
                raise ValueError(f"Unknown journal write type: {kind}")

        sidecars = tuple(
            (name, b"".join(_json_line(record) for record in records))
            for name, records in sidecar_records.items()
        )
        marker = _json_line(
            {"format": _FORMAT_VERSION, "type": "commit", "seq": seq, "writes": main_writes}
        )
        reclaims = self._plan_reclaims(writes, sidecar_records)
        return _PreparedCommit(seq, writes, marker, sidecars, reclaims, candidate)

    def _normalize_write(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError(
                "journal commit accepts raw write mappings; use append() for SessionEntry"
            )
        write = cast(dict[str, Any], _json_copy(dict(raw), "journal write"))
        kind = write.get("type")
        if not isinstance(kind, str):
            raise ValueError("journal write type must be text")
        if kind in {"entry", "conversation", "submission", "task"}:
            if not isinstance(write.get("value"), Mapping):
                raise ValueError(f"{kind} write requires an object value")
            write["value"] = _json_copy(
                dict(cast(Mapping[str, Any], write["value"])), f"{kind} value"
            )
        elif kind == "document.create":
            if not isinstance(write.get("record"), Mapping):
                raise ValueError("document.create requires a record object")
            write["record"] = _json_copy(
                dict(cast(Mapping[str, Any], write["record"])), "document record"
            )
            write["content"] = _validate_content(write.get("content"), "document content")
        elif kind == "document.change":
            write["content"] = _validate_content(write.get("content"), "document content")
        elif kind not in {"document.retire"}:
            raise ValueError(f"Unknown journal write type: {kind}")
        return write

    def _validate_writes(
        self, writes: Sequence[dict[str, Any]], *, seq: int | None = None
    ) -> dict[int, dict[str, Any]]:
        ids = dict(self._id_kinds)
        entry_ids = set(self._entry_ids)
        tasks = copy.deepcopy(self._tasks)
        documents = {
            key: _Document(copy.deepcopy(value.record), copy.deepcopy(value.revisions))
            for key, value in self._documents.items()
        }
        actions: dict[int, dict[str, Any]] = {}
        for write in writes:
            kind = write.get("type")
            if kind == "entry":
                value = cast(dict[str, Any], write["value"])
                entry_id = value.get("id")
                if not isinstance(entry_id, str) or not entry_id:
                    raise ValueError("entry id must be non-empty text")
                if entry_id in entry_ids:
                    raise ValueError(f"entry id already exists: {entry_id}")
                entry_ids.add(entry_id)
                try:
                    entry_from_json_line(json.dumps(value, ensure_ascii=False))
                except Exception as exc:
                    raise ValueError(f"invalid entry value: {entry_id}") from exc
            elif kind == "task":
                value = cast(dict[str, Any], write["value"])
                task_id = _record_id(value, "task")
                existing = ids.get(task_id)
                if existing is not None and existing != "task":
                    raise ValueError(f"ID {task_id} already belongs to {existing}")
                status = _state_status(value, "task")
                if status not in {"pending", "running", "waiting", "completing", "terminal"}:
                    raise ValueError(f"unknown task status: {status}")
                ids[task_id] = "task"
                tasks[task_id] = copy.deepcopy(value)
            elif kind in {"conversation", "submission"}:
                value = cast(dict[str, Any], write["value"])
                record_id = _record_id(value, kind)
                existing = ids.get(record_id)
                table = cast(str, kind)
                if existing is not None and existing != table:
                    raise ValueError(f"ID {record_id} already belongs to {existing}")
                if table == "conversation" and existing == table:
                    raise ValueError(f"conversation already exists: {record_id}")
                ids[record_id] = table
            elif kind == "document.create":
                raw_record = cast(dict[str, Any], write["record"])
                record = _validate_document_record(raw_record, "document record")
                document_id = _record_id(record, "document")
                existing = ids.get(document_id)
                if existing is not None:
                    raise ValueError(f"ID {document_id} already belongs to {existing}")
                if document_id in documents:
                    raise ValueError(f"document already exists: {document_id}")
                action = actions.setdefault(document_id, {"retire": False})
                if "create" in action or "content" in action:
                    raise ValueError(f"Document {document_id} has more than one content command")
                action["create"] = record
                action["content"] = cast(dict[str, Any], write["content"])
                ids[document_id] = "document"
            elif kind == "document.change":
                document_id = _require_int(write.get("id"), "document id")
                action = actions.setdefault(document_id, {"retire": False})
                if "content" in action:
                    raise ValueError(f"Document {document_id} has more than one content command")
                action["content"] = cast(dict[str, Any], write["content"])
            elif kind == "document.retire":
                document_id = _require_int(write.get("id"), "document id")
                action = actions.setdefault(document_id, {"retire": False})
                if action["retire"]:
                    raise ValueError(f"Document {document_id} is retired more than once")
                action["retire"] = True
            else:
                raise ValueError(f"Unknown journal write type: {kind}")

        current_addresses: dict[str, int] = {}
        for document_id, document in documents.items():
            if _retired_at(document.record) is None:
                address = _document_address_key(document.record)
                if address in current_addresses:
                    raise ValueError("document address has more than one current incarnation")
                current_addresses[address] = document_id
        resolved_records: dict[int, dict[str, Any]] = {}
        for document_id, action in actions.items():
            existing_document = documents.get(document_id)
            record_candidate = action.get("create")
            if record_candidate is None:
                if existing_document is None:
                    raise ValueError(f"unknown document: {document_id}")
                if _retired_at(existing_document.record) is not None:
                    raise ValueError(f"document is retired: {document_id}")
                record = existing_document.record
            elif existing_document is not None:
                raise ValueError(f"document already exists: {document_id}")
            else:
                record = cast(dict[str, Any], record_candidate)
            content = action.get("content")
            if content is not None:
                if action.get("create") is not None and content["kind"] != "base":
                    raise ValueError(f"document creation lacks a base: {document_id}")
                if content["kind"] == "delta":
                    previous = (
                        None if existing_document is None else existing_document.revisions[-1]
                    )
                    if previous is None or previous.content["version"] != content["version"]:
                        raise ValueError(
                            f"document version transition requires a base: {document_id}"
                        )
            resolved_records[document_id] = record

        for document_id, action in actions.items():
            if action.get("create") is None and action.get("retire"):
                address = _document_address_key(resolved_records[document_id])
                current_id = current_addresses.get(address)
                if current_id != document_id:
                    raise ValueError("document address does not match the current incarnation")
                current_addresses.pop(address, None)
        for document_id, action in actions.items():
            if action.get("create") is None:
                continue
            address = _document_address_key(resolved_records[document_id])
            current_id = current_addresses.get(address)
            if current_id is not None and not action.get("retire"):
                raise ValueError("document address already has a current incarnation")
            if not action.get("retire"):
                current_addresses[address] = document_id
        return actions

    def _candidate_state(
        self, writes: Sequence[dict[str, Any]], seq: int, actions: Mapping[int, dict[str, Any]]
    ) -> _CandidateState:
        entries = copy.deepcopy(self._entries)
        entry_ids = set(self._entry_ids)
        tasks = copy.deepcopy(self._tasks)
        documents = {
            key: _Document(copy.deepcopy(value.record), copy.deepcopy(value.revisions))
            for key, value in self._documents.items()
        }
        id_kinds = dict(self._id_kinds)
        for write in writes:
            kind = cast(str, write["type"])
            if kind == "entry":
                value = copy.deepcopy(cast(dict[str, Any], write["value"]))
                entries.append(value)
                entry_ids.add(cast(str, value["id"]))
            elif kind in {"conversation", "submission"}:
                value = cast(dict[str, Any], write["value"])
                id_kinds[_record_id(value, kind)] = kind
            elif kind == "task":
                value = copy.deepcopy(cast(dict[str, Any], write["value"]))
                task_id = _record_id(value, "task")
                tasks[task_id] = value
                id_kinds[task_id] = "task"
        for document_id, action in actions.items():
            content = action.get("content")
            if action.get("create") is not None:
                record = _stamp_created(copy.deepcopy(action["create"]), seq)
                documents[document_id] = _Document(
                    record, [_Revision(seq, copy.deepcopy(cast(dict[str, Any], content)))]
                )
                id_kinds[document_id] = "document"
            elif content is not None:
                document = documents[document_id]
                revision = _Revision(seq, copy.deepcopy(content))
                if content["kind"] == "base" and _current_only(document.record):
                    document.revisions = [revision]
                else:
                    document.revisions.append(revision)
            if action["retire"]:
                document = documents[document_id]
                _stamp_retired(document.record, seq)
                if _current_only(document.record):
                    document.revisions = []
        return _CandidateState(tuple(entries), frozenset(entry_ids), tasks, documents, id_kinds)

    @staticmethod
    def _validate_json_value(value: Any, description: str) -> None:
        _json_copy(value, description)

    @staticmethod
    def _validate_document_content_transition(
        document: _Document, content: dict[str, Any], document_id: int
    ) -> None:
        if content["kind"] == "delta":
            if not document.revisions:
                raise ValueError(f"document delta has no base: {document_id}")
            previous = document.revisions[-1].content
            if previous["version"] != content["version"]:
                raise ValueError(f"document version transition requires a base: {document_id}")

    def _plan_reclaims(
        self,
        writes: Sequence[dict[str, Any]],
        sidecar_records: Mapping[str, Sequence[dict[str, Any]]],
    ) -> tuple[tuple[str, bytes | None], ...]:
        current_only_created: set[int] = set()
        retired: set[int] = set()
        bases: set[int] = set()
        final_tasks: dict[int, dict[str, Any]] = {}
        for write in writes:
            kind = write.get("type")
            if kind == "document.create":
                record = cast(dict[str, Any], write["record"])
                if _current_only(record):
                    current_only_created.add(_record_id(record, "document"))
            elif kind == "document.change":
                if cast(dict[str, Any], write["content"])["kind"] == "base":
                    bases.add(_require_int(write.get("id"), "document id"))
            elif kind == "document.retire":
                retired.add(_require_int(write.get("id"), "document id"))
            elif kind == "task":
                value = cast(dict[str, Any], write["value"])
                final_tasks[_record_id(value, "task")] = value

        result: list[tuple[str, bytes | None]] = []
        for document_id in retired:
            if document_id in current_only_created or (
                document_id in self._documents
                and _current_only(self._documents[document_id].record)
            ):
                result.append((_sidecar_name("doc", document_id), None))
        for document_id in bases:
            document = self._documents.get(document_id)
            if document is None or not _current_only(document.record) or document_id in retired:
                continue
            name = _sidecar_name("doc", document_id)
            content = sidecar_records.get(name)
            if content:
                result.append((name, b"".join(_json_line(value) for value in content)))
        for task_id, task in final_tasks.items():
            name = _sidecar_name("task", task_id)
            if _state_status(task, "task") == "terminal" and (
                task_id in self._tasks or name in sidecar_records
            ):
                result.append((name, None))
        return tuple(result)

    def _sidecar_names(self) -> Iterable[tuple[str, Path]]:
        for path in self.directory.iterdir():
            if _SIDECAR_RE.match(path.name):
                yield path.name, path

    def _apply_commit(self, prepared: _PreparedCommit) -> None:
        candidate = prepared.candidate
        self._entries = list(candidate.entries)
        self._entry_ids = set(candidate.entry_ids)
        self._tasks = candidate.tasks
        self._documents = candidate.documents
        self._id_kinds = candidate.id_kinds
        self._sequence = prepared.seq

    def _recover_sync(self) -> None:
        for path in self.directory.iterdir():
            if path.is_file() and _RECLAIM_RE.match(path.name):
                with suppress(OSError):
                    path.unlink()
        main = self._read_file_lines(self.main_path, _MAIN_NAME)
        markers: list[dict[str, Any]] = []
        previous = 0
        for line in main:
            marker = self._parse_marker(line.value, _MAIN_NAME)
            if marker["seq"] <= previous:
                raise JournalCorruptionError(
                    "Commit sequence does not strictly increase in main.jsonl"
                )
            previous = marker["seq"]
            markers.append(marker)

        parsed_sidecars: dict[str, list[_ParsedLine]] = {}
        records: dict[tuple[str, int, int], dict[str, Any]] = {}
        for name, path in sorted(self._sidecar_names()):
            parsed = self._read_file_lines(path, name)
            parsed_sidecars[name] = parsed
            previous_key: tuple[int, int] | None = None
            for line in parsed:
                record = self._parse_sidecar_record(line.value, name)
                key = (record["seq"], record["ordinal"])
                if previous_key is not None and key <= previous_key:
                    raise JournalCorruptionError(f"Sidecar records are out of order in {name}")
                previous_key = key
                records[(name, *key)] = record

        current_only: set[int] = set()
        retired: set[int] = set()
        final_task_live: dict[int, bool] = {}
        latest_base: dict[int, tuple[int, int]] = {}
        for marker in markers:
            for operation in marker["writes"]:
                op_type = operation["type"]
                if op_type == "document.create":
                    document_id = _record_id(cast(dict[str, Any], operation["record"]), "document")
                    if _current_only(cast(dict[str, Any], operation["record"])):
                        current_only.add(document_id)
                elif op_type == "document.retire":
                    retired.add(_require_int(operation.get("id"), "document id"))
                elif op_type == "task":
                    task_id = _record_id(cast(dict[str, Any], operation["value"]), "task")
                    final_task_live[task_id] = False
                elif op_type == "task.sidecar":
                    task_id = _require_int(operation.get("id"), "task id")
                    final_task_live[task_id] = True
                if op_type in {"document.create", "document.change"}:
                    document_id = (
                        _record_id(cast(dict[str, Any], operation["record"]), "document")
                        if op_type == "document.create"
                        else _require_int(operation.get("id"), "document id")
                    )
                    if document_id in current_only:
                        base_record = records.get(
                            (
                                _sidecar_name("doc", document_id),
                                marker["seq"],
                                _require_int(operation.get("ordinal"), "ordinal"),
                            )
                        )
                        if (
                            base_record is not None
                            and cast(dict[str, Any], base_record["payload"]).get("type")
                            == "document"
                        ):
                            base_content = cast(dict[str, Any], base_record["payload"])["content"]
                            if base_content.get("kind") == "base":
                                location = (marker["seq"], operation["ordinal"])
                                previous_base = latest_base.get(document_id)
                                if previous_base is None or location > previous_base:
                                    latest_base[document_id] = location
        terminal_tasks = {task_id for task_id, live in final_task_live.items() if not live}
        retired_current_only = retired & current_only
        confirmed: set[tuple[str, int, int]] = set()
        self._reset_state()
        for marker in markers:
            writes: list[dict[str, Any]] = []
            for operation in marker["writes"]:
                op_type = operation["type"]
                if op_type in {"entry", "conversation", "submission", "task", "document.retire"}:
                    writes.append(copy.deepcopy(operation))
                    continue
                if op_type == "task.sidecar":
                    task_id = _require_int(operation.get("id"), "task id")
                    ordinal = _require_int(operation.get("ordinal"), "ordinal")
                    name = _sidecar_name("task", task_id)
                    optional = not final_task_live.get(task_id, True)
                    sidecar_record = self._confirm_record(
                        marker, name, ordinal, records, confirmed, optional
                    )
                    if sidecar_record is not None:
                        payload = cast(dict[str, Any], sidecar_record["payload"])
                        if payload.get("type") != "task":
                            raise JournalCorruptionError(
                                "Confirmed task sidecar data has an invalid type"
                            )
                        value = cast(dict[str, Any], payload["value"])
                        if _record_id(value, "task") != task_id:
                            raise JournalCorruptionError(
                                "Confirmed task sidecar data does not match commit"
                            )
                        if not optional:
                            writes.append({"type": "task", "value": value})
                    continue
                if op_type in {"document.create", "document.change"}:
                    document_id = (
                        _record_id(cast(dict[str, Any], operation["record"]), "document")
                        if op_type == "document.create"
                        else _require_int(operation.get("id"), "document id")
                    )
                    ordinal = _require_int(operation.get("ordinal"), "ordinal")
                    name = _sidecar_name("doc", document_id)
                    reclaimed = document_id in retired_current_only or (
                        document_id in latest_base
                        and (marker["seq"], ordinal) < latest_base[document_id]
                    )
                    sidecar_record = self._confirm_record(
                        marker, name, ordinal, records, confirmed, reclaimed
                    )
                    content: dict[str, Any] | None = None
                    if sidecar_record is not None:
                        payload = cast(dict[str, Any], sidecar_record["payload"])
                        if payload.get("type") != "document" or payload.get("id") != document_id:
                            raise JournalCorruptionError(
                                "Confirmed document sidecar data does not match commit"
                            )
                        content = cast(dict[str, Any], payload["content"])
                    if op_type == "document.create":
                        if content is not None and content.get("kind") != "base":
                            raise JournalCorruptionError("Document creation lacks a confirmed base")
                        writes.append(
                            {
                                "type": "document.create",
                                "record": copy.deepcopy(cast(dict[str, Any], operation["record"])),
                                "content": content
                                if not reclaimed and content is not None
                                else {"kind": "base", "version": 1, "value": {}},
                            }
                        )
                    elif not reclaimed and content is not None:
                        writes.append(
                            {"type": "document.change", "id": document_id, "content": content}
                        )
            try:
                normalized = tuple(self._normalize_write(write) for write in writes)
                actions = self._validate_writes(normalized, seq=marker["seq"])
                candidate = self._candidate_state(normalized, marker["seq"], actions)
                prepared = _PreparedCommit(marker["seq"], normalized, b"", (), (), candidate)
                self._apply_commit(prepared)
            except Exception as exc:
                raise JournalCorruptionError(
                    f"Invalid committed state at sequence {marker['seq']}"
                ) from exc

        for name, parsed in parsed_sidecars.items():
            first_unconfirmed: int | None = None
            for line in parsed:
                record = self._parse_sidecar_record(line.value, name)
                record_key: tuple[str, int, int] = (
                    name,
                    record["seq"],
                    record["ordinal"],
                )
                if record_key in confirmed:
                    if first_unconfirmed is not None:
                        raise JournalCorruptionError(
                            f"Confirmed record follows an unconfirmed tail in {name}"
                        )
                elif first_unconfirmed is None:
                    first_unconfirmed = line.start
            if first_unconfirmed is not None:
                self._truncate_file(Path(self.directory / name), first_unconfirmed)

        reclaims = self._reclaim_plan_from_recovery(
            parsed_sidecars,
            confirmed,
            current_only,
            retired_current_only,
            latest_base,
            terminal_tasks,
        )
        self._run_reclaims(reclaims)

    def _reset_state(self) -> None:
        self._sequence = 0
        self._entries.clear()
        self._entry_ids.clear()
        self._tasks.clear()
        self._documents.clear()
        self._id_kinds.clear()

    @staticmethod
    def _parse_marker(value: dict[str, Any], name: str) -> dict[str, Any]:
        if value.get("format") != _FORMAT_VERSION or value.get("type") != "commit":
            raise JournalCorruptionError(f"Invalid commit marker in {name}")
        _require_int(value.get("seq"), "commit sequence", positive=True)
        writes = value.get("writes")
        if not isinstance(writes, list):
            raise JournalCorruptionError(f"Invalid commit writes in {name}")
        marker = {"format": 1, "type": "commit", "seq": value["seq"], "writes": []}
        marker["writes"] = [Journal._parse_main_operation(write, name) for write in writes]
        return marker

    @staticmethod
    def _parse_main_operation(value: object, name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
            raise JournalCorruptionError(f"Invalid write in {name}")
        try:
            operation = cast(dict[str, Any], _json_copy(dict(value), f"write in {name}"))
        except (TypeError, ValueError) as exc:
            raise JournalCorruptionError(f"Invalid write in {name}") from exc
        kind = cast(str, operation["type"])
        if kind in {"entry", "conversation", "submission"}:
            if not isinstance(operation.get("value"), Mapping):
                raise JournalCorruptionError(f"Invalid {kind} write in {name}")
            if kind == "entry":
                try:
                    entry_from_json_line(
                        json.dumps(operation["value"], ensure_ascii=False, allow_nan=False)
                    )
                except Exception as exc:
                    raise JournalCorruptionError(f"Invalid entry write in {name}") from exc
            else:
                _record_id(cast(Mapping[str, Any], operation["value"]), kind)
        elif kind == "task":
            value = operation.get("value")
            if (
                not isinstance(value, Mapping)
                or _state_status(cast(Mapping[str, Any], value), "task") != "terminal"
            ):
                raise JournalCorruptionError(f"Invalid terminal task write in {name}")
            _record_id(cast(Mapping[str, Any], value), "task")
        elif kind in {"task.sidecar", "document.create", "document.change"}:
            if not _is_int(operation.get("id")) and kind != "document.create":
                raise JournalCorruptionError(f"Invalid {kind} write in {name}")
            if kind == "document.create":
                record = operation.get("record")
                if not isinstance(record, Mapping):
                    raise JournalCorruptionError(f"Invalid document creation in {name}")
                try:
                    _validate_document_record(record, "document record")
                except (TypeError, ValueError) as exc:
                    raise JournalCorruptionError(f"Invalid document creation in {name}") from exc
            if not _is_int(operation.get("ordinal")) or cast(int, operation["ordinal"]) < 0:
                raise JournalCorruptionError(f"Invalid {kind} ordinal in {name}")
        elif kind == "document.retire":
            if not _is_int(operation.get("id")):
                raise JournalCorruptionError(f"Invalid document retirement in {name}")
        else:
            raise JournalCorruptionError(f"Unknown write type in {name}")
        return operation

    @staticmethod
    def _parse_sidecar_record(value: dict[str, Any], name: str) -> dict[str, Any]:
        if value.get("format") != _FORMAT_VERSION or value.get("type") != "record":
            raise JournalCorruptionError(f"Invalid sidecar record in {name}")
        _require_int(value.get("seq"), "sidecar sequence", positive=True)
        ordinal = _require_int(value.get("ordinal"), "sidecar ordinal")
        payload = value.get("payload")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("type"), str):
            raise JournalCorruptionError(f"Invalid sidecar payload in {name}")
        payload = cast(dict[str, Any], payload)
        if payload["type"] == "task":
            task = payload.get("value")
            if (
                not isinstance(task, Mapping)
                or _state_status(cast(Mapping[str, Any], task), "task") == "terminal"
            ):
                raise JournalCorruptionError(f"Invalid live task record in {name}")
            _record_id(cast(Mapping[str, Any], task), "task")
            if _state_status(cast(Mapping[str, Any], task), "task") not in {
                "pending",
                "running",
                "waiting",
                "completing",
            }:
                raise JournalCorruptionError(f"Invalid live task record in {name}")
            try:
                payload["value"] = cast(dict[str, Any], _json_copy(dict(task), "task"))
            except (TypeError, ValueError) as exc:
                raise JournalCorruptionError(f"Invalid live task record in {name}") from exc
        elif payload["type"] == "document":
            _require_int(payload.get("id"), "document id")
            try:
                _validate_content(payload.get("content"), f"document content in {name}")
            except (TypeError, ValueError) as exc:
                raise JournalCorruptionError(f"Invalid document record in {name}") from exc
        else:
            raise JournalCorruptionError(f"Unknown sidecar record type in {name}")
        return {
            "format": 1,
            "type": "record",
            "seq": value["seq"],
            "ordinal": ordinal,
            "payload": dict(payload),
        }

    @staticmethod
    def _confirm_record(
        marker: Mapping[str, Any],
        name: str,
        ordinal: int,
        records: Mapping[tuple[str, int, int], dict[str, Any]],
        confirmed: set[tuple[str, int, int]],
        optional: bool,
    ) -> dict[str, Any] | None:
        key = (name, cast(int, marker["seq"]), ordinal)
        if key in confirmed:
            raise JournalCorruptionError("Sidecar record is confirmed more than once")
        record = records.get(key)
        if record is None:
            if optional:
                return None
            raise JournalCorruptionError(
                f"Missing confirmed sidecar record {name} at sequence {marker['seq']}"
            )
        confirmed.add(key)
        return record

    def _read_file_lines(self, path: Path, name: str) -> list[_ParsedLine]:
        if not path.exists():
            return []
        raw = path.read_bytes()
        complete_size = len(raw)
        if complete_size and raw[-1] != 0x0A:
            complete_size = raw.rfind(b"\n") + 1
            self._truncate_file(path, complete_size)
        complete = raw[:complete_size]
        lines: list[_ParsedLine] = []
        start = 0
        for end, byte in enumerate(complete):
            if byte != 0x0A:
                continue
            lines.append(
                _ParsedLine(_line_json(complete[start:end], f"{name} line {len(lines) + 1}"), start)
            )
            start = end + 1
        return lines

    def _reclaim_plan_from_recovery(
        self,
        parsed_sidecars: Mapping[str, Sequence[_ParsedLine]],
        confirmed: set[tuple[str, int, int]],
        current_only: set[int],
        retired_current_only: set[int],
        latest_base: Mapping[int, tuple[int, int]],
        terminal_tasks: set[int],
    ) -> tuple[tuple[str, bytes | None], ...]:
        plans: list[tuple[str, bytes | None]] = []
        for name, parsed in parsed_sidecars.items():
            match = _SIDECAR_RE.match(name)
            if match is None:
                continue
            numeric_id = int(match.group("id"))
            confirmed_lines = [
                line
                for line in parsed
                if (name, line.value["seq"], line.value["ordinal"]) in confirmed
            ]
            retained: Sequence[_ParsedLine] | None = None
            if (
                name.startswith("task-")
                and numeric_id in terminal_tasks
                or name.startswith("doc-")
                and numeric_id in retired_current_only
            ):
                retained = []
            elif name.startswith("doc-") and numeric_id in latest_base:
                base = latest_base[numeric_id]
                retained = [
                    line
                    for line in confirmed_lines
                    if (line.value["seq"], line.value["ordinal"]) >= base
                ]
            if retained is not None and (len(retained) < len(confirmed_lines) or not retained):
                plans.append((name, b"".join(_json_line(line.value) for line in retained) or None))
        return tuple(plans)

    def _run_reclaims(self, reclaims: Sequence[tuple[str, bytes | None]]) -> None:
        if reclaims and self.fsync:
            try:
                self._flush_file(self.main_path)
            except Exception:
                return
        for name, content in reclaims:
            with suppress(Exception):
                self._replace_sidecar(name, content)

    def _copy_sync(self, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True, exist_ok=False)
        try:
            for path in sorted(self.directory.iterdir(), key=lambda item: item.name):
                if path.name == _MAIN_NAME or _SIDECAR_RE.match(path.name):
                    shutil.copy2(path, destination / path.name)
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def _append_file(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            written = handle.write(data)
            if written != len(data):
                raise OSError(f"short write to {path}: {written} of {len(data)} bytes")

    def _flush_file(self, path: Path) -> None:
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())

    def _replace_sidecar(self, name: str, content: bytes | None) -> None:
        path = self.directory / name
        if content is None:
            path.unlink(missing_ok=True)
            return
        temporary = path.with_name(path.name + _RECLAIM_SUFFIX)
        with temporary.open("wb") as handle:
            written = handle.write(content)
            if written != len(content):
                raise OSError(
                    f"short reclaim write to {temporary}: {written} of {len(content)} bytes"
                )
        if self.fsync:
            self._flush_file(temporary)
        os.replace(temporary, path)

    def _truncate_file(self, path: Path, size: int) -> None:
        with path.open("r+b") as handle:
            handle.truncate(size)

    def _poison_storage(self, cause: BaseException) -> JournalPoisonedError:
        if self._poison is None:
            self._poison = JournalPoisonedError(cause)
        return self._poison


@dataclass(frozen=True, slots=True)
class _ParsedLine:
    value: dict[str, Any]
    start: int


def _created_at(record: Mapping[str, Any]) -> int:
    value = record.get("createdAt", record.get("created_at", 0))
    return _require_int(value, "document created sequence")


def _retired_at(record: Mapping[str, Any]) -> int | None:
    value = record.get("retiredAt", record.get("retired_at"))
    if value is None:
        return None
    return _require_int(value, "document retired sequence")


def _stamp_created(record: dict[str, Any], seq: int) -> dict[str, Any]:
    record.setdefault("createdAt", seq)
    return record


def _stamp_retired(record: dict[str, Any], seq: int) -> None:
    if "retiredAt" in record or "createdAt" in record:
        record["retiredAt"] = seq
    else:
        record["retired_at"] = seq


def _entry_dict(entry: SessionEntry) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(entry_to_json_line(entry)))


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


async def _await_task(task: asyncio.Task[None]) -> None:
    await _await_thread(task)


async def open_journal(directory: str | Path, *, fsync: bool = False) -> Journal:
    return await Journal.open(directory, fsync=fsync)


open_session_journal = open_journal


__all__ = [
    "Journal",
    "JournalClosedError",
    "JournalCorruptionError",
    "JournalError",
    "JournalPoisonedError",
    "open_journal",
    "open_session_journal",
]
