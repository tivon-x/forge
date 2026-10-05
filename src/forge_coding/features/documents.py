"""Controlled JSON document access on the session commit line."""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol

from forge_agent.session.entries import SessionEntry

DocumentPoint = int | Literal["current"]
DocumentResult = dict[str, Any]
CommitCallback = Callable[[], Awaitable[Any]]


class DocumentStorage(Protocol):
    """Coding-owned storage surface backed by the JSONL journal."""

    async def read_records(self) -> Mapping[str, Any]:
        """Read confirmed records on the storage line."""

    async def commit_records(
        self,
        writes: Sequence[dict[str, Any]],
        entries: Sequence[SessionEntry] = (),
    ) -> int:
        """Validate and commit records under one journal marker."""


class DocumentStore:
    """Address current documents and their rewindable history.

    The journal remains the only validator and source of truth.  ``line`` is
    the same async commit lock used by task/input/session coordination; reads,
    ID allocation and the storage commit therefore share one ordering boundary.
    ``on_commit`` runs only after storage has accepted the marker.
    """

    def __init__(
        self,
        storage: DocumentStorage,
        line: Any,
        on_commit: CommitCallback | None = None,
        *,
        before_change: Callable[[], Any] | None = None,
        on_error: Callable[[BaseException], Any] | None = None,
    ) -> None:
        self.storage = storage
        self.line = line
        self.on_commit = on_commit
        self.before_change = before_change
        self.on_error = on_error

    async def change(
        self,
        kind: str,
        key: str | None = None,
        value: Mapping[str, Any] | None = None,
        ops: Sequence[Any] | None = None,
        *,
        version: int = 1,
        history: Literal["latest", "rewindable"] = "rewindable",
    ) -> int:
        """Create or change one conversation-0 document.

        A new address requires ``value`` so the journal receives a base.  An
        existing address accepts either a complete ``value`` base or explicit
        delta ``ops``.  JSON shape, operation paths and version transitions are
        deliberately validated by the journal's candidate state.
        """

        _validate_address(kind, key)
        if history not in {"latest", "rewindable"}:
            raise ValueError("history must be latest or rewindable")
        if type(version) is not int or version < 1:
            raise ValueError("version must be a positive integer")
        if (value is None) == (ops is None):
            raise ValueError("provide exactly one of value or ops")

        self._before_change()
        async with _locked(self.line):
            self._before_change()
            records = await self.storage.read_records()
            current = _find_document(records, kind, key)
            if current is None:
                if value is None:
                    raise ValueError("a new document requires a complete base value")
                document_id = _allocate_id(records)
                write: dict[str, Any] = {
                    "type": "document.create",
                    "record": {
                        "id": document_id,
                        "kind": kind,
                        "scope": {"kind": "conversation", "conversationId": 0},
                        "history": history,
                    },
                    "content": {"kind": "base", "version": version, "value": dict(value)},
                }
                if key is not None:
                    write["record"]["key"] = key
            else:
                record = _record(current)
                document_id = _document_id(record)
                if ops is not None and not ops and version == current.get("version"):
                    # Empty changes preserve the snapshot and publish nothing.
                    return int(records["sequence"])
                content: dict[str, Any]
                if value is not None:
                    content = {"kind": "base", "version": version, "value": dict(value)}
                else:
                    content = {"kind": "delta", "version": version, "ops": list(ops or ())}
                write = {"type": "document.change", "id": document_id, "content": content}
            return await self._commit(write)

    async def get(
        self, kind: str, key: str | None = None, at: DocumentPoint = "current"
    ) -> DocumentResult | None:
        """Read a current or rewindable as-of document through storage."""

        _validate_address(kind, key)
        if at != "current" and (type(at) is not int or at < 0):
            raise ValueError("at must be current or a non-negative sequence")
        async with _locked(self.line):
            records = await self.storage.read_records()
            current = _find_document(records, kind, key)
            reader = getattr(self.storage, "document", None)
            if current is None:
                if at == "current" or not callable(reader):
                    return None
                stored = await _document_at(records, reader, None, kind, key, at)
                return None if stored is None else copy.deepcopy(dict(stored))
            record = _record(current)
            document_id = _document_id(record)
            if callable(reader):
                if at == "current":
                    stored = await reader(document_id, at)
                else:
                    stored = await _document_at(
                        records,
                        reader,
                        current,
                        kind,
                        key,
                        at,
                    )
                return None if stored is None else copy.deepcopy(dict(stored))
            if at != "current" and record.get("history") == "latest":
                raise ValueError(f"Document {document_id} does not retain historical content")
            return copy.deepcopy(dict(current))

    async def retire(self, kind: str, key: str | None = None) -> int:
        """Retire the current incarnation at one document address."""

        _validate_address(kind, key)
        self._before_change()
        async with _locked(self.line):
            self._before_change()
            records = await self.storage.read_records()
            current = _find_document(records, kind, key)
            if current is None:
                raise ValueError(f"Document {kind!r} does not exist")
            document_id = _document_id(_record(current))
            return await self._commit({"type": "document.retire", "id": document_id})

    async def _commit(self, write: Mapping[str, Any]) -> int:
        reported = False
        try:
            sequence = await self.storage.commit_records([dict(write)])
            try:
                await self._after_commit()
            except BaseException as exc:
                await self._report_error(exc)
                reported = True
                raise
            return sequence
        except BaseException as exc:
            if not reported and self._storage_failed(exc):
                await self._report_error(exc)
            raise

    def _before_change(self) -> None:
        callback = self.before_change
        if callback is not None:
            callback()

    def _storage_failed(self, error: BaseException) -> bool:
        return isinstance(error, (OSError, asyncio.CancelledError)) or bool(
            getattr(self.storage, "poisoned", False)
        )

    async def _report_error(self, error: BaseException) -> None:
        callback = self.on_error
        if callback is None:
            return
        result = callback(error)
        if inspect.isawaitable(result):
            await result

    async def _after_commit(self) -> None:
        callback = self.on_commit
        if callback is None:
            return
        result = callback()
        if inspect.isawaitable(result):
            await result


@asynccontextmanager
async def _locked(line: Any) -> AsyncIterator[None]:
    if line is None:
        yield
        return
    async with line:
        yield


async def _document_at(
    records: Mapping[str, Any],
    reader: Callable[[int, DocumentPoint], Awaitable[Mapping[str, Any] | None]],
    current: Mapping[str, Any] | None,
    kind: str,
    key: str | None,
    at: int,
) -> Mapping[str, Any] | None:
    """Resolve an address across retired incarnations through storage.document."""

    identifiers = records.get("ids", ())
    if not isinstance(identifiers, Sequence) or isinstance(identifiers, (str, bytes)):
        raise TypeError("storage ids must be a sequence")
    current_id = None if current is None else _document_id(_record(current))
    candidates = [] if current_id is None else [current_id]
    candidates.extend(
        value for value in identifiers if type(value) is int and value >= 0 and value != current_id
    )
    for document_id in candidates:
        if document_id == current_id and current is not None:
            created = _record(current).get("createdAt", 0)
            if type(created) is int and at < created:
                continue
        try:
            stored = await reader(document_id, at)
        except ValueError:
            if document_id == current_id:
                raise
            continue
        if stored is not None and _matches_address(stored, kind, key):
            return stored
    return None


def _validate_address(kind: str, key: str | None) -> None:
    if not isinstance(kind, str) or not kind:
        raise ValueError("document kind must be non-empty text")
    if key is not None and not isinstance(key, str):
        raise TypeError("document key must be text")


def _find_document(
    records: Mapping[str, Any], kind: str, key: str | None
) -> Mapping[str, Any] | None:
    documents = records.get("documents", ())
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise TypeError("storage documents must be a sequence")
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        record = document.get("record")
        if not isinstance(record, Mapping):
            continue
        if _matches_address(document, kind, key):
            return document
    return None


def _matches_address(document: Mapping[str, Any], kind: str, key: str | None) -> bool:
    record = document.get("record")
    if not isinstance(record, Mapping):
        return False
    return (
        record.get("scope") == {"kind": "conversation", "conversationId": 0}
        and record.get("kind") == kind
        and record.get("key") == key
    )


def _record(document: Mapping[str, Any]) -> Mapping[str, Any]:
    record = document.get("record")
    if not isinstance(record, Mapping):
        raise ValueError("storage document is missing its record")
    return record


def _document_id(record: Mapping[str, Any]) -> int:
    value = record.get("id")
    if type(value) is not int or value < 0:
        raise ValueError("storage document has an invalid id")
    return value


def _allocate_id(records: Mapping[str, Any]) -> int:
    identifiers = records.get("ids", ())
    if not isinstance(identifiers, Sequence) or isinstance(identifiers, (str, bytes)):
        raise TypeError("storage ids must be a sequence")
    numeric = [value for value in identifiers if type(value) is int and value >= 0]
    return max(numeric, default=0) + 1


__all__ = ["DocumentPoint", "DocumentResult", "DocumentStorage", "DocumentStore"]
