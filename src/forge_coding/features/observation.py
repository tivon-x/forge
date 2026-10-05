"""Committed session snapshots and bounded observation streams.

The observation layer only reads confirmed storage state.  Producers must call
``refresh`` after their commit has returned; this module never publishes a raw
runtime event before that boundary.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from forge_agent.events import AgentEvent, CommittedSnapshotEvent, HumanInputRequestedEvent
from forge_agent.session.entries import SessionEntry
from forge_agent.session.jsonl import entry_to_json_line
from forge_agent.types import JSONValue

MAX_PENDING_FRAMES = 100
"""Maximum number of exact frames retained for one slow observer."""


class ObservationStorage(Protocol):
    """Minimum committed storage surface used by :class:`CommittedView`."""

    async def read_all(self) -> list[SessionEntry]:
        """Read the confirmed transcript entries in storage order."""


type Snapshot = dict[str, JSONValue]
type Event = dict[str, JSONValue]
type Frame = dict[str, JSONValue]
type CommitEvent = AgentEvent | BaseException | None
type CommitQueueItem = tuple[int, CommitEvent] | tuple[int, CommitEvent, int | None]


@dataclass(slots=True)
class _CommitFrame:
    task_id: int
    sequence: int | None
    events: deque[CommitEvent] = field(default_factory=deque)
    control: bool = False


class CommitEventQueue:
    """Bounded commit-frame queue compatible with ``asyncio.Queue`` reads.

    Events carrying the same committed sequence are one pending frame.  Once
    the normal frame budget is full, normal events are replaced by one current
    ``CommittedSnapshotEvent``.  ``None`` and exceptions are control markers;
    they survive coalescing so run settlement cannot be lost.
    """

    def __init__(
        self,
        snapshot: Callable[[], Mapping[str, JSONValue] | CommittedSnapshotEvent],
        *,
        max_pending_frames: int = MAX_PENDING_FRAMES,
    ) -> None:
        if type(max_pending_frames) is not int or max_pending_frames < 1:
            raise ValueError("max_pending_frames must be a positive integer")
        self._snapshot = snapshot
        self.max_pending_frames = max_pending_frames
        self._frames: deque[_CommitFrame] = deque()
        self._wake = asyncio.Event()
        self._snapshot_sequence = -1

    def qsize(self) -> int:
        """Return pending normal commit frames, grouping events by sequence."""

        return self._normal_frame_count()

    def empty(self) -> bool:
        return not self._frames

    def put_nowait(self, item: CommitQueueItem) -> None:
        task_id, event, sequence = _queue_item(item)
        if event is None or isinstance(event, BaseException):
            self._frames.append(_CommitFrame(task_id, sequence, deque((event,)), control=True))
        else:
            if sequence is not None and sequence <= self._snapshot_sequence:
                if isinstance(event, HumanInputRequestedEvent):
                    self._frames.append(
                        _CommitFrame(task_id, sequence, deque((event,)), control=True)
                    )
                    self._wake.set()
                return
            last = self._frames[-1] if self._frames else None
            if (
                last is not None
                and not last.control
                and sequence is not None
                and last.sequence == sequence
                and last.task_id == task_id
            ):
                last.events.append(event)
            elif self._normal_frame_count() >= self.max_pending_frames:
                self._coalesce(task_id, sequence)
                if isinstance(event, HumanInputRequestedEvent):
                    self._frames.append(
                        _CommitFrame(task_id, sequence, deque((event,)), control=True)
                    )
            else:
                self._frames.append(_CommitFrame(task_id, sequence, deque((event,))))
        self._wake.set()

    async def put(self, item: CommitQueueItem) -> None:
        self.put_nowait(item)

    def get_nowait(self) -> tuple[int, CommitEvent]:
        if not self._frames:
            raise asyncio.QueueEmpty
        frame = self._frames[0]
        event = frame.events.popleft()
        if not frame.events:
            self._frames.popleft()
        if not self._frames:
            self._wake.clear()
        return frame.task_id, event

    async def get(self) -> tuple[int, CommitEvent]:
        while self.empty():
            self._wake.clear()
            if not self.empty():
                continue
            await self._wake.wait()
        return self.get_nowait()

    def _normal_frame_count(self) -> int:
        return sum(not frame.control for frame in self._frames)

    def _coalesce(self, task_id: int, sequence: int | None) -> None:
        snapshot = self._snapshot_event()
        covered = snapshot.snapshot.get("sequence", sequence)
        if type(covered) is int:
            self._snapshot_sequence = max(self._snapshot_sequence, covered)
        controls: deque[_CommitFrame] = deque()
        for frame in self._frames:
            if frame.control:
                controls.append(frame)
            else:
                requests: deque[CommitEvent] = deque(
                    event for event in frame.events if isinstance(event, HumanInputRequestedEvent)
                )
                if requests:
                    controls.append(
                        _CommitFrame(frame.task_id, frame.sequence, requests, control=True)
                    )
        # Settlement ends the consumer stream, so its final snapshot must be
        # delivered before retained terminal/error control markers.
        controls.appendleft(_CommitFrame(task_id, None, deque((snapshot,))))
        self._frames = controls

    def _snapshot_event(self) -> CommittedSnapshotEvent:
        value = self._snapshot()
        if isinstance(value, CommittedSnapshotEvent):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("snapshot callback must return a JSON object")
        return CommittedSnapshotEvent(snapshot=copy.deepcopy(dict(value)))


def _queue_item(item: CommitQueueItem) -> tuple[int, CommitEvent, int | None]:
    if len(item) == 2:
        task_id, event = item
        sequence = None
    elif len(item) == 3:
        task_id, event, sequence = item
    else:
        raise TypeError("commit queue items need task id, event, and optional sequence")
    if type(task_id) is not int:
        raise TypeError("commit queue task id must be an integer")
    if sequence is not None and (type(sequence) is not int or sequence < 0):
        raise ValueError("commit queue sequence must be a non-negative integer")
    return task_id, event, sequence


class CommittedView:
    """A cached view derived from one committed storage line.

    ``read_records`` is optional so the view also works with the small in-memory
    storage used by print-mode tests.  Durable storage implementations should
    provide it; when absent, task/document tables are empty and the sequence is
    derived from the number of entries.
    """

    def __init__(
        self,
        storage: ObservationStorage,
        *,
        max_pending_frames: int = MAX_PENDING_FRAMES,
    ) -> None:
        if type(max_pending_frames) is not int or max_pending_frames < 1:
            raise ValueError("max_pending_frames must be a positive integer")
        self.storage = storage
        self.max_pending_frames = max_pending_frames
        self._snapshot: Snapshot | None = None
        self._observers: set[_ObservationStream] = set()
        self._refresh_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(cls, storage: ObservationStorage) -> CommittedView:
        """Build a view from confirmed storage without publishing a frame."""

        view = cls(storage)
        await view._refresh(events=None, publish=False)
        return view

    @property
    def closed(self) -> bool:
        return self._closed

    def snapshot(self) -> Snapshot:
        """Return a detached copy of the last confirmed snapshot."""

        if self._snapshot is None:
            raise RuntimeError("CommittedView.open() must complete before snapshot()")
        return copy.deepcopy(self._snapshot)

    async def refresh(self, *, events: Sequence[Mapping[str, JSONValue]] | None = None) -> Snapshot:
        """Read a post-commit snapshot and publish one bounded frame.

        The caller owns the commit boundary.  ``events`` are attached only to
        the newly read snapshot, so a failed commit cannot reach an observer.
        """

        return await self._refresh(events=events, publish=True)

    async def publish(self, *, events: Sequence[Mapping[str, JSONValue]] | None = None) -> Snapshot:
        """Compatibility alias for ``refresh`` used by commit adapters."""

        return await self.refresh(events=events)

    def observe(self) -> AsyncIterator[Frame]:
        """Return a stream whose first frame is the current complete snapshot."""

        if self._closed:
            raise RuntimeError("CommittedView is closed")
        if self._snapshot is None:
            raise RuntimeError("CommittedView.open() must complete before observe()")
        stream = _ObservationStream(self.snapshot(), self.max_pending_frames, self._detach)
        self._observers.add(stream)
        return stream

    async def close(self) -> None:
        """Stop all observers and reject future observation."""

        if self._closed:
            return
        self._closed = True
        for stream in tuple(self._observers):
            stream.close()
        self._observers.clear()

    async def _refresh(
        self,
        *,
        events: Sequence[Mapping[str, JSONValue]] | None,
        publish: bool,
    ) -> Snapshot:
        if self._closed:
            raise RuntimeError("CommittedView is closed")
        normalized_events = _normalize_events(events)
        async with self._refresh_lock:
            candidate = await self._read_consistent()
            previous = self._snapshot
            self._snapshot = candidate
            if publish and previous is not None and candidate["sequence"] != previous["sequence"]:
                frame = _frame(candidate, normalized_events)
                for stream in tuple(self._observers):
                    stream.push(frame)
            return copy.deepcopy(candidate)

    async def _read_consistent(self) -> Snapshot:
        """Read entries and tables around a stable sequence number."""

        read_records = getattr(self.storage, "read_records", None)
        if not callable(read_records):
            entries = await self.storage.read_all()
            return _snapshot_from({}, entries)

        # Storage reads release their own lock between calls.  Re-check the
        # sequence after reading entries so a frame never mixes two commits.
        for _ in range(32):
            before = _records(await read_records())
            entries = await self.storage.read_all()
            after = _records(await read_records())
            if before.get("sequence") == after.get("sequence"):
                return _snapshot_from(after, entries)
        raise RuntimeError("committed storage changed continuously while building a snapshot")

    def _detach(self, stream: _ObservationStream) -> None:
        self._observers.discard(stream)


class _ObservationStream:
    """Async iterator for one observer, with Pi's bounded overflow behavior."""

    def __init__(
        self,
        initial: Snapshot,
        max_pending_frames: int,
        detach: Any,
    ) -> None:
        self._initial = _frame(initial, ())
        self._pending: deque[Frame] = deque()
        self._max_pending_frames = max_pending_frames
        self._detach_callback = detach
        self._wake = asyncio.Event()
        self._started = False
        self._initial_sent = False
        self._closed = False

    def __aiter__(self) -> _ObservationStream:
        return self

    async def __anext__(self) -> Frame:
        if self._closed:
            raise StopAsyncIteration
        if not self._initial_sent:
            self._initial_sent = True
            self._started = True
            return copy.deepcopy(self._initial)
        while True:
            if self._pending:
                return copy.deepcopy(self._pending.popleft())
            if self._closed:
                raise StopAsyncIteration
            await self._wake.wait()
            self._wake.clear()

    def push(self, frame: Frame) -> None:
        if self._closed:
            return
        if len(self._pending) >= self._max_pending_frames:
            self._pending.clear()
            # Overflow is a complete latest snapshot.  Its event batch is
            # intentionally dropped because the snapshot supersedes deltas.
            frame = _frame(cast(Snapshot, frame["snapshot"]), ())
        self._pending.append(copy.deepcopy(frame))
        self._wake.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._detach_callback(self)
        self._wake.set()

    async def aclose(self) -> None:
        self.close()


def _frame(snapshot: Snapshot, events: Sequence[Mapping[str, JSONValue]]) -> Frame:
    frame: Frame = {
        "sequence": cast(int, snapshot["sequence"]),
        "snapshot": copy.deepcopy(snapshot),
    }
    if events:
        frame["events"] = _json_copy(list(events), "events")
    return frame


def _normalize_events(events: Sequence[Mapping[str, JSONValue]] | None) -> list[Event]:
    if events is None:
        return []
    copied = _json_copy(list(events), "events")
    if not isinstance(copied, list) or any(not isinstance(item, dict) for item in copied):
        raise TypeError("events must be a sequence of JSON objects")
    return cast(list[Event], copied)


def _records(value: Any) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise TypeError("read_records() must return a mapping")
    copied = _json_copy(dict(value), "storage records")
    if not isinstance(copied, dict):
        raise TypeError("storage records must be a JSON object")
    return copied


def _snapshot_from(records: Mapping[str, JSONValue], entries: Sequence[SessionEntry]) -> Snapshot:
    encoded_entries: list[JSONValue] = []
    for entry in entries:
        encoded_entries.append(cast(JSONValue, json.loads(entry_to_json_line(entry))))

    sequence = records.get("sequence")
    if type(sequence) is not int or sequence < 0:
        sequence = len(encoded_entries)

    tasks = _live_records(records.get("tasks"))
    documents = _record_list(records.get("documents"), sort=True)
    if not documents:
        documents = _memory_documents(entries)
    submissions = _record_list(records.get("submissions"), sort=True)
    conversations = _record_list(records.get("conversations"), sort=True)
    return {
        "sequence": sequence,
        "entries": encoded_entries,
        "tasks": tasks,
        "documents": documents,
        "submissions": submissions,
        "conversations": conversations,
    }


def _live_records(value: JSONValue | None) -> list[JSONValue]:
    records = _record_list(value, sort=True)
    return [record for record in records if not _is_terminal_record(record)]


def _is_terminal_record(record: JSONValue) -> bool:
    if not isinstance(record, dict):
        return False
    state = record.get("state")
    return isinstance(state, dict) and state.get("status") == "terminal"


def _record_list(value: JSONValue | None, *, sort: bool) -> list[JSONValue]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("storage record tables must be JSON arrays")
    records = [_json_copy(item, "storage record") for item in value]
    if any(not isinstance(record, dict) for record in records):
        raise TypeError("storage record tables must contain JSON objects")
    if sort:
        records.sort(key=_record_sort_key)
    return records


def _record_sort_key(value: JSONValue) -> tuple[int, str]:
    if isinstance(value, dict):
        identifier = value.get("id")
        if type(identifier) is int:
            return identifier, json.dumps(value, sort_keys=True)
    return 0, json.dumps(value, sort_keys=True)


def _memory_documents(entries: Sequence[SessionEntry]) -> list[JSONValue]:
    """Project the latest in-memory live custom entry for print-mode storage."""

    latest: dict[str, tuple[int, dict[str, JSONValue]]] = {}
    for index, entry in enumerate(entries):
        if getattr(entry, "type", None) != "custom":
            continue
        namespace = getattr(entry, "namespace", None)
        data = getattr(entry, "data", None)
        if namespace != "forge.live.v1" or not isinstance(data, dict):
            continue
        latest[namespace] = (index, cast(dict[str, JSONValue], _json_copy(data, "live document")))
    return [
        {
            "record": {
                "id": index,
                "kind": namespace,
                "scope": {"kind": "session"},
            },
            "version": 1,
            "value": value,
        }
        for namespace, (index, value) in sorted(latest.items(), key=lambda item: item[1][0])
    ]


def _json_copy(value: Any, description: str) -> JSONValue:
    try:
        return cast(JSONValue, json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{description} must contain JSON values") from exc


__all__ = [
    "MAX_PENDING_FRAMES",
    "CommitEventQueue",
    "CommittedView",
    "ObservationStorage",
]
