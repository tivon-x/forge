"""D3 observation acceptance against the confirmed JSONL journal.

Failure matrix written before implementation:

* uncommitted input: no refresh means no observer frame;
* terminal race: one commit exposes final documents while terminal tasks leave
  the live task projection;
* late join: the first frame is the latest complete snapshot;
* slow consumer: the 101st pending frame replaces all older frames with the
  latest complete snapshot;
* restart: a new process reconstructs the same confirmed snapshot;
* memory compatibility: ``forge.live.v1`` custom entries project as documents;
* controlled documents: invalid base/delta candidates do not publish, history
  remains readable, and retirement removes the current address;
* lifecycle callbacks: close admission rejects before a write and adoption
  failure is reported after the journal commit;
* queue controls: sequence groups coalesce while ``None`` and exceptions
  survive the snapshot replacement.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from forge_agent import CommittedSnapshotEvent, QueueUpdateEvent
from forge_agent.session import CustomEntry, MessageEntry
from forge_coding.features.documents import DocumentStore
from forge_coding.features.observation import CommitEventQueue, CommittedView
from forge_coding.sessions.journal import Journal
from forge_coding.sessions.storage import JsonlSessionStorage


def _task(task_id: int, status: str = "waiting") -> dict[str, Any]:
    state: dict[str, Any] = {
        "status": status,
        "checkpoint": {"phase": "observe"},
    }
    if status == "waiting":
        state.update(on=[task_id + 1], policy="allSettled")
    if status == "completing":
        state = {"status": "completing", "outcome": {"status": "completed", "result": None}}
    if status == "terminal":
        state = {"status": "terminal", "outcome": {"status": "completed", "result": None}}
    return {
        "id": task_id,
        "kind": "d3.observe",
        "version": 1,
        "input": {"case": "observation"},
        "conversationId": 0,
        "owner": 7,
        "background": True,
        "abortRequested": False,
        "state": state,
    }


def _document(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "document.create",
        "record": {
            "id": 2,
            "kind": "forge.live.v1",
            "scope": {"kind": "session"},
        },
        "content": {"kind": "base", "version": 1, "value": value},
    }


async def _commit_state(
    storage: JsonlSessionStorage, *, value: dict[str, Any], status: str
) -> None:
    records = await storage.read_records()
    writes: list[dict[str, Any]] = []
    if not records["tasks"]:
        writes.append({"type": "task", "value": _task(1, status)})
        writes.append({"type": "task", "value": _task(7, "running")})
    else:
        current = next(item for item in records["tasks"] if item["id"] == 1)
        updated = dict(current)
        updated["state"] = _task(1, status)["state"]
        writes.append({"type": "task", "value": updated})
    if not records["documents"]:
        writes.append(_document(value))
    else:
        writes.append(
            {
                "type": "document.change",
                "id": records["documents"][0]["record"]["id"],
                "content": {"kind": "base", "version": 1, "value": value},
            }
        )
    await storage.commit_records(writes)


def test_observation_publishes_only_confirmed_state_and_live_tasks(tmp_path: Path) -> None:
    asyncio.run(_test_observation_publishes_only_confirmed_state_and_live_tasks(tmp_path))


async def _test_observation_publishes_only_confirmed_state_and_live_tasks(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session")
    await storage.append(MessageEntry(message=HumanMessage(content="confirmed")))
    await _commit_state(storage, value={"assistant": "partial"}, status="waiting")
    view = await CommittedView.open(storage)
    initial = view.snapshot()
    assert initial["sequence"] == 2
    assert len(initial["entries"]) == 1
    assert len(initial["tasks"]) == 2
    assert initial["documents"][0]["value"] == {"assistant": "partial"}

    stream = view.observe()
    first = await anext(stream)
    assert first["snapshot"] == initial

    await _commit_state(storage, value={"assistant": "done"}, status="terminal")
    await view.refresh(events=[{"type": "assistant_delta", "delta": "done"}])
    frame = await anext(stream)
    assert frame["events"] == [{"type": "assistant_delta", "delta": "done"}]
    assert frame["snapshot"]["documents"][0]["value"] == {"assistant": "done"}
    assert [task["id"] for task in frame["snapshot"]["tasks"]] == [7]
    await view.refresh(events=[{"type": "duplicate", "ignored": True}])
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(anext(stream), timeout=0.02)
    await stream.aclose()
    await view.close()
    await storage.close()


def test_failed_commit_does_not_publish_and_late_join_reads_latest(tmp_path: Path) -> None:
    asyncio.run(_test_failed_commit_does_not_publish_and_late_join_reads_latest(tmp_path))


async def _test_failed_commit_does_not_publish_and_late_join_reads_latest(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session")
    await _commit_state(storage, value={"assistant": "before"}, status="running")
    view = await CommittedView.open(storage)
    stream = view.observe()
    await anext(stream)
    before = view.snapshot()

    with pytest.raises(ValueError):
        await storage.commit_records(
            [
                {
                    "type": "document.change",
                    "id": 999,
                    "content": {"kind": "delta", "version": 1, "ops": []},
                }
            ]
        )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(anext(stream), timeout=0.02)

    await _commit_state(storage, value={"assistant": "after"}, status="completing")
    await view.refresh()
    late = view.observe()
    late_frame = await anext(late)
    assert late_frame["snapshot"] == view.snapshot()
    assert late_frame["snapshot"]["sequence"] > before["sequence"]
    assert late_frame["snapshot"]["documents"][0]["value"] == {"assistant": "after"}
    await stream.aclose()
    await late.aclose()
    await view.close()
    await storage.close()


@pytest.mark.anyio
async def test_observation_reads_committed_records_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another adapter's commit must appear in one read of the records and transcript."""
    reader = JsonlSessionStorage(tmp_path / "session")
    writer = JsonlSessionStorage(tmp_path / "session")
    view = await CommittedView.open(reader)
    entry = MessageEntry(message=HumanMessage(content="confirmed"))
    await writer.commit_records([_document({"assistant": "confirmed"})], [entry])
    reads = 0
    original_read = Journal.read_records

    async def count_read(journal: Journal) -> dict[str, Any]:
        nonlocal reads
        reads += 1
        return await original_read(journal)

    monkeypatch.setattr(Journal, "read_records", count_read)
    try:
        snapshot = await view.refresh()
        assert snapshot["sequence"] == 1
        assert snapshot["entries"][0]["id"] == entry.id
        assert snapshot["documents"][0]["value"] == {"assistant": "confirmed"}
        assert reads == 1
    finally:
        await view.close()
        await reader.close()
        await writer.close()


def test_slow_consumer_coalesces_after_100_pending_frames(tmp_path: Path) -> None:
    asyncio.run(_test_slow_consumer_coalesces_after_100_pending_frames(tmp_path))


async def _test_slow_consumer_coalesces_after_100_pending_frames(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session")
    await _commit_state(storage, value={"assistant": "0"}, status="running")
    view = await CommittedView.open(storage)
    stream = view.observe()
    await anext(stream)
    for index in range(101):
        await storage.append(CustomEntry(namespace="d3.note", data={"index": index}))
        await view.refresh(events=[{"type": "note", "index": index}])
    overflow = await anext(stream)
    snapshot = overflow["snapshot"]
    assert overflow.get("events") is None
    assert snapshot["sequence"] == 102
    assert len(snapshot["entries"]) == 101
    await stream.aclose()
    await view.close()
    await storage.close()


def test_memory_custom_live_document_is_projected(tmp_path: Path) -> None:
    asyncio.run(_test_memory_custom_live_document_is_projected())


async def _test_memory_custom_live_document_is_projected() -> None:
    class MemoryStorage:
        def __init__(self) -> None:
            self.entries: list[MessageEntry | CustomEntry] = []

        async def read_all(self) -> list[MessageEntry | CustomEntry]:
            return list(self.entries)

    storage = MemoryStorage()
    storage.entries.append(CustomEntry(namespace="forge.live.v1", data={"assistant": "memory"}))
    view = await CommittedView.open(storage)  # type: ignore[arg-type]
    assert view.snapshot()["documents"][0]["value"] == {"assistant": "memory"}
    await view.close()


def test_observation_survives_real_subprocess_restart(tmp_path: Path) -> None:
    session = tmp_path / "restart"
    result = tmp_path / "result.json"
    driver = Path(__file__).with_name("durable_d3_driver.py")
    for action in ("seed", "resume"):
        subprocess.run(
            [
                sys.executable,
                str(driver),
                action,
                "--session",
                str(session),
                "--result",
                str(result),
            ],
            check=True,
            cwd=Path(__file__).parents[1],
        )
        payload = json.loads(result.read_text(encoding="utf-8"))
        assert payload["snapshot"]["documents"][0]["value"]["assistant"] == "restarted"
    assert payload["action"] == "resume"


def test_controlled_documents_preserve_history_and_reject_invalid_changes(tmp_path: Path) -> None:
    asyncio.run(_test_controlled_documents_preserve_history_and_reject_invalid_changes(tmp_path))


async def _test_controlled_documents_preserve_history_and_reject_invalid_changes(
    tmp_path: Path,
) -> None:
    journal = await Journal.open(tmp_path / "documents")
    published: list[int] = []

    async def on_commit() -> None:
        published.append((await journal.read_records())["sequence"])

    documents = DocumentStore(journal, asyncio.Lock(), on_commit)
    first = await documents.change("d3.history", value={"count": 0}, history="rewindable")
    second = await documents.change(
        "d3.history",
        ops=[{"op": "set", "path": ["count"], "value": 1}],
    )
    current = await documents.get("d3.history")
    historical = await documents.get("d3.history", at=first)
    assert second > first
    assert current is not None and current["value"] == {"count": 1}
    assert historical is not None and historical["value"] == {"count": 0}
    unchanged = await documents.change("d3.history", ops=[])
    assert unchanged == second
    assert len(published) == 2

    with pytest.raises(ValueError):
        await documents.change("d3.invalid", ops=[])
    with pytest.raises(TypeError):
        await documents.change("d3.invalid", value={"bad": object()})
    assert len(published) == 2

    latest = await documents.change("d3.latest", value={"count": 0}, history="latest")
    await documents.change("d3.latest", value={"count": 1})
    with pytest.raises(ValueError):
        await documents.get("d3.latest", at=latest)

    await documents.retire("d3.history")
    assert await documents.get("d3.history") is None
    replacement = await documents.change("d3.history", value={"count": 9}, history="latest")
    historical_replacement = await documents.get("d3.history", at=second)
    assert replacement > second
    assert historical_replacement is not None
    assert historical_replacement["value"] == {"count": 1}
    await documents.change("d3.history", value={"count": 10})
    with pytest.raises(ValueError):
        await documents.get("d3.history", at=replacement)
    await documents.change("d3.keyed", "left", value={"name": "left"})
    await documents.change("d3.keyed", "right", value={"name": "right"})
    left = await documents.get("d3.keyed", "left")
    assert left is not None and left["value"] == {"name": "left"}
    assert await documents.get("d3.history", at=0) is None
    await journal.close()


def test_document_callbacks_guard_close_and_report_adoption_failure(tmp_path: Path) -> None:
    asyncio.run(_test_document_callbacks_guard_close_and_report_adoption_failure(tmp_path))


async def _test_document_callbacks_guard_close_and_report_adoption_failure(
    tmp_path: Path,
) -> None:
    journal = await Journal.open(tmp_path / "callbacks")
    closed = False
    failures: list[BaseException] = []

    def before_change() -> None:
        if closed:
            raise RuntimeError("closed")

    def on_error(error: BaseException) -> None:
        failures.append(error)

    documents = DocumentStore(
        journal,
        asyncio.Lock(),
        before_change=before_change,
        on_error=on_error,
    )
    await documents.change("d3.callbacks", value={"ok": True})
    with pytest.raises(ValueError):
        await documents.change("d3.invalid", ops=[])
    assert failures == []

    async def fail_adoption() -> None:
        raise RuntimeError("adoption failed")

    documents.on_commit = fail_adoption
    with pytest.raises(RuntimeError, match="adoption failed"):
        await documents.change("d3.adopt", value={"ok": True})
    assert [str(error) for error in failures] == ["adoption failed"]

    closed = True
    with pytest.raises(RuntimeError, match="closed"):
        await documents.change("d3.closed", value={"ok": True})
    await journal.close()


def test_commit_event_queue_coalesces_frames_and_keeps_controls() -> None:
    snapshot = {"sequence": 101, "documents": [], "tasks": []}
    queue = CommitEventQueue(lambda: snapshot)
    for sequence in range(99):
        queue.put_nowait((7, QueueUpdateEvent(), sequence))
    queue.put_nowait((7, QueueUpdateEvent(), 99))
    queue.put_nowait((7, QueueUpdateEvent(), 99))
    assert queue.qsize() == 100

    queue.put_nowait((7, None, 100))
    failure = RuntimeError("publication failed")
    queue.put_nowait((7, failure, 100))
    assert queue.qsize() == 100
    queue.put_nowait((7, QueueUpdateEvent(), 101))
    assert queue.qsize() == 1
    task_id, event = queue.get_nowait()
    assert task_id == 7 and isinstance(event, CommittedSnapshotEvent)
    assert event.snapshot == snapshot
    controls = [queue.get_nowait(), queue.get_nowait()]
    assert controls[0] == (7, None)
    assert controls[1][0] == 7 and controls[1][1] is failure
    assert queue.empty()


@pytest.mark.parametrize("action", ["new", "resume"])
def test_session_replacement_closes_previous_observer(tmp_path, monkeypatch, action):
    from conftest import isolate_home
    from fake_models import ScriptedChatModel
    from forge_coding import CodingSession, CodingSessionConfig
    from forge_coding.paths import ForgePaths
    from forge_coding.sessions.manager import SessionManager

    isolate_home(monkeypatch, tmp_path)

    async def run():
        manager = SessionManager(
            ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
        )
        record = manager.create_session(cwd=tmp_path, model="fake", provider_name="fake")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=ScriptedChatModel(),
                model="fake",
                system="offline",
                cwd=tmp_path,
                storage=JsonlSessionStorage(record.path),
                session_id=record.id,
                session_manager=manager,
                enable_subagents=False,
            )
        )
        observer = session.observe()
        await anext(observer)
        try:
            if action == "new":
                await session.new_session()
            else:
                target = manager.create_session(cwd=tmp_path, model="fake")
                await session.resume(target.id)
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(observer), 0.5)
            current = session.observe()
            assert (await anext(current))["snapshot"] == session.snapshot()
            await current.aclose()
        finally:
            await observer.aclose()
            await session.aclose()

    asyncio.run(run())


def test_slow_consumer_retains_committed_questionnaire_control():
    from forge_agent import HumanInputRequestedEvent

    question = HumanInputRequestedEvent(
        interrupt_id="offline",
        tool_call_id="q",
        tool_name="ask_user_question",
        arguments={"questions": []},
    )
    snapshot = {"sequence": 101, "documents": [], "tasks": []}
    queue = CommitEventQueue(lambda: snapshot)
    for sequence in range(1, 101):
        queue.put_nowait((7, QueueUpdateEvent(), sequence))
    queue.put_nowait((7, question, 101))
    assert queue.qsize() <= 100
    delivered = []
    while not queue.empty():
        _, event = queue.get_nowait()
        delivered.append(event)
    assert question in delivered
    # Requests already pending must also survive a later snapshot replacement.
    for sequence in range(102, 202):
        queue.put_nowait((7, QueueUpdateEvent(), sequence))
    queue.put_nowait((7, question, 202))
    for sequence in range(203, 304):
        snapshot["sequence"] = sequence
        queue.put_nowait((7, QueueUpdateEvent(), sequence))
    delivered = []
    while not queue.empty():
        _, event = queue.get_nowait()
        delivered.append(event)
    assert question in delivered


def test_memory_session_close_ends_committed_observer(tmp_path, monkeypatch):
    from conftest import isolate_home
    from fake_models import ScriptedChatModel
    from forge_coding import CodingSession, CodingSessionConfig

    class MemoryStorage:
        def __init__(self):
            self.entries = []

        async def read_all(self):
            return self.entries[:]

        async def commit(self, entries):
            self.entries.extend(entries)
            return len(self.entries)

        async def append(self, entry):
            return await self.commit([entry])

    isolate_home(monkeypatch, tmp_path)

    async def run():
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=ScriptedChatModel(),
                model="fake",
                system="offline",
                cwd=tmp_path,
                storage=MemoryStorage(),
                enable_subagents=False,
            )
        )
        observer = session.observe()
        await anext(observer)
        await session.aclose()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(observer), 0.5)
        await observer.aclose()

    asyncio.run(run())
