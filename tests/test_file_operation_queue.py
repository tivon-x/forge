from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from forge_coding.tools.file_operation_queue import FileOperationQueue


@pytest.mark.anyio
async def test_same_path_operations_are_fifo_and_non_overlapping(tmp_path: Path) -> None:
    queue = FileOperationQueue()
    path = tmp_path / "file.txt"
    events: list[str] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with queue.operation(path):
            events.append("first-start")
            first_started.set()
            await release_first.wait()
            events.append("first-end")

    async def second() -> None:
        await first_started.wait()
        async with queue.operation(path):
            events.append("second-start")
            events.append("second-end")

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await first_started.wait()
    await asyncio.sleep(0)
    assert events == ["first-start"]
    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert events == ["first-start", "first-end", "second-start", "second-end"]
    assert queue.path_state_count == 0


@pytest.mark.anyio
async def test_different_paths_can_overlap(tmp_path: Path) -> None:
    queue = FileOperationQueue()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold(path: Path) -> None:
        async with queue.operation(path):
            entered.set()
            await release.wait()

    first = asyncio.create_task(hold(tmp_path / "a"))
    await entered.wait()
    second = asyncio.create_task(hold(tmp_path / "b"))
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    await asyncio.gather(first, second)
    assert queue.path_state_count == 0


@pytest.mark.anyio
async def test_cancelled_waiter_and_failed_owner_release_queue(tmp_path: Path) -> None:
    queue = FileOperationQueue()
    path = tmp_path / "file.txt"
    entered = asyncio.Event()
    release = asyncio.Event()

    async def owner() -> None:
        with pytest.raises(RuntimeError):
            async with queue.operation(path):
                entered.set()
                await release.wait()
                raise RuntimeError("boom")

    owner_task = asyncio.create_task(owner())
    await entered.wait()
    waiter = asyncio.create_task(_touch(queue, path))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await owner_task
    assert queue.path_state_count == 0


async def _touch(queue: FileOperationQueue, path: Path) -> None:
    async with queue.operation(path):
        return


@pytest.mark.anyio
async def test_wait_setup_failure_does_not_leak_the_path_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FileOperationQueue()
    path = tmp_path / "executor-shutdown.txt"
    owner_entered = asyncio.Event()
    release_owner = asyncio.Event()

    async def owner() -> None:
        async with queue.operation(path):
            owner_entered.set()
            await release_owner.wait()

    owner_task = asyncio.create_task(owner())
    await owner_entered.wait()

    async def fail_to_thread(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Executor shutdown has been called")

    monkeypatch.setattr(asyncio, "to_thread", fail_to_thread)
    with pytest.raises(RuntimeError, match="Executor shutdown"):
        await _touch(queue, path)

    release_owner.set()
    await owner_task
    assert queue.path_state_count == 0

    async with queue.operation(path):
        pass
    assert queue.path_state_count == 0


def test_queue_can_be_used_by_independent_event_loops(tmp_path: Path) -> None:
    queue = FileOperationQueue()

    async def use() -> None:
        async with queue.operation(tmp_path / "loop.txt"):
            pass

    asyncio.run(use())
    asyncio.run(use())
    assert queue.path_state_count == 0


def test_same_path_operations_are_serialized_across_event_loops(tmp_path: Path) -> None:
    queue = FileOperationQueue()
    path = tmp_path / "threaded.txt"
    owner_entered = threading.Event()
    release_owner = threading.Event()
    waiter_entered = threading.Event()
    events: list[str] = []
    errors: list[BaseException] = []
    events_lock = threading.Lock()

    def record(event: str) -> None:
        with events_lock:
            events.append(event)

    async def owner() -> None:
        async with queue.operation(path):
            record("owner-start")
            owner_entered.set()
            await asyncio.to_thread(release_owner.wait)
            record("owner-end")

    async def waiter() -> None:
        async with queue.operation(path):
            record("waiter-start")
            waiter_entered.set()
            record("waiter-end")

    def run(coroutine: object) -> None:
        try:
            asyncio.run(coroutine)  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001 - surface thread failures below
            errors.append(exc)

    owner_thread = threading.Thread(target=run, args=(owner(),), daemon=True)
    owner_thread.start()
    assert owner_entered.wait(timeout=2)

    waiter_thread = threading.Thread(target=run, args=(waiter(),), daemon=True)
    waiter_thread.start()
    assert not waiter_entered.wait(timeout=0.05)

    release_owner.set()
    owner_thread.join(timeout=2)
    waiter_thread.join(timeout=2)

    assert not owner_thread.is_alive()
    assert not waiter_thread.is_alive()
    assert errors == []
    assert events == ["owner-start", "owner-end", "waiter-start", "waiter-end"]
    assert queue.path_state_count == 0
