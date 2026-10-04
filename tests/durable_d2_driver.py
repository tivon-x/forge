"""Subprocess driver for the D2 durable-task acceptance contract.

The driver is intentionally small and process oriented.  Every action receives
an isolated Journal directory, records controlled effects beside it, and only
uses offline task definitions.  Crash actions call ``os._exit`` after the
durable boundary they name; the parent test then opens the same directory in a
fresh interpreter.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _result_path(args: argparse.Namespace) -> Path:
    return Path(args.result) if args.result else Path(args.session) / "d2-driver-result.json"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _effect_path(session: Path) -> Path:
    return session / "d2-effects.jsonl"


def _effect(session: Path, event: str, **fields: Any) -> None:
    payload = {"event": event, "pid": os.getpid(), **fields}
    path = _effect_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_effects(session: Path) -> list[dict[str, Any]]:
    path = _effect_path(session)
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _task_id(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("id", value.get("task_id"))
    return getattr(value, "id", value)


def _state(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("state", value)
    return getattr(value, "state", value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    for method in ("model_dump", "to_dict"):
        converter = getattr(value, method, None)
        if callable(converter):
            try:
                return _jsonable(converter())
            except TypeError:
                pass
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return repr(value)


async def _maybe_await[T](value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


async def _open(session: Path, definitions: Sequence[Any]) -> tuple[Any, Any]:
    """Open the D2 coordinator on the coding-owned Journal."""

    from forge_coding.features.tasks import TaskCoordinator
    from forge_coding.sessions.journal import Journal

    storage = await Journal.open(session)
    opened = TaskCoordinator.open(storage, definitions)
    coordinator = await _maybe_await(opened)
    return storage, coordinator


def _definition(
    kind: str,
    version: int,
    run: Callable[..., Any],
    abort: Callable[..., Any],
    migrate: Callable[..., Any] | None = None,
) -> Any:
    from forge_coding.features.tasks import TaskDefinition

    values: dict[str, Any] = {
        "kind": kind,
        "version": version,
        "run": run,
        "abort": abort,
    }
    if migrate is not None:
        values["migrate"] = migrate
    return TaskDefinition(**values)


async def _call_create(
    coordinator: Any,
    kind: str,
    input_value: Any,
    checkpoint: Any,
    *,
    owner: Any = None,
    background: bool = False,
) -> Any:
    kwargs: dict[str, Any] = {"owner": owner, "background": background}
    created = coordinator.create(kind, input_value, checkpoint, **kwargs)
    return await _maybe_await(created)


async def _resume(coordinator: Any) -> Any:
    return await _maybe_await(coordinator.resume())


async def _wait(coordinator: Any, task_id: Any, timeout: float | None = None) -> Any:
    return await _maybe_await(coordinator.wait(task_id, timeout=timeout))


async def _abort(coordinator: Any, task_id: Any, *, include_background: bool = False) -> Any:
    return await _maybe_await(coordinator.abort(task_id, include_background=include_background))


async def _inspect(coordinator: Any) -> Any:
    return await _maybe_await(coordinator.inspect())


async def _close(coordinator: Any, storage: Any) -> None:
    close = getattr(coordinator, "close", None)
    if close is not None:
        await _maybe_await(close())
    storage_close = getattr(storage, "close", None)
    if storage_close is not None:
        await _maybe_await(storage_close())


def _task_record(records: Mapping[str, Any], task_id: Any) -> Any:
    for record in records.get("tasks", ()):
        if _task_id(record) == task_id:
            return record
    return None


async def _records(storage: Any) -> dict[str, Any]:
    read = getattr(storage, "read_records", None)
    if read is None:
        return {}
    return _jsonable(await _maybe_await(read()))


async def _run_no_progress(args: argparse.Namespace) -> int:
    session = Path(args.session)

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "run", task_id=_task_id(task))

    async def abort(task: Any, runtime: Any) -> None:
        _effect(session, "abort", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definition = _definition("d2.no-progress", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    created = await _call_create(
        coordinator, "d2.no-progress", {"case": "no-progress"}, {"phase": "run"}
    )
    task_id = _task_id(created)
    _write_json(session / "task-id.json", {"task_id": task_id})
    await _resume(coordinator)
    record = await _wait(coordinator, task_id, timeout=5)
    _write_json(
        _result_path(args),
        {
            "action": args.action,
            "task_id": task_id,
            "record": _jsonable(record),
            "state": _jsonable(_state(record)),
            "records": await _records(storage),
            "effects": _read_effects(session),
            "pid": os.getpid(),
        },
    )
    await _close(coordinator, storage)
    return 0


async def _run_crash_after_checkpoint(args: argparse.Namespace) -> int:
    session = Path(args.session)

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "run-start", task_id=_task_id(task))
        await _maybe_await(runtime.checkpoint({"phase": "resume"}))
        _effect(session, "checkpoint-committed", task_id=_task_id(task))
        _write_json(
            session / "task-id.json",
            {"task_id": _task_id(task), "checkpoint": {"phase": "resume"}},
        )
        os._exit(args.exit_code)

    async def abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definition = _definition("d2.restartable", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    created = await _call_create(
        coordinator, "d2.restartable", {"case": "restart"}, {"phase": "start"}
    )
    _write_json(session / "task-id.json", {"task_id": _task_id(created)})
    await _resume(coordinator)
    await asyncio.sleep(30)
    return 2


async def _run_reopen_inspect(args: argparse.Namespace) -> int:
    session = Path(args.session)
    task_id = json.loads((session / "task-id.json").read_text(encoding="utf-8"))["task_id"]

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "reopen-run", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"resumed": True}))

    async def abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definition = _definition("d2.restartable", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    inspection = await _inspect(coordinator)
    records = await _records(storage)
    record = _task_record(records, task_id)
    _write_json(
        _result_path(args),
        {
            "action": args.action,
            "task_id": task_id,
            "inspection": _jsonable(inspection),
            "record": record,
            "effects": _read_effects(session),
            "records": records,
            "pid": os.getpid(),
        },
    )
    await _close(coordinator, storage)
    return 0


async def _run_resume_after_restart(args: argparse.Namespace) -> int:
    session = Path(args.session)
    task_id = json.loads((session / "task-id.json").read_text(encoding="utf-8"))["task_id"]

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "resume-run", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"resumed": True}))

    async def abort(task: Any, runtime: Any) -> None:
        _effect(session, "abort", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definition = _definition("d2.restartable", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    await _resume(coordinator)
    record = await _wait(coordinator, task_id, timeout=5)
    _write_json(
        _result_path(args),
        {
            "action": args.action,
            "task_id": task_id,
            "record": _jsonable(record),
            "effects": _read_effects(session),
            "records": await _records(storage),
            "pid": os.getpid(),
        },
    )
    await _close(coordinator, storage)
    return 0


async def _run_abort_crash(args: argparse.Namespace) -> int:
    session = Path(args.session)
    ready = asyncio.Event()

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "run", task_id=_task_id(task))
        _write_json(session / "running.json", {"task_id": _task_id(task)})
        ready.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _effect(session, "run-cancelled", task_id=_task_id(task))
            raise

    async def abort(task: Any, runtime: Any) -> None:
        _effect(session, "abort-start", task_id=_task_id(task))
        _write_json(session / "abort-stage.json", {"task_id": _task_id(task), "stage": "handler"})
        os._exit(args.exit_code)

    definition = _definition("d2.abort-restart", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    created = await _call_create(
        coordinator, "d2.abort-restart", {"case": "abort"}, {"phase": "run"}
    )
    task_id = _task_id(created)
    _write_json(session / "task-id.json", {"task_id": task_id})
    await _resume(coordinator)
    for _ in range(500):
        if ready.is_set():
            break
        await asyncio.sleep(0.01)
    if not ready.is_set():
        raise RuntimeError("abort crash task did not start")
    await _abort(coordinator, task_id)
    # The abort mark is durable before the handler is dispatched. Keep this
    # process alive until the crash point in the handler is reached.
    for _ in range(500):
        if any(item.get("event") == "abort-start" for item in _read_effects(session)):
            await asyncio.sleep(30)
            return 2
        await asyncio.sleep(0.01)
    raise RuntimeError("abort handler did not start")


async def _run_reopen_abort(args: argparse.Namespace) -> int:
    session = Path(args.session)
    task_id = json.loads((session / "task-id.json").read_text(encoding="utf-8"))["task_id"]

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "unexpected-run", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"resumed": True}))

    async def abort(task: Any, runtime: Any) -> None:
        _effect(session, "abort-restart", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"reason": "recovered"}))

    definition = _definition("d2.abort-restart", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    await _resume(coordinator)
    record = await _wait(coordinator, task_id, timeout=5)
    _write_json(
        _result_path(args),
        {
            "action": args.action,
            "task_id": task_id,
            "record": _jsonable(record),
            "effects": _read_effects(session),
            "records": await _records(storage),
            "pid": os.getpid(),
        },
    )
    await _close(coordinator, storage)
    return 0


async def _run_migration_prepare(args: argparse.Namespace) -> int:
    session = Path(args.session)

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "old-run", task_id=_task_id(task))

    async def abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definition = _definition("d2.versioned", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    created = await _call_create(coordinator, "d2.versioned", {"version": 1}, {"phase": "run"})
    _write_json(session / "task-id.json", {"task_id": _task_id(created)})
    await _close(coordinator, storage)
    _write_json(
        _result_path(args),
        {"action": args.action, "task_id": _task_id(created), "pid": os.getpid()},
    )
    return 0


async def _run_migration(args: argparse.Namespace, *, fail: bool) -> int:
    session = Path(args.session)
    task_id = json.loads((session / "task-id.json").read_text(encoding="utf-8"))["task_id"]
    migrations = {"count": 0}

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "new-run", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"version": 2}))

    async def abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    def migrate(input_value: Any, checkpoint: Any, from_version: int) -> dict[str, Any]:
        migrations["count"] += 1
        _effect(session, "migration", from_version=from_version)
        if fail:
            raise RuntimeError("D2 migration failure")
        return {"input": {"version": 2}, "checkpoint": checkpoint}

    definition = _definition("d2.versioned", 2, run, abort, migrate)
    storage, coordinator = await _open(session, [definition])
    await _resume(coordinator)
    await asyncio.sleep(0.1)
    records = await _records(storage)
    record = _task_record(records, task_id)
    if fail:
        inspection = await _inspect(coordinator)
        payload = {
            "action": args.action,
            "task_id": task_id,
            "record": record,
            "inspection": _jsonable(inspection),
            "migration_count": migrations["count"],
            "effects": _read_effects(session),
            "records": records,
            "pid": os.getpid(),
        }
    else:
        settled = await _wait(coordinator, task_id, timeout=5)
        payload = {
            "action": args.action,
            "task_id": task_id,
            "record": _jsonable(settled),
            "migration_count": migrations["count"],
            "effects": _read_effects(session),
            "records": await _records(storage),
            "pid": os.getpid(),
        }
    _write_json(_result_path(args), payload)
    await _close(coordinator, storage)
    return 0


async def _run_orphan_prepare(args: argparse.Namespace) -> int:
    session = Path(args.session)

    async def run(task: Any, runtime: Any) -> None:
        _effect(session, "unexpected-orphan-run", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"unexpected": True}))

    async def abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definition = _definition("d2.orphan", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    created = await _call_create(coordinator, "d2.orphan", {"case": "orphan"}, {"phase": "run"})
    task_id = _task_id(created)
    _write_json(session / "task-id.json", {"task_id": task_id})
    await _close(coordinator, storage)
    _write_json(_result_path(args), {"action": args.action, "task_id": task_id, "pid": os.getpid()})
    return 0


async def _run_orphan_abort(args: argparse.Namespace) -> int:
    session = Path(args.session)
    task_id = json.loads((session / "task-id.json").read_text(encoding="utf-8"))["task_id"]
    storage, coordinator = await _open(session, [])
    await _resume(coordinator)
    blocked = await _inspect(coordinator)
    await _abort(coordinator, task_id)
    record = await _wait(coordinator, task_id, timeout=5)
    payload = {
        "action": args.action,
        "task_id": task_id,
        "blocked": _jsonable(blocked),
        "record": _jsonable(record),
        "effects": _read_effects(session),
        "records": await _records(storage),
        "pid": os.getpid(),
    }
    _write_json(_result_path(args), payload)
    await _close(coordinator, storage)
    return 0


async def _run_abort_tree_crash(args: argparse.Namespace) -> int:
    session = Path(args.session)
    owner_started = asyncio.Event()
    child_started = asyncio.Event()

    async def owner_run(task: Any, runtime: Any) -> None:
        _effect(session, "owner-run", task_id=_task_id(task))
        owner_started.set()
        await asyncio.Event().wait()

    async def child_run(task: Any, runtime: Any) -> None:
        _effect(session, "child-run", task_id=_task_id(task))
        child_started.set()
        await asyncio.Event().wait()

    async def owner_abort(task: Any, runtime: Any) -> None:
        _effect(session, "owner-abort-start", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"reason": "owner-aborted"}))

    async def child_abort(task: Any, runtime: Any) -> None:
        _effect(session, "child-abort-start", task_id=_task_id(task))
        os._exit(args.exit_code)

    definitions = [
        _definition("d2.abort-owner", 1, owner_run, owner_abort),
        _definition("d2.abort-child", 1, child_run, child_abort),
    ]
    storage, coordinator = await _open(session, definitions)
    owner = await _call_create(coordinator, "d2.abort-owner", {}, {"phase": "run"})
    owner_id = _task_id(owner)
    child = await _call_create(
        coordinator,
        "d2.abort-child",
        {},
        {"phase": "run"},
        owner=owner_id,
    )
    child_id = _task_id(child)
    _write_json(session / "task-ids.json", {"owner": owner_id, "child": child_id})
    await _resume(coordinator)
    for _ in range(500):
        if owner_started.is_set() and child_started.is_set():
            break
        await asyncio.sleep(0.01)
    if not (owner_started.is_set() and child_started.is_set()):
        raise RuntimeError("abort tree did not start")
    await _abort(coordinator, owner_id)
    for _ in range(500):
        if any(item.get("event") == "child-abort-start" for item in _read_effects(session)):
            await asyncio.sleep(30)
            return 2
        await asyncio.sleep(0.01)
    raise RuntimeError("child abort handler did not start")


async def _run_abort_tree_reopen(args: argparse.Namespace) -> int:
    session = Path(args.session)
    ids = json.loads((session / "task-ids.json").read_text(encoding="utf-8"))

    async def owner_run(task: Any, runtime: Any) -> None:
        _effect(session, "unexpected-owner-run", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"unexpected": True}))

    async def child_run(task: Any, runtime: Any) -> None:
        _effect(session, "unexpected-child-run", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"unexpected": True}))

    async def owner_abort(task: Any, runtime: Any) -> None:
        _effect(session, "owner-abort-restart", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"reason": "owner-restarted"}))

    async def child_abort(task: Any, runtime: Any) -> None:
        _effect(session, "child-abort-restart", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"reason": "child-restarted"}))

    definitions = [
        _definition("d2.abort-owner", 1, owner_run, owner_abort),
        _definition("d2.abort-child", 1, child_run, child_abort),
    ]
    storage, coordinator = await _open(session, definitions)
    await _resume(coordinator)
    owner_record = await _wait(coordinator, ids["owner"], timeout=5)
    child_record = await _wait(coordinator, ids["child"], timeout=5)
    payload = {
        "action": args.action,
        "owner": _jsonable(owner_record),
        "child": _jsonable(child_record),
        "effects": _read_effects(session),
        "records": await _records(storage),
        "pid": os.getpid(),
    }
    _write_json(_result_path(args), payload)
    await _close(coordinator, storage)
    return 0


async def _run_completing_crash(args: argparse.Namespace) -> int:
    session = Path(args.session)
    child_started = asyncio.Event()

    async def owner_run(task: Any, runtime: Any) -> None:
        _effect(session, "owner-completing", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"owner": "held"}))
        os._exit(args.exit_code)

    async def child_run(task: Any, runtime: Any) -> None:
        _effect(session, "child-running", task_id=_task_id(task))
        child_started.set()
        await asyncio.Event().wait()

    async def abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definitions = [
        _definition("d2.completing-owner", 1, owner_run, abort),
        _definition("d2.completing-child", 1, child_run, abort),
    ]
    storage, coordinator = await _open(session, definitions)
    owner = await _call_create(coordinator, "d2.completing-owner", {}, {"phase": "run"})
    owner_id = _task_id(owner)
    child = await _call_create(
        coordinator,
        "d2.completing-child",
        {},
        {"phase": "run"},
        owner=owner_id,
    )
    _write_json(session / "task-ids.json", {"owner": owner_id, "child": _task_id(child)})
    await _resume(coordinator)
    for _ in range(500):
        if child_started.is_set():
            await asyncio.sleep(30)
            return 2
        await asyncio.sleep(0.01)
    raise RuntimeError("completing child did not start")


async def _run_completing_reopen(args: argparse.Namespace) -> int:
    session = Path(args.session)
    ids = json.loads((session / "task-ids.json").read_text(encoding="utf-8"))

    async def owner_run(task: Any, runtime: Any) -> None:
        _effect(session, "unexpected-owner-rerun", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"unexpected": True}))

    async def child_run(task: Any, runtime: Any) -> None:
        _effect(session, "child-resume", task_id=_task_id(task))
        await _maybe_await(runtime.complete({"child": "settled"}))

    async def abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definitions = [
        _definition("d2.completing-owner", 1, owner_run, abort),
        _definition("d2.completing-child", 1, child_run, abort),
    ]
    storage, coordinator = await _open(session, definitions)
    await _resume(coordinator)
    owner_record = await _wait(coordinator, ids["owner"], timeout=5)
    child_record = await _wait(coordinator, ids["child"], timeout=5)
    payload = {
        "action": args.action,
        "owner": _jsonable(owner_record),
        "child": _jsonable(child_record),
        "effects": _read_effects(session),
        "records": await _records(storage),
        "pid": os.getpid(),
    }
    _write_json(_result_path(args), payload)
    await _close(coordinator, storage)
    return 0


async def _run_stale_handle(args: argparse.Namespace) -> int:
    session = Path(args.session)
    reached = asyncio.Event()
    release = asyncio.Event()
    holder: dict[str, Any] = {}

    async def run(task: Any, runtime: Any) -> None:
        holder["runtime"] = runtime
        _effect(session, "handle-created", task_id=_task_id(task))
        reached.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            _effect(session, "run-cancelled", task_id=_task_id(task))
            # Keep the old callback alive long enough to prove the coordinator gate.
            await release.wait()
        try:
            await _maybe_await(holder["runtime"].complete({"late": True}))
        except BaseException as exc:  # noqa: BLE001 - persisted contract evidence
            _effect(session, "stale-rejected", error=type(exc).__name__, message=str(exc))

    async def abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"reason": "aborted"}))

    definition = _definition("d2.stale", 1, run, abort)
    storage, coordinator = await _open(session, [definition])
    created = await _call_create(coordinator, "d2.stale", {"case": "stale"}, {"phase": "run"})
    task_id = _task_id(created)
    await _resume(coordinator)
    for _ in range(500):
        if reached.is_set():
            break
        await asyncio.sleep(0.01)
    if not reached.is_set():
        raise RuntimeError("stale-handle task did not start")
    closing = asyncio.create_task(_close(coordinator, storage))
    await asyncio.sleep(0.05)
    release.set()
    await closing
    _write_json(
        _result_path(args),
        {
            "action": args.action,
            "task_id": task_id,
            "effects": _read_effects(session),
            "pid": os.getpid(),
        },
    )
    return 0


async def _run_loop_shutdown(args: argparse.Namespace) -> int:
    """Leave one storage reader waiting while ``asyncio.run`` closes the loop."""

    from forge_coding.sessions.storage import JsonlSessionStorage, _path_lock

    session = Path(args.session)
    session.mkdir(parents=True, exist_ok=True)
    storage = JsonlSessionStorage(session / "main.jsonl")
    path_lock = _path_lock(session)
    if not path_lock.acquire(blocking=False):
        raise RuntimeError("loop-shutdown path lock was already held")
    pending = asyncio.create_task(storage.read_records(), name="d2-storage-waiter")
    try:
        await asyncio.sleep(0.05)
        _write_json(
            _result_path(args),
            {
                "action": args.action,
                "pid": os.getpid(),
                "lock_held": True,
                "pending": not pending.done(),
            },
        )
        # Returning with the waiter alive exercises the real asyncio/AnyIO
        # shutdown cancellation path in the subprocess.
        return 0
    finally:
        path_lock.release()


async def _run_sdk_submissions(args: argparse.Namespace) -> int:
    """Exercise the public CodingSession submission receipt in isolation."""

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    from forge_coding import CodingSession, CodingSessionConfig, ForgePaths, ForgeResourcePaths
    from forge_coding.sessions.storage import JsonlSessionStorage

    class ProbeModel(BaseChatModel):
        def __init__(
            self,
            releases: list[asyncio.Event],
            *,
            fail_indexes: set[int] | None = None,
            tool_indexes: set[int] | None = None,
        ) -> None:
            super().__init__()
            object.__setattr__(self, "releases", releases)
            object.__setattr__(self, "fail_indexes", fail_indexes or set())
            object.__setattr__(self, "tool_indexes", tool_indexes or set())
            object.__setattr__(self, "started", [asyncio.Event() for _ in releases])
            object.__setattr__(self, "calls", [])
            object.__setattr__(self, "call_index", 0)

        @property
        def _llm_type(self) -> str:
            return "forge-d2-submission-probe"

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
            del tools, tool_choice, kwargs
            return self

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            del stop, run_manager, kwargs
            index = int(self.call_index)
            object.__setattr__(self, "call_index", index + 1)
            self.calls.append(list(messages))
            self.started[index].set()
            await self.releases[index].wait()
            if index in self.fail_indexes:
                raise RuntimeError("D2 controlled model failure")
            if index in self.tool_indexes:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_calls=[
                            {
                                "name": "effect",
                                "args": {},
                                "id": "effect-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
                return
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"answer-{index}"))

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            del messages, stop, run_manager, kwargs
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="fallback"))])

    async def load_model_session(
        root: Path, model: BaseChatModel, *, tools: Sequence[Any] = ()
    ) -> tuple[Any, Any]:
        workspace = root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        forge_home = root / "forge-home"
        agents_home = root / "agents-home"
        paths = ForgePaths(home=forge_home, agents_home=agents_home)
        resources = ForgeResourcePaths(
            root=forge_home,
            cwd=workspace,
            agents_root=agents_home,
            paths=paths,
        )
        storage = JsonlSessionStorage(root / "native")
        coding = await CodingSession.load(
            CodingSessionConfig(
                provider=model,
                model="fake",
                system="D2 offline submission acceptance",
                storage=storage,
                cwd=workspace,
                resource_paths=resources,
                tools=tools,
                enable_subagents=False,
                auto_compact_enabled=False,
            )
        )
        return coding, storage

    session = Path(args.session)
    releases = [asyncio.Event() for _ in range(4)]
    model = ProbeModel(releases)
    coding, storage = await load_model_session(session, model)
    failure_root = session / "failure-retention"
    failure_releases = [asyncio.Event()]
    failure_model = ProbeModel(failure_releases, fail_indexes={0})
    failing, failure_storage = await load_model_session(failure_root, failure_model)
    writer: Any | None = None
    tooling: Any | None = None
    type_conflict = False
    busy_reject = False
    observer_timeout = False
    observer_cancelled = False
    try:
        first = await coding.submit("first", request_id="same-request")
        duplicate = await coding.submit("different content", request_id="same-request")
        _effect(session, "submit-deduplicated", id=first.id, duplicate_id=duplicate.id)
        retry = await coding.submit(
            "ignored retry payload", request_id="same-request", when_busy="invalid-retry-mode"
        )
        assert retry.id == first.id
        try:
            await coding.submit("write payload", request_id="same-request", type="write")
        except ValueError:
            type_conflict = True
        await asyncio.wait_for(model.started[0].wait(), timeout=5)
        _effect(session, "first-model-started")
        steer = await coding.submit("steer", request_id="steer-request", when_busy="steer")
        second = await coding.submit("second", request_id="second-request", when_busy="follow_up")
        third = await coding.submit("third", request_id="third-request", when_busy="follow_up")
        try:
            await coding.submit("reject me", when_busy="reject")
        except RuntimeError:
            busy_reject = True
        _effect(session, "queued-inputs", steer_id=steer.id, second_id=second.id, third_id=third.id)
        try:
            await third.wait(timeout=0.01)
        except TimeoutError:
            observer_timeout = True
        queued_before = await third.status()
        releases[0].set()
        await asyncio.wait_for(model.started[1].wait(), timeout=5)
        _effect(session, "steer-model-started")
        placed_after_first = await steer.status()
        queued_after_first = await second.status()
        queued_after_second = await third.status()
        observer = asyncio.create_task(second.wait())
        await asyncio.sleep(0)
        observer.cancel()
        try:
            await observer
        except asyncio.CancelledError:
            observer_cancelled = True
        releases[1].set()
        await asyncio.wait_for(model.started[2].wait(), timeout=5)
        _effect(session, "follow-up-model-started", index=2)
        placed_after_second = await second.status()
        queued_after_second = await third.status()
        placed_after_third = await third.status()
        releases[2].set()
        await first.wait(timeout=5)
        await steer.wait(timeout=5)
        await second.wait(timeout=5)
        await third.wait(timeout=5)
        settled = {
            "first": await first.status(),
            "steer": await steer.status(),
            "second": await second.status(),
            "third": await third.status(),
        }
        await coding.aclose()
        coding, storage = await load_model_session(session, ProbeModel([asyncio.Event()]))
        reopened_receipt = await coding.submission(first.id)
        if reopened_receipt is None:
            raise RuntimeError("reopened submission receipt is missing")
        reopened_receipt_status = await reopened_receipt.status()

        _effect(failure_root, "failure-submit-start")
        failed = await failing.submit("will fail", request_id="failed")
        _effect(failure_root, "failure-submit-done", id=failed.id)
        await asyncio.wait_for(failure_model.started[0].wait(), timeout=5)
        _effect(failure_root, "failure-model-started")
        retained = await failing.submit("retain after failure", request_id="retained")
        failure_releases[0].set()
        failed_result = await failed.wait(timeout=5)
        retained_result = await retained.status()
        failure_records = await failure_storage.read_records()
        failure_inbox = [
            _jsonable(document.get("value", {}))
            for document in failure_records.get("documents", [])
            if document.get("record", {}).get("kind") == "forge.inbox.v1"
        ]

        write_root = session / "write-idle"
        write_model = ProbeModel([asyncio.Event()])
        writer, write_storage = await load_model_session(write_root, write_model)
        write_submission = await writer.submit("idle write", request_id="write-idle", type="write")
        write_status = await write_submission.status()
        write_records = await write_storage.read_records()

        from conftest import make_native_tool
        from forge_agent.tools import AgentToolResult

        tool_root = session / "tool-boundary"
        tool_releases = [asyncio.Event() for _ in range(3)]
        tool_model = ProbeModel(tool_releases, tool_indexes={0})
        tool_effects = 0
        tool_finished = asyncio.Event()

        async def effect(arguments: Mapping[str, Any], signal: Any = None, context: Any = None):
            del arguments, signal, context
            nonlocal tool_effects
            tool_effects += 1
            _effect(tool_root, "tool-effect", count=tool_effects)
            tool_finished.set()
            return AgentToolResult(
                tool_call_id="",
                name="effect",
                ok=True,
                content="controlled effect",
            )

        effect_tool = make_native_tool(
            name="effect",
            description="Record one controlled D2 effect.",
            input_schema={"type": "object", "properties": {}},
            executor=effect,
        )
        tooling, tool_storage = await load_model_session(tool_root, tool_model, tools=[effect_tool])
        tool_first = await tooling.submit("tool first", request_id="tool-first")
        await asyncio.wait_for(tool_model.started[0].wait(), timeout=5)
        tool_releases[0].set()
        await asyncio.wait_for(tool_finished.wait(), timeout=5)
        post_tools = await tooling.submit("after tools", request_id="post-tools", when_busy="steer")
        await asyncio.wait_for(tool_model.started[1].wait(), timeout=5)
        post_tools_status = await post_tools.status()
        final_follow_up = await tooling.submit(
            "final follow up", request_id="final-follow-up", when_busy="follow_up"
        )
        final_queued_status = await final_follow_up.status()
        tool_releases[1].set()
        await asyncio.wait_for(tool_model.started[2].wait(), timeout=5)
        final_placed_status = await final_follow_up.status()
        tool_releases[2].set()
        await tool_first.wait(timeout=5)
        await final_follow_up.wait(timeout=5)
        tool_call_ids = [
            str(call.get("id"))
            for message in tooling.messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        tool_message_ids = [
            str(message.tool_call_id)
            for message in tooling.messages
            if isinstance(message, ToolMessage)
        ]
        tool_boundary = {
            "effect_calls": tool_effects,
            "model_calls": len(tool_model.calls),
            "post_tools": _jsonable(post_tools_status),
            "final_queued": _jsonable(final_queued_status),
            "final_placed": _jsonable(final_placed_status),
            "paired_tool_call_ids": tool_call_ids == tool_message_ids,
            "tool_call_ids": tool_call_ids,
            "tool_message_ids": tool_message_ids,
            "records": _jsonable(await tool_storage.read_records()),
        }
        payload = {
            "action": args.action,
            "pid": os.getpid(),
            "first_id": first.id,
            "duplicate_id": duplicate.id,
            "type_conflict": type_conflict,
            "busy_reject": busy_reject,
            "observer_timeout": observer_timeout,
            "observer_cancelled": observer_cancelled,
            "queued_before": _jsonable(queued_before),
            "placed_after_first": _jsonable(placed_after_first),
            "queued_after_first": _jsonable(queued_after_first),
            "queued_after_second": _jsonable(queued_after_second),
            "placed_after_second": _jsonable(placed_after_second),
            "placed_after_third": _jsonable(placed_after_third),
            "settled": _jsonable(settled),
            "reopened_receipt": _jsonable(reopened_receipt_status),
            "model_calls": len(model.calls),
            "failure": {
                "failed": _jsonable(failed_result),
                "retained": _jsonable(retained_result),
                "inbox": failure_inbox,
            },
            "write_idle": {
                "model_calls": len(write_model.calls),
                "submission": _jsonable(write_status),
                "records": _jsonable(write_records),
            },
            "tool_boundary": tool_boundary,
            "records": _jsonable(
                (await storage.read_records()) if hasattr(storage, "read_records") else {}
            ),
        }
        _write_json(_result_path(args), payload)
    finally:
        await coding.aclose()
        await failing.aclose()
        if writer is not None:
            await writer.aclose()
        if tooling is not None:
            await tooling.aclose()
    return 0


async def _run_boundary_safety(args: argparse.Namespace) -> int:
    from forge_coding.features.tasks import TaskCoordinator, TaskDefinition
    from forge_coding.sessions.storage import JsonlSessionStorage

    failures: list[str] = []
    calls: list[int] = []

    async def effect(record: dict[str, Any], runtime: Any) -> None:
        calls.append(record["id"])
        await runtime.complete()

    class FailedReservationStorage(JsonlSessionStorage):
        async def commit_records(self, records: Any, entries: Any = ()) -> int:
            if any(
                write.get("type") == "task" and write["value"]["state"]["status"] == "running"
                for write in records
            ):
                raise OSError("injected reservation publication failure")
            return await super().commit_records(records, entries)

    storage = FailedReservationStorage(args.session / "reservation")
    coordinator = await TaskCoordinator.open(storage, [TaskDefinition("effect", 1, run=effect)])
    await coordinator.create("effect", {})
    await coordinator.resume()
    await asyncio.sleep(0.1)
    if calls:
        failures.append("uncommitted-handler-dispatched")
    await coordinator.close()
    await storage.close()

    from fake_models import ScriptedChatModel
    from forge_coding import CodingSession, CodingSessionConfig

    storage = FailedReservationStorage(args.session / "sdk-publication")
    coding = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            storage=storage,
            cwd=args.session,
            system="offline",
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        await asyncio.wait_for(_collect(coding.prompt("fail reservation")), 0.5)
    except TimeoutError:
        failures.append("sdk-observer-hung-after-publication-error")
    except (OSError, RuntimeError):
        pass
    finally:
        await coding.aclose()

    async def failed_adoption(tx: Any, sequence: int) -> None:
        del sequence
        if any(record["state"]["status"] == "running" for record in tx.tasks.values()):
            raise OSError("injected adoption failure")

    storage = JsonlSessionStorage(args.session / "adoption")
    coordinator = await TaskCoordinator.open(
        storage, [TaskDefinition("effect", 1, run=effect)], on_commit=failed_adoption
    )
    await coordinator.create("effect", {})
    await coordinator.resume()
    await asyncio.sleep(0.1)
    if calls:
        failures.append("failed-adoption-dispatched-handler")
    await coordinator.close()
    await storage.close()

    storage = JsonlSessionStorage(args.session / "observer")
    coordinator = await TaskCoordinator.open(storage, [])
    task_id = await coordinator.create("missing", {})
    observer = asyncio.create_task(coordinator.wait_task(task_id))
    await asyncio.sleep(0)
    await coordinator.close()
    try:
        await asyncio.wait_for(observer, 0.2)
        failures.append("close-did-not-reject-observer")
    except TimeoutError:
        failures.append("observer-survived-close")
    except RuntimeError:
        pass
    await storage.close()

    storage = JsonlSessionStorage(args.session / "migration")
    coordinator = await TaskCoordinator.open(storage, [TaskDefinition("upgrade", 1, run=effect)])
    task_id = await coordinator.create("upgrade", {})
    coordinator.register(TaskDefinition("upgrade", 2, run=effect))
    await coordinator.resume()
    await coordinator.abort(task_id)
    try:
        settled = await coordinator.wait_task(task_id, timeout=0.2)
        if settled["state"]["outcome"]["status"] != "orphaned":
            failures.append("blocked-abort-not-orphaned")
    except TimeoutError:
        failures.append("blocked-abort-remained-pending")
    await coordinator.close()
    await storage.close()
    _write_json(
        _result_path(args), {"pid": os.getpid(), "failures": failures, "handler_calls": calls}
    )
    return 0


async def _collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


async def run(args: argparse.Namespace) -> int:
    action = args.action
    if action == "boundary-safety":
        return await _run_boundary_safety(args)
    if action == "no-progress":
        return await _run_no_progress(args)
    if action == "crash-after-checkpoint":
        return await _run_crash_after_checkpoint(args)
    if action == "reopen-inspect":
        return await _run_reopen_inspect(args)
    if action == "resume-after-restart":
        return await _run_resume_after_restart(args)
    if action == "abort-crash":
        return await _run_abort_crash(args)
    if action == "reopen-abort":
        return await _run_reopen_abort(args)
    if action == "migration-prepare":
        return await _run_migration_prepare(args)
    if action == "migration-fail":
        return await _run_migration(args, fail=True)
    if action == "migration-success":
        return await _run_migration(args, fail=False)
    if action == "orphan-prepare":
        return await _run_orphan_prepare(args)
    if action == "orphan-abort":
        return await _run_orphan_abort(args)
    if action == "abort-tree-crash":
        return await _run_abort_tree_crash(args)
    if action == "abort-tree-reopen":
        return await _run_abort_tree_reopen(args)
    if action == "completing-crash":
        return await _run_completing_crash(args)
    if action == "completing-reopen":
        return await _run_completing_reopen(args)
    if action == "stale-handle":
        return await _run_stale_handle(args)
    if action == "loop-shutdown":
        return await _run_loop_shutdown(args)
    if action == "sdk-submissions":
        return await _run_sdk_submissions(args)
    raise ValueError(f"unknown D2 driver action: {action}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "boundary-safety",
            "no-progress",
            "crash-after-checkpoint",
            "reopen-inspect",
            "resume-after-restart",
            "abort-crash",
            "reopen-abort",
            "migration-prepare",
            "migration-fail",
            "migration-success",
            "orphan-prepare",
            "orphan-abort",
            "abort-tree-crash",
            "abort-tree-reopen",
            "completing-crash",
            "completing-reopen",
            "stale-handle",
            "loop-shutdown",
            "sdk-submissions",
        ),
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--exit-code", type=int, default=17)
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except BaseException as exc:  # noqa: BLE001 - subprocess evidence
        _write_json(
            _result_path(args),
            {
                "action": args.action,
                "error": type(exc).__name__,
                "message": str(exc),
                "pid": os.getpid(),
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
