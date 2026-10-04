"""Durable task stages and input-admission transactions for Forge sessions.

The coordinator deliberately owns no model or tool loop.  It only reserves a
committed task, invokes its current Python phase, and persists the next
checkpoint before another phase can run.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from forge_agent.session.entries import SessionEntry

Json = Any
TaskHandler = Callable[..., Any]
Migration = Callable[..., Any]
_MISSING = object()


class TaskError(RuntimeError):
    """Base error for task lifecycle failures."""


class TaskClosedError(TaskError):
    """The coordinator no longer accepts work."""


class TaskNotFoundError(TaskError):
    """A requested task or submission does not exist."""


class TaskStaleError(TaskError):
    """A runtime callback arrived after its invocation ended."""


class TaskDefinitionError(TaskError, ValueError):
    """A task definition or checkpoint violates the durable contract."""


class TaskStorage(Protocol):
    """The public D1 storage surface used by the coordinator."""

    async def read_records(self) -> Mapping[str, Any]: ...

    async def commit_records(
        self, writes: Sequence[dict[str, Any]], entries: Sequence[SessionEntry] = ()
    ) -> int: ...


class TaskDefinition:
    """Executable definition for one persisted task kind.

    ``run`` is the compact one-phase form.  ``phases`` may instead map the
    checkpoint's ``phase`` value to handlers.  Every handler receives the
    persisted task record and its invocation-bound runtime.
    """

    def __init__(
        self,
        kind: str,
        version: int = 1,
        run: TaskHandler | None = None,
        abort: TaskHandler | None = None,
        migrate: Migration | None = None,
        initial: Callable[..., Json] | Json | None = None,
        phases: Mapping[str, TaskHandler] | None = None,
    ) -> None:
        if not isinstance(kind, str) or not kind:
            raise TaskDefinitionError("task kind is required")
        if type(version) is not int or version < 1:
            raise TaskDefinitionError("task version must be a positive integer")
        if run is None and not phases:
            raise TaskDefinitionError("task definition requires run or phases")
        self.kind = kind
        self.version = version
        self.run = run
        self.abort = abort
        self.migrate = migrate
        self.initial = initial
        self.phases = dict(phases or {})

    def first_checkpoint(self, value: Json) -> Json:
        if self.initial is None:
            return {"phase": "run"}
        if callable(self.initial):
            return _call_sync(self.initial, value)
        return copy.deepcopy(self.initial)

    def handler(self, checkpoint: Json) -> TaskHandler:
        phase = checkpoint.get("phase") if isinstance(checkpoint, Mapping) else None
        if self.phases:
            if phase is not None and phase in self.phases:
                return self.phases[phase]
            raise TaskDefinitionError(f"task {self.kind} has no phase {phase!r}")
        if self.run is not None:
            return self.run
        raise TaskDefinitionError(f"task {self.kind} has no phase {phase!r}")


@dataclass(slots=True)
class _Invocation:
    task_id: int
    definition: TaskDefinition
    mode: str
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    ended: bool = False
    progress: bool = False
    task: asyncio.Task[None] | None = None


class TaskTransaction:
    """Mutable candidate view passed to one atomic coordinator change."""

    def __init__(
        self,
        coordinator: TaskCoordinator,
        tasks: Mapping[int, dict[str, Any]],
        submissions: Mapping[int, dict[str, Any]],
        conversations: Mapping[int, dict[str, Any]],
        ids: Collection[int],
    ) -> None:
        self.coordinator = coordinator
        self.tasks: dict[int, dict[str, Any]] = copy.deepcopy(dict(tasks))
        self.submissions: dict[int, dict[str, Any]] = copy.deepcopy(dict(submissions))
        self.conversations: dict[int, dict[str, Any]] = copy.deepcopy(dict(conversations))
        self._ids = set(ids)
        self._writes: list[dict[str, Any]] = []
        self.entries: list[Any] = []
        self._terminal: dict[int, dict[str, Any]] = {}
        self._abort_signals: set[int] = set()

    @property
    def writes(self) -> list[dict[str, Any]]:
        return self._writes

    def allocate(self) -> int:
        candidate = max(self._ids, default=0) + 1
        while candidate in self._ids:
            candidate += 1
        self._ids.add(candidate)
        return candidate

    def create(
        self,
        kind: str,
        input: Json,
        checkpoint: Json | None = None,
        owner: int | None = None,
        background: bool = False,
        conversation_id: int = 0,
    ) -> int:
        definition = self.coordinator.definition(kind)
        selected_checkpoint = (
            copy.deepcopy(checkpoint)
            if checkpoint is not None
            else definition.first_checkpoint(input)
            if definition is not None
            else {"phase": "run"}
        )
        task_id = self.allocate()
        record: dict[str, Any] = {
            "id": task_id,
            "kind": kind,
            "version": definition.version if definition is not None else 1,
            "input": copy.deepcopy(input),
            "conversationId": conversation_id,
            "background": bool(background),
            "abortRequested": False,
            "state": {"status": "pending", "checkpoint": selected_checkpoint},
        }
        if owner is not None:
            record["owner"] = owner
        self.set_task(record)
        return task_id

    def set_task(self, record: Mapping[str, Any]) -> None:
        value = _normalize_task(record)
        task_id = _task_id(value)
        previous = self.tasks.get(task_id)
        if (
            previous is not None
            and previous.get("state", {}).get("status") == "terminal"
            and value["state"]["status"] != "terminal"
        ):
            raise TaskStaleError(f"task {task_id} is terminal")
        self.tasks[task_id] = copy.deepcopy(value)
        self._replace_write("task", task_id, {"type": "task", "value": copy.deepcopy(value)})
        if value["state"]["status"] in {"terminal"}:
            committed = self.coordinator._tasks.get(task_id)  # noqa: SLF001
            if committed is None or committed.get("state", {}).get("status") != "terminal":
                self._terminal[task_id] = copy.deepcopy(value)

    def set_submission(self, record: Mapping[str, Any]) -> None:
        value = _normalize_submission(record)
        submission_id = _record_id(value)
        self.submissions[submission_id] = copy.deepcopy(value)
        self._replace_write(
            "submission", submission_id, {"type": "submission", "value": copy.deepcopy(value)}
        )

    def set_conversation(self, record: Mapping[str, Any]) -> None:
        value = copy.deepcopy(dict(record))
        conversation_id = _record_id(value)
        self.conversations[conversation_id] = value
        self._replace_write(
            "conversation", conversation_id, {"type": "conversation", "value": copy.deepcopy(value)}
        )

    def add_entry(self, entry: Any) -> None:
        self.entries.append(entry)

    def stage(self, write: Mapping[str, Any]) -> None:
        self._writes.append(copy.deepcopy(dict(write)))

    def terminal_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._terminal.values())

    def _replace_write(self, kind: str, record_id: int, write: dict[str, Any]) -> None:
        for index, existing in enumerate(self._writes):
            if existing.get("type") != kind:
                continue
            value = existing.get("value")
            if isinstance(value, Mapping) and value.get("id") == record_id:
                self._writes[index] = write
                return
        self._writes.append(write)


class TaskRuntime:
    """Invocation-bound runtime; every mutating method checks its invocation."""

    def __init__(self, coordinator: TaskCoordinator, invocation: _Invocation) -> None:
        self._coordinator = coordinator
        self._invocation = invocation
        self.task_id = invocation.task_id

    @property
    def conversation_id(self) -> int:
        record = self._coordinator._tasks.get(self.task_id)  # noqa: SLF001
        return int(record.get("conversationId", 0)) if record is not None else 0

    @property
    def signal(self) -> asyncio.Event:
        return self._invocation.cancelled

    @property
    def registry(self) -> Mapping[str, TaskDefinition]:
        return dict(self._coordinator._definitions)  # noqa: SLF001

    @property
    def task(self) -> dict[str, Any]:
        record = self._coordinator._tasks.get(self.task_id)  # noqa: SLF001
        if record is None:
            raise TaskStaleError(f"task {self.task_id} invocation has ended")
        return copy.deepcopy(record)

    async def commit(self, change: Any = None, **kwargs: Any) -> dict[str, Any] | None:
        return await self._coordinator._runtime_commit(self._invocation, change, kwargs)  # noqa: SLF001

    async def checkpoint(self, checkpoint: Json) -> dict[str, Any] | None:
        return await self.commit({"status": "running", "checkpoint": copy.deepcopy(checkpoint)})

    async def waiting(
        self,
        on: Sequence[int],
        policy: str = "allSettled",
        checkpoint: Json | None = None,
    ) -> dict[str, Any] | None:
        if self._invocation.mode == "abort":
            raise TaskDefinitionError("abort handler cannot wait")
        current = self.task
        selected = copy.deepcopy(checkpoint)
        if selected is None:
            selected = copy.deepcopy(current.get("state", {}).get("checkpoint", {"phase": "run"}))
        return await self.commit(
            {"status": "waiting", "checkpoint": selected, "on": list(on), "policy": _policy(policy)}
        )

    async def complete(self, result: Json = None) -> dict[str, Any] | None:
        return await self.commit(
            {"status": "terminal", "outcome": {"status": "completed", "result": result}}
        )

    async def fail(self, error: Any, result: Json = None) -> dict[str, Any] | None:
        outcome: dict[str, Any] = {"status": "failed", "error": _error_value(error)}
        if result is not None:
            outcome["result"] = copy.deepcopy(result)
        return await self.commit({"status": "terminal", "outcome": outcome})

    async def spawn(
        self,
        kind: str,
        input: Json,
        checkpoint: Json | None = None,
        *,
        owner: int | None = None,
        background: bool = False,
        conversation_id: int | None = None,
    ) -> int:
        self._coordinator._check_invocation(self._invocation)  # noqa: SLF001

        def change(tx: TaskTransaction) -> int:
            self._coordinator._check_invocation(self._invocation)  # noqa: SLF001
            return tx.create(
                kind,
                input,
                checkpoint,
                self.task_id if owner is None else owner,
                background,
                self.conversation_id if conversation_id is None else conversation_id,
            )

        return cast(int, await self._coordinator.change(change))

    async def memo(self, name: str, value: Any = _MISSING) -> Any:
        self._coordinator._check_invocation(self._invocation)  # noqa: SLF001
        if value is _MISSING:

            def read(tx: TaskTransaction) -> Any:
                self._coordinator._check_invocation(self._invocation)  # noqa: SLF001
                record = tx.tasks.get(self.task_id)
                if record is None:
                    raise TaskStaleError(f"task {self.task_id} invocation has ended")
                return copy.deepcopy(record.get("memos", {}).get(name))

            return await self._coordinator.change(read)

        def change(tx: TaskTransaction) -> Any:
            self._coordinator._check_invocation(self._invocation)  # noqa: SLF001
            current = tx.tasks.get(self.task_id)
            if current is None:
                raise TaskStaleError(f"task {self.task_id} invocation has ended")
            memos = current.setdefault("memos", {})
            if name in memos:
                return copy.deepcopy(memos[name])
            memos[name] = copy.deepcopy(value)
            tx.set_task(current)
            return copy.deepcopy(value)

        return await self._coordinator.change(change)


class TaskCoordinator:
    """One serialized durable task line for a session."""

    def __init__(
        self,
        storage: TaskStorage,
        definitions: Mapping[str, TaskDefinition] | Sequence[TaskDefinition] = (),
        *,
        on_terminal: Callable[..., Any] | None = None,
        on_commit: Callable[..., Any] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.storage = storage
        self.line = asyncio.Lock()
        self._definitions: dict[str, TaskDefinition] = _definition_map(definitions)
        self._on_terminal = on_terminal
        self._on_commit = on_commit
        self._on_error = on_error
        self._failure: BaseException | None = None
        self._tasks: dict[int, dict[str, Any]] = {}
        self._submissions: dict[int, dict[str, Any]] = {}
        self._conversations: dict[int, dict[str, Any]] = {}
        self._ids: set[int] = set()
        self._sequence = 0
        self._waiters: dict[int, set[asyncio.Future[dict[str, Any]]]] = {}
        self._idle_waiters: dict[int, set[asyncio.Future[None]]] = {}
        self._invocations: dict[int, _Invocation] = {}
        self._migration_failures: dict[
            tuple[int, str, int, int], tuple[TaskDefinition, BaseException]
        ] = {}
        self._enabled = False
        self._closing = False
        self._opened = False
        self._drain_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._dirty = False
        self._reports: list[BaseException] = []

    @classmethod
    async def open(
        cls,
        storage: TaskStorage,
        definitions: Mapping[str, TaskDefinition] | Sequence[TaskDefinition] = (),
        **kwargs: Any,
    ) -> TaskCoordinator:
        coordinator = cls(storage, definitions, **kwargs)
        await coordinator._open()  # noqa: SLF001
        return coordinator

    async def _open(self) -> None:
        async with self.line:
            await self._reload_locked()
            running = [
                {
                    **record,
                    "state": {
                        "status": "pending",
                        "checkpoint": copy.deepcopy(record["state"]["checkpoint"]),
                    },
                }
                for record in self._tasks.values()
                if record.get("state", {}).get("status") == "running"
            ]
            if running:
                writes = [{"type": "task", "value": record} for record in running]
                await self.storage.commit_records(writes, ())
                await self._reload_locked()
            self._opened = True

    async def resume(self) -> None:
        if self._failure is not None:
            raise TaskError("task publication failed; reopen the session") from self._failure
        if self._closing:
            raise TaskClosedError("task coordinator is closed")
        self._enabled = True
        self._dirty = True
        task = self._kick()
        if task is not None and task is not asyncio.current_task():
            await asyncio.shield(task)

    def resume_nowait(self) -> asyncio.Task[None] | None:
        self._enabled = True
        self._dirty = True
        return self._kick()

    async def join(self) -> None:
        tasks = [inv.task for inv in self._invocations.values() if inv.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self._closing = True
        self._enabled = False
        groups: list[set[asyncio.Future[Any]]] = [
            *self._waiters.values(),
            *self._idle_waiters.values(),
        ]
        for waiters in groups:
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_exception(TaskClosedError("task coordinator is closed"))
            waiters.clear()
        if self._drain_task is not None:
            await asyncio.shield(self._drain_task)
        # A caller may already have entered the publication line before close.
        async with self.line:
            pass
        invocations = list(self._invocations.values())
        for invocation in invocations:
            invocation.cancelled.set()
            if invocation.task is not None:
                invocation.task.cancel()
        if invocations:
            await asyncio.gather(
                *(invocation.task for invocation in invocations if invocation.task is not None),
                return_exceptions=True,
            )

    def definition(self, kind: str) -> TaskDefinition | None:
        return self._definitions.get(kind)

    def register(self, definition: TaskDefinition) -> None:
        self._definitions[definition.kind] = definition
        self._dirty = True
        self._kick()

    def set_definitions(
        self, definitions: Mapping[str, TaskDefinition] | Sequence[TaskDefinition]
    ) -> None:
        self._definitions = _definition_map(definitions)
        self._dirty = True
        self._kick()

    def report(self, error: BaseException) -> None:
        self._reports.append(error)

    async def change(self, callback: Callable[[TaskTransaction], Any]) -> Any:
        if self._failure is not None:
            raise TaskError("task publication failed; reopen the session") from self._failure
        if self._closing:
            raise TaskClosedError("task coordinator is closed")
        async with self.line:
            if self._failure is not None:
                raise TaskError("task publication failed; reopen the session") from self._failure
            if self._closing:
                raise TaskClosedError("task coordinator is closed")
            await self._reload_locked()
            tx = TaskTransaction(
                self, self._tasks, self._submissions, self._conversations, self._ids
            )
            result = callback(tx)
            if inspect.isawaitable(result):
                result = await result
            for record in tx.terminal_records():
                if self._on_terminal is not None:
                    await _call_callback(self._on_terminal, tx, record)
            changed = bool(tx.writes or tx.entries)
            if changed:
                try:
                    self._sequence = await self.storage.commit_records(tx.writes, tx.entries)
                    self._adopt(tx)
                    if self._on_commit is not None:
                        await _call_callback(self._on_commit, tx, self._sequence)
                except BaseException as error:
                    self._failure = error
                    self._enabled = False
                    groups: list[set[asyncio.Future[Any]]] = [
                        *self._waiters.values(),
                        *self._idle_waiters.values(),
                    ]
                    for waiters in groups:
                        for waiter in waiters:
                            if not waiter.done():
                                waiter.set_exception(
                                    TaskError("task publication failed; reopen the session")
                                )
                        waiters.clear()
                    if self._on_error is not None:
                        self._on_error(error)
                    raise
                for task_id in tx._abort_signals:
                    self._signal_abort(task_id)
                self._notify_waiters()
            else:
                self._adopt(tx, commit=False)
            if changed:
                self._dirty = True
        if changed:
            self._kick()
        return result

    async def create(
        self,
        kind: str,
        input: Json,
        checkpoint: Json | None = None,
        owner: int | None = None,
        background: bool = False,
        conversation_id: int = 0,
    ) -> int:
        return cast(
            int,
            await self.change(
                lambda tx: tx.create(kind, input, checkpoint, owner, background, conversation_id)
            ),
        )

    async def abort(self, task_id: int, include_background: bool = False) -> str:
        result = await self.change(lambda tx: self._mark_abort(tx, task_id, include_background))
        self._dirty = True
        self._kick()
        invocation = self._invocations.get(task_id)
        if (
            result == "marked"
            and invocation is not None
            and invocation.mode == "run"
            and invocation.task is not None
            and invocation.task is not asyncio.current_task()
        ):
            await asyncio.shield(invocation.task)
        return cast(str, result)

    async def abort_task(self, task_id: int, include_background: bool = False) -> str:
        return await self.abort(task_id, include_background)

    async def abort_conversation(
        self,
        conversation_id: int,
        include_background: bool = False,
        *,
        before_mark: Callable[[TaskTransaction], Any] | None = None,
    ) -> None:
        async def mark(tx: TaskTransaction) -> list[int]:
            if before_mark is not None:
                staged = before_mark(tx)
                if inspect.isawaitable(staged):
                    await staged
            reached: list[int] = []
            for task in list(tx.tasks.values()):
                if task.get("state", {}).get("status") == "terminal":
                    continue
                if task.get("background") and not include_background:
                    continue
                if self._in_scope(
                    task, conversation_id, include_background, tx.tasks, tx.conversations
                ):
                    reached.append(_task_id(task))
                    task["abortRequested"] = True
                    self._prepare_abort_record(tx, task)
            for submission in list(tx.submissions.values()):
                if (
                    submission.get("type") == "input"
                    and submission.get("status") == "queued"
                    and self._in_scope(
                        {
                            "conversationId": submission.get("conversationId", 0),
                        },
                        conversation_id,
                        include_background,
                        tx.tasks,
                        tx.conversations,
                    )
                ):
                    submission["status"] = "unanswered"
                    submission["reason"] = "aborted"
                    tx.set_submission(submission)
            return reached

        reached = cast(list[int], await self.change(mark))
        await self.resume()
        if include_background:
            for task_id in reached:
                try:
                    await self.wait_task(task_id)
                except TaskNotFoundError:
                    continue
        await self.wait_idle(conversation_id)

    async def wait(self, task_id: int, timeout: float | None = None) -> dict[str, Any]:
        return await self.wait_task(task_id, timeout)

    async def wait_task(self, task_id: int, timeout: float | None = None) -> dict[str, Any]:
        async with self.line:
            if self._failure is not None:
                raise TaskError("task publication failed; reopen the session") from self._failure
            if self._closing:
                raise TaskClosedError("task coordinator is closed")
            await self._reload_locked()
            record = self._tasks.get(task_id)
            if record is None:
                raise TaskNotFoundError(f"task {task_id} does not exist")
            if record.get("state", {}).get("status") == "terminal":
                return copy.deepcopy(record)
            waiter: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._waiters.setdefault(task_id, set()).add(waiter)
        try:
            if timeout is None:
                return await asyncio.shield(waiter)
            return await asyncio.wait_for(asyncio.shield(waiter), timeout)
        finally:
            self._waiters.get(task_id, set()).discard(waiter)

    async def wait_idle(self, conversation_id: int = 0, timeout: float | None = None) -> None:
        """Wait for foreground work in one conversation to settle.

        Background tasks and descendants owned by a background task never keep
        the conversation busy.  This is an observer wait only; it does not
        wake or cancel work.
        """

        async with self.line:
            if self._failure is not None:
                raise TaskError("task publication failed; reopen the session") from self._failure
            if self._closing:
                raise TaskClosedError("task coordinator is closed")
            await self._reload_locked()
            if self._conversation_idle(conversation_id):
                return
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._idle_waiters.setdefault(conversation_id, set()).add(waiter)
        try:
            if timeout is None:
                await asyncio.shield(waiter)
            else:
                await asyncio.wait_for(asyncio.shield(waiter), timeout)
        finally:
            self._idle_waiters.get(conversation_id, set()).discard(waiter)

    async def inspect(self, task_id: int | None = None) -> dict[str, Any]:
        async with self.line:
            await self._reload_locked()
            records = [self._inspection(record) for record in self._tasks.values()]
            if task_id is not None:
                records = [record for record in records if record.get("id") == task_id]
            return {
                "scheduling": "closing"
                if self._closing
                else "running"
                if self._enabled
                else "paused",
                "tasks": copy.deepcopy(records),
                "submissions": copy.deepcopy(list(self._submissions.values())),
                "sequence": self._sequence,
            }

    async def submission(self, submission_id: int) -> dict[str, Any] | None:
        async with self.line:
            await self._reload_locked()
            value = self._submissions.get(submission_id)
            return copy.deepcopy(value) if value is not None else None

    async def _runtime_commit(
        self, invocation: _Invocation, change: Any, options: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        self._check_invocation(invocation)

        def commit(tx: TaskTransaction) -> dict[str, Any]:
            self._check_invocation(invocation)
            current = tx.tasks.get(invocation.task_id)
            if current is None:
                raise TaskStaleError(f"task {invocation.task_id} invocation has ended")
            if current.get("state", {}).get("status") != "running":
                raise TaskStaleError(f"task {invocation.task_id} is no longer running")
            if invocation.mode == "run" and current.get("abortRequested"):
                raise TaskStaleError(f"task {invocation.task_id} has a durable abort mark")
            next_state = _state_from_change(change, current, options)
            if next_state is None:
                return current
            state = _normalize_state(next_state)
            if state["status"] == "waiting":
                if invocation.mode == "abort":
                    raise TaskDefinitionError("abort handler cannot wait")
                self._validate_wait(invocation.task_id, state, tx.tasks, tx.conversations)
            if state["status"] == "terminal":
                self._set_terminal_or_completing(tx, current, state["outcome"])
            else:
                current["state"] = state
                tx.set_task(current)
            invocation.progress = True
            if state["status"] in {"waiting", "terminal"}:
                invocation.ended = True
            return copy.deepcopy(current)

        result = await self.change(commit)
        return cast(dict[str, Any] | None, result)

    def _set_terminal_or_completing(
        self, tx: TaskTransaction, current: dict[str, Any], outcome: dict[str, Any]
    ) -> None:
        current.pop("memos", None)
        if self._has_owned_live(current["id"], tx.tasks, tx.conversations):
            current["state"] = {"status": "completing", "outcome": copy.deepcopy(outcome)}
        else:
            current["state"] = {"status": "terminal", "outcome": copy.deepcopy(outcome)}
        tx.set_task(current)

    async def _run_invocation(self, invocation: _Invocation) -> None:
        try:
            while not invocation.ended and not self._closing:
                current = self._tasks.get(invocation.task_id)
                if current is None or current.get("state", {}).get("status") != "running":
                    return
                if invocation.cancelled.is_set() and invocation.mode == "run":
                    return
                checkpoint = copy.deepcopy(
                    current.get("state", {}).get("checkpoint", {"phase": "run"})
                )
                invocation.progress = False
                runtime = TaskRuntime(self, invocation)
                try:
                    handler = invocation.definition.handler(checkpoint)
                    await _call_handler(handler, current, runtime)
                except asyncio.CancelledError:
                    return
                except BaseException as error:
                    if (
                        self._closing
                        or invocation.cancelled.is_set()
                        or current.get("abortRequested")
                    ):
                        return
                    await self._fault(invocation.task_id, error)
                    return
                if invocation.ended or self._closing:
                    return
                latest = self._tasks.get(invocation.task_id)
                if latest is None or latest.get("state", {}).get("status") != "running":
                    return
                if latest.get("abortRequested"):
                    return
                latest_checkpoint = latest.get("state", {}).get("checkpoint")
                if _json_equal(checkpoint, latest_checkpoint):
                    phase = (
                        checkpoint.get("phase") if isinstance(checkpoint, Mapping) else checkpoint
                    )
                    await self._fault(
                        invocation.task_id,
                        TaskError(
                            f"Task {latest.get('kind')} phase {phase!r} returned without "
                            "durable progress"
                        ),
                    )
                    return
                replacement = self._definitions.get(str(latest.get("kind")))
                if (
                    replacement is not None
                    and replacement is not invocation.definition
                    and _can_reserve(replacement, latest)
                ):

                    def handover(tx: TaskTransaction) -> None:
                        record = tx.tasks.get(invocation.task_id)
                        if (
                            record is not None
                            and record.get("state", {}).get("status") == "running"
                        ):
                            record["state"]["status"] = "pending"
                            tx.set_task(record)

                    await self.change(handover)
                    return
        finally:
            self._finish_invocation(invocation)
            current_task = asyncio.current_task()
            externally_cancelled = bool(
                current_task is not None
                and current_task.cancelling()
                and not invocation.cancelled.is_set()
            )
            if invocation.mode == "run" and not self._closing and not externally_cancelled:
                self._schedule_abort_requeue(invocation.task_id)
            self._dirty = True
            if not externally_cancelled:
                self._kick()

    async def _requeue_aborted(self, task_id: int) -> None:
        """Release a cancelled run reservation so its abort handler can run."""

        def requeue(tx: TaskTransaction) -> None:
            record = tx.tasks.get(task_id)
            if record is None or not record.get("abortRequested"):
                return
            if record.get("state", {}).get("status") != "running":
                return
            record["state"] = {
                "status": "pending",
                "checkpoint": copy.deepcopy(record["state"].get("checkpoint", {"phase": "run"})),
            }
            tx.set_task(record)

        try:
            await self.change(requeue)
        except (TaskClosedError, TaskStaleError, asyncio.CancelledError):
            return
        except BaseException as error:
            self.report(error)

    def _schedule_abort_requeue(self, task_id: int) -> None:
        record = self._tasks.get(task_id)
        if (
            record is None
            or not record.get("abortRequested")
            or record.get("state", {}).get("status") != "running"
        ):
            return
        asyncio.create_task(self._requeue_aborted(task_id))

    async def _run_abort(self, invocation: _Invocation) -> None:
        try:
            current = self._tasks.get(invocation.task_id)
            if current is None or current.get("state", {}).get("status") not in {
                "running",
                "pending",
            }:
                return
            if invocation.definition.abort is None:
                await self._fault(
                    invocation.task_id,
                    TaskError("task definition has no abort handler"),
                    force=True,
                )
                return
            runtime = TaskRuntime(self, invocation)
            await _call_handler(invocation.definition.abort, current, runtime)
            if not invocation.ended:
                await self._fault(
                    invocation.task_id,
                    TaskError(
                        f"abort handler of task {invocation.task_id} returned without an outcome"
                    ),
                    force=True,
                )
        except asyncio.CancelledError:
            return
        except BaseException as error:
            await self._fault(invocation.task_id, error, force=True)
        finally:
            self._finish_invocation(invocation)
            self._dirty = True
            current_task = asyncio.current_task()
            if not (
                current_task is not None
                and current_task.cancelling()
                and not invocation.cancelled.is_set()
            ):
                self._kick()

    async def _fault(self, task_id: int, error: BaseException, *, force: bool = False) -> None:
        def fault(tx: TaskTransaction) -> None:
            current = tx.tasks.get(task_id)
            if current is None or current.get("state", {}).get("status") != "running":
                return
            if current.get("abortRequested") and not force:
                return
            self._set_terminal_or_completing(
                tx, current, {"status": "faulted", "error": _error_value(error)}
            )

        try:
            await self.change(fault)
        except BaseException as commit_error:
            self.report(commit_error)

    def _finish_invocation(self, invocation: _Invocation) -> None:
        if self._invocations.get(invocation.task_id) is invocation:
            self._invocations.pop(invocation.task_id, None)
        invocation.ended = True

    def _check_invocation(self, invocation: _Invocation) -> None:
        if (
            self._closing
            or invocation.ended
            or self._invocations.get(invocation.task_id) is not invocation
        ):
            raise TaskStaleError(f"task {invocation.task_id} invocation has ended")

    def _kick(self) -> asyncio.Task[None] | None:
        if self._closing or not self._enabled or self._drain_task is not None:
            return self._drain_task
        try:
            task = asyncio.create_task(self._drain())
        except RuntimeError:
            return None
        self._drain_task = task
        return task

    async def _drain(self) -> None:
        try:
            while self._dirty and self._enabled and not self._closing:
                self._dirty = False
                await self._reconcile()
                reservations = await self._reserve()
                for invocation in reservations:
                    if self._closing:
                        self._finish_invocation(invocation)
                        continue
                    task = asyncio.create_task(
                        self._run_abort(invocation)
                        if invocation.mode == "abort"
                        else self._run_invocation(invocation)
                    )
                    invocation.task = task
        except asyncio.CancelledError:
            # A loop shutdown can cancel the scheduler before its owner closes
            # the coordinator.  Do not recreate a drain task from ``finally``.
            self._enabled = False
            self._dirty = False
        except BaseException as error:
            if not self._closing:
                self.report(error)
        finally:
            self._drain_task = None
            if self._dirty and self._enabled and not self._closing:
                self._kick()

    async def _reconcile(self) -> None:
        def reconcile(tx: TaskTransaction) -> None:
            for record in list(tx.tasks.values()):
                status = record.get("state", {}).get("status")
                if status == "terminal":
                    continue
                if (
                    self._below_cancelled(record, tx.tasks, tx.conversations)
                    and not record.get("background")
                    and not record.get("abortRequested")
                ):
                    record["abortRequested"] = True
                    self._prepare_abort_record(tx, record)
                if status == "waiting" and record.get("state", {}).get("policy") == "failFast":
                    on = record.get("state", {}).get("on", [])
                    if any(_failed(tx.tasks.get(int(item))) for item in on):
                        for item in on:
                            member = tx.tasks.get(int(item))
                            if (
                                member is not None
                                and member.get("state", {}).get("status") != "terminal"
                                and not member.get("abortRequested")
                            ):
                                member["abortRequested"] = True
                                self._prepare_abort_record(tx, member)
            self._finalize_completing(tx)

        try:
            await self.change(reconcile)
        except TaskClosedError:
            return
        except BaseException as error:
            self.report(error)

    def _finalize_completing(self, tx: TaskTransaction) -> None:
        changed = True
        while changed:
            changed = False
            for record in list(tx.tasks.values()):
                if record.get("state", {}).get("status") != "completing":
                    continue
                if self._has_owned_live(record["id"], tx.tasks, tx.conversations):
                    continue
                outcome = copy.deepcopy(record["state"].get("outcome"))
                record.pop("memos", None)
                record["state"] = {"status": "terminal", "outcome": outcome}
                tx.set_task(record)
                changed = True

    async def _reserve(self) -> list[_Invocation]:
        reservations: list[_Invocation] = []

        def reserve(tx: TaskTransaction) -> None:
            if not self._enabled or self._closing:
                return
            for record in list(tx.tasks.values()):
                task_id = _task_id(record)
                if task_id in self._invocations:
                    continue
                status = record.get("state", {}).get("status")
                if status == "completing":
                    continue
                if status == "waiting":
                    if record.get("abortRequested"):
                        if self._has_owned_live(task_id, tx.tasks, tx.conversations):
                            continue
                    else:
                        on = [int(item) for item in record.get("state", {}).get("on", [])]
                        live = [tx.tasks[item] for item in on if item in tx.tasks]
                        if any(item.get("state", {}).get("status") != "terminal" for item in live):
                            continue
                    record["state"] = {
                        "status": "pending",
                        "checkpoint": copy.deepcopy(
                            record["state"].get("checkpoint", {"phase": "run"})
                        ),
                    }
                    tx.set_task(record)
                    status = "pending"
                if status != "pending":
                    continue
                if record.get("abortRequested") and self._has_owned_live(
                    task_id, tx.tasks, tx.conversations
                ):
                    continue
                definition = self._definitions.get(str(record.get("kind")))
                if definition is None:
                    if record.get("abortRequested"):
                        record["state"] = {
                            "status": "terminal",
                            "outcome": {"status": "orphaned", "reason": "missing_task"},
                        }
                        tx.set_task(record)
                    continue
                resolved = self._resolve_definition(record, definition)
                if resolved is None:
                    if record.get("abortRequested"):
                        record["state"] = {
                            "status": "terminal",
                            "outcome": {
                                "status": "orphaned",
                                "reason": self._inspection(record)["blocked"]["reason"],
                            },
                        }
                        record.pop("memos", None)
                        tx.set_task(record)
                    continue
                definition, resolved_record = resolved
                if resolved_record != record:
                    record = resolved_record
                    tx.set_task(record)
                record["state"] = {
                    "status": "running",
                    "checkpoint": copy.deepcopy(
                        record["state"].get("checkpoint", {"phase": "run"})
                    ),
                }
                tx.set_task(record)
                invocation = _Invocation(
                    task_id=task_id,
                    definition=definition,
                    mode="abort" if record.get("abortRequested") else "run",
                )
                self._invocations[task_id] = invocation
                reservations.append(invocation)

        try:
            await self.change(reserve)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            for invocation in reservations:
                self._invocations.pop(invocation.task_id, None)
            if not self._closing:
                self.report(error)
            reservations.clear()
        return reservations

    def _resolve_definition(
        self, record: dict[str, Any], definition: TaskDefinition
    ) -> tuple[TaskDefinition, dict[str, Any]] | None:
        version = int(record.get("version", 1))
        if definition.version == version:
            return definition, record
        if definition.version < version:
            return None
        if definition.migrate is None:
            return None
        key = (_task_id(record), definition.kind, version, definition.version)
        failed = self._migration_failures.get(key)
        if failed is not None and failed[0] is definition:
            return None
        try:
            result = _call_sync(
                definition.migrate, record.get("input"), record["state"].get("checkpoint"), version
            )
            if inspect.isawaitable(result):
                raise TaskDefinitionError("task migration must be synchronous")
            if isinstance(result, Mapping):
                if "input" not in result or "checkpoint" not in result:
                    raise TaskDefinitionError("task migration must return input and checkpoint")
                input_value = result.get("input")
                checkpoint = result.get("checkpoint")
            elif isinstance(result, Sequence) and len(result) == 2:
                input_value, checkpoint = result
            else:
                raise TaskDefinitionError("task migration must return input and checkpoint")
            if checkpoint is None:
                raise TaskDefinitionError("task migration checkpoint cannot be null")
            migrated = copy.deepcopy(record)
            migrated["version"] = definition.version
            migrated["input"] = copy.deepcopy(input_value)
            migrated["state"]["checkpoint"] = copy.deepcopy(checkpoint)
            return definition, migrated
        except BaseException as error:
            self._migration_failures[key] = (definition, error)
            self.report(error)
            return None

    async def _reload_locked(self, *, recover_running: bool = False) -> None:
        data = await self.storage.read_records()
        self._tasks = {
            _task_id(record): _normalize_task(record, recover_running=recover_running)
            for record in _records(data.get("tasks", []))
        }
        self._submissions = {
            _record_id(record): _normalize_submission(record)
            for record in _records(data.get("submissions", []))
        }
        self._conversations = {
            _record_id(record): copy.deepcopy(record)
            for record in _records(data.get("conversations", []))
        }
        self._ids = {int(value) for value in data.get("ids", []) if type(value) is int}
        self._ids.update(self._tasks)
        self._ids.update(self._submissions)
        self._ids.update(self._conversations)
        self._sequence = int(data.get("sequence", 0))

    def _adopt(self, tx: TaskTransaction, *, commit: bool = True) -> None:
        self._tasks = copy.deepcopy(tx.tasks)
        self._submissions = copy.deepcopy(tx.submissions)
        self._conversations = copy.deepcopy(tx.conversations)
        self._ids.update(tx._ids)  # noqa: SLF001
        if not commit:
            self._sequence = self._sequence

    def _notify_waiters(self) -> None:
        for task_id, waiters in list(self._waiters.items()):
            record = self._tasks.get(task_id)
            if record is None or record.get("state", {}).get("status") != "terminal":
                continue
            for waiter in list(waiters):
                if not waiter.done():
                    waiter.set_result(copy.deepcopy(record))
            waiters.clear()
        for conversation_id, idle_waiters in list(self._idle_waiters.items()):
            if not self._conversation_idle(conversation_id):
                continue
            for idle_waiter in list(idle_waiters):
                if not idle_waiter.done():
                    idle_waiter.set_result(None)
            idle_waiters.clear()

    def _conversation_idle(self, conversation_id: int) -> bool:
        return not any(
            record.get("state", {}).get("status") != "terminal"
            and not record.get("background")
            and self._in_scope(
                record,
                conversation_id,
                False,
                self._tasks,
                self._conversations,
            )
            for record in self._tasks.values()
        )

    def _mark_abort(self, tx: TaskTransaction, task_id: int, include_background: bool) -> str:
        record = tx.tasks.get(task_id)
        if record is None:
            raise TaskNotFoundError(f"task {task_id} does not exist")
        state = record.get("state", {}).get("status")
        if state == "terminal":
            return "terminal"
        if not record.get("abortRequested"):
            record["abortRequested"] = True
            self._prepare_abort_record(tx, record)
        else:
            tx._abort_signals.add(task_id)
        for child in list(tx.tasks.values()):
            if (
                child.get("state", {}).get("status") == "terminal"
                or child.get("background")
                and not include_background
            ):
                continue
            if self._owned_by(child, task_id, tx.tasks, tx.conversations, include_background):
                child["abortRequested"] = True
                self._prepare_abort_record(tx, child)
        return "marked"

    def _prepare_abort_record(self, tx: TaskTransaction, record: dict[str, Any]) -> None:
        """Persist cancellation before signalling the live invocation."""

        tx.set_task(record)
        tx._abort_signals.add(_task_id(record))

    def _signal_abort(self, task_id: int) -> None:
        invocation = self._invocations.get(task_id)
        if invocation is not None and invocation.mode == "run":
            invocation.cancelled.set()
            if invocation.task is not None:
                invocation.task.cancel()

    def _inspection(self, record: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(record)
        task_id = _task_id(record)
        definition = self._definitions.get(str(record.get("kind")))
        state = record.get("state", {}).get("status")
        if state == "pending" and definition is None:
            result["blocked"] = {"reason": "missing_task"}
        elif state == "pending" and definition is not None:
            version = int(record.get("version", 1))
            if definition.version < version:
                result["blocked"] = {"reason": "task_too_old"}
            elif definition.version > version and definition.migrate is None:
                result["blocked"] = {"reason": "migration_failed", "error": "missing migration"}
            elif (
                task_id,
                definition.kind,
                version,
                definition.version,
            ) in self._migration_failures and self._migration_failures[
                (task_id, definition.kind, version, definition.version)
            ][0] is definition:
                result["blocked"] = {
                    "reason": "migration_failed",
                    "error": str(
                        self._migration_failures[
                            (task_id, definition.kind, version, definition.version)
                        ][1]
                    ),
                }
        if task_id in self._invocations:
            result["derived"] = "running"
        return result

    def _parent(
        self, record: Mapping[str, Any], conversations: Mapping[int, Mapping[str, Any]]
    ) -> tuple[str, int] | None:
        owner = record.get("owner")
        if isinstance(owner, Mapping):
            owner = owner.get("taskId", owner.get("task_id"))
        if owner is not None:
            return "task", int(owner)
        conversation_id = int(record.get("conversationId", record.get("conversation_id", 0)))
        conversation = conversations.get(conversation_id)
        if conversation is not None:
            owner_value = conversation.get("owner")
            if isinstance(owner_value, Mapping):
                owner_value = owner_value.get("taskId", owner_value.get("task_id"))
            if owner_value is not None:
                return "task", int(owner_value)
        return "conversation", conversation_id

    def _owned_by(
        self,
        record: Mapping[str, Any],
        owner_id: int,
        tasks: Mapping[int, Mapping[str, Any]],
        conversations: Mapping[int, Mapping[str, Any]],
        include_background: bool = False,
    ) -> bool:
        current = self._parent(record, conversations)
        seen: set[tuple[str, int]] = set()
        while current is not None and current not in seen:
            seen.add(current)
            kind, value = current
            if kind == "task":
                if value == owner_id:
                    return True
                parent = tasks.get(value)
                if parent is None:
                    return False
                if parent.get("background") and not include_background:
                    return False
                current = self._parent(parent, conversations)
            else:
                return False
        return False

    def _in_scope(
        self,
        record: Mapping[str, Any],
        conversation_id: int,
        include_background: bool,
        tasks: Mapping[int, Mapping[str, Any]],
        conversations: Mapping[int, Mapping[str, Any]],
    ) -> bool:
        current = self._parent(record, conversations)
        seen: set[tuple[str, int]] = set()
        while current is not None and current not in seen:
            seen.add(current)
            kind, value = current
            if kind == "conversation":
                return value == conversation_id
            owner = tasks.get(value)
            if owner is None:
                return False
            if owner.get("background") and not include_background:
                return False
            current = self._parent(owner, conversations)
        return False

    def _below_cancelled(
        self,
        record: Mapping[str, Any],
        tasks: Mapping[int, Mapping[str, Any]],
        conversations: Mapping[int, Mapping[str, Any]],
    ) -> bool:
        current = self._parent(record, conversations)
        seen: set[tuple[str, int]] = set()
        while current is not None and current not in seen:
            seen.add(current)
            kind, value = current
            if kind == "conversation":
                return False
            owner = tasks.get(value)
            if owner is None:
                return False
            state = owner.get("state", {})
            outcome = state.get("outcome", {}) if isinstance(state, Mapping) else {}
            if state.get("status") != "terminal" and (
                owner.get("abortRequested")
                or outcome.get("status")
                in {
                    "failed",
                    "aborted",
                    "faulted",
                    "orphaned",
                }
            ):
                return True
            if owner.get("background"):
                return False
            current = self._parent(owner, conversations)
        return False

    def _has_owned_live(
        self,
        owner_id: int,
        tasks: Mapping[int, Mapping[str, Any]],
        conversations: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> bool:
        edges = self._conversations if conversations is None else conversations
        for task in tasks.values():
            if task.get("state", {}).get("status") == "terminal":
                continue
            if task.get("background"):
                continue
            if self._owned_by(task, owner_id, tasks, edges):
                return True
        return False

    def _validate_wait(
        self,
        task_id: int,
        state: Mapping[str, Any],
        tasks: Mapping[int, Mapping[str, Any]],
        conversations: Mapping[int, Mapping[str, Any]],
    ) -> None:
        on = state.get("on")
        if not isinstance(on, Sequence) or isinstance(on, (str, bytes)):
            raise TaskDefinitionError("waiting state requires task IDs")
        if state.get("policy") not in {"allSettled", "failFast"}:
            raise TaskDefinitionError("waiting policy must be allSettled or failFast")
        for item in on:
            dependency = int(item)
            if dependency == task_id or self._owned_by(
                tasks[task_id], dependency, tasks, conversations, include_background=True
            ):
                raise TaskDefinitionError(
                    f"task {task_id} cannot wait on itself or its owner {dependency}"
                )
            member = tasks.get(dependency)
            if member is None:
                raise TaskNotFoundError(f"task {dependency} does not exist")
            if state.get("policy") == "failFast":
                parent = self._parent(member, conversations)
                if parent != ("task", task_id):
                    raise TaskDefinitionError(
                        f"task {task_id} can wait failFast only on tasks it owns; "
                        f"{dependency} is not one"
                    )


def _definition_map(
    definitions: Mapping[str, TaskDefinition] | Sequence[TaskDefinition],
) -> dict[str, TaskDefinition]:
    if isinstance(definitions, Mapping):
        return dict(definitions)
    return {definition.kind: definition for definition in definitions}


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [
            copy.deepcopy(cast(dict[str, Any], item))
            for item in value.values()
            if isinstance(item, Mapping)
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            copy.deepcopy(cast(dict[str, Any], item)) for item in value if isinstance(item, Mapping)
        ]
    return []


def _record_id(record: Mapping[str, Any]) -> int:
    raw = record.get("id")
    if type(raw) is not int or raw < 1:
        raise TaskDefinitionError("record id must be a positive integer")
    return raw


def _task_id(record: Mapping[str, Any]) -> int:
    return _record_id(record)


def _normalize_task(record: Mapping[str, Any], *, recover_running: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(dict(record))
    for snake, camel in (
        ("conversation_id", "conversationId"),
        ("abort_requested", "abortRequested"),
    ):
        if camel not in value and snake in value:
            value[camel] = value.pop(snake)
    value.setdefault("conversationId", 0)
    value.setdefault("background", False)
    value.setdefault("abortRequested", False)
    value.setdefault("version", 1)
    state = value.get("state")
    if not isinstance(state, Mapping):
        state = {"status": "pending", "checkpoint": {"phase": "run"}}
    state_value = dict(state)
    status = state_value.get("status")
    if recover_running and status == "running":
        state_value = {
            "status": "pending",
            "checkpoint": copy.deepcopy(state_value.get("checkpoint", {"phase": "run"})),
        }
    value["state"] = state_value
    _record_id(value)
    return value


def _normalize_submission(record: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(record))
    if "conversationId" not in value and "conversation_id" in value:
        value["conversationId"] = value.pop("conversation_id")
    value.setdefault("status", "queued")
    value.setdefault("type", "input")
    _record_id(value)
    return value


def _normalize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(state))
    status = value.get("status")
    if status not in {"running", "waiting", "terminal"}:
        raise TaskDefinitionError(f"invalid next task status: {status!r}")
    if status in {"running", "waiting"} and "checkpoint" not in value:
        raise TaskDefinitionError(f"{status} state requires checkpoint")
    if status == "waiting":
        value["on"] = [int(item) for item in value.get("on", [])]
        value["policy"] = _policy(str(value.get("policy", "allSettled")))
    if status == "terminal":
        outcome = value.get("outcome")
        if not isinstance(outcome, Mapping) or outcome.get("status") not in {
            "completed",
            "failed",
            "aborted",
            "faulted",
            "orphaned",
        }:
            raise TaskDefinitionError("terminal state requires a valid outcome")
        value["outcome"] = copy.deepcopy(dict(outcome))
    return value


def _state_from_change(
    change: Any, current: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any] | None:
    if "state" in options:
        change = options["state"]
    if change is None:
        return None
    if callable(change):
        change = _call_sync(change, copy.deepcopy(dict(current)))
    if change is None:
        return None
    if isinstance(change, Mapping):
        if "status" in change:
            return dict(change)
        return {"status": "running", "checkpoint": copy.deepcopy(dict(change))}
    raise TaskDefinitionError("runtime commit expects a state mapping or callback")


def _policy(value: str) -> str:
    normalized = {
        "all_settled": "allSettled",
        "all-settled": "allSettled",
        "allSettled": "allSettled",
        "fail_fast": "failFast",
        "fail-fast": "failFast",
        "failFast": "failFast",
    }.get(value)
    if normalized is None:
        raise TaskDefinitionError("waiting policy must be allSettled or failFast")
    return normalized


def _failed(record: Mapping[str, Any] | None) -> bool:
    if record is None:
        return False
    state = record.get("state")
    if not isinstance(state, Mapping):
        return False
    outcome = state.get("outcome")
    return isinstance(outcome, Mapping) and outcome.get("status") in {
        "failed",
        "aborted",
        "faulted",
        "orphaned",
    }


def _can_reserve(definition: TaskDefinition, record: Mapping[str, Any]) -> bool:
    version = int(record.get("version", 1))
    return definition.version == version or (
        definition.version > version and definition.migrate is not None
    )


def _error_value(error: Any) -> dict[str, Any]:
    if isinstance(error, Mapping):
        value = dict(error)
        value.setdefault("message", str(value.get("error", "task failed")))
        return copy.deepcopy(value)
    return {"message": str(error)}


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
    ):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _call_sync(function: Callable[..., Any], *args: Any) -> Any:
    signature = inspect.signature(function)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
    ]
    if any(
        parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values()
    ):
        return function(*args)
    return function(*args[: len(positional)])


async def _call_handler(
    function: TaskHandler, record: Mapping[str, Any], runtime: TaskRuntime
) -> Any:
    result = function(copy.deepcopy(dict(record)), runtime)
    if inspect.isawaitable(result):
        return await result
    return result


async def _call_callback(function: Callable[..., Any], *args: Any) -> Any:
    result = _call_sync(function, *args)
    if inspect.isawaitable(result):
        return await result
    return result
