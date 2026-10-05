"""Conversation-scoped durable admission around the existing LangChain run.

The inbox is a journal product snapshot, never a second model transcript.
Only committed boundaries append its messages to the native graph.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from forge_agent import (
    AgentEvent,
    ErrorEvent,
    HumanInputRequestedEvent,
    QueueUpdateEvent,
)
from forge_agent.session import CustomEntry, LeafEntry, MessageEntry
from forge_agent.session.entries import SessionEntry
from forge_agent.session.jsonl import entry_from_json_line, entry_to_json_line
from forge_coding.features.observation import CommitEventQueue
from forge_coding.features.subagent_runs import RUN_NAMESPACE
from forge_coding.features.tasks import (
    TaskClosedError,
    TaskCoordinator,
    TaskDefinition,
    TaskRuntime,
    TaskTransaction,
)

if TYPE_CHECKING:
    from forge_coding.sessions.session import CodingSession

INBOX_NAMESPACE = "forge.inbox.v1"
RUN_KIND = "forge.run"
WhenBusy = Literal["follow_up", "steer", "reject"]


@dataclass(frozen=True, slots=True)
class Submission:
    """A durable receipt; waiting belongs to the observer, never to the work."""

    id: int
    _inputs: DurableInputs

    async def status(self) -> dict[str, Any]:
        return await self._inputs.status(self.id)

    async def wait(self, *, timeout: float | None = None) -> dict[str, Any]:
        return await self._inputs.wait(self.id, timeout=timeout)

    async def abort(self) -> str:
        return await self._inputs.withdraw(self.id)


class InboxMiddleware(AgentMiddleware):
    """Use the public native boundary after a fully paired ToolNode batch."""

    def __init__(self, inputs: DurableInputs) -> None:
        self.inputs = inputs

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del runtime
        messages = state.get("messages", [])
        # First request is already placed. Steers wait for postTools or final.
        if not messages or not isinstance(messages[-1], ToolMessage):
            return None
        placed = await self.inputs.boundary("postTools", messages)
        return {"messages": placed} if placed else None


class DurableInputs:
    """Admission and queue placement on the coordinator's single commit line."""

    def __init__(self, session: CodingSession) -> None:
        self.session = session
        self.host = uuid4().hex
        self.inbox: dict[str, Any] = {"items": []}
        self.inbox_id: int | None = None
        self.updated = asyncio.Event()
        self.closed = False
        self.coordinator: TaskCoordinator
        self.listeners: set[CommitEventQueue] = set()
        self.interactive_done = asyncio.Event()
        self.interactive_failed = False
        self.next_runs: dict[int, int] = {}

    @classmethod
    async def open(cls, session: CodingSession) -> DurableInputs:
        self = cls(session)
        self.restore(await cast(Any, session._config.storage).read_records())
        definition = TaskDefinition(
            kind=RUN_KIND,
            version=1,
            run=self.run,
            abort=self.abort_run,
        )
        self.coordinator = await TaskCoordinator.open(
            cast(Any, session._config.storage),
            [definition],
            on_terminal=self.terminal,
            on_commit=self.adopt,
            on_error=self.publication_failed,
        )
        return self

    def publication_failed(self, error: BaseException) -> None:
        self.session._invalidate_storage_session(error)
        self.updated.set()
        self.emit(self.inbox.get("run", {}).get("taskId", 0), error)

    def restore(self, records: dict[str, Any]) -> None:
        for document in records.get("documents", []):
            record = document["record"]
            if record["kind"] == INBOX_NAMESPACE and record["scope"] == {
                "kind": "conversation",
                "conversationId": 0,
            }:
                self.inbox_id = record["id"]
                self.inbox = copy.deepcopy(document["value"])

    async def adopt(self, tx: TaskTransaction, sequence: int) -> None:
        """Adopt only after main publication, before handlers may be dispatched."""
        del sequence
        entries = await self.session._config.storage.read_all()
        previous_run = self.inbox.get("run")
        self.restore(await cast(Any, self.session._config.storage).read_records())
        current_run = self.inbox.get("run")
        if previous_run and (not current_run or previous_run["taskId"] != current_run["taskId"]):
            self.emit(previous_run["taskId"], self.queue_event())
        if (
            self.listeners
            and previous_run
            and current_run
            and previous_run["taskId"] != current_run["taskId"]
        ):
            self.next_runs[previous_run["taskId"]] = current_run["taskId"]
        self.session._pending_initial_entries = ()
        leaf = next(
            (entry.entry_id for entry in reversed(entries) if isinstance(entry, LeafEntry)),
            self.session._last_parent_id,
        )
        await self.session._refresh_persisted_state(leaf_id=leaf)
        self.session._last_parent_id = self.session._state.active_leaf_id
        await self.session._refresh_committed_view()
        accepted = {
            cast(str, entry.data["run_id"]): entry.parent_id
            for entry in tx.entries
            if isinstance(entry, CustomEntry)
            and entry.namespace == RUN_NAMESPACE
            and isinstance(entry.data.get("run_id"), str)
            and entry.data.get("status") == "queued"
        }
        self.session._accept_message_runs(accepted)
        if self.session._config.index_on_first_persist:
            self.session._index_current_session()
        self.updated.set()
        self.updated = asyncio.Event()

    def save_inbox(self, tx: TaskTransaction, inbox: dict[str, Any]) -> None:
        content = {"kind": "base", "version": 1, "value": inbox}
        if self.inbox_id is None:
            tx.stage(
                {
                    "type": "document.create",
                    "record": {
                        "id": tx.allocate(),
                        "kind": INBOX_NAMESPACE,
                        "scope": {"kind": "conversation", "conversationId": 0},
                        "history": "latest",
                    },
                    "content": content,
                }
            )
        else:
            tx.stage({"type": "document.change", "id": self.inbox_id, "content": content})

    def append_messages(
        self,
        tx: TaskTransaction,
        messages: Sequence[AnyMessage],
        *,
        parent: str | None = None,
    ) -> list[str]:
        entries, parent, _ = self.session._message_entries(messages, parent=parent)
        ids = [entry.id for entry in entries if isinstance(entry, MessageEntry)]
        for entry in entries:
            tx.add_entry(entry)
        if ids:
            tx.add_entry(LeafEntry(parent_id=parent, entry_id=parent))
        return ids

    def start(self, tx: TaskTransaction, inbox: dict[str, Any], inputs: list[int]) -> int:
        task_id = tx.create(
            RUN_KIND,
            {"inputs": inputs},
            {"phase": "prepare"},
            conversation_id=0,
        )
        inbox["run"] = {"taskId": task_id, "inputs": inputs}
        return task_id

    async def submit(
        self,
        content: str | SessionEntry,
        *,
        request_id: str | None = None,
        when_busy: WhenBusy = "follow_up",
        type: Literal["input", "write"] = "input",
    ) -> Submission:
        if request_id is not None and not isinstance(request_id, str):
            raise TypeError("request_id must be text")

        def admit(tx: TaskTransaction) -> int:
            if request_id is not None:
                for existing in tx.submissions.values():
                    if existing["conversationId"] == 0 and existing.get("requestId") == request_id:
                        if existing["type"] != type:
                            raise ValueError(
                                f"Request {request_id} already identifies type {existing['type']}"
                            )
                        return cast(int, existing["id"])
            if when_busy not in {"follow_up", "steer", "reject"}:
                raise ValueError("when_busy must be follow_up, steer or reject")
            if type not in {"input", "write"}:
                raise ValueError("submission type must be input or write")
            if type == "input" and not isinstance(content, str):
                raise TypeError("input content must be text")
            if self.session.is_waiting_for_input and type == "input" and when_busy == "reject":
                raise RuntimeError("CodingSession is waiting for human input")
            inbox = copy.deepcopy(self.inbox)
            busy = "run" in inbox or self.session.is_running
            if busy and type == "input" and when_busy == "reject":
                raise RuntimeError("CodingSession is already running")
            for entry in self.session._pending_initial_entries:
                tx.add_entry(entry)
            submission_id = tx.allocate()
            record: dict[str, Any] = {
                "id": submission_id,
                "conversationId": 0,
                "type": type,
                "status": "queued",
            }
            if request_id is not None:
                record["requestId"] = request_id
            tx.set_submission(record)
            item: dict[str, Any] = {"id": submission_id}
            if type == "write":
                entry = (
                    content
                    if not isinstance(content, str)
                    else CustomEntry(
                        namespace="forge.write.v1",
                        data={"content": content},
                    )
                )
                item.update(mode="write", entry=json.loads(entry_to_json_line(entry)))
            else:
                item.update(
                    mode="steer" if when_busy == "steer" else "follow_up",
                    content=self.session.expand_prompt_text(cast(str, content)),
                )
            inbox["items"].append(item)
            if not busy:
                users, _ = self.place(tx, inbox, "final")
                if users:
                    self.start(tx, inbox, users)
            self.save_inbox(tx, inbox)
            return submission_id

        submission_id = await self.coordinator.change(admit)
        await self.coordinator.resume()
        return Submission(submission_id, self)

    async def start_continuation(self, options: dict[str, Any]) -> int | None:
        expected = options.get("expected_goal_id")
        goal = self.session.goal
        if expected is not None and (
            goal is None or goal.id != expected or goal.status != "active"
        ):
            return None
        if self.session.is_waiting_for_input:
            raise RuntimeError("CodingSession is waiting for human input")
        owned = options.get("run_already_owned", False)
        if self.session._goal_replace_pending and not owned:
            raise RuntimeError("Goal replacement is in progress")

        def start(tx: TaskTransaction) -> int:
            inbox = copy.deepcopy(self.inbox)
            if "run" in inbox or self.session.is_running and not owned:
                raise RuntimeError("CodingSession is already running")
            for entry in self.session._pending_initial_entries:
                tx.add_entry(entry)
            task_id = self.start(tx, inbox, [])
            tx.tasks[task_id]["input"]["continuation"] = copy.deepcopy(options)
            tx.set_task(tx.tasks[task_id])
            self.save_inbox(tx, inbox)
            return task_id

        task_id = cast(int, await self.coordinator.change(start))
        await self.coordinator.resume()
        return task_id

    def place(
        self,
        tx: TaskTransaction,
        inbox: dict[str, Any],
        at: str,
        messages: Sequence[AnyMessage] = (),
    ) -> tuple[list[int], list[HumanMessage]]:
        items = inbox["items"]
        writes = [item for item in items if item["mode"] == "write"]
        users: list[dict[str, Any]] = []
        for mode in ("steer", "follow_up"):
            if mode == "follow_up" and at != "final":
                continue
            selected = [item for item in items if item["mode"] == mode]
            users.extend(
                selected[:1] if self.session._harness.config.queue_mode != "all" else selected
            )
        users.sort(key=lambda item: item["id"])
        if not writes and not users:
            return [], []
        # Native state is paired here even if the v3 projector is still draining.
        self.append_messages(tx, messages[len(self.session._state.messages) :])
        parent = next(
            (entry.entry_id for entry in reversed(tx.entries) if isinstance(entry, LeafEntry)),
            self.session._last_parent_id,
        )
        for item in writes:
            entry = entry_from_json_line(json.dumps(item["entry"]))
            entry = entry.model_copy(update={"parent_id": parent})
            tx.add_entry(entry)
            parent = entry.id
            record = {**tx.submissions[item["id"]], "status": "done", "entry": entry.id}
            tx.set_submission(record)
        human = [HumanMessage(content=item["content"]) for item in users]
        entry_ids = self.append_messages(tx, human, parent=parent)
        if writes and not human:
            tx.add_entry(LeafEntry(parent_id=parent, entry_id=parent))
        for item, entry_id in zip(users, entry_ids, strict=True):
            tx.set_submission({**tx.submissions[item["id"]], "status": "placed", "entry": entry_id})
        removed = {item["id"] for item in [*writes, *users]}
        inbox["items"] = [item for item in items if item["id"] not in removed]
        return [item["id"] for item in users], human

    async def boundary(self, at: str, messages: Sequence[AnyMessage]) -> list[HumanMessage]:
        def change(tx: TaskTransaction) -> list[HumanMessage]:
            inbox = copy.deepcopy(self.inbox)
            if "run" not in inbox:
                return []
            task = tx.tasks[inbox["run"]["taskId"]]
            if task["abortRequested"] or task["state"]["status"] != "running":
                return []
            users, placed = self.place(tx, inbox, at, messages)
            if users or any(item["mode"] == "write" for item in self.inbox["items"]):
                inbox["run"]["inputs"].extend(users)
                self.save_inbox(tx, inbox)
            return placed

        placed = await self.coordinator.change(change)
        if placed:
            self.emit(self.inbox["run"]["taskId"], self.queue_event())
        return cast(list[HumanMessage], placed)

    def terminal(self, tx: TaskTransaction, record: dict[str, Any]) -> None:
        if record["kind"] != RUN_KIND or self.inbox.get("run", {}).get("taskId") != record["id"]:
            return
        inbox = copy.deepcopy(self.inbox)
        outcome = record["state"]["outcome"]
        answer = next(
            (
                entry.id
                for entry in reversed(self.session._state.entries)
                if isinstance(entry, MessageEntry)
                and isinstance(entry.message, AIMessage)
                and not entry.message.tool_calls
            ),
            None,
        )
        done = outcome["status"] == "completed" and answer is not None
        reason = outcome.get(
            "reason", outcome.get("error", {}).get("detail", {}).get("reason", outcome["status"])
        )
        for input_id in inbox["run"]["inputs"]:
            submission = tx.submissions[input_id]
            if submission["status"] != "placed":
                continue
            settlement = (
                {"status": "done", "answer": answer}
                if done
                else {
                    "status": "unanswered",
                    "reason": reason,
                }
            )
            tx.set_submission({**submission, **settlement})
        del inbox["run"]
        if done and not self.closed:
            users, _ = self.place(tx, inbox, "final")
            if users:
                self.start(tx, inbox, users)
        self.save_inbox(tx, inbox)

    async def run(self, record: dict[str, Any], runtime: TaskRuntime) -> None:
        checkpoint = record["state"]["checkpoint"]
        if checkpoint["phase"] == "prepare":
            await runtime.checkpoint({"phase": "execute", "host": self.host})
            return
        if checkpoint.get("host") != self.host:
            await runtime.commit(
                {
                    "status": "terminal",
                    "outcome": {
                        "status": "failed",
                        "error": {
                            "message": "Model/tool execution interrupted; D4 recovery unavailable",
                            "detail": {"reason": "interrupted"},
                        },
                    },
                }
            )
            return
        failed = False
        self.interactive_failed = False
        question: HumanInputRequestedEvent | None = None
        self.session._harness.replace_messages(self.session._state.messages)
        try:
            content = next(
                (
                    cast(str, message.content)
                    for message in reversed(self.session._state.messages)
                    if isinstance(message, HumanMessage)
                ),
                "",
            )
            continuation = record["input"].get("continuation")
            events = (
                self.session._continue_legacy(
                    _run_already_owned=continuation.get("run_already_owned", False),
                    _expected_goal_id=continuation.get("expected_goal_id"),
                )
                if continuation is not None
                else self.session._prompt_legacy(content, _placed=True)
            )
            async for event in events:
                failed = failed or isinstance(event, ErrorEvent) and not event.recoverable
                if isinstance(event, HumanInputRequestedEvent):
                    question = event
                else:
                    self.emit(record["id"], event)
            if self.session.is_waiting_for_input:
                self.interactive_done.clear()
                if question is not None:
                    self.emit(record["id"], question)
                await self.interactive_done.wait()
                failed = failed or self.interactive_failed
            if failed:
                await runtime.fail("Agent run failed")
            else:
                await runtime.complete(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.emit(record["id"], exc)
            await runtime.fail(str(exc))
        finally:
            self.emit(record["id"], None)

    async def abort_run(self, record: dict[str, Any], runtime: TaskRuntime) -> None:
        del record
        self.session._harness.cancel_pending_input()
        await self.session._persist_messages_since(len(self.session._state.messages))
        await runtime.commit({"status": "terminal", "outcome": {"status": "aborted"}})

    async def withdraw_latest(self, mode: str | None = None) -> str | None:
        def change(tx: TaskTransaction) -> str | None:
            inbox = copy.deepcopy(self.inbox)
            item = next(
                (
                    item
                    for item in reversed(inbox["items"])
                    if item["mode"] != "write" and (mode is None or item["mode"] == mode)
                ),
                None,
            )
            if item is None:
                return None
            inbox["items"].remove(item)
            tx.set_submission(
                {**tx.submissions[item["id"]], "status": "unanswered", "reason": "aborted"}
            )
            self.save_inbox(tx, inbox)
            return cast(str, item["content"])

        return cast(str | None, await self.coordinator.change(change))

    def emit(self, task_id: int, event: AgentEvent | BaseException | None) -> None:
        sequence = self.session.snapshot()["sequence"]
        for listener in self.listeners:
            listener.put_nowait((task_id, event, cast(int, sequence)))

    def queue_event(self) -> QueueUpdateEvent:
        return QueueUpdateEvent(
            steering=tuple(
                item["content"] for item in self.inbox["items"] if item["mode"] == "steer"
            ),
            follow_up=tuple(
                item["content"] for item in self.inbox["items"] if item["mode"] == "follow_up"
            ),
        )

    async def events(
        self,
        content: str | None,
        *,
        when_busy: WhenBusy,
        continuation: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        listener = CommitEventQueue(self.session.snapshot)
        self.listeners.add(listener)
        try:
            task_id = None
            submission = None
            async with self.session._switch_lock:
                self.session._assert_session_usable()
                current = self.session._durable_inputs
                if current is self:
                    if continuation is not None:
                        task_id = await self.start_continuation(continuation)
                        if task_id is None:
                            return
                    else:
                        assert content is not None
                        submission = await self.submit(content, when_busy=when_busy)
            if current is not self:
                if current is None:
                    raise RuntimeError("Durable session was replaced")
                async for forwarded in current.events(
                    content, when_busy=when_busy, continuation=continuation
                ):
                    yield forwarded
                return
            if submission is not None and (await submission.status())["status"] == "queued":
                yield self.queue_event()
                return
            if task_id is None:
                task_id = self.inbox.get("run", {}).get("taskId")
            while True:
                event_task, event = await listener.get()
                if task_id is None:
                    task_id = event_task
                if event_task != task_id:
                    continue
                if event is None:
                    if self.closed:
                        raise asyncio.CancelledError
                    settled = await self.coordinator.wait_task(task_id)
                    if settled["state"]["outcome"]["status"] == "aborted":
                        raise asyncio.CancelledError
                    task_id = self.next_runs.pop(task_id, None)
                    if task_id is None:
                        break
                    continue
                if isinstance(event, BaseException):
                    raise event
                yield event
                if isinstance(event, HumanInputRequestedEvent):
                    return
        except asyncio.CancelledError:
            # The historical prompt consumer cancels its turn. SDK wait does not.
            if not self.closed:
                try:
                    await asyncio.shield(self.abort())
                except TaskClosedError:
                    if not self.closed:
                        raise
            raise
        finally:
            self.listeners.discard(listener)
            if not self.listeners:
                self.next_runs.clear()

    async def status(self, submission_id: int) -> dict[str, Any]:
        def read(tx: TaskTransaction) -> dict[str, Any]:
            record = tx.submissions.get(submission_id)
            if record is None or record["conversationId"] != 0:
                raise ValueError(f"Submission {submission_id} does not exist")
            return copy.deepcopy(record)

        return cast(dict[str, Any], await self.coordinator.change(read))

    async def wait(self, submission_id: int, *, timeout: float | None = None) -> dict[str, Any]:
        await self.coordinator.resume()
        async with asyncio.timeout(timeout):
            while True:
                changed = self.updated
                record = await self.status(submission_id)
                if record["status"] in {"done", "unanswered"}:
                    return record
                if self.closed:
                    raise RuntimeError("CodingSession is closed")
                await changed.wait()

    async def withdraw(self, submission_id: int) -> str:
        def change(tx: TaskTransaction) -> str:
            record = tx.submissions.get(submission_id)
            if record is None:
                raise ValueError(f"Submission {submission_id} does not exist")
            if record["status"] != "queued":
                return "already_placed" if record["status"] == "placed" else "settled"
            inbox = copy.deepcopy(self.inbox)
            inbox["items"] = [item for item in inbox["items"] if item["id"] != submission_id]
            tx.set_submission({**record, "status": "unanswered", "reason": "aborted"})
            self.save_inbox(tx, inbox)
            return "aborted"

        return cast(str, await self.coordinator.change(change))

    async def abort(self, *, include_background: bool = False) -> None:
        def withdraw_inputs(tx: TaskTransaction) -> None:
            inbox = copy.deepcopy(self.inbox)
            for item in inbox["items"]:
                if item["mode"] != "write":
                    tx.set_submission(
                        {**tx.submissions[item["id"]], "status": "unanswered", "reason": "aborted"}
                    )
            inbox["items"] = [item for item in inbox["items"] if item["mode"] == "write"]
            self.save_inbox(tx, inbox)

        await self.coordinator.abort_conversation(
            0, include_background=include_background, before_mark=withdraw_inputs
        )

    async def close(self) -> None:
        self.closed = True
        self.updated.set()
        await self.coordinator.close()
