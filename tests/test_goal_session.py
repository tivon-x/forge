from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent import GoalUpdateEvent, TodoItem, TodoUpdateEvent
from forge_agent.session import CustomEntry, JsonlSessionStorage, MessageEntry
from forge_coding import CodingSession, CodingSessionConfig, GoalCommandAction
from forge_coding.features.goals import (
    GOAL_MAX_AUTOMATIC_RUNS,
    GOAL_NAMESPACE,
    latest_goal_snapshot,
)
from forge_coding.paths import ForgePaths
from forge_coding.session_manager import SessionManager


def _config(
    tmp_path: Path,
    provider: BaseChatModel,
    storage: JsonlSessionStorage,
) -> CodingSessionConfig:
    return CodingSessionConfig(
        provider=provider,
        model="fake",
        system="You are Forge.",
        storage=storage,
        cwd=tmp_path,
        enable_subagents=False,
    )


async def _collect_session_events(stream: object) -> list[object]:
    return [event async for event in stream]  # type: ignore[attr-defined]


async def _persist_active_goal(session: CodingSession, objective: str) -> str:
    snapshot = session._goal_controller.start(objective)  # noqa: SLF001 - integration seam
    session._goal_dirty = True  # noqa: SLF001 - integration seam
    await session._persist_goal_update()  # noqa: SLF001 - integration seam
    return snapshot.id


class DelayedCancelChatModel(BaseChatModel):
    """Provider that remains in-flight until its cancellation cleanup is released."""

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "started", asyncio.Event())
        object.__setattr__(self, "cancellation_seen", asyncio.Event())
        object.__setattr__(self, "release", asyncio.Event())
        object.__setattr__(self, "closed", False)
        object.__setattr__(self, "calls", 0)

    @property
    def _llm_type(self) -> str:
        return "forge-delayed-cancel"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        calls = int(getattr(self, "calls", 0))
        object.__setattr__(self, "calls", calls + 1)
        if calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="replacement"))
            return
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellation_seen.set()
            await self.release.wait()
            raise
        yield ChatGenerationChunk(message=AIMessageChunk(content="unreachable"))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        raise AssertionError("the async stream should be used")

    async def aclose(self) -> None:
        object.__setattr__(self, "closed", True)


@pytest.mark.anyio
async def test_goal_start_continues_without_forging_human_messages(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="same progress")])
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    events = [
        event
        async for event in session.apply_goal_action(
            GoalCommandAction(action="start", objective="Ship and verify")
        )
    ]

    assert len(provider.calls) == 4  # initial run + three automatic continuations
    assert not any(isinstance(message, HumanMessage) for message in session.messages)
    assert session.goal is not None
    assert session.goal.status == "blocked"
    assert session.goal.stop_reason == "no_progress"
    assert any(isinstance(event, GoalUpdateEvent) for event in events)


@pytest.mark.anyio
async def test_goal_tool_completion_persists_snapshot_and_pairs_tool_message(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel()
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    goal_id = await _persist_active_goal(session, "Ship and verify")
    provider.responses = [
        tool_call_ai("complete-1", "goal_complete", {"goal_id": goal_id, "summary": "Verified"}),
        AIMessage(content="Goal verified."),
    ]

    events = [event async for event in session.continue_()]

    assert len(provider.calls) == 1
    assert session.goal is not None
    assert session.goal.status == "complete"
    assert session.goal.completion_summary == "Verified"
    assert any(
        isinstance(event, GoalUpdateEvent)
        and event.goal is not None
        and event.goal.status == "complete"
        for event in events
    )
    tool_calls = {
        call["id"]
        for message in session.messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    tool_results = {
        message.tool_call_id for message in session.messages if isinstance(message, ToolMessage)
    }
    assert tool_calls == {"complete-1"}
    assert tool_results == tool_calls

    entries = await storage.read_all()
    goal_entries = [
        entry
        for entry in entries
        if isinstance(entry, CustomEntry) and entry.namespace == GOAL_NAMESPACE
    ]
    assert latest_goal_snapshot(goal_entries) == session.goal

    restored_provider = ScriptedChatModel([AIMessage(content="must not run on load")])
    restored = await CodingSession.load(_config(tmp_path, restored_provider, storage))
    assert restored.goal == session.goal
    assert restored_provider.calls == []


@pytest.mark.anyio
async def test_restored_active_goal_is_paused_until_explicit_resume(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    old_goal_id = await _persist_active_goal(first, "Resume this goal")

    provider = ScriptedChatModel([AIMessage(content="resumed progress")])
    restored = await CodingSession.load(_config(tmp_path, provider, storage))

    assert provider.calls == []
    assert restored.goal is not None
    assert restored.goal.id == old_goal_id
    assert restored.goal.status == "paused"
    assert restored.goal.stop_reason == "session_restored"

    _events = [
        event async for event in restored.apply_goal_action(GoalCommandAction(action="resume"))
    ]

    assert provider.calls
    assert restored.goal is not None
    assert restored.goal.id != old_goal_id


@pytest.mark.anyio
async def test_goal_waits_for_pending_input_without_auto_continuing(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel(
        [
            tool_call_ai(
                "ask-1",
                "ask_user_question",
                {
                    "questions": [
                        {
                            "header": "Mode",
                            "question": "Which mode?",
                            "options": [
                                {"label": "Fast", "description": "Prioritize speed."},
                                {"label": "Careful", "description": "Prioritize review."},
                            ],
                            "multi_select": False,
                        }
                    ]
                },
            )
        ]
    )
    config = _config(tmp_path, provider, storage)
    config = replace(config, interactive=True)
    session = await CodingSession.load(config)

    _events = [
        event
        async for event in session.apply_goal_action(
            GoalCommandAction(action="start", objective="Ask before finishing")
        )
    ]

    assert session.is_waiting_for_input is True
    assert session.goal is not None
    assert session.goal.status == "active"
    assert len(provider.calls) == 1
    goal_id = session.goal.id
    provider.responses.append(
        tool_call_ai(
            "complete-after-answer", "goal_complete", {"goal_id": goal_id, "summary": "Verified"}
        )
    )

    _resumed = [event async for event in session.respond_to_human_input("[answer]")]

    assert session.is_waiting_for_input is False
    assert session.goal is not None
    assert session.goal.status == "complete"
    assert len(provider.calls) == 2
    assert not any(isinstance(message, HumanMessage) for message in session.messages)
    assert any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == "ask-1"
        and message.content == "[answer]"
        for message in session.messages
    )


@pytest.mark.anyio
async def test_goal_clear_writes_tombstone_and_stops_future_replay(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    await _persist_active_goal(session, "Clear me")

    events = [event async for event in session.apply_goal_action(GoalCommandAction(action="clear"))]

    assert session.goal is None
    assert len(events) == 1
    assert isinstance(events[0], GoalUpdateEvent)
    assert events[0].goal is None
    restored = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    assert restored.goal is None


@pytest.mark.anyio
async def test_session_close_waits_for_cancelled_run_before_provider_teardown(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = DelayedCancelChatModel()
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    session._owned_providers.append(provider)  # noqa: SLF001 - provider ownership seam

    async def run_prompt() -> None:
        async for _event in session.prompt("wait"):
            pass

    run_task = asyncio.create_task(run_prompt())
    await provider.started.wait()
    close_task = asyncio.create_task(session.aclose())
    await provider.cancellation_seen.wait()
    await asyncio.sleep(0.01)

    assert provider.closed is False
    provider.release.set()
    with pytest.raises(asyncio.CancelledError):
        await run_task
    await close_task
    assert provider.closed is True


@pytest.mark.anyio
async def test_cleared_goal_does_not_start_after_initial_update_is_yielded(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="must not run")])
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    start_stream = session.apply_goal_action(
        GoalCommandAction(action="start", objective="Clear before running")
    )

    first = await anext(start_stream)
    assert isinstance(first, GoalUpdateEvent)
    assert first.goal is not None
    assert provider.calls == []

    await _collect_session_events(session.apply_goal_action(GoalCommandAction(action="clear")))
    await _collect_session_events(start_stream)

    assert session.goal is None
    assert provider.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["pause", "clear"])
async def test_goal_stop_action_waits_for_cancelled_run_to_settle(
    tmp_path: Path,
    action: str,
) -> None:
    storage = JsonlSessionStorage(tmp_path / f"{action}.jsonl")
    provider = DelayedCancelChatModel()
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    await _persist_active_goal(session, "Stop only after the run settles")

    old_run = asyncio.create_task(_collect_session_events(session.continue_()))
    await provider.started.wait()
    stop = asyncio.create_task(
        _collect_session_events(session.apply_goal_action(GoalCommandAction(action=action)))
    )
    await provider.cancellation_seen.wait()
    await asyncio.sleep(0.01)

    assert stop.done() is False
    assert session.is_running is True

    provider.release.set()
    with pytest.raises(asyncio.CancelledError):
        await old_run
    await stop

    assert session.is_running is False
    if action == "pause":
        assert session.goal is not None
        assert session.goal.status == "paused"
    else:
        assert session.goal is None


@pytest.mark.anyio
async def test_goal_stale_guard_is_checked_inside_the_session_lock(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    old_goal_id = await _persist_active_goal(session, "Old goal")

    async with session._switch_lock:  # noqa: SLF001 - force the stale-check race
        edit = asyncio.create_task(
            _collect_session_events(
                session.apply_goal_action(
                    GoalCommandAction(
                        action="edit",
                        objective="Must not replace the new goal",
                        goal_id=old_goal_id,
                    )
                )
            )
        )
        await asyncio.sleep(0)
        session._goal_controller.clear(goal_id=old_goal_id)  # noqa: SLF001
        new_goal = session._goal_controller.start("New goal")  # noqa: SLF001

    with pytest.raises(RuntimeError, match="Goal changed"):
        await edit
    assert session.goal == new_goal


@pytest.mark.anyio
async def test_goal_replacement_waits_then_commits_clear_and_start_atomically(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = DelayedCancelChatModel()
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    session._owned_providers.append(provider)  # noqa: SLF001 - provider ownership seam
    old_goal_id = await _persist_active_goal(session, "Old goal")

    async def run_current_goal() -> None:
        async for _event in session.continue_():
            pass

    old_run = asyncio.create_task(run_current_goal())
    await provider.started.wait()
    replacement = asyncio.create_task(
        _collect_session_events(
            session.apply_goal_action(
                GoalCommandAction(
                    action="start",
                    objective="New goal",
                    goal_id=old_goal_id,
                    replace=True,
                )
            )
        )
    )
    await provider.cancellation_seen.wait()
    await asyncio.sleep(0.01)

    assert replacement.done() is False
    assert session.goal is not None
    assert session.goal.id == old_goal_id
    assert session.goal.objective == "Old goal"

    provider.release.set()
    with pytest.raises(asyncio.CancelledError):
        await old_run
    await replacement

    assert session.goal is not None
    assert session.goal.objective == "New goal"
    assert session.goal.id != old_goal_id
    entries = await storage.read_all()
    goal_rows = [
        entry
        for entry in entries
        if isinstance(entry, CustomEntry) and entry.namespace == GOAL_NAMESPACE
    ]
    assert any(entry.data.get("goal") is None for entry in goal_rows)


@pytest.mark.anyio
async def test_session_close_cancels_a_pending_goal_replacement(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = DelayedCancelChatModel()
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    session._owned_providers.append(provider)  # noqa: SLF001 - provider ownership seam
    old_goal_id = await _persist_active_goal(session, "Keep the old goal")

    async def run_current_goal() -> None:
        async for _event in session.continue_():
            pass

    old_run = asyncio.create_task(run_current_goal())
    await provider.started.wait()
    replacement = asyncio.create_task(
        _collect_session_events(
            session.apply_goal_action(
                GoalCommandAction(
                    action="start",
                    objective="Must not start after close",
                    goal_id=old_goal_id,
                    replace=True,
                )
            )
        )
    )
    await provider.cancellation_seen.wait()
    assert session._goal_replace_pending is True  # noqa: SLF001
    assert session._goal_replace_task is replacement  # noqa: SLF001
    close_task = asyncio.create_task(session.aclose())
    await asyncio.sleep(0)
    assert replacement.cancelling() > 0
    provider.release.set()

    with pytest.raises(asyncio.CancelledError):
        await old_run
    await close_task
    with pytest.raises(asyncio.CancelledError):
        await replacement

    assert session.goal is not None
    assert session.goal.objective == "Keep the old goal"
    assert session.goal.status == "paused"


@pytest.mark.anyio
async def test_goal_and_todo_snapshots_remain_independent(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    await session._persist_todo_update(  # noqa: SLF001 - persistence projection seam
        TodoUpdateEvent(todos=(TodoItem(content="Current step", status="in_progress"),)),
        len(session.messages),
    )
    await _persist_active_goal(session, "Exit condition")

    assert session.todos == (TodoItem(content="Current step", status="in_progress"),)
    assert session.goal is not None
    assert session.goal.objective == "Exit condition"
    entries = await storage.read_all()
    namespaces = {entry.namespace for entry in entries if isinstance(entry, CustomEntry)}
    assert {"forge.todo.v1", GOAL_NAMESPACE} <= namespaces


@pytest.mark.anyio
async def test_goal_branch_and_compaction_restore_the_active_snapshot(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="compaction summary")])
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    session._harness.append_message(HumanMessage(content="before Goal"))  # noqa: SLF001
    await session._persist_messages_since(len(session._state.messages))  # noqa: SLF001
    goal_id = await _persist_active_goal(session, "Keep the Goal")
    session._harness.append_message(AIMessage(content="after Goal"))  # noqa: SLF001
    await session._persist_messages_since(len(session._state.messages))  # noqa: SLF001

    entries = await storage.read_all()
    message_entries = [entry for entry in entries if isinstance(entry, MessageEntry)]
    before_entry = next(
        entry for entry in message_entries if entry.message.content == "before Goal"
    )
    after_entry = next(entry for entry in message_entries if entry.message.content == "after Goal")

    await session.branch_to_entry(before_entry.id)
    assert session.goal is None
    await session.branch_to_entry(after_entry.id)
    assert session.goal is not None
    assert session.goal.id == goal_id
    assert session.goal.status == "paused"
    assert session.goal.stop_reason == "session_restored"

    provider.responses = [AIMessage(content="resumed after branch")]
    branch_goal_id = session.goal.id
    _events = [
        event async for event in session.apply_goal_action(GoalCommandAction(action="resume"))
    ]
    assert session.goal is not None
    assert session.goal.id != branch_goal_id
    assert provider.calls
    resumed_goal_id = session.goal.id

    await session.compact()
    restored = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    assert restored.goal is not None
    assert restored.goal.id == resumed_goal_id


@pytest.mark.anyio
async def test_session_switch_pauses_the_old_active_goal_without_running_target_model(
    tmp_path: Path,
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / "home", agents_home=tmp_path / "agents"))
    second_cwd = tmp_path / "second"
    second_cwd.mkdir()
    first = manager.create_session(cwd=tmp_path, model="fake")
    second = manager.create_session(cwd=second_cwd, model="fake")
    provider = ScriptedChatModel()
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(first.path),
            cwd=first.cwd,
            session_id=first.id,
            session_manager=manager,
            enable_subagents=False,
        )
    )
    await _persist_active_goal(session, "Stop on switch")

    assert await session.resume(second.id) == f"Resumed session: {second.id}"
    assert provider.calls == []
    restored_old = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(first.path),
            cwd=first.cwd,
            enable_subagents=False,
        )
    )
    assert restored_old.goal is not None
    assert restored_old.goal.status == "paused"
    assert restored_old.goal.stop_reason == "cancelled"


@pytest.mark.anyio
async def test_session_close_pauses_an_idle_active_goal(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    await _persist_active_goal(session, "Stop on close")

    await session.aclose()

    restored = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    assert restored.goal is not None
    assert restored.goal.status == "paused"
    assert restored.goal.stop_reason == "cancelled"


@pytest.mark.anyio
async def test_goal_automatic_limit_is_enforced_after_25_continuations(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content=f"progress-{index}") for index in range(30)])
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    await _persist_active_goal(session, "Keep going")
    events = [event async for event in session.continue_()]

    assert len(provider.calls) == 26
    assert session.goal is not None
    assert session.goal.status == "paused"
    assert session.goal.automatic_runs == 25
    assert session.goal.stop_reason == "automatic_limit"
    assert any(
        isinstance(event, GoalUpdateEvent)
        and event.goal is not None
        and event.goal.stop_reason == "automatic_limit"
        for event in events
    )


@pytest.mark.anyio
async def test_no_progress_at_automatic_limit_keeps_blocked_terminal_state(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel(
        [
            AIMessage(content="initial"),
            *[
                AIMessage(content=f"progress-{index}")
                for index in range(GOAL_MAX_AUTOMATIC_RUNS - 3)
            ],
            AIMessage(content="repeat"),
            AIMessage(content="repeat"),
            AIMessage(content="repeat"),
        ]
    )
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    await _persist_active_goal(session, "Keep working")
    _events = [event async for event in session.continue_()]

    assert len(provider.calls) == GOAL_MAX_AUTOMATIC_RUNS + 1
    assert session.goal is not None
    assert session.goal.status == "blocked"
    assert session.goal.stop_reason == "no_progress"
    assert session.goal.automatic_runs == GOAL_MAX_AUTOMATIC_RUNS
