from __future__ import annotations

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import SystemMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import ValidationError

from forge_agent import GoalSnapshot, GoalUpdateEvent
from forge_agent.session import CustomEntry
from forge_agent.tool_execution import (
    SEQUENTIAL_TOOL_EXECUTION_MODE,
    TOOL_EXECUTION_MODE_METADATA_KEY,
)
from forge_coding.features.goals import (
    GOAL_MAX_AUTOMATIC_RUNS,
    GOAL_MAX_NO_PROGRESS_RUNS,
    GOAL_NAMESPACE,
    GoalController,
    GoalMiddleware,
    GoalStateError,
    StaleGoalError,
    build_goal_prompt,
    create_goal_blocked_tool,
    create_goal_complete_tool,
    goal_entry_data,
    goal_tombstone_data,
    latest_goal_snapshot,
)


def _controller() -> GoalController:
    ids = iter(("goal-1", "goal-2", "goal-3"))
    return GoalController(clock=lambda: 10.0, id_factory=lambda: next(ids))


def test_goal_snapshot_is_strict_and_event_is_public_projection() -> None:
    snapshot = GoalSnapshot(
        id="goal-1",
        objective="Ship the parser",
        status="active",
        started_at=1,
        updated_at=1,
    )
    assert GoalUpdateEvent(goal=snapshot).type == "goal_update"
    with pytest.raises(ValidationError):
        GoalSnapshot(
            id="goal-1",
            objective="Ship the parser",
            status="active",
            started_at=1,
            updated_at=1,
            unexpected=True,
        )
    with pytest.raises(ValidationError):
        GoalSnapshot(
            id="goal-1",
            objective="Ship the parser",
            status="active",
            started_at=1,
            updated_at=1,
            automatic_runs=-1,
        )


def test_controller_transitions_rotate_ids_and_reject_stale_turns() -> None:
    controller = _controller()
    active = controller.start("Ship it")
    paused = controller.pause(goal_id=active.id, reason="user")
    with pytest.raises(GoalStateError):
        controller.pause(goal_id=paused.id)
    resumed = controller.resume(goal_id=paused.id)
    assert resumed.status == "active"
    assert resumed.id != active.id
    with pytest.raises(GoalStateError):
        controller.resume(goal_id=resumed.id)
    with pytest.raises(StaleGoalError):
        controller.complete(active.id, "done")
    edited = controller.edit("Ship and verify it", goal_id=resumed.id)
    complete = controller.complete(edited.id, "All requirements verified.")
    assert complete.status == "complete"
    assert complete.completion_summary == "All requirements verified."


def test_start_after_complete_rotates_a_max_length_reused_id() -> None:
    repeated_id = "g" * 200
    controller = GoalController(clock=lambda: 10.0, id_factory=lambda: repeated_id)
    first = controller.start("First")
    controller.complete(first.id, "Verified.")
    second = controller.start("Second")
    assert second.id != first.id
    assert len(second.id) <= 200


def test_completion_summary_is_structural_not_english_semantic_validation() -> None:
    controller = _controller()
    active = controller.start("Verify the previous failure is fixed")

    completed = controller.complete(
        active.id,
        'Verified that the previous "tests still fail" condition is resolved.',
    )

    assert completed.status == "complete"


def test_start_after_clear_does_not_reuse_a_stale_id() -> None:
    controller = GoalController(clock=lambda: 10.0, id_factory=lambda: "same-id")
    first = controller.start("First")
    controller.clear(goal_id=first.id)
    second = controller.start("Second")
    assert second.id != first.id


def test_goal_timestamps_remain_monotonic_when_the_clock_moves_backwards() -> None:
    controller = GoalController(clock=lambda: 10.0)
    first = controller.start("Clock")
    paused = controller.pause(goal_id=first.id, now=5.0)
    assert paused.updated_at == first.updated_at


def test_controller_safety_guards_pause_and_reset_progress() -> None:
    controller = _controller()
    snapshot = controller.start("Keep working")
    for _ in range(GOAL_MAX_AUTOMATIC_RUNS - 1):
        snapshot = controller.record_automatic_run(goal_id=snapshot.id)
    assert snapshot.status == "active"
    snapshot = controller.record_automatic_run(goal_id=snapshot.id)
    assert snapshot.status == "paused"
    assert snapshot.stop_reason == "automatic_limit"

    controller = _controller()
    controller.start("Check output")
    snapshot = controller.snapshot
    assert snapshot is not None
    snapshot = controller.record_output("...", goal_id=snapshot.id)
    snapshot = controller.reset_no_progress(goal_id=snapshot.id)
    assert snapshot.no_progress_runs == 0
    for _ in range(GOAL_MAX_NO_PROGRESS_RUNS):
        snapshot = controller.record_output("...", goal_id=snapshot.id)
    assert snapshot.status == "blocked"
    assert snapshot.stop_reason == "no_progress"


def test_goal_codec_honors_tombstone_and_ignores_malformed_rows() -> None:
    controller = _controller()
    snapshot = controller.start("Persist me")
    entries = [
        CustomEntry(namespace=GOAL_NAMESPACE, data=goal_entry_data(snapshot)),
        CustomEntry(namespace=GOAL_NAMESPACE, data={"goal": {"bad": True}}),
        CustomEntry(namespace=GOAL_NAMESPACE, data=goal_tombstone_data()),
    ]
    assert latest_goal_snapshot(entries) is None


def test_goal_prompt_escapes_objective_and_middleware_filters_tools() -> None:
    controller = _controller()
    model = FakeListChatModel(responses=["ok"])
    middleware = GoalMiddleware(controller)
    request = ModelRequest(model=model, messages=[], tools=[*middleware.tools])

    inactive = middleware._request(request)
    assert inactive.tools == []
    active = controller.start("Use <safe> & finish")
    assert "&lt;safe&gt; &amp;" in build_goal_prompt(active)
    active_request = middleware._request(request)
    assert {getattr(tool, "name", None) for tool in active_request.tools} == {
        "goal_complete",
        "goal_blocked",
    }
    assert isinstance(active_request.system_message, SystemMessage)
    assert "<goal_id>" in active_request.system_message.content
    controller.pause(goal_id=active.id)
    inactive_again = middleware._request(active_request)
    assert inactive_again.tools == []
    assert "Active /goal:" not in inactive_again.system_message.content


def test_goal_middleware_bounds_invalid_model_arguments() -> None:
    middleware = GoalMiddleware(_controller())
    request = ToolCallRequest(
        tool_call={
            "name": "goal_blocked",
            "id": "call-1",
            "type": "tool_call",
            "args": {
                "goal_id": "old",
                "reason": "r" * 2_000,
                "evidence": "e" * 8_000,
                "repeated_turns": 1,
            },
        },
        tool=middleware.goal_tools[1],
        state={"messages": []},
        runtime=None,  # type: ignore[arg-type]
    )
    result = middleware.wrap_tool_call(request, lambda _request: pytest.fail("handler called"))
    assert result.status == "error"
    assert len(result.content) <= 1_000


def test_goal_blocked_coerces_integer_strings_but_not_booleans() -> None:
    controller = _controller()
    active = controller.start("Wait")
    tool = create_goal_blocked_tool(controller)
    validated = tool.args_schema.model_validate(
        {
            "goal_id": active.id,
            "reason": "external",
            "evidence": "unavailable for three turns",
            "repeated_turns": "3",
        }
    )

    assert validated.repeated_turns == 3
    with pytest.raises(ValidationError):
        tool.args_schema.model_validate(
            {
                "goal_id": active.id,
                "reason": "external",
                "evidence": "unavailable for three turns",
                "repeated_turns": True,
            }
        )


@pytest.mark.anyio
async def test_goal_tools_use_native_runtime_and_stale_errors_are_bounded() -> None:
    controller = _controller()
    complete = create_goal_complete_tool(controller)
    assert complete.metadata == {TOOL_EXECUTION_MODE_METADATA_KEY: SEQUENTIAL_TOOL_EXECUTION_MODE}
    assert await complete.ainvoke({"goal_id": "old", "summary": "done"}) == "no Goal is active"

    active = controller.start("Ship")
    assert await complete.ainvoke({"goal_id": active.id, "summary": "Verified."}) == (
        "Goal marked complete."
    )
    assert controller.snapshot is not None
    assert controller.snapshot.status == "complete"

    stale_controller = _controller()
    stale_tool = create_goal_complete_tool(stale_controller)
    old = stale_controller.start("Rotate me")
    paused = stale_controller.pause(goal_id=old.id)
    stale_controller.resume(goal_id=paused.id)
    assert await stale_tool.ainvoke({"goal_id": old.id, "summary": "Verified."}) == (
        "Goal id is stale"
    )

    blocked_controller = _controller()
    blocked = create_goal_blocked_tool(blocked_controller)
    assert blocked.metadata == {TOOL_EXECUTION_MODE_METADATA_KEY: SEQUENTIAL_TOOL_EXECUTION_MODE}
    blocked_active = blocked_controller.start("Wait")
    assert (
        await blocked.ainvoke(
            {
                "goal_id": blocked_active.id,
                "reason": "external dependency",
                "evidence": "The dependency remains unavailable across three turns.",
                "repeated_turns": 3,
            }
        )
        == "Goal marked blocked."
    )
    assert blocked_controller.snapshot is not None
    assert blocked_controller.snapshot.status == "blocked"

    blocked_controller = _controller()
    blocked = create_goal_blocked_tool(blocked_controller)
    blocked_active = blocked_controller.start("Wait")
    with pytest.raises(ValidationError):
        await blocked.ainvoke(
            {
                "goal_id": blocked_active.id,
                "reason": "external",
                "evidence": "provider is unavailable",
                "repeated_turns": 2,
            }
        )
