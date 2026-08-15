from types import SimpleNamespace
from typing import cast

import pytest
from rich.console import Console
from textual.app import App, ComposeResult
from textual.widgets import ListView, Static, TextArea

from forge_agent import AgentEvent
from forge_cli.tui.adapter import TuiEventAdapter
from forge_cli.tui.app import ForgeTuiApp
from forge_cli.tui.goals import (
    GoalAction,
    GoalManagerScreen,
    goal_manager_actions,
    render_goal_status,
)
from forge_cli.tui.state import TuiState


class _GoalHost(App[None]):
    def get_theme_variable_defaults(self) -> dict[str, str]:
        return {
            "forge-screen-background": "#101010",
            "forge-screen-text": "#f0f0f0",
            "forge-chrome-background": "#202020",
            "forge-chrome-text": "#f0f0f0",
            "forge-muted-text": "#aaaaaa",
            "forge-border": "#666666",
            "forge-transcript-background": "#181818",
            "forge-prompt-background": "#202020",
        }

    def compose(self) -> ComposeResult:
        yield Static("base")


def _goal(status: str, **kwargs: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": "goal-1",
        "objective": "Refactor [parser] and verify",
        "status": status,
        "automatic_runs": 3,
        "stop_reason": None,
        "completion_summary": None,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_goal_status_line_is_compact_and_literal() -> None:
    console = Console(record=True, width=80)

    console.print(render_goal_status(_goal("active")))

    assert console.export_text().strip() == "🎯 active · automatic 3/25"


def test_goal_status_uses_reason_and_terminal_status() -> None:
    paused = _goal("paused", stop_reason="no_progress")
    blocked = _goal("blocked")
    complete = _goal("complete")

    assert render_goal_status(paused).plain == "🎯 paused · no progress"
    assert render_goal_status(blocked).plain == "🎯 blocked"
    assert render_goal_status(complete).plain == "🎯 complete"


def test_goal_manager_actions_follow_snapshot_status() -> None:
    assert goal_manager_actions(None)[0] == ("start", "Start a goal…")
    assert goal_manager_actions(_goal("active"))[0] == ("pause", "Pause goal")
    assert goal_manager_actions(_goal("paused"))[0] == ("resume", "Resume goal")
    assert goal_manager_actions(_goal("paused", stop_reason="automatic_limit"))[0] == (
        "resume",
        "Review and continue…",
    )
    assert goal_manager_actions(_goal("blocked", stop_reason="no_progress"))[0] == (
        "resume",
        "Review and continue…",
    )
    assert goal_manager_actions(_goal("complete"))[0] == ("start", "Start a goal…")


def test_tui_adapter_projects_goal_update_without_transcript_row() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    goal = _goal("active")

    adapter.apply(cast(AgentEvent, SimpleNamespace(type="goal_update", goal=goal)))

    assert state.goal is goal
    assert state.items == []


def test_completed_goal_is_hidden_after_next_user_turn() -> None:
    state = TuiState()
    state.update_goal(_goal("complete"))

    assert state.goal is not None
    state.add_user_message("What is next?")

    assert state.goal is None


@pytest.mark.anyio
async def test_goal_manager_navigates_literal_text_and_closes() -> None:
    goal = _goal("active")
    app = _GoalHost()

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(GoalManagerScreen(goal))
        await pilot.pause()

        screen = app.screen
        objective = screen.query_one("#goal-manager-objective", Static)
        rendered = objective.render()
        assert getattr(rendered, "plain", str(rendered)) == ("Refactor [parser] and verify")
        actions = screen.query_one("#goal-manager-list", ListView)
        assert actions.index == 0
        await pilot.press("j")
        assert actions.index == 1
        await pilot.press("escape")
        assert not isinstance(app.screen, GoalManagerScreen)


@pytest.mark.anyio
async def test_goal_manager_emits_pause_resume_edit_and_clear_actions() -> None:
    cases = (("active", "pause", 0), ("paused", "resume", 0), ("active", "clear", 3))
    for status, expected, index in cases:
        actions: list[object] = []
        app = _GoalHost()
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(GoalManagerScreen(_goal(status), on_action=actions.append))
            await pilot.pause()
            for _ in range(index):
                await pilot.press("j")
            await pilot.press("enter")
            if expected == "clear":
                await pilot.press("y")
            await pilot.pause()

        assert getattr(actions[0], "kind", None) == expected

    actions = []
    app = _GoalHost()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(GoalManagerScreen(_goal("active"), on_action=actions.append))
        await pilot.pause()
        await pilot.press("j")
        await pilot.press("enter")
        await pilot.press("y")
        await pilot.pause()
        editor = app.screen.query_one("#goal-editor", TextArea)
        editor.text = "Updated [goal]"
        await pilot.press("ctrl+enter")
        await pilot.pause()

    assert getattr(actions[0], "kind", None) == "edit"
    assert getattr(actions[0], "objective", None) == "Updated [goal]"


@pytest.mark.anyio
async def test_goal_replacement_keeps_id_captured_before_confirmation() -> None:
    seen: list[object] = []

    async def capture(action: object) -> None:
        seen.append(action)

    # The session has changed since the confirmation was opened.  The
    # replacement intent must retain the ID captured by the original action.
    app = cast(
        ForgeTuiApp,
        SimpleNamespace(
            session=SimpleNamespace(goal=SimpleNamespace(id="goal-new")),
            _run_goal_action=capture,
        ),
    )
    action = GoalAction("start", objective="new objective", goal_id="goal-old")

    await ForgeTuiApp._replace_goal_then_start(app, action)

    assert len(seen) == 1
    replacement = seen[0]
    assert getattr(replacement, "goal_id", None) == "goal-old"
    assert getattr(replacement, "replace", False) is True


@pytest.mark.anyio
async def test_goal_edit_keeps_id_captured_before_confirmation() -> None:
    actions: list[object] = []
    app = _GoalHost()
    manager = GoalManagerScreen(
        _goal("active", id="goal-old"),
        on_action=actions.append,
    )

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(manager)
        await pilot.pause()
        await pilot.press("j")
        await pilot.press("enter")
        manager.update_goal(_goal("active", id="goal-new", objective="New goal"))
        await pilot.press("y")
        await pilot.pause()
        editor = app.screen.query_one("#goal-editor", TextArea)
        editor.text = "Edited old goal"
        await pilot.press("ctrl+enter")
        await pilot.pause()

    assert len(actions) == 1
    assert getattr(actions[0], "kind", None) == "edit"
    assert getattr(actions[0], "goal_id", None) == "goal-old"


@pytest.mark.anyio
async def test_goal_clear_keeps_id_captured_before_confirmation() -> None:
    actions: list[object] = []
    app = _GoalHost()
    manager = GoalManagerScreen(
        _goal("active", id="goal-old"),
        on_action=actions.append,
    )

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(manager)
        await pilot.pause()
        for _ in range(3):
            await pilot.press("j")
        await pilot.press("enter")
        manager.update_goal(_goal("active", id="goal-new"))
        await pilot.press("y")
        await pilot.pause()

    assert len(actions) == 1
    assert getattr(actions[0], "kind", None) == "clear"
    assert getattr(actions[0], "goal_id", None) == "goal-old"
