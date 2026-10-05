from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from textual.widgets import Button, Label, ListView, Static

from forge_agent import AgentEvent, AgentToolResult, ToolExecutionEndEvent
from forge_cli.tui.app import ForgeTuiApp
from forge_cli.tui.fleet import FleetRun, FleetScreen, _detail_text, _run_row
from forge_cli.tui.prompt import PromptInput
from forge_cli.tui.widgets import TranscriptView, transcript_item_selection_text
from test_tui_app import FakeSession, _wait_until


def _run(
    run_id: str,
    *,
    status: str,
    notification: bool = False,
    agent: str = "reviewer",
    model: str = "fake-model",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "agent": agent,
        "status": status,
        "model": model,
        "duration_ms": 1234,
        "tool_calls": 3,
        "input_tokens": 120,
        "output_tokens": 80,
        "total_tokens": 200,
        "activity": "working" if status == "running" else status,
        "notification": notification,
        "baseline": "abc1234" if status == "completed" else None,
    }


class FleetSession(FakeSession):
    def __init__(self) -> None:
        super().__init__(messages=[HumanMessage(content="parent")])
        self._runs: tuple[dict[str, object], ...] = (
            _run("run-running", status="running", agent="scout"),
            _run("run-completed", status="completed", notification=True),
            _run("run-queued", status="queued", model="slow-model"),
        )
        self.cancelled: list[str] = []
        self.inspected: list[str] = []

    @property
    def background_subagents(self) -> tuple[dict[str, object], ...]:
        return self._runs

    def inspect_background_subagent(self, run_id: str) -> dict[str, object]:
        self.inspected.append(run_id)
        return {
            "run_id": run_id,
            "trace": {
                "items": [
                    {"kind": "tool_call", "tool": "read", "text": "[literal trace]"},
                ],
            },
            "result": {
                "final_output": "[literal result]",
                "error": None,
            },
        }

    async def cancel_background_subagent(self, run_id: str) -> dict[str, object]:
        self.cancelled.append(run_id)
        self._runs = tuple(
            {**run, "status": "cancelled", "activity": "cancelled"}
            if run["run_id"] == run_id
            else run
            for run in self._runs
        )
        return {"run_id": run_id, "status": "cancelled"}

    async def prompt(
        self,
        text: str,
        *,
        streaming_behavior: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        del streaming_behavior
        self.prompt_texts.append(text)
        if False:
            yield AgentEvent()  # pragma: no cover


class RefreshingFleetSession(FleetSession):
    def inspect_background_subagent(self, run_id: str) -> dict[str, object]:
        self.inspected.append(run_id)
        run = next(item for item in self._runs if item["run_id"] == run_id)
        status = run["status"]
        return {
            "run_id": run_id,
            "actual_model": run.get("actual_model", "primary"),
            "model_attempts": run.get("model_attempts", 1),
            "result": {
                "final_output": ("old result" if status == "running" else "new completed result"),
                "error": None,
            },
        }


@pytest.mark.anyio
async def test_fleet_refresh_during_partial_mount_retains_latest_runs() -> None:
    session = FleetSession()
    app = ForgeTuiApp(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        await pilot.pause()
        fleet = app.screen
        assert isinstance(fleet, FleetScreen)
        await fleet.query_one("#fleet-cancel", Button).remove()
        session._runs = (_run("new-committed", status="completed"),)
        app._sync_fleet_subagents()
        await fleet.query_one("#fleet-actions").mount(
            Button("Cancel selected", id="fleet-cancel"), before="#fleet-collect"
        )
        app._sync_fleet_subagents()
        await pilot.pause()
        assert any("new-committed" in str(label.render()) for label in fleet.query(Label))


@pytest.mark.anyio
async def test_fleet_summary_and_rows_keep_all_runs_visible() -> None:
    app = ForgeTuiApp(FleetSession())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        summary = app.query_one("#fleet-summary", Static)
        assert summary.display is True
        assert "3 runs" in str(summary.render())
        assert "1 running" in str(summary.render())
        assert "1 queued" in str(summary.render())
        assert "1 completed" in str(summary.render())

        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        fleet = app.screen
        assert await _wait_until(
            pilot,
            lambda: (
                bool(fleet.query("#fleet-list"))
                and bool(fleet.query_one("#fleet-list", ListView).children)
            ),
        )
        await pilot.pause()
        labels = [str(label.render()) for label in fleet.query(Label)]
        assert any("run-running" in label for label in labels)
        assert any("run-completed" in label for label in labels)
        assert any("run-queued" in label for label in labels)
        assert any("fake-model" in label and "200 tokens" in label for label in labels)
        assert isinstance(fleet.query_one("#fleet-list"), ListView)


@pytest.mark.anyio
async def test_fleet_summary_click_opens_inspector() -> None:
    app = ForgeTuiApp(FleetSession())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert await pilot.click("#fleet-summary")
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))


@pytest.mark.anyio
async def test_task_control_projection_hides_payload_but_fleet_inspector_keeps_result() -> None:
    markers = (
        "SECRET_CONTENT",
        "SECRET_RESULT",
        "SECRET_TRACE",
        "SECRET_UNKNOWN",
    )
    event = ToolExecutionEndEvent(
        result=AgentToolResult(
            tool_call_id="control-1",
            name="task_control",
            ok=True,
            content="SECRET_CONTENT",
            data={
                "run_id": "run-completed",
                "agent": "scout",
                "status": "completed",
                "duration_ms": 1234,
                "tool_calls": 3,
                "input_tokens": 120,
                "output_tokens": 80,
                "total_tokens": 200,
                "result": {"final_output": "SECRET_RESULT"},
                "trace": {"items": [{"text": "SECRET_TRACE"}]},
                "unknown": "SECRET_UNKNOWN",
            },
        )
    )
    app = ForgeTuiApp(FleetSession())

    async with app.run_test(size=(80, 24)) as pilot:
        app.adapter.apply(event)
        await app._apply_streaming_transcript_event(event)
        await pilot.pause()

        item = app.state.items[-1]
        assert item.tool_result_text is not None
        assert item.tool_result_text.startswith("✓ task_control")
        assert "run-completed" in item.tool_result_text
        assert "scout" in item.tool_result_text
        assert "200 tokens" in item.tool_result_text
        assert not any(marker in item.tool_result_text for marker in markers)

        app.action_toggle_tool_results()
        await pilot.pause()
        visible = transcript_item_selection_text(item, show_tool_results=True)
        mounted = "\n".join(
            child.selection_text
            for child in app.query_one("#transcript", TranscriptView).children
            if hasattr(child, "selection_text")
        )
        assert "task_control" in visible
        assert not any(marker in visible for marker in markers)
        assert not any(marker in mounted for marker in markers)

        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        fleet = app.screen
        assert await _wait_until(
            pilot,
            lambda: (
                bool(fleet.query("#fleet-list"))
                and bool(fleet.query_one("#fleet-list", ListView).children)
            ),
        )
        await pilot.press("down", "enter")
        assert await _wait_until(
            pilot,
            lambda: isinstance(app.screen, FleetScreen) and app.screen.view == "detail",
        )
        detail = app.screen.query_one("#fleet-detail", Static)
        assert "[literal result]" in str(detail.render())


@pytest.mark.anyio
async def test_fleet_enter_detail_esc_back_and_literal_result_is_unformatted(
    tmp_path: Path,
) -> None:
    app = ForgeTuiApp(FleetSession())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        fleet = app.screen
        assert await _wait_until(
            pilot,
            lambda: (
                bool(fleet.query("#fleet-list"))
                and bool(fleet.query_one("#fleet-list", ListView).children)
            ),
        )

        await pilot.press("down", "enter")
        assert await _wait_until(
            pilot,
            lambda: isinstance(app.screen, FleetScreen) and app.screen.view == "detail",
        )
        detail = app.screen.query_one("#fleet-detail", Static)
        rendered = str(detail.render())
        assert "[literal result]" in rendered
        assert "[literal trace]" in rendered
        assert "run-completed" in rendered
        assert "baseline: abc1234" in rendered
        assert app.session.inspected == ["run-completed"]  # type: ignore[attr-defined]
        screenshot = app.export_screenshot()
        artifact = tmp_path / "fleet-80x24.svg"
        artifact.write_text(screenshot, encoding="utf-8")
        assert "Fleet" in screenshot
        assert "run-completed" in screenshot

        await pilot.press("escape")
        assert isinstance(app.screen, FleetScreen)
        assert app.screen.view == "list"


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["running", "completed"])
async def test_fleet_refreshes_selected_detail_when_run_changes(status: str) -> None:
    session = RefreshingFleetSession()
    app = ForgeTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        fleet = app.screen
        assert await _wait_until(
            pilot,
            lambda: (
                bool(fleet.query("#fleet-list"))
                and bool(fleet.query_one("#fleet-list", ListView).children)
            ),
        )

        await pilot.press("enter")
        assert await _wait_until(
            pilot,
            lambda: isinstance(app.screen, FleetScreen) and app.screen.view == "detail",
        )
        detail = app.screen.query_one("#fleet-detail", Static)
        assert "old result" in str(detail.render())

        session._runs = tuple(
            {
                **run,
                "status": status,
                "activity": status,
                "actual_model": "fallback",
                "model_attempts": 2,
            }
            if run["run_id"] == "run-running"
            else run
            for run in session._runs
        )
        app._poll_fleet_subagents()
        await pilot.pause()

        assert ("new completed result" in str(detail.render())) == (status == "completed")
        assert "actual model: fallback" in str(detail.render())
        assert "model attempts: 2" in str(detail.render())
        assert session.inspected == ["run-running", "run-running"]


class HiddenReasoningFleetSession(FleetSession):
    def inspect_background_subagent(self, run_id: str) -> dict[str, object]:
        self.inspected.append(run_id)
        return {
            "run_id": run_id,
            "result": {
                "final_output": [
                    {"type": "reasoning", "text": "private chain"},
                    {"type": "text", "text": "Visible result"},
                ],
                "error": None,
            },
        }


@pytest.mark.anyio
async def test_fleet_hides_reasoning_in_historical_structured_result() -> None:
    app = ForgeTuiApp(HiddenReasoningFleetSession())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        fleet = app.screen
        assert await _wait_until(
            pilot,
            lambda: (
                bool(fleet.query("#fleet-list"))
                and bool(fleet.query_one("#fleet-list", ListView).children)
            ),
        )
        await pilot.press("enter")
        assert await _wait_until(
            pilot,
            lambda: isinstance(app.screen, FleetScreen) and app.screen.view == "detail",
        )

        rendered = str(app.screen.query_one("#fleet-detail", Static).render())
        assert "Visible result" in rendered
        assert "private chain" not in rendered


def test_fleet_hides_explicit_reasoning_tags_in_legacy_result_string() -> None:
    run = FleetRun.from_snapshot(_run("run-completed", status="completed"))
    assert run is not None

    rendered = _detail_text(
        run,
        {"result": {"final_output": "<think>private chain</think>Visible result"}},
    )

    assert "Visible result" in rendered
    assert "private chain" not in rendered


def test_fleet_r2_facts_are_projected_without_provider_payloads() -> None:
    snapshot = _run("run-r2", status="completed", model="requested-model")
    snapshot.update(
        {
            "actual_provider": "gateway-b",
            "actual_model": "fallback-model",
            "context": "fork",
            "inherited_messages": 6,
            "context_truncated": True,
            "model_attempts": 2,
            "fallback_reasons": [
                {"kind": "transient", "message": "SECRET_PROVIDER_ERROR"},
                {"kind": "quota", "metadata": "SECRET_METADATA"},
            ],
            "provider_metadata": "SECRET_METADATA",
        }
    )
    run = FleetRun.from_snapshot(snapshot)
    assert run is not None
    assert run.actual_provider == "gateway-b"
    assert run.actual_model == "fallback-model"
    assert run.context == "fork"
    assert run.inherited_messages == 6
    assert run.context_truncated is True
    assert run.model_attempts == 2
    assert run.fallback_reasons == ("transient", "quota")

    rendered = _detail_text(run, {"result": {"final_output": "visible"}})
    row = _run_row(run)
    assert "fallback-model" in row
    assert "gateway-b/fallback-model" in row
    assert "actual provider: gateway-b" in rendered
    assert "fork" in rendered
    assert "6 inherited" in rendered
    assert "truncated" in rendered
    assert "model attempts: 2" in rendered
    assert "fallback: transient, quota" in rendered
    assert "SECRET_" not in rendered
    assert "SECRET_" not in row


def test_fleet_detail_does_not_render_raw_error_or_unknown_metadata() -> None:
    run = FleetRun.from_snapshot(_run("run-failed", status="failed"))
    assert run is not None

    rendered = _detail_text(
        run,
        {
            "context": "summary",
            "inherited_messages": 3,
            "context_truncated": False,
            "model_attempts": 1,
            "fallback_reasons": [{"kind": "transient", "error": "SECRET_ERROR"}],
            "result": {
                "final_output": "",
                "error": "SECRET_RAW_PROVIDER_ERROR",
                "response_metadata": "SECRET_METADATA",
            },
            "metadata": "SECRET_METADATA",
        },
    )

    assert "summary" in rendered
    assert "3 inherited" in rendered
    assert "transient" in rendered
    assert "SECRET_" not in rendered


@pytest.mark.anyio
async def test_fleet_mounted_r2_artifact_keeps_only_allowlisted_facts(tmp_path) -> None:
    class R2FleetSession(FleetSession):
        def inspect_background_subagent(self, run_id: str) -> dict[str, object]:
            self.inspected.append(run_id)
            return {
                "run_id": run_id,
                "actual_model": "fallback-model",
                "context": "summary",
                "inherited_messages": 4,
                "context_truncated": False,
                "model_attempts": 2,
                "fallback_reasons": [{"kind": "transient", "detail": "SECRET"}],
                "result": {
                    "final_output": "mounted result",
                    "error": "SECRET_ERROR",
                    "provider_metadata": "SECRET_METADATA",
                },
                "provider_metadata": "SECRET_METADATA",
            }

    app = ForgeTuiApp(R2FleetSession())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        fleet = app.screen
        assert await _wait_until(
            pilot,
            lambda: (
                bool(fleet.query("#fleet-list"))
                and bool(fleet.query_one("#fleet-list", ListView).children)
            ),
        )
        await pilot.press("enter")
        assert await _wait_until(
            pilot,
            lambda: isinstance(app.screen, FleetScreen) and app.screen.view == "detail",
        )
        rendered = str(app.screen.query_one("#fleet-detail", Static).render())

        assert "fallback-model" in rendered
        assert "summary" in rendered
        assert "4 inherited" in rendered
        assert "model attempts: 2" in rendered
        assert "mounted result" in rendered
        assert "SECRET_" not in rendered
        (tmp_path / "fleet-r2.txt").write_text(rendered, encoding="utf-8")
        (tmp_path / "fleet-r2.svg").write_text(app.export_screenshot(), encoding="utf-8")


@pytest.mark.anyio
async def test_fleet_cancel_uses_exact_selected_run_id() -> None:
    session = FleetSession()
    app = ForgeTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        assert await _wait_until(
            pilot,
            lambda: (
                bool(app.screen.query("#fleet-list"))
                and bool(app.screen.query_one("#fleet-list", ListView).children)
            ),
        )
        await pilot.press("c")
        await pilot.pause()

        assert session.cancelled == ["run-running"]
        assert app.screen.view == "list"


@pytest.mark.anyio
async def test_fleet_collect_places_task_control_prompt_in_composer() -> None:
    app = ForgeTuiApp(FleetSession())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        assert await _wait_until(
            pilot,
            lambda: (
                bool(app.screen.query("#fleet-list"))
                and bool(app.screen.query_one("#fleet-list", ListView).children)
            ),
        )
        await pilot.press("down", "enter", "r")
        await pilot.pause()

        assert not isinstance(app.screen, FleetScreen)
        assert (
            app.query_one("#prompt", PromptInput).text
            == "Read completed background task run-completed with task_control and explain"
        )


@pytest.mark.anyio
async def test_fleet_copy_copies_only_visible_inspector() -> None:
    app = ForgeTuiApp(FleetSession())
    copied: list[str] = []
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        assert await _wait_until(
            pilot,
            lambda: (
                bool(app.screen.query("#fleet-list"))
                and bool(app.screen.query_one("#fleet-list", ListView).children)
            ),
        )
        await pilot.press("down", "enter", "y")
        await pilot.pause()

        assert copied
        assert "[literal result]" in copied[0]
        assert "parent" not in copied[0]


@pytest.mark.anyio
async def test_fleet_completion_notification_is_deduplicated_by_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FleetSession()
    app = ForgeTuiApp(session)
    notifications: list[str] = []
    monkeypatch.setattr(app, "_notify", lambda message, **kwargs: notifications.append(message))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._poll_fleet_subagents()
        app._poll_fleet_subagents()
        await pilot.pause()

        matching = [message for message in notifications if "run-completed" in message]
        assert len(matching) == 1
        assert "/fleet" in matching[0]
