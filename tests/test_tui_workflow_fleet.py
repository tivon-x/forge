from __future__ import annotations

import json

import pytest
from textual.widgets import Label, ListView, Static

from forge_cli.tui.app import ForgeTuiApp
from forge_cli.tui.fleet import FleetScreen
from test_tui_app import _wait_until
from test_tui_fleet import FleetSession, _run


class WorkflowFleetSession(FleetSession):
    def __init__(self) -> None:
        super().__init__()
        self._runs = (
            {
                **_run("run-scout", status="running", agent="scout"),
                "workflow_id": "wf-release",
                "step_key": "scout",
            },
            {
                **_run("run-worker", status="running", agent="worker"),
                "workflow_id": "wf-release",
                "step_key": "worker",
            },
            {
                **_run("run-review", status="completed", agent="reviewer"),
                "workflow_id": "wf-release",
                "step_key": "review",
                "result": {"report": {"outcome": "blocked"}},
            },
        )


@pytest.mark.anyio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
@pytest.mark.parametrize("theme", ["forge-dark", "forge-light", "high-contrast"])
async def test_mounted_workflow_fleet_groups_steps_and_controls_exact_run(
    tmp_path, size, theme
) -> None:
    session = WorkflowFleetSession()
    app = ForgeTuiApp(session)

    async with app.run_test(size=size) as pilot:
        app.theme = theme
        await pilot.pause()
        app.action_open_fleet()
        assert await _wait_until(pilot, lambda: isinstance(app.screen, FleetScreen))
        fleet = app.screen
        assert await _wait_until(
            pilot,
            lambda: (
                bool(fleet.query("#fleet-list"))
                and bool(fleet.query_one("#fleet-list", ListView).children)
                and all(
                    any(text in str(label.render()).lower() for label in fleet.query(Label))
                    for text in ("workflow wf-release", "step scout", "step worker", "step review")
                )
            ),
        )

        labels = [str(label.render()) for label in fleet.query(Label)]
        assert any("workflow wf-release" in label.lower() for label in labels)
        assert any("step scout" in label.lower() for label in labels)
        assert any("step worker" in label.lower() for label in labels)
        assert any("step review" in label.lower() for label in labels)

        await pilot.press("enter")
        assert await _wait_until(
            pilot,
            lambda: isinstance(app.screen, FleetScreen) and app.screen.view == "detail",
        )
        detail = str(app.screen.query_one("#fleet-detail", Static).render())
        assert "workflow: wf-release" in detail
        assert "step: scout" in detail

        await pilot.press("escape", "down", "c")
        await pilot.pause()
        assert session.cancelled == ["run-worker"]

        (tmp_path / f"fleet-workflow-{size[0]}x{size[1]}-{theme}.svg").write_text(
            app.export_screenshot(), encoding="utf-8"
        )
        (tmp_path / "fleet-workflow.txt").write_text(
            json.dumps(
                {
                    "cancelled": session.cancelled,
                    "labels": labels,
                    "detail": detail,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
