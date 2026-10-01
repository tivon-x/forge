from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent import SubagentRunner, SubagentRuntime, SubagentSpec
from forge_agent.tool_execution import tool_exposure
from forge_coding.commands import create_default_command_registry
from forge_coding.features.subagents import (
    coding_subagent_tool_set,
    create_task_control_tool_definition,
    create_task_tool_definition,
    ensure_task_name_available,
)
from forge_coding.paths import ForgePaths
from forge_coding.resources import ForgeResourcePaths
from forge_coding.resources.subagent_profiles import (
    CodingSubagentProfile,
    load_subagent_profiles,
)
from forge_coding.tools import ToolSet


class _Background:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, str]] = []
        self.status = "failed"

    def submit(self, *, agent: str, instruction: str) -> dict[str, object]:
        self.submitted.append((agent, instruction))
        return {
            "kind": "subagent_receipt",
            "version": 1,
            "run_id": "run-1",
            "agent": agent,
            "status": "accepted",
        }

    async def control(
        self,
        *,
        action: str,
        run_id: str | None = None,
        timeout_seconds: float = 10,
    ) -> dict[str, object]:
        return {
            "kind": "subagent_control",
            "version": 1,
            "action": action,
            "run_id": run_id,
            "timeout_seconds": timeout_seconds,
            "status": self.status,
            "error": "child failed" if self.status == "failed" else None,
        }


def _runner() -> SubagentRunner:
    return SubagentRunner(
        lambda: SubagentRuntime(provider=None),  # type: ignore[arg-type]
        [SubagentSpec("scout", "Inspect code.", "Inspect.")],
    )


def test_profile_mode_defaults_and_loads_from_frontmatter(tmp_path: Path) -> None:
    project = tmp_path / "project"
    forge_home = tmp_path / ".forge"
    paths = ForgePaths(home=forge_home)
    project_agents = paths.project_forge_agents_dir(project) / "oracle"
    project_agents.mkdir(parents=True)
    (project_agents / "AGENT.md").write_text(
        "---\ndescription: Background role\nmode: background\n---\n\nReview.",
        encoding="utf-8",
    )

    loaded = load_subagent_profiles(
        paths=ForgeResourcePaths(
            root=forge_home,
            cwd=project,
            agents_root=None,
            paths=paths,
            project_resources_allowed=True,
        )
    )

    oracle = next(profile for profile in loaded.profiles if profile.name == "oracle")
    assert oracle.mode == "background"
    assert (
        CodingSubagentProfile("role", "Role", "Prompt", (), 1, 1024, "project").mode == "foreground"
    )


@pytest.mark.anyio
async def test_background_task_returns_receipt_without_starting_child() -> None:
    background = _Background()
    tool = create_task_tool_definition(
        _runner(),
        background=background,
        interactive=True,
        profile_modes={"scout": "background"},
    ).tool

    result = await tool.execute({"agent": "scout", "instruction": "Inspect"})

    assert result.ok is True
    assert result.data == {
        "kind": "subagent_receipt",
        "version": 1,
        "run_id": "run-1",
        "agent": "scout",
        "status": "accepted",
    }
    assert background.submitted == [("scout", "Inspect")]


@pytest.mark.anyio
async def test_background_task_rejects_noninteractive_or_missing_manager() -> None:
    with pytest.raises(ValueError, match="interactive"):
        await create_task_tool_definition(
            _runner(),
            background=_Background(),
            interactive=False,
            profile_modes={"scout": "background"},
        ).tool.execute({"agent": "scout", "instruction": "Inspect"})
    with pytest.raises(ValueError, match="Background"):
        await create_task_tool_definition(
            _runner(), interactive=True, profile_modes={"scout": "background"}
        ).tool.execute({"agent": "scout", "instruction": "Inspect"})


@pytest.mark.anyio
async def test_task_control_classifies_failed_status_and_validates_input() -> None:
    background = _Background()
    tool = create_task_control_tool_definition(background).tool

    failed = await tool.execute({"action": "status", "run_id": "run-1"})
    assert failed.ok is False
    assert failed.data is not None
    assert failed.data["status"] == "failed"

    with pytest.raises(ValueError, match="timeout"):
        await tool.execute({"action": "wait", "run_id": "run-1", "timeout_seconds": 61})
    with pytest.raises(ValueError, match="timeout"):
        await tool.execute({"action": "wait", "run_id": "run-1", "timeout_seconds": True})
    with pytest.raises(ValueError, match="Unexpected"):
        await tool.execute({"action": "list", "unknown": "value"})


def test_task_control_is_model_only_and_reserved() -> None:
    control = create_task_control_tool_definition(_Background()).tool
    assert tool_exposure(control) == "model-only"
    assert all(
        item.name != "task_control"
        for item in coding_subagent_tool_set(ToolSet.from_tools([control]))
    )

    with pytest.raises(ValueError, match="task_control"):
        ensure_task_name_available([control])


def test_fleet_command_returns_ui_request_and_fallback_message(tmp_path: Path) -> None:
    from types import SimpleNamespace

    result = create_default_command_registry().execute(SimpleNamespace(), "/fleet")

    assert result.handled is True
    assert result.fleet_requested is True
    assert result.message and "Fleet" in result.message
