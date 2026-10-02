"""Offline provider-aware background run acceptance."""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from textual.widgets import ListView

from fake_models import ScriptedChatModel, ScriptedErrorChatModel
from forge_agent import (
    SubagentModelCandidate,
    SubagentRunner,
    SubagentRuntime,
    SubagentSpec,
)
from forge_agent.context import ForgeRuntimeContext
from forge_agent.session import CustomEntry
from forge_cli.tui.app import ForgeTuiApp
from forge_cli.tui.fleet import FleetScreen
from forge_coding.features.subagent_runs import RUN_NAMESPACE, SubagentRuns
from test_tui_app import FakeSession, _wait_until


class BlockingChatModel(ScriptedChatModel):
    def __init__(self) -> None:
        super().__init__([AIMessage(content="done")])
        object.__setattr__(self, "started", asyncio.Event())
        object.__setattr__(self, "gate", asyncio.Event())

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        self._record_call(messages)
        self.started.set()
        await self.gate.wait()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])


class RestoredFleetSession(FakeSession):
    def __init__(self, snapshot: dict[str, object], detail: dict[str, object]) -> None:
        super().__init__(messages=[HumanMessage(content="parent")])
        self._snapshot = snapshot
        self._detail = detail

    @property
    def background_subagents(self) -> tuple[dict[str, object], ...]:
        return (self._snapshot,)

    def inspect_background_subagent(self, run_id: str) -> dict[str, object]:
        assert run_id == self._snapshot["run_id"]
        return self._detail


@pytest.mark.anyio
async def test_provider_switch_survives_background_snapshot_restore_and_fleet(tmp_path) -> None:
    primary = ScriptedErrorChatModel(
        [AIMessage(content="unused")],
        error_message="temporary 503",
    )
    fallback = ScriptedChatModel([AIMessage(content="provider B result")])
    runtime = SubagentRuntime(
        provider=primary,
        model="same-model",
        candidates=(
            SubagentModelCandidate(
                provider=fallback,
                provider_name="gateway-b",
                model="same-model",
                context_window=16_384,
            ),
            SubagentModelCandidate(
                provider=primary,
                provider_name="gateway-a",
                model="same-model",
                context_window=16_384,
            ),
        ),
        runtime_context=ForgeRuntimeContext(workspace_root=str(tmp_path)),
    )
    runner = SubagentRunner(
        lambda: runtime,
        [
            SubagentSpec(
                name="scout",
                description="inspect",
                system_prompt="You inspect code.",
                model="gateway-a/same-model",
                fallback_models=("gateway-b/same-model",),
            )
        ],
    )
    runs = SubagentRuns(
        runner=runner,
        changed=lambda: None,
        origin=lambda: "offline",
        concurrency=1,
        queue_limit=0,
        timeout_seconds=5,
    )
    receipt = runs.submit("scout", "inspect")
    run_id = receipt["run_id"]
    assert isinstance(run_id, str)
    assert runs.runs[run_id].snapshot()["actual_provider"] == "gateway-a"
    runs.accept(run_id)
    detail = await runs.control("wait", run_id, 5)
    assert detail["status"] == "completed"
    assert detail["actual_provider"] == "gateway-b"
    assert detail["actual_model"] == "same-model"
    assert detail["fallback_reasons"] == ["transient"]

    restored = SubagentRuns(
        runner=None,
        changed=lambda: None,
        origin=lambda: "offline",
    )
    restored.restore(
        [CustomEntry(namespace=RUN_NAMESPACE, data=detail)],
    )
    restored_detail = restored.get(run_id).snapshot(detail=True)
    assert restored_detail["actual_provider"] == "gateway-b"
    assert restored_detail["fallback_reasons"] == ["transient"]

    session = RestoredFleetSession(restored.get(run_id).snapshot(), restored_detail)
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
        rendered = str(app.screen.query_one("#fleet-detail").render())
        assert "actual provider: gateway-b" in rendered
        assert "actual model: same-model" in rendered
        assert "fallback: transient" in rendered
        assert "provider B result" in rendered
        (tmp_path / "provider-fleet-summary.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "actual_provider": restored_detail["actual_provider"],
                    "fallback_reasons": restored_detail["fallback_reasons"],
                    "fleet_detail_contains_provider": "actual provider: gateway-b" in rendered,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "provider-fleet.svg").write_text(app.export_screenshot(), encoding="utf-8")

    await runs.stop()
    await restored.stop()


@pytest.mark.anyio
async def test_background_invocation_clients_close_on_discard_and_completion(tmp_path) -> None:
    closed: list[str] = []
    created = 0

    def runtime_reader() -> SubagentRuntime:
        nonlocal created
        created += 1
        label = f"client-{created}"
        model = ScriptedChatModel([AIMessage(content="done")])
        released = False

        async def cleanup() -> None:
            nonlocal released
            if not released:
                released = True
                closed.append(label)

        return SubagentRuntime(
            provider=model,
            model="same-model",
            runtime_context=ForgeRuntimeContext(workspace_root=str(tmp_path)),
            cleanup=cleanup,
        )

    runner = SubagentRunner(
        runtime_reader,
        [
            SubagentSpec(
                name="scout",
                description="inspect",
                system_prompt="You inspect code.",
            )
        ],
    )
    runs = SubagentRuns(
        runner=runner,
        changed=lambda: None,
        origin=lambda: "offline",
        concurrency=1,
        queue_limit=0,
        timeout_seconds=5,
    )

    provisional = runs.submit("scout", "discard me")
    assert isinstance(provisional["run_id"], str)
    await runs.discard_provisional()
    assert closed == ["client-1"]

    receipt = runs.submit("scout", "complete me")
    run_id = receipt["run_id"]
    assert isinstance(run_id, str)
    runs.accept(run_id)
    result = await runs.control("wait", run_id, 5)
    assert result["status"] == "completed"
    assert closed == ["client-1", "client-2"]
    await runs.stop()
    assert closed == ["client-1", "client-2"]
    (tmp_path / "provider-cleanup-summary.json").write_text(
        json.dumps({"passed": True, "created": created, "closed": closed}),
        encoding="utf-8",
    )


@pytest.mark.anyio
async def test_background_cancel_timeout_queue_and_close_cancellation_settle(tmp_path) -> None:
    def runtime(model, label: str, closed: list[str]) -> SubagentRuntime:
        released = False

        async def cleanup() -> None:
            nonlocal released
            if not released:
                released = True
                closed.append(label)

        return SubagentRuntime(
            provider=model,
            model="same-model",
            runtime_context=ForgeRuntimeContext(workspace_root=str(tmp_path)),
            cleanup=cleanup,
        )

    def runner_for(runtime_reader) -> SubagentRunner:
        return SubagentRunner(
            runtime_reader,
            [
                SubagentSpec(
                    name="scout",
                    description="inspect",
                    system_prompt="You inspect code.",
                )
            ],
        )

    closed: list[str] = []
    first_model = BlockingChatModel()
    second_model = ScriptedChatModel([AIMessage(content="queued done")])
    models = iter(
        (
            runtime(first_model, "active", closed),
            runtime(second_model, "queued", closed),
        )
    )
    runs = SubagentRuns(
        runner=runner_for(lambda: next(models)),
        changed=lambda: None,
        origin=lambda: "offline",
        concurrency=1,
        queue_limit=1,
        timeout_seconds=5,
    )
    first = runs.submit("scout", "active")
    first_id = first["run_id"]
    assert isinstance(first_id, str)
    runs.accept(first_id)
    await first_model.started.wait()
    second = runs.submit("scout", "queued")
    second_id = second["run_id"]
    assert isinstance(second_id, str)
    runs.accept(second_id)
    assert runs.get(second_id).status == "queued"
    cancelled_queued = await runs.control("cancel", second_id)
    assert cancelled_queued["status"] == "cancelled"
    assert runs.get(second_id).settled.is_set()
    assert closed == ["queued"]
    cancelled_active = await runs.control("cancel", first_id)
    assert cancelled_active["status"] == "cancelled"
    assert runs.get(first_id).settled.is_set()
    assert closed == ["queued", "active"]
    await runs.stop()

    before_start_closed: list[str] = []
    before_start_runs = SubagentRuns(
        runner=runner_for(
            lambda: runtime(
                ScriptedChatModel([AIMessage(content="must not run")]),
                "before-start",
                before_start_closed,
            )
        ),
        changed=lambda: None,
        origin=lambda: "offline",
        concurrency=1,
        queue_limit=0,
        timeout_seconds=5,
    )
    before_start = before_start_runs.submit("scout", "before-start")
    before_start_id = before_start["run_id"]
    assert isinstance(before_start_id, str)
    before_start_runs.accept(before_start_id)
    before_start_result = await before_start_runs.control("cancel", before_start_id)
    assert before_start_result["status"] == "cancelled"
    assert before_start_runs.get(before_start_id).settled.is_set()
    assert before_start_closed == ["before-start"]
    await before_start_runs.stop()

    timeout_closed: list[str] = []
    timeout_model = BlockingChatModel()
    timeout_runtime = runtime(timeout_model, "timeout", timeout_closed)
    timeout_runs = SubagentRuns(
        runner=runner_for(lambda: timeout_runtime),
        changed=lambda: None,
        origin=lambda: "offline",
        concurrency=1,
        queue_limit=0,
        timeout_seconds=0.01,
    )
    timed = timeout_runs.submit("scout", "timeout")
    timed_id = timed["run_id"]
    assert isinstance(timed_id, str)
    timeout_runs.accept(timed_id)
    timed_result = await timeout_runs.control("wait", timed_id, 5)
    assert timed_result["status"] == "failed"
    assert timeout_runs.get(timed_id).settled.is_set()
    assert timeout_closed == ["timeout"]
    await timeout_runs.stop()

    close_cancel_runtime = SubagentRuntime(
        provider=ScriptedChatModel([AIMessage(content="done")]),
        model="same-model",
        runtime_context=ForgeRuntimeContext(workspace_root=str(tmp_path)),
        cleanup=lambda: _raise_cancelled(),
    )
    close_cancel_runs = SubagentRuns(
        runner=runner_for(lambda: close_cancel_runtime),
        changed=lambda: None,
        origin=lambda: "offline",
        concurrency=1,
        queue_limit=0,
        timeout_seconds=5,
    )
    close_cancelled = close_cancel_runs.submit("scout", "close-cancel")
    close_cancelled_id = close_cancelled["run_id"]
    assert isinstance(close_cancelled_id, str)
    close_cancel_runs.accept(close_cancelled_id)
    close_cancel_task = close_cancel_runs.get(close_cancelled_id).task
    assert close_cancel_task is not None
    with pytest.raises(asyncio.CancelledError):
        await close_cancel_task
    close_cancel_run = close_cancel_runs.get(close_cancelled_id)
    assert close_cancel_run.settled.is_set()
    assert close_cancel_run.prepared is None
    await close_cancel_runs.stop()

    (tmp_path / "provider-lifecycle-summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cancelled": cancelled_active["status"],
                "queued_cancelled": cancelled_queued["status"],
                "timeout": timed_result["status"],
                "close_callback_cancelled": close_cancel_run.settled.is_set(),
            }
        ),
        encoding="utf-8",
    )


async def _raise_cancelled() -> None:
    raise asyncio.CancelledError()
