"""Real QuickJS worker contracts for the Codemode workflow host."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

import forge_coding.codemode.runtime as runtime_module
from forge_coding.codemode.runtime import CodemodeRuntime


class _Discovery:
    def reader(self) -> tuple[()]:
        return ()

    def search(self, query: str, *, limit: int = 5, namespace: str | None = None) -> list[Any]:
        del query, limit, namespace
        return []


class _Nested:
    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError(f"unexpected native tool call: {name}({arguments})")


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CodemodeRuntime:
    monkeypatch.setattr(runtime_module, "get_nested_tool_executor", lambda: _Nested())

    async def persist(namespace: str, data: dict[str, Any]) -> None:
        del namespace, data

    return CodemodeRuntime(tmp_path, _Discovery(), persist)


@pytest.mark.anyio
async def test_real_worker_exposes_workflow_runs_and_freezes_args(
    runtime: CodemodeRuntime, tmp_path: Path
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def runs_handler(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, dict(arguments)))
        if name == "runs.run":
            return {"run_id": "r1", "status": "completed", "outcome": "blocked"}
        if name == "runs.control":
            return {"run_id": "r1", "status": "cancelled"}
        raise AssertionError(name)

    value = {
        "marker": "literal'); text('not-code'); //",
        "nested": {"value": 1},
    }
    _, artifact = await runtime.execute(
        """
        const before = workflowArgs.marker;
        try { workflowArgs.marker = 'changed'; } catch (error) {}
        try { workflowArgs.nested.value = 9; } catch (error) {}
        const receipt = await runs.run({step_key:'s1', agent:'worker', instruction:'inspect'});
        const stopped = await runs.control({action:'cancel', run_id:receipt.run_id});
        return {before, after:workflowArgs.marker, nested:workflowArgs.nested.value,
            receipt, stopped};
        """,
        workflow_args=value,
        runs_handler=runs_handler,
        timeout_seconds=5,
    )

    payload = artifact["forge.codemode.v1"]
    assert payload["ok"] is True
    assert payload["value"]["before"] == value["marker"]
    assert payload["value"]["after"] == value["marker"]
    assert payload["value"]["nested"] == 1
    assert [name for name, _ in calls] == ["runs.run", "runs.control"]
    assert calls[0][1]["step_key"] == "s1"
    assert runtime.workers == set()

    evidence = tmp_path / "workflow-worker-evidence.json"
    evidence.write_text(
        json.dumps(
            {"scope": "workflow", "runs": [name for name, _ in calls], "workers": 0},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert evidence.is_file()


@pytest.mark.anyio
async def test_plain_codemode_has_no_workflow_host(runtime: CodemodeRuntime) -> None:
    _, artifact = await runtime.execute("return typeof runs;", timeout_seconds=5)
    payload = artifact["forge.codemode.v1"]
    assert payload["ok"] is True
    assert payload["value"] == "undefined"


@pytest.mark.anyio
async def test_workflow_report_limit_fails_without_json_truncation(
    runtime: CodemodeRuntime,
) -> None:
    content, artifact = await runtime.execute(
        "return 'x'.repeat(70000);",
        workflow_args={},
        runs_handler=None,
        timeout_seconds=5,
        max_report_bytes=65536,
    )
    payload = artifact["forge.codemode.v1"]
    assert payload["ok"] is False
    assert "report" in str(payload["error"]).lower()
    assert payload.get("truncated") is not True
    assert artifact["forge.status"] == "error"
    assert content.startswith("Script failed\n")


@pytest.mark.anyio
async def test_workflow_timeout_cleans_real_worker(runtime: CodemodeRuntime) -> None:
    _, artifact = await runtime.execute(
        "while (true) {}",
        workflow_args={},
        timeout_seconds=1,
    )
    assert artifact["forge.codemode.v1"]["ok"] is False
    assert runtime.workers == set()


@pytest.mark.anyio
async def test_workflow_cancellation_drains_host_call_and_worker(
    runtime: CodemodeRuntime,
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def runs_handler(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del name, arguments
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()
        return {}

    task = asyncio.create_task(
        runtime.execute(
            "await runs.run({step_key:'s1', agent:'worker', instruction:'wait'});",
            workflow_args={},
            runs_handler=runs_handler,
            timeout_seconds=30,
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert runtime.workers == set()


@pytest.mark.anyio
async def test_repeated_cancellation_waits_for_blocked_cleanup(
    runtime: CodemodeRuntime,
) -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_gate = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def runs_handler(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del name, arguments
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            cleanup_started.set()
            await cleanup_gate.wait()
            cleanup_finished.set()
        return {}

    task = asyncio.create_task(
        runtime.execute(
            "await runs.run({step_key:'s1', agent:'worker', instruction:'wait'});",
            workflow_args={},
            runs_handler=runs_handler,
            timeout_seconds=30,
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    await asyncio.wait_for(cleanup_started.wait(), 5)
    task.cancel()
    cleanup_gate.set()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert cleanup_finished.is_set()
    assert runtime.workers == set()


@pytest.mark.anyio
async def test_cancel_during_normal_completion_cleanup_propagates(
    runtime: CodemodeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    kill_started = asyncio.Event()
    kill_gate = threading.Event()
    original_kill = runtime_module._kill_process_tree

    def gated_kill(process: asyncio.subprocess.Process) -> None:
        loop.call_soon_threadsafe(kill_started.set)
        if not kill_gate.wait(5):
            raise RuntimeError("cleanup gate timed out")
        original_kill(process)

    monkeypatch.setattr(runtime_module, "_kill_process_tree", gated_kill)

    task = asyncio.create_task(
        runtime.execute(
            "return 42;",
            workflow_args={},
            timeout_seconds=30,
        )
    )
    await asyncio.wait_for(kill_started.wait(), 5)
    task.cancel()
    kill_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert runtime.workers == set()
