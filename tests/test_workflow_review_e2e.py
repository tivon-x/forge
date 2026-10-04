"""Focused R3 workflow review regressions using real session and QuickJS boundaries."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from conftest import isolate_home
from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent.session import CustomEntry, MessageEntry
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.storage import JsonlSessionStorage
from test_workflows_e2e import WorkflowModel


class BlockingWorkflowModel(ScriptedChatModel):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "child_started", asyncio.Event())
        object.__setattr__(self, "child_gate", asyncio.Event())

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        self._record_call(messages)
        if any(isinstance(message, ToolMessage) for message in messages):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="parent done"))]
            )
        human = next(message for message in reversed(messages) if isinstance(message, HumanMessage))
        if str(human.content) == "child":
            self.child_started.set()
            await self.child_gate.wait()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="child done"))])
        if str(human.content) == "execute":
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=tool_call_ai(
                            "flow",
                            "workflow",
                            {"name": "case", "args": {}},
                        )
                    )
                ]
            )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="parent done"))])


async def _consume(events) -> None:
    async for _ in events:
        pass


@pytest.mark.parametrize("ending", ["cancel", "timeout"])
def test_workflow_cancellation_persists_parent_tool_pair(tmp_path, monkeypatch, ending) -> None:
    isolate_home(monkeypatch, tmp_path)

    async def run() -> None:
        workflows = tmp_path / ".forge" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "case.js").write_text(
            ('// @options: {"timeout_ms":5000}\n' if ending == "timeout" else "")
            + "await runs.run({step_key:'a',agent:'worker',instruction:'child'});return 'done';",
            encoding="utf-8",
        )
        profile = tmp_path / ".forge" / "agents" / "worker"
        profile.mkdir(parents=True)
        (profile / "AGENT.md").write_text(
            "---\ndescription: Offline worker\n---\nReturn a bounded result.",
            encoding="utf-8",
        )
        model = BlockingWorkflowModel()
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=model,
                model="fake",
                system="offline",
                tools=[],
                trust_override="allow",
                auto_compact_enabled=False,
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
            )
        )
        try:
            prompt = asyncio.create_task(_consume(session.prompt("execute")))
            await asyncio.wait_for(model.child_started.wait(), 5)
            if ending == "cancel":
                prompt.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(prompt, 10)

            calls = [
                call
                for message in session.messages
                if isinstance(message, AIMessage)
                for call in message.tool_calls
                if call.get("id") == "flow"
            ]
            assert calls
            assert any(
                isinstance(message, ToolMessage) and message.tool_call_id == "flow"
                for message in session.messages
            )

            entries = await session._config.storage.read_all()
            assert any(
                isinstance(entry, MessageEntry)
                and isinstance(entry.message, ToolMessage)
                and entry.message.tool_call_id == "flow"
                for entry in entries
            )
            report = [
                e.data
                for e in entries
                if isinstance(e, CustomEntry) and e.namespace == "forge.workflow.v1"
            ][-1]
            assert report["status"] == ("cancelled" if ending == "cancel" else "failed")
            assert report["steps"][0]["status"] == "cancelled"
            assert session.active_background_subagents == session.active_codemode_workers == 0
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "report": report,
                        "paired_results": 1,
                        "residual_workers": 0,
                        "residual_children": 0,
                        "parent": "parent.jsonl",
                    }
                ),
                encoding="utf-8",
            )
        finally:
            await session.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("action", ["wait", "cancel", "exit"])
def test_workflow_drains_completed_child_cleanup(tmp_path, monkeypatch, action) -> None:
    isolate_home(monkeypatch, tmp_path)

    async def run() -> None:
        folder = tmp_path / ".forge" / "workflows"
        folder.mkdir(parents=True)
        control = (
            f"return await runs.control({{action:'{action}',run_id:r.run_id,timeout_seconds:0}});"
            if action != "exit"
            else "return 'done';"
        )
        (folder / "case.js").write_text(
            "const r=await runs.run({step_key:'a',agent:'scout',"
            "instruction:'child',mode:'background'});"
            "let s;do{s=await runs.control({action:'status',run_id:r.run_id});}"
            "while(s.status==='queued'||s.status==='running');" + control,
            encoding="utf-8",
        )
        gate, cleanup_started = asyncio.Event(), asyncio.Event()
        original = CodingSession._subagent_model_runtime

        async def runtime(self, spec):
            result = await original(self, spec)

            async def cleanup():
                cleanup_started.set()
                await gate.wait()
                await result.aclose()

            return replace(result, cleanup=cleanup)

        monkeypatch.setattr(CodingSession, "_subagent_model_runtime", runtime)
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=WorkflowModel(),
                model="fake",
                system="offline",
                tools=[],
                trust_override="allow",
                auto_compact_enabled=False,
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
            )
        )
        prompt = asyncio.create_task(_consume(session.prompt("execute")))
        try:
            await asyncio.wait_for(cleanup_started.wait(), 10)
            done, _ = await asyncio.wait({prompt}, timeout=1)
            assert not done, "Workflow returned while child cleanup was pending"
            assert session.active_background_subagents == 1
            gate.set()
            await asyncio.wait_for(prompt, 10)
            child = next(iter(session._subagent_runs.runs.values()))
            assert child.settled.is_set() and child.task.done()
            result = next(m for m in session.messages if isinstance(m, ToolMessage)).artifact[
                "data"
            ]
            assert result["status"] == ("completed" if action == "cancel" else "failed")
            assert session.active_background_subagents == session.active_codemode_workers == 0
            (tmp_path / "summary.json").write_text(
                json.dumps({"passed": True, "result": result, "settled": True, "task_done": True}),
                encoding="utf-8",
            )
        finally:
            gate.set()
            await asyncio.gather(prompt, return_exceptions=True)
            await session.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("blocked_role", ["scout", "worker"])
def test_sequential_workflow_stops_at_blocked_dependency(
    tmp_path, monkeypatch, blocked_role
) -> None:
    isolate_home(monkeypatch, tmp_path)

    async def run() -> None:
        folder = tmp_path / ".forge" / "workflows"
        folder.mkdir(parents=True)
        (folder / "case.js").write_text(
            Path("docs/workflows/sequential-scout-worker-review.js").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        for role in ("scout", "worker", "reviewer"):
            profile = tmp_path / ".forge" / "agents" / role
            profile.mkdir(parents=True)
            (profile / "AGENT.md").write_text(
                "---\ndescription: Offline role\nresult-format: task-report\n---\nReturn a report.",
                encoding="utf-8",
            )
        reports = [
            tool_call_ai(
                role,
                "TaskReport",
                {
                    "outcome": "blocked" if role == blocked_role else "done",
                    "summary": "Missing required input" if role == blocked_role else "done",
                    "findings": [],
                    "evidence": [],
                    "question": "Which project?" if role == blocked_role else None,
                },
            )
            for role in ("scout", "worker")[: 1 if blocked_role == "scout" else 2]
        ]
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=ScriptedChatModel(
                    [
                        tool_call_ai("flow", "workflow", {"name": "case", "args": {}}),
                        *reports,
                        AIMessage(content="done"),
                    ]
                ),
                model="fake",
                system="offline",
                tools=[],
                trust_override="allow",
                auto_compact_enabled=False,
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
            )
        )
        try:
            await _consume(session.prompt("execute"))
            result = next(m for m in session.messages if isinstance(m, ToolMessage)).artifact[
                "data"
            ]
            assert result["status"] == "completed"
            assert len(result["steps"]) == len(reports)
            assert result["value"]["outcome"] == "blocked"
            assert result["value"]["report"]["question"] == "Which project?"
            (tmp_path / "summary.json").write_text(json.dumps(result), encoding="utf-8")
        finally:
            await session.aclose()

    asyncio.run(run())
