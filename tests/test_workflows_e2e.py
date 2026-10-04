"""R3 failure contract: scope isolation, idempotent steps, bounded launch,
safe resources and args, paired parent calls, and cleanup with partial reports.
Real session graphs and QuickJS workers use offline models exclusively.
"""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from conftest import isolate_home
from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent.session import CustomEntry, MessageEntry
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.resources import ForgeResourcePaths
from forge_coding.sessions.storage import JsonlSessionStorage
from forge_coding.tools import create_write_tool


def test_workflow_session_sequence_and_idempotency(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    folder = tmp_path / ".forge" / "workflows"
    folder.mkdir(parents=True)
    (folder / "sequence.js").write_text(
        "const a = await runs.run({step_key:'read',agent:'scout',instruction:'child'});"
        "const b = await runs.run({step_key:'read',agent:'scout',instruction:'child'});"
        "if(a.run_id !== b.run_id) throw Error('duplicate');"
        "return {arg:workflowArgs.value, status:b.status};",
        encoding="utf-8",
    )

    async def run():
        model = ScriptedChatModel(
            [
                tool_call_ai(
                    "flow", "workflow", {"name": "sequence", "args": {"value": "';exit();"}}
                ),
                AIMessage(content="CHILD_RESULT"),
                AIMessage(content="PARENT_DONE"),
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=model,
                model="fake",
                system="offline",
                tools=[],
                resource_paths=ForgeResourcePaths(cwd=tmp_path, project_resources_allowed=True),
                trust_override="allow",
                auto_compact_enabled=False,
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
            )
        )
        try:
            assert [w.name for w in session.workflows] == ["sequence"]
            async for _ in session.prompt("execute workflow"):
                pass
            result = next(m for m in session.messages if isinstance(m, ToolMessage))
            report = result.artifact["data"]
            assert report["status"] == "completed", result
            assert report["value"] == {"arg": "';exit();", "status": "completed"}
            assert len(report["steps"]) == 1
            assert session.active_background_subagents == session.active_codemode_workers == 0
            assert len(model.calls) == 3
            assert len([m for m in session.messages if isinstance(m, ToolMessage)]) == 1
            (tmp_path / "summary.json").write_text(json.dumps(report), encoding="utf-8")
        finally:
            await session.aclose()

    asyncio.run(run())


class WorkflowModel(ScriptedChatModel):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._record_call(messages)
        human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        if str(human.content).startswith("child"):
            await asyncio.sleep(0.02)
            message = AIMessage(content="CHILD_RESULT")
        elif not isinstance(messages[-1], ToolMessage):
            message = tool_call_ai("flow", "workflow", {"name": "case", "args": {}})
        else:
            message = AIMessage(content="PARENT_DONE")
        return ChatResult(generations=[ChatGeneration(message=message)])


class ReportWorkflowModel(ScriptedChatModel):
    def __init__(self, reviews_with_findings=0, blocked=False):
        super().__init__()
        object.__setattr__(self, "remaining_reviews", reviews_with_findings)
        object.__setattr__(self, "blocked_review", blocked)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._record_call(messages)
        human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        text = str(human.content)
        if text == "execute saved":
            message = (
                AIMessage(content="done")
                if isinstance(messages[-1], ToolMessage)
                else tool_call_ai(
                    "flow", "workflow", {"name": "case", "args": {"target": "offline target"}}
                )
            )
        else:
            reviewer = text.startswith("Review")
            findings = ["Fix the concrete issue"] if reviewer and self.remaining_reviews else []
            if reviewer and self.remaining_reviews:
                object.__setattr__(self, "remaining_reviews", self.remaining_reviews - 1)
            report = {
                "outcome": "blocked" if reviewer and self.blocked_review else "done",
                "summary": "offline handoff",
                "findings": findings,
                "evidence": ["synthetic evidence"],
                "question": "Need user input" if reviewer and self.blocked_review else None,
            }
            message = tool_call_ai("report-" + str(len(self.calls)), "TaskReport", report)
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.parametrize(
    "example,reviews,blocked,steps",
    [
        ("sequential-scout-worker-review.js", 0, False, 3),
        ("parallel-review.js", 0, False, 2),
        ("parallel-review.js", 0, True, 2),
        ("review-fix-loop.js", 1, False, 3),
        ("review-fix-loop.js", 8, False, 6),
        ("review-fix-loop.js", 0, True, 1),
    ],
)
def test_saved_examples_with_real_structured_children(
    tmp_path, monkeypatch, example, reviews, blocked, steps
):
    isolate_home(monkeypatch, tmp_path)
    folder = tmp_path / ".forge" / "workflows"
    folder.mkdir(parents=True)
    code = (Path(__file__).resolve().parents[1] / "docs" / "workflows" / example).read_text(
        encoding="utf-8"
    )
    (folder / "case.js").write_text(code, encoding="utf-8")
    for role in ("scout", "worker", "reviewer"):
        root = tmp_path / ".forge" / "agents" / role
        root.mkdir(parents=True)
        (root / "AGENT.md").write_text(
            "---\ndescription: Offline role\nresult-format: task-report\n---\n"
            "Return a bounded report.",
            encoding="utf-8",
        )

    async def run():
        model = ReportWorkflowModel(reviews, blocked)
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
            async for _ in session.prompt("execute saved"):
                pass
            tool = next(m for m in session.messages if isinstance(m, ToolMessage))
            report = tool.artifact["data"]
            assert report["status"] == "completed", tool
            assert len(report["steps"]) == steps
            if blocked:
                assert report["value"]["outcome"] == "blocked"
            assert all(
                row["result"]["report"]["summary"] == "offline handoff" for row in report["steps"]
            )
            assert session.active_codemode_workers == session.active_background_subagents == 0
            await session.flush_background_subagents()
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "example": example,
                        "report": report,
                        "parent": "parent.jsonl",
                        "residual_workers": 0,
                        "residual_children": 0,
                    }
                ),
                encoding="utf-8",
            )
        finally:
            await session.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "code,status,count",
    [
        (
            "const rows=await Promise.allSettled(['a',"
            "'b'].map(step_key=>runs.run({step_key,agent:'scout',"
            "instruction:'child'})));return rows.map(r=>r.value.status);",
            "completed",
            2,
        ),
        (
            "const a=await runs.run({step_key:'a',agent:'scout',"
            "instruction:'child'});"
            "if(a.status==='completed')await runs.run({step_key:'b',agent:'scout',"
            "instruction:'child'});return 'branch';",
            "completed",
            2,
        ),
        (
            "await runs.run({step_key:'a',agent:'scout',instruction:'child'});"
            "throw Error('later failure');",
            "failed",
            1,
        ),
        (
            "for(let i=0;i<17;"
            "i++)await runs.run({step_key:'s'+i,agent:'scout',instruction:'child'});",
            "failed",
            16,
        ),
        ("await runs.control({action:'cancel',run_id:'foreign'});", "failed", 0),
        (
            "await runs.run({step_key:'a',agent:'scout',instruction:'child'});"
            "await runs.run({step_key:'a',agent:'worker',instruction:'child'});",
            "failed",
            1,
        ),
        (
            "const r=await runs.run({step_key:'a',agent:'scout',instruction:'child',"
            "mode:'background'});"
            "await runs.control({action:'wait',run_id:r.run_id,timeout_seconds:5});"
            "return 'settled';",
            "completed",
            1,
        ),
        (
            "await runs.run({step_key:'a',agent:'scout',instruction:'child',"
            "mode:'background'});return 'abandoned';",
            "failed",
            1,
        ),
        ("return 'x'.repeat(65536);", "failed", 0),
        ("return {status:'failed'};", "failed", 0),
    ],
)
def test_workflow_limits_and_partial_reports(tmp_path, monkeypatch, code, status, count):
    isolate_home(monkeypatch, tmp_path)
    folder = tmp_path / ".forge" / "workflows"
    folder.mkdir(parents=True)
    (folder / "case.js").write_text(code, encoding="utf-8")

    async def run():
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
        try:
            async for _ in session.prompt("execute"):
                pass
            result = next(m for m in session.messages if isinstance(m, ToolMessage))
            report = result.artifact["data"]
            assert report["status"] == status, result
            assert len(report["steps"]) == count
            assert len(json.dumps(report, ensure_ascii=False).encode()) <= 65536
            assert session.active_background_subagents == session.active_codemode_workers == 0
            await session.flush_background_subagents()
            entries = await session._config.storage.read_all()
            messages = [e.message for e in entries if isinstance(e, MessageEntry)]
            assert (
                len(
                    [m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "flow"]
                )
                == 1
            )
            terminal = [
                e
                for e in entries
                if isinstance(e, CustomEntry) and e.namespace == "forge.workflow.v1"
            ][-1]
            assert terminal.data["status"] == status
            for row in report["steps"]:
                assert row["workflow_id"] == report["workflow_id"]
                assert row["status"] in {"completed", "failed", "cancelled"}
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "report": report,
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


def test_workflow_resource_trust_reload_and_listing(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    folder = tmp_path / ".forge" / "workflows"
    folder.mkdir(parents=True)
    (folder / "case.js").write_text("return workflowArgs;", encoding="utf-8")

    async def run():
        model = ScriptedChatModel([AIMessage(content="parent")])
        denied = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=model,
                model="fake",
                tools=[],
                trust_override="deny",
                storage=JsonlSessionStorage(tmp_path / "denied.jsonl"),
            )
        )
        try:
            assert not denied.workflows
            assert "workflow" not in [t.name for t in denied.tools]
        finally:
            await denied.aclose()
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=model,
                model="fake",
                tools=[],
                trust_override="allow",
                storage=JsonlSessionStorage(tmp_path / "allowed.jsonl"),
            )
        )
        try:
            assert len(session.workflows) == 1
            before = len(session.messages)
            listed = session.handle_command("/workflows")
            assert listed.handled and "case" in listed.message
            assert len(session.messages) == before
            assert not model.calls
            (folder / "case.js").write_text("return 2;", encoding="utf-8")
            session.reload()
            assert session.workflows[0].code == "return 2;"
            (folder / "bad name.js").write_text("return 3;", encoding="utf-8")
            session.reload()
            assert not session.workflows
            assert "workflow" not in [t.name for t in session.tools]
        finally:
            await session.aclose()

    asyncio.run(run())


def test_workflow_keeps_real_file_effects_and_native_tool_boundary(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    folder = tmp_path / ".forge" / "workflows"
    folder.mkdir(parents=True)
    (folder / "case.js").write_text(
        "await tools.write({path:'script.txt',content:'native script write'});"
        "const args={step_key:'write',agent:'worker',instruction:'write once'};"
        "await runs.run(args);await runs.run(args);"
        "await runs.run({step_key:'fail',agent:'missing-role',instruction:'fail'});",
        encoding="utf-8",
    )
    agent = tmp_path / ".forge" / "agents" / "worker"
    agent.mkdir(parents=True)
    (agent / "AGENT.md").write_text(
        "---\ndescription: Write once\ntools: write\nresult-format: task-report\n---\nWrite once.",
        encoding="utf-8",
    )

    async def run():
        report = {
            "outcome": "done",
            "summary": "wrote marker",
            "findings": [],
            "evidence": ["marker.txt"],
            "question": None,
        }
        child = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "write-file",
                    "name": "write",
                    "args": {"path": "marker.txt", "content": "one effect"},
                    "type": "tool_call",
                },
                {"id": "report-file", "name": "TaskReport", "args": report, "type": "tool_call"},
            ],
        )
        model = ScriptedChatModel(
            [
                tool_call_ai("flow", "workflow", {"name": "case", "args": {}}),
                child,
                AIMessage(content="done"),
            ]
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=model,
                model="fake",
                system="offline",
                tools=[create_write_tool(cwd=tmp_path)],
                trust_override="allow",
                auto_compact_enabled=False,
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
            )
        )
        try:
            async for _ in session.prompt("run"):
                pass
            result = next(m for m in session.messages if isinstance(m, ToolMessage))
            assert result.artifact["data"]["status"] == "failed", result
            assert len(result.artifact["data"]["steps"]) == 1
            assert result.artifact["data"]["steps"][0]["tool_calls"] == 1
            assert (tmp_path / "marker.txt").read_text() == "one effect"
            assert (tmp_path / "script.txt").read_text() == "native script write"
            assert len(model.calls) == 3
            assert all(
                not isinstance(m, ToolMessage) or m.tool_call_id == "flow" for m in session.messages
            )
            assert session.active_codemode_workers == session.active_background_subagents == 0
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "result": result.artifact["data"],
                        "effects": ["marker.txt", "script.txt"],
                        "parent": "parent.jsonl",
                        "residual_workers": 0,
                    }
                ),
                encoding="utf-8",
            )
        finally:
            await session.aclose()

    asyncio.run(run())


def test_workflow_limits_live_children_even_with_larger_session_capacity(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    folder = tmp_path / ".forge" / "workflows"
    folder.mkdir(parents=True)
    (folder / "case.js").write_text(
        "return await Promise.all(Array.from({length:8},(_,i)=>runs.run("
        "{step_key:'s'+i,agent:'scout',instruction:'child'}).then(r=>r.status)));",
        encoding="utf-8",
    )

    class CapacityModel(WorkflowModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
            if human.content == "child":
                object.__setattr__(self, "live", self.live + 1)
                object.__setattr__(self, "peak", max(self.peak, self.live))
                try:
                    await asyncio.sleep(0.2)
                    return await super()._agenerate(messages, stop, run_manager, **kwargs)
                finally:
                    object.__setattr__(self, "live", self.live - 1)
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    async def run():
        model = CapacityModel()
        object.__setattr__(model, "live", 0)
        object.__setattr__(model, "peak", 0)
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=model,
                model="fake",
                system="offline",
                tools=[],
                trust_override="allow",
                background_subagent_concurrency=8,
                auto_compact_enabled=False,
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
            )
        )
        try:
            async for _ in session.prompt("execute"):
                pass
            result = next(m for m in session.messages if isinstance(m, ToolMessage)).artifact[
                "data"
            ]
            assert result["status"] == "completed"
            assert len(result["steps"]) == 8
            assert model.peak == 4
            assert model.live == session.active_codemode_workers == 0
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "peak_children": model.peak,
                        "report": result,
                        "parent": "parent.jsonl",
                        "residual_workers": 0,
                    }
                ),
                encoding="utf-8",
            )
        finally:
            await session.aclose()

    asyncio.run(run())
