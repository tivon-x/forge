"""D3 session acceptance: committed visibility, terminal drain and publication failure.

Failure matrix (written before implementation):
* partial answer: no delta before marker; terminal drains every accepted delta;
* marker failure: no visible event from that commit, session invalidated;
* close/cancel: drain admitted progress, no old write after terminal;
* background: commit while root busy, Fleet reads only confirmed values;
* reopen: partial view survives without invoking the model;
* late/slow subscriber: snapshot first, bounded frames, latest complete view.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from conftest import isolate_home
from fake_models import StreamingScriptedChatModel
from forge_agent import MessageDeltaEvent
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.storage import JsonlSessionStorage


def test_d3_session_publishes_only_committed_events(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def scenario():
        storage = JsonlSessionStorage(tmp_path / "session")
        model = StreamingScriptedChatModel([AIMessage(content="confirmed partial")])
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=model,
                model="fake",
                system="offline",
                storage=storage,
                cwd=tmp_path,
                enable_subagents=False,
            )
        )
        try:
            observer = session.observe()
            await anext(observer)
            deltas = []
            async for event in session.prompt("test"):
                snapshot = session.snapshot()
                records = await storage.read_records()
                assert snapshot["sequence"] <= records["sequence"]
                if isinstance(event, MessageDeltaEvent):
                    deltas.append(event.delta)
                    live = next(
                        d["value"]
                        for d in records["documents"]
                        if d["record"]["kind"] == "forge.live.v1"
                    )
                    assert "confirmed partial" in live["assistant"]
            assert "".join(deltas) == "confirmed partial"
            assert len(model.calls) == 1
            assert session.live_activity["assistant"] == ""
            while True:
                frame = await anext(observer)
                snapshot = frame["snapshot"]
                if any(e.get("message", {}).get("type") == "ai" for e in snapshot["entries"]):
                    live = next(
                        d["value"]
                        for d in snapshot["documents"]
                        if d["record"]["kind"] == "forge.live.v1"
                    )
                    assert live["assistant"] == "", "terminal snapshot duplicated its partial"
                    break
            await observer.aclose()
        finally:
            await session.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["abort", "close"])
def test_d3_terminal_waits_for_admitted_progress(tmp_path, monkeypatch, operation):
    isolate_home(monkeypatch, tmp_path)

    async def scenario():
        from durable_progress_driver import PausedModel

        storage = JsonlSessionStorage(tmp_path / "session")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=PausedModel(),
                model="fake",
                system="offline",
                storage=storage,
                cwd=tmp_path,
                enable_subagents=False,
            )
        )
        entered, release = asyncio.Event(), asyncio.Event()
        original = storage.commit_records

        async def hold(writes, entries=()):
            if any(
                op.get("path") == ["assistant"] and op.get("op") == "append"
                for write in writes
                for op in write.get("content", {}).get("ops", [])
            ):
                entered.set()
                await release.wait()
            return await original(writes, entries)

        monkeypatch.setattr(storage, "commit_records", hold)
        received = []

        async def collect():
            async for event in session.prompt("controlled terminal race"):
                received.append(event)

        consumer = asyncio.create_task(collect())
        await asyncio.wait_for(entered.wait(), 5)
        terminal = asyncio.create_task(
            session.abort() if operation == "abort" else session.aclose()
        )
        await asyncio.sleep(0.02)
        assert not terminal.done()
        release.set()
        await asyncio.wait_for(terminal, 8)
        await asyncio.gather(consumer, return_exceptions=True)
        live = session.live_activity
        assert "已确认的部分回答" in live["assistant"]
        assert session._storage_failure is None
        if operation == "abort":
            records = await storage.read_records()
            assert all(task["state"]["status"] == "terminal" for task in records["tasks"])
            await session.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["partial", "background"])
def test_d3_partial_process_exit_reopens_last_visible_snapshot(tmp_path, fault):
    driver = Path(__file__).with_name("durable_progress_driver.py")
    root = tmp_path / "case"
    commands = []
    for action, expected in ((fault, 17), ("reopen", 0)):
        command = [sys.executable, str(driver), action, str(root)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=50)
        commands.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        assert result.returncode == expected, result.stderr
    confirmed = json.loads((root / "confirmed.json").read_text(encoding="utf-8"))
    reopened = json.loads((root / "reopened.json").read_text(encoding="utf-8"))
    previous_live = next(
        d["value"] for d in confirmed["documents"] if d["record"]["kind"] == "forge.live.v1"
    )
    assert reopened["live"] == previous_live
    assert reopened["model_calls"] == 0
    if fault == "partial":
        assert previous_live["assistant"] == "已确认的部分回答"
    else:
        assert previous_live["background"][0]["output"] == "已确认的部分回答"
    (root / "commands.json").write_text(json.dumps(commands), encoding="utf-8")
    destination = os.environ.get("FORGE_D3_ARTIFACT_DIR")
    if destination:
        shutil.copytree(root, Path(destination) / f"{fault}-process", dirs_exist_ok=True)


def test_d3_failed_progress_marker_never_reaches_renderer(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def scenario():
        storage = JsonlSessionStorage(tmp_path / "session")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=StreamingScriptedChatModel([AIMessage(content="invisible")]),
                model="fake",
                system="offline",
                storage=storage,
                cwd=tmp_path,
                enable_subagents=False,
            )
        )
        original = storage.commit_records

        async def fail(writes, entries=()):
            if any(
                w.get("record", {}).get("kind") == "forge.live.v1"
                or w.get("type") == "document.change"
                for w in writes
            ):
                raise OSError("controlled progress failure")
            return await original(writes, entries)

        monkeypatch.setattr(storage, "commit_records", fail)
        visible = []
        try:
            try:
                async for event in session.prompt("test"):
                    if isinstance(event, MessageDeltaEvent):
                        visible.append(event.delta)
            except (OSError, RuntimeError):
                pass
            assert not visible
            assert session._storage_failure is not None
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_d3_background_partial_is_committed_while_parent_busy(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def scenario():
        from test_durable_recovery_e2e import _BackgroundAcceptanceModel

        model = _BackgroundAcceptanceModel()
        storage = JsonlSessionStorage(tmp_path / "session")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=model,
                model="fake",
                system="offline",
                storage=storage,
                cwd=tmp_path,
                tools=[],
                interactive=True,
                enable_subagents=True,
            )
        )
        try:
            observer = session.observe()
            await anext(observer)
            _ = [event async for event in session.prompt("launch background")]
            await asyncio.wait_for(model.started.wait(), 5)
            run = next(iter(session._subagent_runs.runs.values()))
            session._run_active = True
            session._subagent_runs._update(run, {"activity": "controlled background output"})
            await asyncio.sleep(0.25)
            assert session.background_subagents[0]["activity"] == "controlled background output"
            model.gate.set()
            await asyncio.wait_for(run.settled.wait(), 5)
            await asyncio.wait_for(session.flush_background_subagents(), 5)
            assert session.background_subagents[0]["status"] == "completed"
            assert not session._subagent_runs.dirty
            while True:
                frame = await anext(observer)
                snapshot = frame["snapshot"]
                terminal = any(
                    e.get("namespace") == "forge.subagent_run.v1"
                    and e.get("data", {}).get("status") == "completed"
                    for e in snapshot["entries"]
                )
                live = next(
                    (
                        d["value"]
                        for d in snapshot["documents"]
                        if d["record"]["kind"] == "forge.live.v1"
                    ),
                    {},
                )
                if live.get("background") and live["background"][0]["status"] == "completed":
                    assert terminal, "Fleet completion preceded its settlement marker"
                if terminal:
                    assert live["background"][0]["status"] == "completed"
                    break
            await observer.aclose()
        finally:
            session._run_active = False
            await session.aclose()

    asyncio.run(scenario())


def test_d3_background_model_partial_reaches_committed_fleet(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def scenario():
        from durable_progress_driver import PausedModel
        from fake_models import ScriptedChatModel, tool_call_ai
        from forge_agent import SubagentRuntime

        parent = ScriptedChatModel(
            [
                tool_call_ai(
                    "bg-partial",
                    "task",
                    {
                        "agent": "scout",
                        "instruction": "partial",
                        "mode": "background",
                        "context": "fresh",
                    },
                ),
                AIMessage(content="parent ready"),
            ]
        )
        storage = JsonlSessionStorage(tmp_path / "session")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=parent,
                model="fake",
                system="offline",
                storage=storage,
                cwd=tmp_path,
                tools=[],
                interactive=True,
                enable_subagents=True,
            )
        )
        child = PausedModel()

        async def runtime(spec):
            del spec
            return SubagentRuntime(provider=child, model="fake")

        monkeypatch.setattr(session, "_subagent_model_runtime", runtime)
        try:
            _ = [event async for event in session.prompt("launch")]
            for _ in range(250):
                if session.background_subagents and session.background_subagents[0].get("output"):
                    break
                await asyncio.sleep(0.02)
            assert session.background_subagents[0].get("output"), session.background_subagents
            assert session.background_subagents[0]["output"] == "已确认的部分回答"
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_d3_abandoned_tool_pair_clears_committed_activity(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def scenario():
        from langchain_core.messages import ToolMessage

        from conftest import make_native_tool
        from fake_models import ScriptedChatModel, tool_call_ai

        entered = asyncio.Event()

        async def execute(arguments, signal=None, context=None):
            del arguments, signal, context
            entered.set()
            await asyncio.sleep(30)

        tool = make_native_tool(
            name="blocked",
            description="controlled tool",
            input_schema={"type": "object", "properties": {}},
            executor=execute,
        )
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=ScriptedChatModel([tool_call_ai("abandoned", "blocked", {})]),
                model="fake",
                system="offline",
                storage=JsonlSessionStorage(tmp_path / "session"),
                cwd=tmp_path,
                tools=[tool],
                enable_subagents=False,
            )
        )

        async def consume():
            _ = [event async for event in session.prompt("test cancellation")]

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(session.abort(), 8)
            await asyncio.gather(consumer, return_exceptions=True)
            assert any(
                isinstance(m, ToolMessage) and m.tool_call_id == "abandoned"
                for m in session.messages
            )
            assert not session.live_activity["tools"]
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_d3_trailing_progress_is_single_flight_and_terminal_drains(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def scenario():
        from time import monotonic

        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        chunks = []

        class BurstModel(StreamingScriptedChatModel):
            async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
                self._record_call(messages)
                for _ in range(30):
                    chunks.append(monotonic())
                    yield ChatGenerationChunk(message=AIMessageChunk(content="x", id="burst"))
                    # Stay above Windows' monotonic timer resolution.
                    await asyncio.sleep(0.03)

        storage = JsonlSessionStorage(tmp_path / "session")
        model = BurstModel()
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=model,
                model="fake",
                system="offline",
                storage=storage,
                cwd=tmp_path,
                enable_subagents=False,
            )
        )
        original = storage.commit_records
        active, maximum = 0, 0
        admissions = []

        async def delayed(writes, entries=()):
            nonlocal active, maximum
            is_partial = any(
                op.get("path") == ["assistant"] and op.get("op") == "append"
                for w in writes
                for op in w.get("content", {}).get("ops", [])
            )
            if is_partial:
                admissions.append(monotonic())
                active += 1
                maximum = max(maximum, active)
                await asyncio.sleep(0.12)
            try:
                return await original(writes, entries)
            finally:
                if is_partial:
                    active -= 1

        monkeypatch.setattr(storage, "commit_records", delayed)
        try:
            events = [event async for event in session.prompt("burst")]
            assert "".join(e.delta for e in events if isinstance(e, MessageDeltaEvent)) == "x" * 30
            assert maximum == 1
            assert admissions[0] - chunks[0] >= 0.08, (admissions, chunks)
            assert admissions[0] - chunks[0] < 0.5
            assert session.live_activity["assistant"] == ""
            from forge_agent.message_codec import message_text

            assert message_text(session.messages[-1]) == "x" * 30
            result = {
                "maximum_progress_commits_in_flight": maximum,
                "first_delay_seconds": admissions[0] - chunks[0],
                "progress_commits": len(admissions),
                "chunks": len(chunks),
                "model_calls": len(model.calls),
                "terminal_drained": True,
            }
            destination = os.environ.get("FORGE_D3_ARTIFACT_DIR")
            if destination:
                path = Path(destination) / "trailing"
                path.mkdir(parents=True, exist_ok=True)
                (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_live_document_checkpoints_complete_base_between_generations(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def scenario():
        storage = JsonlSessionStorage(tmp_path / "session")
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=StreamingScriptedChatModel(
                    [AIMessage(content="one"), AIMessage(content="two")]
                ),
                model="fake",
                system="offline",
                cwd=tmp_path,
                storage=storage,
                enable_subagents=False,
            )
        )
        facts = []
        try:
            for prompt in ("first", "second"):
                _ = [event async for event in session.prompt(prompt)]
                records = await storage.read_records()
                live = next(
                    d for d in records["documents"] if d["record"]["kind"] == "forge.live.v1"
                )
                document = await storage.document(live["record"]["id"])
                assert document is not None
                assert document["deltas_since_base"] == 0
                assert live["record"]["history"] == "latest"
                facts.append({"sequence": records["sequence"], "deltas_since_base": 0})
            output = Path(os.environ.get("FORGE_D3_ARTIFACT_DIR", str(tmp_path))) / "idle-base"
            output.mkdir(parents=True, exist_ok=True)
            (output / "result.json").write_text(json.dumps(facts), encoding="utf-8")
        finally:
            await session.aclose()

    asyncio.run(scenario())
