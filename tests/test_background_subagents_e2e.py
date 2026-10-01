"""Offline R1 acceptance: real parent/child graphs and durable session artifacts.

Failure contract: a receipt is not a result; cancelled siblings stay alive;
pending calls remain paired; persistence gates launch; close stops children;
timeouts release capacity; resumed runs never restart; foreign ids fail closed.
"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from conftest import isolate_home
from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent.session import CustomEntry, JsonlSessionStorage, MessageEntry
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.manager import SessionManager


class BackgroundModel(ScriptedChatModel):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "gate", asyncio.Event())
        object.__setattr__(self, "started", asyncio.Event())
        object.__setattr__(self, "parent_started", asyncio.Event())
        object.__setattr__(self, "parent_gate", asyncio.Event())

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._record_call(messages)
        human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        if str(human.content).startswith("child"):
            self.started.set()
            await self.gate.wait()
            message = AIMessage(
                content="CHILD_RESULT",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
            )
        elif human.content == "blocking discussion":
            self.parent_started.set()
            await self.parent_gate.wait()
            message = AIMessage(content="PARENT_DONE")
        elif str(human.content).startswith("launch") and not isinstance(messages[-1], ToolMessage):
            message = tool_call_ai(
                "submit" if human.content == "launch" else "submit2",
                "task",
                {
                    "agent": "scout",
                    "instruction": "child one",
                    "mode": "background",
                },
            )
        else:
            message = AIMessage(content="PARENT_READY")
        return ChatResult(generations=[ChatGeneration(message=message)])


async def load_session(tmp_path, model, **limits):
    return await CodingSession.load(
        CodingSessionConfig(
            cwd=tmp_path,
            provider=model,
            model="fake",
            system="Forge offline acceptance",
            tools=[],
            interactive=True,
            auto_compact_enabled=False,
            storage=JsonlSessionStorage(tmp_path / "acceptance.jsonl"),
            **limits,
        )
    )


def test_background_parent_child_acceptance(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def run():
        model = BackgroundModel()
        session = await load_session(tmp_path, model)
        try:
            await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            await asyncio.wait_for(model.started.wait(), 5)
            row = session.background_subagents[0]
            run_id = row["run_id"]
            assert row["status"] == "running"
            await asyncio.wait_for(_consume(session.prompt("continue discussion")), 5)
            assert session.messages[-1].content == "PARENT_READY"
            pending = await session._subagent_runs.control("wait", run_id, 0.01)
            assert pending["status"] in {"queued", "running"}
            assert session.background_subagents[0]["status"] == "running"
            model.gate.set()
            await asyncio.wait_for(session._subagent_runs.control("wait", run_id, 5), 6)
            await session.flush_background_subagents()
            detail = session.inspect_background_subagent(run_id)
            assert detail["status"] == "completed"
            assert detail["result"]["final_output"] == "CHILD_RESULT"
            assert all(m.content != "CHILD_RESULT" for m in session.messages)
            entries = await session._config.storage.read_all()
            messages = [e.message for e in entries if isinstance(e, MessageEntry)]
            receipts = [
                m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "submit"
            ]
            assert len(receipts) == 1
            runs = [
                e
                for e in entries
                if isinstance(e, CustomEntry) and e.namespace == "forge.subagent_run.v1"
            ]
            assert [e.data["status"] for e in runs] == ["queued", "completed"]
            usage = [
                e.data
                for e in entries
                if isinstance(e, CustomEntry)
                and e.namespace == "forge.usage.v1"
                and e.data.get("purpose") == "subagent"
            ]
            assert len(usage) == 1
            assert usage[0]["total_tokens"] == 12
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "case": "background-parent-child",
                        "passed": True,
                        "receipts": len(receipts),
                        "terminal": detail["status"],
                        "artifact": "acceptance.jsonl",
                    }
                ),
                encoding="utf-8",
            )
        finally:
            await session.aclose()
        restored = await load_session(tmp_path, model)
        try:
            assert restored.background_subagents[0]["status"] == "completed"
            with pytest.raises(ValueError):
                restored.inspect_background_subagent("foreign")
        finally:
            await restored.aclose()

    asyncio.run(run())


def test_background_cancel_queue_timeout_and_close(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def run():
        model = BackgroundModel()
        session = await load_session(
            tmp_path,
            model,
            background_subagent_concurrency=1,
            background_subagent_queue_limit=1,
            background_subagent_timeout_seconds=0.5,
        )
        try:
            await _consume(session.prompt("launch"))
            await asyncio.wait_for(model.started.wait(), 5)
            first = session.background_subagents[0]["run_id"]
            with pytest.raises(RuntimeError, match="background"):
                session.reload()
            with pytest.raises(RuntimeError, match="background"):
                session.set_model("other")
            await session._subagent_runs.control("wait", first, 1)
            await session.flush_background_subagents()
            assert session.inspect_background_subagent(first)["status"] == "failed"
        finally:
            await session.aclose()
        assert not session._subagent_runs.active_count

    asyncio.run(run())


def test_background_queue_cancel_sibling_and_close(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def run():
        model = BackgroundModel()
        session = await load_session(
            tmp_path, model, background_subagent_concurrency=1, background_subagent_queue_limit=1
        )
        await asyncio.wait_for(_consume(session.prompt("launch")), 5)
        await asyncio.wait_for(model.started.wait(), 5)
        first = session.background_subagents[0]["run_id"]
        await asyncio.wait_for(_consume(session.prompt("launch two")), 5)
        second = session.background_subagents[1]["run_id"]
        assert session.inspect_background_subagent(second)["status"] == "queued"
        with pytest.raises(ValueError, match="queue is full"):
            session._subagent_runs.submit("scout", "child excess")
        await session.cancel_background_subagent(second)
        assert session.inspect_background_subagent(second)["status"] == "cancelled"
        assert session.inspect_background_subagent(first)["status"] == "running"
        await session.aclose()
        assert not session.active_background_subagents
        restored = await load_session(tmp_path, model)
        try:
            assert all(row["status"] == "cancelled" for row in restored.background_subagents)
        finally:
            await restored.aclose()

    asyncio.run(run())


def test_background_receipt_persistence_gates_launch(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def run():
        model = BackgroundModel()
        session = await load_session(tmp_path, model)
        storage = session._config.storage
        append = storage.append

        async def reject_receipt(entry):
            if (
                isinstance(entry, MessageEntry)
                and isinstance(entry.message, ToolMessage)
                and entry.message.name == "task"
            ):
                raise OSError("offline receipt persistence failure")
            await append(entry)

        monkeypatch.setattr(storage, "append", reject_receipt)
        try:
            with pytest.raises(OSError, match="receipt persistence"):
                await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            assert not model.started.is_set()
            assert not session.active_background_subagents
        finally:
            monkeypatch.setattr(storage, "append", append)
            await asyncio.wait_for(_consume(session.prompt("launch two")), 5)
            assert session.background_subagents[0]["status"] == "interrupted"
            assert session.background_subagents[1]["status"] == "running"
            await session.aclose()
        calls_before_reload = len(model.calls)
        restored = await load_session(tmp_path, model)
        try:
            assert restored.background_subagents[0]["status"] == "interrupted"
            assert not restored.active_background_subagents
            assert len(model.calls) == calls_before_reload
        finally:
            await restored.aclose()

    asyncio.run(run())


def test_background_branch_switch_cancels_and_hides_old_run(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def run():
        model = BackgroundModel()
        session = await load_session(tmp_path, model)
        try:
            await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            await asyncio.wait_for(model.started.wait(), 5)
            run_id = session.background_subagents[0]["run_id"]
            entries = await session._config.storage.read_all()
            origin = next(
                e.id
                for e in entries
                if isinstance(e, MessageEntry) and isinstance(e.message, HumanMessage)
            )
            await session.branch_to_entry(origin)
            assert not session.active_background_subagents
            assert not session.background_subagents
            with pytest.raises(ValueError):
                session.inspect_background_subagent(run_id)
            await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            assert session.background_subagents[0]["run_id"] != run_id
        finally:
            await session.aclose()

    asyncio.run(run())


async def _consume(stream):
    return [event async for event in stream]


def test_parent_cancel_keeps_accepted_background_run(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def run():
        model = BackgroundModel()
        session = await load_session(tmp_path, model)
        try:
            await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            await asyncio.wait_for(model.started.wait(), 5)
            run_id = session.background_subagents[0]["run_id"]
            parent = asyncio.create_task(_consume(session.prompt("blocking discussion")))
            await asyncio.wait_for(model.parent_started.wait(), 5)
            session.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(parent, 5)
            assert session.inspect_background_subagent(run_id)["status"] == "running"
            model.gate.set()
            await session._subagent_runs.control("wait", run_id, 5)
            await session.flush_background_subagents()
            assert session.inspect_background_subagent(run_id)["status"] == "completed"
        finally:
            await session.aclose()

    asyncio.run(run())


def test_new_session_stops_old_background_and_rebinds_controls(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def run():
        model = BackgroundModel()
        session = await load_session(tmp_path, model, session_manager=SessionManager())
        try:
            await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            await asyncio.wait_for(model.started.wait(), 5)
            run_id = session.background_subagents[0]["run_id"]
            old_storage = session._config.storage
            await session.new_session()
            assert not session.background_subagents
            with pytest.raises(ValueError):
                session.inspect_background_subagent(run_id)
            entries = await old_storage.read_all()
            terminal = [
                e
                for e in entries
                if isinstance(e, CustomEntry) and e.namespace == "forge.subagent_run.v1"
            ]
            assert terminal[-1].data["status"] == "cancelled"
            await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            assert session.background_subagents[0]["run_id"] != run_id
        finally:
            await session.aclose()

    asyncio.run(run())


def test_failed_branch_keeps_background_submission_available(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def fail_summary(*args, **kwargs):
        raise RuntimeError("offline branch summary failure")

    async def run():
        model = BackgroundModel()
        session = await load_session(tmp_path, model)
        try:
            await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            origin = session.state.context_entry_ids[0]
            monkeypatch.setattr(session, "_summarize_branch_messages", fail_summary)
            with pytest.raises(RuntimeError, match="branch summary failure"):
                await session.branch_to_entry(origin, summarize=True)
            await asyncio.wait_for(_consume(session.prompt("launch two")), 5)
            assert session.background_subagents[-1]["status"] == "running"
        finally:
            await session.aclose()

    asyncio.run(run())


def test_failed_adoption_keeps_background_submission_available(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    class FailedClose:
        async def aclose(self):
            raise RuntimeError("offline resource close failure")

    async def run():
        model = BackgroundModel()
        session = await load_session(tmp_path, model, session_manager=SessionManager())
        try:
            await asyncio.wait_for(_consume(session.prompt("launch")), 5)
            session._mcp_runtime = FailedClose()
            with pytest.raises(RuntimeError, match="resource close failure"):
                await session.new_session()
            session._mcp_runtime = None
            await asyncio.wait_for(_consume(session.prompt("launch two")), 5)
            assert session.background_subagents[-1]["status"] == "running"
        finally:
            session._mcp_runtime = None
            await session.aclose()

    asyncio.run(run())


def test_helper_usage_and_manual_flush_preserve_current_branch(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    async def run():
        session = await load_session(tmp_path, BackgroundModel())
        entered, release = asyncio.Event(), asyncio.Event()
        original = session._append_usage_message

        async def blocked_usage(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        try:
            await _consume(session.prompt("launch"))
            monkeypatch.setattr(session, "_append_usage_message", blocked_usage)
            helper = asyncio.create_task(
                session._persist_helper_usage_messages(
                    [
                        AIMessage(
                            content="title",
                            usage_metadata={
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        )
                    ],
                    purpose="auto_name",
                )
            )
            session._auto_name_task = helper
            await entered.wait()
            session._config.provider.gate.set()
            run_id = session.background_subagents[0]["run_id"]
            await session._subagent_runs.control("wait", run_id, 5)
            flush = asyncio.create_task(session.flush_background_subagents())
            for _ in range(10):
                await asyncio.sleep(0)
            release.set()
            await asyncio.gather(helper, flush)
            current = session.state.custom_entries
            assert any(
                e.namespace == "forge.subagent_run.v1" and e.data["status"] == "completed"
                for e in current
            )
            purposes = {e.data.get("purpose") for e in current if e.namespace == "forge.usage.v1"}
            assert {"auto_name", "subagent"} <= purposes
            (tmp_path / "helper-flush-summary.json").write_text(
                json.dumps({"passed": True, "purposes": sorted(purposes)}),
                encoding="utf-8",
            )
        finally:
            release.set()
            await session.aclose()

    asyncio.run(run())


def test_completion_during_flush_is_drained_without_another_turn(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)

    class TwoChildren(BackgroundModel):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, "gates", [asyncio.Event(), asyncio.Event()])
            object.__setattr__(self, "child_count", 0)

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
            if str(human.content).startswith("child"):
                index = self.child_count
                object.__setattr__(self, "child_count", index + 1)
                await self.gates[index].wait()
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="DONE"))])
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    async def run():
        model = TwoChildren()
        session = await load_session(tmp_path, model)
        original = session._append_session_entry
        injected = False

        async def append(entry):
            nonlocal injected
            if (
                not injected
                and isinstance(entry, CustomEntry)
                and entry.namespace == "forge.subagent_run.v1"
                and entry.data["status"] == "completed"
            ):
                injected = True
                model.gates[1].set()
                second = list(session._subagent_runs.runs.values())[1]
                await asyncio.wait_for(second.settled.wait(), 5)
            await original(entry)

        try:
            await _consume(session.prompt("launch"))
            await _consume(session.prompt("launch two"))
            while model.child_count < 2:
                await asyncio.sleep(0)
            monkeypatch.setattr(session, "_append_session_entry", append)
            model.gates[0].set()
            first = list(session._subagent_runs.runs.values())[0]
            await asyncio.wait_for(first.settled.wait(), 5)
            await asyncio.wait_for(session._background_flush_task, 5)
            assert not session._subagent_runs.dirty
            assert all(r.persisted for r in session._subagent_runs.runs.values())
            (tmp_path / "drain-summary.json").write_text(
                json.dumps({"passed": True, "saved": 2, "dirty": 0}),
                encoding="utf-8",
            )
        finally:
            await session.aclose()

    asyncio.run(run())
