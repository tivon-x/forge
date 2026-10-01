"""Offline tool discovery and JS acceptance, written before implementation.

Failure modes: undeclared direct execution, lost branch loadout/store, globals
leaking between scripts, unsafe runtime APIs, unbounded loops and orphan children.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from conftest import isolate_home
from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent.session import JsonlSessionStorage
from forge_coding.sessions.session import CodingSession, CodingSessionConfig
from forge_coding.tools import ToolDefinition, create_coding_tool_set


@pytest.mark.anyio
async def test_discovery_and_codemode_branch_state(tmp_path: Path, monkeypatch) -> None:
    isolate_home(monkeypatch, tmp_path)
    catalog = create_coding_tool_set(cwd=tmp_path)
    remote = StructuredTool.from_function(
        func=lambda value: value, name="deferred_echo", description="Echo deferred values"
    )
    catalog = catalog.with_tools(ToolDefinition(remote, label="echo", exposure="deferred"))
    model = ScriptedChatModel(
        [
            tool_call_ai("forbidden", "deferred_echo", {"value": "before loading"}),
            tool_call_ai("search", "tool_search", {"query": "deferred echo"}),
            tool_call_ai("echo", "deferred_echo", {"value": "loaded"}),
            tool_call_ai(
                "script",
                "codemode",
                {
                    "code": """
            await tools.write({path:'note.txt',content:'hello'});
            text(await tools.read({path:'note.txt'}));
            const found=await searchTools('read',{namespace:'builtin',limit:1});
            text(found); text(await describeTool('read',{offset:0,limit:64}));
            text([typeof process, typeof fetch, typeof require]);
            store('answer',{value:42});
            return 42;
        """
                },
            ),
            AIMessage(content="done"),
        ]
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    config = CodingSessionConfig(
        provider=model,
        model="fake",
        storage=storage,
        cwd=tmp_path,
        tools=catalog,
        codemode="on",
        enable_subagents=False,
        auto_compact_enabled=False,
    )
    session = await CodingSession.load(config)
    evidence = tmp_path / "tool-features-evidence"
    evidence.mkdir()
    try:
        events = [event async for event in session.prompt("use deferred echo then JS")]
        results = [m for m in session.messages if isinstance(m, ToolMessage)]
        assert [r.status for r in results] == ["error", "success", "success", "success"]
        assert "deferred_echo" not in {t.name for t in model.calls[0]["tools"]}
        assert "deferred_echo" in {t.name for t in model.calls[2]["tools"]}
        assert (tmp_path / "note.txt").read_text() == "hello"
        assert session.codemode_store == {"answer": {"value": 42}}
        assert "hello" in str(results[-1].artifact["forge.codemode.v1"])
        assert "next_offset" in str(results[-1].artifact["forge.codemode.v1"])
        assert results[-1].artifact["forge.nested_calls.v1"]["total"] == 2
        await session.export(evidence / "session.html")
        (evidence / "events.json").write_text(
            json.dumps([e.model_dump(mode="json") for e in events])
        )
        (evidence / "summary.json").write_text(
            json.dumps({"loaded": True, "stored": True, "paired": True})
        )
        print(f"Tool feature evidence: {evidence}")
    finally:
        await session.aclose()

    restored_model = ScriptedChatModel(
        [
            tool_call_ai(
                "restore",
                "codemode",
                {"code": "text(load('answer')); store('failed',1); throw Error('fixture');"},
            ),
            AIMessage(content="done"),
        ]
    )
    restored = await CodingSession.load(
        CodingSessionConfig(
            provider=restored_model,
            model="fake",
            storage=storage,
            cwd=tmp_path,
            tools=catalog,
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        assert restored.codemode_store == {"answer": {"value": 42}}
        assert "codemode" in {t.name for t in restored.tools}
        _ = [event async for event in restored.prompt("fail without committing")]
        assert restored.codemode_store == {"answer": {"value": 42}}
        assert [m for m in restored.messages if isinstance(m, ToolMessage)][-1].status == "error"
        assert "deferred_echo" in {t.name for t in restored_model.calls[0]["tools"]}
    finally:
        await restored.aclose()


@pytest.mark.anyio
async def test_codemode_cancellation_drains_children_and_keeps_store(
    tmp_path: Path, monkeypatch
) -> None:
    isolate_home(monkeypatch, tmp_path)
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def wait_child() -> str:
        started.set()
        try:
            await asyncio.sleep(30)
            return "done"
        finally:
            stopped.set()

    wait = StructuredTool.from_function(coroutine=wait_child, name="wait_child", description="Wait")
    model = ScriptedChatModel(
        [
            tool_call_ai(
                "script",
                "codemode",
                {"code": "store('uncommitted',1); await tools.wait_child({});"},
            ),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[wait],
            codemode="on",
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )

    async def run():
        return [e async for e in session.prompt("cancel waiting tool")]

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(started.wait(), 20)
        session.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        assert stopped.is_set() and session.active_codemode_workers == 0
        assert session.codemode_store == {}
        results = [m for m in session.messages if isinstance(m, ToolMessage)]
        assert results and results[-1].status == "error"
        evidence = tmp_path / "cancellation-evidence.json"
        evidence.write_text(
            json.dumps({"workers": 0, "child_stopped": True, "store_unchanged": True})
        )
        print(f"Cancellation evidence: {evidence}")
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await session.aclose()


@pytest.mark.anyio
async def test_search_same_batch_and_session_branch(tmp_path: Path, monkeypatch) -> None:
    isolate_home(monkeypatch, tmp_path)
    remote = StructuredTool.from_function(
        func=lambda value: value, name="deferred_echo", description="Echo"
    )
    ToolDefinition(remote, label="echo", exposure="deferred")
    model = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tool_search", "args": {"names": ["deferred_echo"]}, "id": "search"},
                    {"name": "deferred_echo", "args": {"value": "premature"}, "id": "early"},
                ],
            ),
            tool_call_ai("later", "deferred_echo", {"value": "allowed"}),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[remote],
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        _ = [e async for e in session.prompt("load")]
        results = [m for m in session.messages if isinstance(m, ToolMessage)]
        assert [m.status for m in results] == ["success", "error", "success"]
        user = next(
            entry
            for entry in await session._config.storage.read_all()
            if entry.type == "message" and entry.message.type == "human"
        )
        await session.branch_to_entry(user.id)
        assert session._tool_discovery.loaded == {}
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_codemode_only_and_bounded_loop(tmp_path: Path, monkeypatch) -> None:
    isolate_home(monkeypatch, tmp_path)
    model = ScriptedChatModel(
        [
            tool_call_ai(
                "loop", "codemode", {"code": '// @options: {"timeout_ms": 1500}\nwhile(true){}'}
            ),
            tool_call_ai("later", "codemode", {"code": "text(typeof leaked); return 1;"}),
            tool_call_ai(
                "tiny",
                "codemode",
                {"code": '// @options: {"max_output_tokens": 1}\ntext("x".repeat(10000));'},
            ),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            codemode="only",
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        _ = [e async for e in session.prompt("timeout then recover")]
        assert "write" not in {t.name for t in model.calls[0]["tools"]}
        assert "codemode" in {t.name for t in model.calls[0]["tools"]}
        results = [m for m in session.messages if isinstance(m, ToolMessage)]
        assert [m.status for m in results] == ["error", "success", "success"]
        assert len(results[-1].content) < 500
        full = results[-1].artifact["forge.codemode.v1"]["full_output_path"]
        assert Path(full).is_file()
        assert session.active_codemode_workers == 0
    finally:
        await session.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("attempt", range(8))
async def test_codemode_exit_and_untrusted_store(tmp_path: Path, monkeypatch, attempt: int) -> None:
    isolate_home(monkeypatch, tmp_path)
    model = ScriptedChatModel(
        [
            tool_call_ai(
                "exit",
                "codemode",
                {
                    "code": "store('before', 1); try { exit(); } catch(e) {} "
                    "await tools.write({path:'after.txt', content:'bad'});"
                },
            ),
            tool_call_ai("store", "codemode", {"code": "__forge_store.bad='x'.repeat(70000);"}),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            codemode="on",
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        _ = [e async for e in session.prompt("exit and validate")]
        results = [m for m in session.messages if isinstance(m, ToolMessage)]
        assert [m.status for m in results] == ["success", "error"], [m.content for m in results]
        assert not (tmp_path / "after.txt").exists()
        assert session.codemode_store == {"before": 1}
        assert session.active_codemode_workers == 0
        (tmp_path / "boundary-evidence.json").write_text(
            json.dumps(
                {
                    "exit_stopped_tool": True,
                    "store_checked_by_parent": True,
                    "workers": 0,
                }
            )
        )
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_shared_native_tools_keep_session_loadouts_separate(
    tmp_path: Path, monkeypatch
) -> None:
    isolate_home(monkeypatch, tmp_path)
    remote = StructuredTool.from_function(
        func=lambda value: value, name="shared_echo", description="Echo"
    )
    ToolDefinition(remote, label="echo", exposure="deferred")
    first_model = ScriptedChatModel(
        [
            tool_call_ai("load", "tool_search", {"names": ["shared_echo"]}),
            AIMessage(content="loaded"),
            tool_call_ai("allowed", "shared_echo", {"value": "first"}),
            AIMessage(content="done"),
        ]
    )
    first = await CodingSession.load(
        CodingSessionConfig(
            provider=first_model,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "first.jsonl"),
            cwd=tmp_path,
            tools=[remote],
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    second = None
    try:
        _ = [e async for e in first.prompt("load")]
        second = await CodingSession.load(
            CodingSessionConfig(
                provider=ScriptedChatModel(
                    [
                        tool_call_ai("blocked", "shared_echo", {"value": "second"}),
                        AIMessage(content="done"),
                    ]
                ),
                model="fake",
                storage=JsonlSessionStorage(tmp_path / "second.jsonl"),
                cwd=tmp_path,
                tools=[remote],
                enable_subagents=False,
                auto_compact_enabled=False,
            )
        )
        _ = [e async for e in first.prompt("use loaded")]
        _ = [e async for e in second.prompt("not loaded")]
        assert [m.status for m in first.messages if isinstance(m, ToolMessage)] == [
            "success",
            "success",
        ]
        assert [m.status for m in second.messages if isinstance(m, ToolMessage)] == ["error"]
        (tmp_path / "isolation-evidence.json").write_text(json.dumps({"loadouts_isolated": True}))
    finally:
        await first.aclose()
        if second is not None:
            await second.aclose()


@pytest.mark.anyio
async def test_codemode_artifact_write_failure_is_paired(tmp_path: Path, monkeypatch) -> None:
    isolate_home(monkeypatch, tmp_path)
    (tmp_path / ".forge").mkdir()
    (tmp_path / ".forge/artifacts").write_text("blocking file")
    model = ScriptedChatModel(
        [
            tool_call_ai(
                "blocked",
                "codemode",
                {"code": '// @options: {"max_output_tokens": 1}\ntext("x".repeat(1000));'},
            ),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            codemode="on",
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        _ = [e async for e in session.prompt("save output")]
        result = next(m for m in session.messages if isinstance(m, ToolMessage))
        assert result.status == "error" and "Collected output:" not in result.content
        assert session.active_codemode_workers == 0
        (tmp_path / "write-failure-evidence.json").write_text(json.dumps({"paired_error": True}))
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_unknown_feature_versions_are_ignored_with_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    from forge_agent.session import CustomEntry

    isolate_home(monkeypatch, tmp_path)
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    await storage.append(CustomEntry(namespace="forge.tool_loadout.v1", data={"version": 99}))
    await storage.append(CustomEntry(namespace="forge.codemode_store.v1", data={"version": 99}))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel([AIMessage(content="done")]),
            model="fake",
            storage=storage,
            cwd=tmp_path,
            codemode="on",
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        _ = [e async for e in session.prompt("resume unknown snapshots")]
        assert session.codemode_store == {} and session._tool_discovery.loaded == {}
        assert {d.kind for d in session.resource_diagnostics} >= {"tool_loadout", "codemode_store"}
        (tmp_path / "unknown-version-evidence.json").write_text(
            json.dumps({"ignored_with_diagnostics": True})
        )
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_codemode_backpressure_keeps_parallel_calls_within_budget(
    tmp_path: Path, monkeypatch
) -> None:
    isolate_home(monkeypatch, tmp_path)
    echo = StructuredTool.from_function(func=lambda value: value, name="echo", description="Echo")
    model = ScriptedChatModel(
        [
            tool_call_ai(
                "parallel",
                "codemode",
                {
                    "code": "await Promise.all(Array.from({length:100}, "
                    "(_,i)=>tools.echo({value:i}))); return 100;"
                },
            ),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[echo],
            codemode="on",
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        _ = [e async for e in session.prompt("parallel calls")]
        result = next(m for m in session.messages if isinstance(m, ToolMessage))
        assert result.status == "success" and result.artifact["forge.codemode.v1"]["value"] == 100
        assert result.artifact["forge.nested_calls.v1"]["total"] == 100
        assert session.active_codemode_workers == 0
        (tmp_path / "backpressure-evidence.json").write_text(
            json.dumps({"completed": 100, "workers": 0})
        )
    finally:
        await session.aclose()
