"""Offline acceptance of the native tool boundary, including nested calls.

Failure modes: forged runtime, undeclared control tools, schema errors, path
escape, leaked child results, deadlocked sequential batches and orphan tasks.
Evidence is emitted even when an assertion fails; no real provider is used.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.types import Command

from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent import AgentHarness, AgentHarnessConfig
from forge_agent.context import ForgeRuntimeContext
from forge_agent.events import ToolExecutionEndEvent, ToolExecutionStartEvent
from forge_agent.session import MessageEntry
from forge_agent.tool_execution import get_nested_tool_executor
from forge_cli.tui.adapter import TuiEventAdapter
from forge_cli.tui.state import TuiState
from forge_coding.sessions.compaction import extract_file_operations
from forge_coding.sessions.export import export_session_html, export_session_jsonl
from forge_coding.tools import ToolDefinition, ToolSet, create_coding_tool_set


@pytest.mark.anyio
async def test_native_nested_tool_boundary_end_to_end(tmp_path: Path) -> None:
    catalog = create_coding_tool_set(cwd=tmp_path)
    ToolDefinition(catalog.by_name["read"].tool, label="read", exposure="deferred")
    observed = []
    events = []
    evidence = tmp_path / "tool-ecosystem-evidence"
    evidence.mkdir()

    @tool
    async def compose(runtime: ToolRuntime[ForgeRuntimeContext]) -> str:
        """Exercise native file operations through the shared nested boundary."""
        nested = get_nested_tool_executor()
        written = await nested.call("write", {"path": "note.txt", "content": "hello"})
        assert written.status == "success"
        results = await asyncio.gather(
            nested.call("read", {"path": "note.txt", "runtime": "forged"}),
            nested.call("read", {"path": "../escape.txt"}),
            nested.call("read", {"path": 1}),
            nested.call("blocked", {}),
            nested.call("compose", {}),
        )
        observed.extend(results)
        assert runtime.context.workspace_root == str(tmp_path)
        return "composition finished"

    blocked = StructuredTool.from_function(
        func=lambda: "private child payload", name="blocked", description="Must not run"
    )
    catalog = catalog.with_tools(
        ToolDefinition(compose, label="compose", exposure="model-only"),
        ToolDefinition(blocked, label="blocked", exposure="hidden"),
    )
    assert blocked not in catalog.declared_tools
    assert compose not in catalog.callable_tools
    compose.metadata = {**(compose.metadata or {}), "forge.execution_mode": "sequential"}
    model = ScriptedChatModel([tool_call_ai("parent", "compose", {}), AIMessage(content="done")])
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=model,
            tools=catalog.registered_tools,
            runtime_context=ForgeRuntimeContext(workspace_root=str(tmp_path)),
        )
    )
    passed = False
    try:
        async with asyncio.timeout(10):
            events = [event async for event in harness.prompt("compose a note")]
        assert [item.status for item in observed] == ["success", "error", "error", "error", "error"]
        assert (tmp_path / "note.txt").read_text() == "hello"
        messages = harness.messages
        results = [message for message in messages if isinstance(message, ToolMessage)]
        assert len(results) == 1 and results[0].tool_call_id == "parent"
        assert results[0].status == "success"
        declared_names = [tool.name for tool in model.calls[0]["tools"]]
        assert "read" not in declared_names and "blocked" not in declared_names
        assert "compose" in declared_names
        record = results[0].artifact["forge.nested_calls.v1"]
        assert record["total"] == 6 and record["complete"]
        assert len({call["id"] for call in record["calls"]}) == 6
        assert all("content" not in call for call in record["calls"])
        nested_events = [
            event
            for event in events
            if isinstance(event, (ToolExecutionStartEvent, ToolExecutionEndEvent))
            and event.parent_tool_call_id
        ]
        assert len(nested_events) == 12
        assert all(event.parent_tool_call_id == "parent" for event in nested_events)
        assert "hello" not in json.dumps(record)
        assert "private child payload" not in json.dumps(record)
        operations = extract_file_operations(messages)
        assert "note.txt" in operations.written and "note.txt" in operations.read
        state = TuiState()
        adapter = TuiEventAdapter(state)
        for event in events:
            adapter.apply(event)
        tool_rows = [item for item in state.items if item.role == "tool"]
        assert len(tool_rows) == 1
        assert "Nested tool calls:" in tool_rows[0].tool_result_text
        restored = TuiState()
        restored.load_messages(messages)
        restored_tool = next(item for item in restored.items if item.role == "tool")
        assert restored_tool.tool_result_text == tool_rows[0].tool_result_text
        exported = export_session_html(
            [MessageEntry(message=message) for message in messages], evidence / "session.html"
        ).read_text(encoding="utf-8")
        assert "Nested tool calls (6)" in exported
        assert "private child payload" not in exported
        passed = True
    finally:
        entries = [MessageEntry(message=message) for message in harness.messages]
        export_session_jsonl(entries, evidence / "session.jsonl")
        export_session_html(entries, evidence / "session.html")
        (evidence / "events.json").write_text(
            json.dumps([event.model_dump(mode="json") for event in events], ensure_ascii=False),
            encoding="utf-8",
        )
        (evidence / "summary.json").write_text(
            json.dumps({"passed": passed, "observed_statuses": [x.status for x in observed]}),
            encoding="utf-8",
        )
        print(f"Tool ecosystem evidence: {evidence}")


@pytest.mark.anyio
async def test_nested_cancellation_drains_children(tmp_path: Path) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    @tool
    async def waiting() -> str:
        """Wait until cancelled and record cleanup."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return "unreachable"

    @tool
    async def compose() -> str:
        """Start a child then abandon the parent."""
        nested = get_nested_tool_executor()
        task = asyncio.create_task(nested.call("waiting", {}))
        await started.wait()
        assert not task.done()
        raise asyncio.CancelledError

    catalog = ToolSet.from_tools([waiting, compose])
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=ScriptedChatModel([tool_call_ai("parent", "compose", {})]),
            tools=catalog.tools,
        )
    )
    async with asyncio.timeout(5):
        events = [event async for event in harness.prompt("cancel composition")]
    assert stopped.is_set()
    pending = [task for task in asyncio.all_tasks() if "NestedToolExecutor.call" in repr(task)]
    assert not pending
    evidence = tmp_path / "tool-ecosystem-evidence"
    evidence.mkdir()
    (evidence / "cleanup.json").write_text(
        json.dumps(
            {
                "child_stopped": stopped.is_set(),
                "pending_children": len(pending),
                "event_count": len(events),
            }
        ),
        encoding="utf-8",
    )
    print(f"Tool cleanup evidence: {evidence}")


@pytest.mark.anyio
async def test_nested_parallel_limits_and_state_rejection(tmp_path: Path) -> None:
    both_started = asyncio.Event()
    active = 0
    denied = 0
    observed = []

    @tool
    async def parallel() -> str:
        """Require overlapping native child calls to finish."""
        nonlocal active
        active += 1
        if active == 2:
            both_started.set()
        await both_started.wait()
        return "payload never retained in trace"

    @tool
    async def state_change(runtime: ToolRuntime) -> Command:
        """Return a graph update that compositions must reject."""
        return Command(
            update={"messages": [ToolMessage(content="changed", tool_call_id=runtime.tool_call_id)]}
        )

    @tool
    async def control() -> str:
        """Record if an excluded tool unexpectedly ran."""
        nonlocal denied
        denied += 1
        return "must not execute"

    @tool
    async def compose() -> str:
        """Exercise concurrency, rejection, output pairing and byte budgets."""
        nested = get_nested_tool_executor()
        observed.extend(
            await asyncio.gather(nested.call("parallel", {}), nested.call("parallel", {}))
        )
        observed.append(await nested.call("state_change", {}))
        observed.append(await nested.call("control", {}))
        observed.append(await nested.call("parallel", {"unexpected": "x" * 65537}))
        for _ in range(251):
            await nested.call("missing", {})
        with pytest.raises(ValueError, match="budget"):
            await nested.call("missing", {})
        return "done"

    catalog = ToolSet.from_tools([parallel, state_change, compose]).with_tools(
        ToolDefinition(control, label="control", exposure="model-only")
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=ScriptedChatModel(
                [tool_call_ai("parent", "compose", {}), AIMessage(content="done")]
            ),
            tools=catalog.tools,
        )
    )
    async with asyncio.timeout(10):
        events = [event async for event in harness.prompt("verify boundaries")]
    assert denied == 0 and active == 2
    assert [result.status for result in observed] == [
        "success",
        "success",
        "error",
        "error",
        "error",
    ]
    parent = next(message for message in harness.messages if isinstance(message, ToolMessage))
    assert parent.status == "success"
    trace = parent.artifact["forge.nested_calls.v1"]
    assert trace["total"] == 256 and len(trace["calls"]) == 256
    assert not trace["complete"]
    assert "payload never retained" not in json.dumps(trace)
    assert len([message for message in harness.messages if isinstance(message, ToolMessage)]) == 1
    evidence = tmp_path / "tool-ecosystem-evidence"
    evidence.mkdir()
    export_session_jsonl(
        [MessageEntry(message=message) for message in harness.messages], evidence / "session.jsonl"
    )
    (evidence / "summary.json").write_text(
        json.dumps(
            {
                "parallel_children": active,
                "denied_executions": denied,
                "trace_count": trace["total"],
                "event_count": len(events),
            }
        ),
        encoding="utf-8",
    )
    print(f"Tool budget evidence: {evidence}")
