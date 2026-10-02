"""R3 review reproductions for cross-provider and capacity boundaries."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from fake_models import ScriptedChatModel
from forge_agent.context import ForgeRuntimeContext
from forge_agent.subagents import (
    SubagentModelCandidate,
    SubagentRunner,
    SubagentRuntime,
    SubagentSpec,
    TaskReport,
    _estimate_message_tokens,
)
from forge_coding.tools import create_write_tool


def _report_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "report-1",
                "name": "TaskReport",
                "args": {
                    "outcome": "done",
                    "summary": "The marker was read.",
                    "findings": [],
                    "evidence": ["marker"],
                    "question": None,
                },
                "type": "tool_call",
            }
        ],
    )


@pytest.mark.anyio
async def test_task_report_capacity_counts_virtual_schema_after_native_tool() -> None:
    """A later create_agent model request must include the TaskReport schema budget."""

    @tool
    def read_marker() -> str:
        """Read the marker."""

        return "x" * 100

    model = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "read-1",
                        "name": "read_marker",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
            _report_call(),
        ]
    )
    first_messages = (HumanMessage(content="i"),)
    system_bytes = len(b"s")
    native_schema_bytes = len(
        json.dumps(convert_to_openai_tool(read_marker), ensure_ascii=False).encode("utf-8")
    )
    report_schema_bytes = len(
        json.dumps(convert_to_openai_tool(TaskReport), ensure_ascii=False).encode("utf-8")
    )
    context_window = (
        _estimate_message_tokens(first_messages)
        + system_bytes
        + native_schema_bytes
        + report_schema_bytes
        + 1
    )
    second_messages = first_messages + (
        model.responses[0],
        ToolMessage(content="x" * 100, name="read_marker", tool_call_id="read-1"),
    )
    assert (
        _estimate_message_tokens(second_messages)
        + system_bytes
        + native_schema_bytes
        + report_schema_bytes
        + 1
        > context_window
    )
    candidate = SubagentModelCandidate(
        model,
        "fake",
        "fake",
        context_window,
        1,
    )
    runtime = SubagentRuntime(
        provider=model,
        model="fake",
        candidates=(candidate,),
        runtime_context=ForgeRuntimeContext(workspace_root="."),
    )
    runner = SubagentRunner(
        lambda: runtime,
        [
            SubagentSpec(
                name="worker",
                description="worker",
                system_prompt="s",
                tools=[read_marker],
                result_format="task-report",
            )
        ],
    )

    result = await runner.run("worker", "i")

    # The provider must not receive the over-capacity second request. The graph
    # may still count the rejected logical request before the guard runs.
    assert result.status == "failed"
    assert len(model.calls) == 1
    assert "context" in (result.error or "").lower()


@pytest.mark.anyio
async def test_task_report_mixed_batch_executes_real_forge_write(tmp_path) -> None:
    """The native Forge write tool remains executable beside TaskReport."""

    report = {
        "outcome": "done",
        "summary": "Wrote the marker.",
        "findings": [],
        "evidence": ["marker.txt"],
        "question": None,
    }
    model = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "write-1",
                        "name": "write",
                        "args": {"path": "marker.txt", "content": "one effect"},
                        "type": "tool_call",
                    },
                    {
                        "id": "report-1",
                        "name": "TaskReport",
                        "args": report,
                        "type": "tool_call",
                    },
                ],
            )
        ]
    )
    runner = SubagentRunner(
        lambda: SubagentRuntime(
            provider=model,
            model="fake",
            runtime_context=ForgeRuntimeContext(workspace_root=str(tmp_path)),
        ),
        [
            SubagentSpec(
                name="worker",
                description="worker",
                system_prompt="Return a bounded report.",
                tools=[create_write_tool(cwd=tmp_path)],
                result_format="task-report",
            )
        ],
    )

    result = await runner.run("worker", "Write the marker")

    assert result.status == "completed"
    assert result.tool_calls == 1
    assert result.report is not None
    assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "one effect"
