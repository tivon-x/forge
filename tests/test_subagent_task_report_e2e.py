from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from fake_models import ScriptedChatModel
from forge_agent.context import ForgeRuntimeContext
from forge_agent.subagents import (
    SubagentRunner,
    SubagentRuntime,
    SubagentSpec,
    TaskReport,
)


def _runner(model: ScriptedChatModel, *, result_format: str = "task-report", **kwargs: object):
    return SubagentRunner(
        lambda: SubagentRuntime(
            provider=model,
            model="fake",
            runtime_context=ForgeRuntimeContext(workspace_root="."),
        ),
        [
            SubagentSpec(
                name="worker",
                description="worker",
                system_prompt="Return a bounded report.",
                result_format=result_format,
                **kwargs,
            )
        ],
    )


def _report_call(
    *,
    call_id: str = "report-1",
    outcome: str = "done",
    summary: str = "Implemented the change.",
    findings: list[str] | None = None,
    evidence: list[str] | None = None,
    question: str | None = None,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "name": "TaskReport",
                "args": {
                    "outcome": outcome,
                    "summary": summary,
                    "findings": findings or [],
                    "evidence": evidence or [],
                    "question": question,
                },
                "type": "tool_call",
            }
        ],
    )


@pytest.mark.anyio
async def test_task_report_uses_structured_response_and_keeps_blocked_execution_completed(
    tmp_path,
) -> None:
    model = ScriptedChatModel(
        [
            _report_call(
                outcome="blocked",
                summary="The required provider is unavailable.",
                question="Configure the provider before retrying.",
                evidence=["Provider catalog has no authorized entry."],
            )
        ]
    )

    result = await _runner(model).run("worker", "Implement the task")

    assert result.status == "completed"
    assert result.report is not None
    assert result.report.outcome == "blocked"
    assert json.loads(result.content)["outcome"] == "blocked"
    assert result.to_artifact()["report"] == result.report.model_dump(mode="json")
    assert result.to_artifact()["status"] == "completed"
    restored = type(result).from_artifact(result.to_artifact())
    assert restored.report == result.report
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {"case": "blocked-report", "status": result.status, "outcome": result.report.outcome},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.anyio
async def test_invalid_structured_report_retries_within_existing_model_budget() -> None:
    model = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "bad",
                        "name": "TaskReport",
                        "args": {"outcome": "done", "summary": "missing fields"},
                        "type": "tool_call",
                    }
                ],
            ),
            _report_call(summary="Corrected after validation."),
        ]
    )

    result = await _runner(model, max_model_calls=2).run("worker", "Implement the task")

    assert result.status == "completed"
    assert result.report is not None
    assert result.report.summary == "Corrected after validation."
    assert result.model_calls == 2


@pytest.mark.anyio
async def test_structured_output_tool_is_not_counted_as_a_file_tool_in_a_mixed_batch() -> None:
    @tool
    def read_marker() -> str:
        """Read the marker."""
        return "marker"

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
                    },
                    {
                        "id": "report-1",
                        "name": "TaskReport",
                        "args": {
                            "outcome": "done",
                            "summary": "Read the marker.",
                            "findings": [],
                            "evidence": ["marker"],
                            "question": None,
                        },
                        "type": "tool_call",
                    },
                ],
            )
        ]
    )

    result = await _runner(model, tools=(read_marker,)).run("worker", "Read the marker")

    assert result.status == "completed"
    assert result.report is not None
    assert result.tool_calls == 1


@pytest.mark.anyio
async def test_missing_structured_response_cannot_fall_back_to_plain_text() -> None:
    model = ScriptedChatModel([AIMessage(content="The work is complete.")])

    result = await _runner(model).run("worker", "Implement the task")

    assert result.status == "failed"
    assert result.report is None
    assert result.final_output == ""
    assert "structured" in (result.error or "").lower()


@pytest.mark.anyio
async def test_task_report_over_budget_fails_without_truncating_json() -> None:
    model = ScriptedChatModel(
        [
            _report_call(
                summary="x" * 3_000,
                evidence=["e" * 2_000],
            )
        ]
    )

    result = await _runner(model, max_result_bytes=1_024).run("worker", "Implement the task")

    assert result.status == "failed"
    assert result.report is None
    assert result.final_output == ""
    assert result.truncated is False
    assert "budget" in (result.error or "").lower()


@pytest.mark.anyio
async def test_background_task_report_reads_structured_response_from_v3_values() -> None:
    model = ScriptedChatModel([_report_call(summary="Background report.")])
    runner = _runner(model)
    prepared = runner.prepare("worker", "Implement in the background")
    updates: list[dict[str, object]] = []

    result = await runner.run_background(prepared, updates.append)

    assert result.status == "completed"
    assert result.report is not None
    assert result.report.summary == "Background report."
    assert all(update.get("tool_calls") != 1 for update in updates)


def test_task_report_rejects_unbounded_or_unknown_fields() -> None:
    with pytest.raises(ValueError):
        TaskReport.model_validate(
            {
                "outcome": "done",
                "summary": "ok",
                "findings": ["x"] * 17,
                "evidence": [],
                "question": None,
            }
        )
    with pytest.raises(ValueError):
        TaskReport.model_validate(
            {
                "outcome": "done",
                "summary": "ok",
                "findings": [],
                "evidence": [],
                "question": None,
                "extra": "reject",
            }
        )
