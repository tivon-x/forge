"""Focused background usage regression coverage."""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from fake_models import ScriptedChatModel
from forge_agent import (
    ForgeRuntimeContext,
    SubagentRunner,
    SubagentRuntime,
    SubagentSpec,
)
from forge_coding.features.subagent_runs import SubagentRuns


def test_failed_background_run_keeps_usage_from_progress_updates(tmp_path) -> None:
    async def run() -> None:
        @tool
        def echo(value: str) -> str:
            """Echo a value."""
            return value

        first_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "echo-1",
                    "name": "echo",
                    "args": {"value": "ok"},
                    "type": "tool_call",
                }
            ],
            usage_metadata={"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
        )
        model = ScriptedChatModel([first_response])
        runtime = SubagentRuntime(
            provider=model,
            model="fake",
            runtime_context=ForgeRuntimeContext(workspace_root="."),
        )
        runner = SubagentRunner(
            lambda: runtime,
            [
                SubagentSpec(
                    name="scout",
                    description="inspect",
                    system_prompt="You inspect code.",
                    tools=[echo],
                    max_model_calls=1,
                )
            ],
        )
        runs = SubagentRuns(
            runner=runner,
            changed=lambda: None,
            origin=lambda: "test",
            concurrency=1,
            queue_limit=0,
            timeout_seconds=5,
        )
        receipt = runs.submit("scout", "echo")
        run_id = receipt["run_id"]
        assert isinstance(run_id, str)
        runs.accept(run_id)

        result = await runs.control("wait", run_id, 5)

        assert result["status"] == "failed"
        assert result["input_tokens"] == 9
        assert result["output_tokens"] == 4
        assert result["total_tokens"] == 13
        facts = runs.get(run_id).usage_facts
        assert len(facts) == 1
        assert facts[0]["input_tokens"] == 9
        assert facts[0]["output_tokens"] == 4
        assert facts[0]["total_tokens"] == 13
        (tmp_path / "summary.json").write_text(
            json.dumps(
                {
                    "case": "failed-background-usage",
                    "passed": True,
                    "status": result["status"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                    "usage_facts": len(facts),
                }
            ),
            encoding="utf-8",
        )

    asyncio.run(run())
