"""Offline end-to-end checks for R2 subagent model and context contracts."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool

from fake_models import ScriptedChatModel, ScriptedErrorChatModel, tool_call_ai
from forge_agent import (
    SubagentModelCandidate,
    SubagentRunner,
    SubagentRuntime,
    SubagentSpec,
)
from forge_agent.context import ForgeRuntimeContext


def _runner(
    primary: ScriptedChatModel,
    fallback: ScriptedChatModel | None = None,
    *,
    spec: SubagentSpec | None = None,
) -> SubagentRunner:
    candidates = [
        SubagentModelCandidate(
            provider=primary,
            provider_name="fake-primary",
            model="primary",
            context_window=16_384,
        )
    ]
    if fallback is not None:
        candidates.append(
            SubagentModelCandidate(
                provider=fallback,
                provider_name="fake-fallback",
                model="fallback",
                context_window=16_384,
            )
        )
    runtime = SubagentRuntime(
        provider=primary,
        model="primary",
        candidates=tuple(candidates),
        runtime_context=ForgeRuntimeContext(workspace_root="."),
    )
    return SubagentRunner(
        lambda: runtime,
        [
            spec
            or SubagentSpec(
                name="scout",
                description="inspect",
                system_prompt="You inspect code.",
                model="primary",
                fallback_models=("fallback",) if fallback is not None else (),
            )
        ],
    )


@pytest.mark.anyio
async def test_transient_fallback_retries_only_current_model_request() -> None:
    primary = ScriptedErrorChatModel(
        [AIMessage(content="unused")],
        error_message="temporary 503",
    )
    fallback = ScriptedChatModel(
        [
            AIMessage(
                content="fallback answer",
                usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            )
        ]
    )

    result = await _runner(primary, fallback).run("scout", "inspect")

    assert result.status == "completed"
    assert result.final_output == "fallback answer"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    attempts = result.to_artifact()["attempts"]
    assert isinstance(attempts, list)
    assert [attempt["model"] for attempt in attempts] == ["primary", "fallback"]
    assert attempts[0]["reason"] == "transient"
    assert result.usage_facts[-1].requested_model == "fallback"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        "429 quota exhausted",
        "401 unauthorized",
        "429 insufficient_quota",
        "400 invalid request",
        "context length exceeded",
        "synthetic unknown",
    ],
)
async def test_non_transient_model_errors_do_not_fallback(error: str) -> None:
    primary = ScriptedErrorChatModel([AIMessage(content="unused")], error_message=error)
    fallback = ScriptedChatModel([AIMessage(content="must not run")])

    result = await _runner(primary, fallback).run("scout", "inspect")

    assert result.status == "failed"
    assert len(primary.calls) == 1
    assert fallback.calls == []
    attempts = result.to_artifact()["attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["reason"] in {"quota", "auth", "invalid_request", "overflow", "unknown"}


@pytest.mark.anyio
async def test_fallback_does_not_repeat_a_completed_tool_call() -> None:
    writes: list[str] = []

    @tool
    def write_marker(value: str) -> str:
        """Record one marker."""
        writes.append(value)
        return "written"

    primary = ScriptedErrorChatModel(
        [tool_call_ai("write-1", "write_marker", {"value": "once"})],
        error_on_call=2,
        error_message="temporary 503",
    )
    fallback = ScriptedChatModel([AIMessage(content="recovered")])
    runner = _runner(primary, fallback)
    runner.replace_specs(
        [
            SubagentSpec(
                name="scout",
                description="inspect",
                system_prompt="You inspect code.",
                tools=[write_marker],
                model="primary",
                fallback_models=("fallback",),
            )
        ]
    )

    result = await runner.run("scout", "write")

    assert result.status == "completed"
    assert result.final_output == "recovered"
    assert writes == ["once"]
    assert result.tool_calls == 1
    assert len(primary.calls) == 2
    assert len(fallback.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("pending_id", ["task-1", None])
async def test_fork_snapshot_is_paired_redacted_and_isolated(pending_id: str | None) -> None:
    model = ScriptedChatModel([AIMessage(content="done")])
    runner = _runner(model)
    parent_messages = [
        HumanMessage(content="old user", response_metadata={"secret": "drop"}),
        AIMessage(
            content="old assistant",
            tool_calls=[
                {"id": "read-1", "name": "read", "args": {"path": "a.txt"}, "type": "tool_call"}
            ],
            additional_kwargs={"reasoning_content": "drop"},
            response_metadata={"provider": "drop"},
        ),
        ToolMessage(
            content="old tool result",
            name="read",
            tool_call_id="read-1",
            artifact={"secret": "drop"},
            response_metadata={"provider": "drop"},
        ),
        HumanMessage(content="current user"),
        AIMessage(
            content="pending",
            tool_calls=[
                {
                    "id": pending_id,
                    "name": "task",
                    "args": {"instruction": "child"},
                    "type": "tool_call",
                }
            ],
        ),
    ]
    original = deepcopy(parent_messages)

    prepared = runner.prepare(
        "scout",
        "child instruction",
        parent_messages=parent_messages,
        context="fork",
    )

    assert prepared.context_mode == "fork"
    assert prepared.context_truncated is False
    assert prepared.inherited_message_count == 4
    assert [message.type for message in prepared.messages] == [
        "human",
        "ai",
        "tool",
        "human",
        "human",
    ]
    assert prepared.messages[-1].content == "child instruction"
    assert prepared.messages[1].additional_kwargs == {}
    assert prepared.messages[1].response_metadata == {}
    assert prepared.messages[2].artifact is None
    assert parent_messages == original

    parent_messages[0].content = "mutated after prepare"
    assert prepared.messages[0].content == "old user"

    result = await runner.run_prepared(prepared)
    submitted = model.calls[0]["messages"]
    assert [message.type for message in submitted] == [
        "system",
        "human",
        "ai",
        "tool",
        "human",
        "human",
    ]
    assert result.context_mode == "fork"


@pytest.mark.anyio
async def test_fork_reports_window_failure_instead_of_dropping_history() -> None:
    model = ScriptedChatModel([AIMessage(content="must not run")])
    runner = _runner(model)
    runtime = SubagentRuntime(
        provider=model,
        model="primary",
        candidates=(
            SubagentModelCandidate(
                provider=model,
                model="primary",
                context_window=8,
            ),
        ),
        runtime_context=ForgeRuntimeContext(workspace_root="."),
    )
    runner = SubagentRunner(
        lambda: runtime,
        [
            SubagentSpec(
                name="scout",
                description="inspect",
                system_prompt="You inspect code.",
                model="primary",
                context="fork",
            )
        ],
    )

    result = await runner.run(
        "scout",
        "new instruction",
        parent_messages=[HumanMessage(content="a very long parent fact")],
        context="fork",
    )

    assert result.status == "failed"
    assert "summary" in (result.error or "")
    assert model.calls == []


@pytest.mark.anyio
async def test_logical_requests_and_actual_attempts_are_separately_bounded(tmp_path):
    class Unavailable(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._record_call(messages)
            raise RuntimeError("429 temporary rate limit")

    marker = tmp_path / "writes.txt"

    @tool
    def append_marker() -> str:
        """Append one synthetic acceptance marker."""
        with marker.open("a", encoding="utf-8") as output:
            output.write("once\n")
        return "written"

    class ToolLoop(ScriptedChatModel):
        def _next_response(self):
            return tool_call_ai(f"write-{len(self.calls)}", "append_marker", {})

    models = [Unavailable(), Unavailable(), ToolLoop()]
    runner = SubagentRunner(
        lambda: SubagentRuntime(
            models[0],
            "a",
            candidates=tuple(
                SubagentModelCandidate(
                    model,
                    name,
                    "fake",
                    64000,
                    1024,
                )
                for model, name in zip(models, ("a", "b", "c"), strict=True)
            ),
        ),
        [
            SubagentSpec(
                "worker",
                "write",
                "Offline",
                tools=[append_marker],
                model="a",
                fallback_models=("b", "c"),
            )
        ],
    )
    result = await runner.run("worker", "append within budget")
    assert result.status == "failed"
    assert [len(model.calls) for model in models] == [8, 8, 8]
    assert result.model_calls == 8
    assert len(result.attempts) == 24
    assert len(result.usage_facts) == 24
    assert marker.read_text(encoding="utf-8").splitlines() == ["once"] * 8
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "case": "attempt_budget",
                "pass": True,
                "status": result.status,
                "logical_requests": result.model_calls,
                "attempts": len(result.attempts),
                "paired_calls": 8,
                "residual_tasks": 0,
                "artifacts": ["writes.txt"],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.anyio
async def test_partial_usage_before_fallback_and_cancel_is_preserved(tmp_path):
    class Partial(ScriptedChatModel):
        streaming: bool = True

        def __init__(self, *, block=False):
            super().__init__()
            self.streaming = True
            object.__setattr__(self, "started", asyncio.Event())
            object.__setattr__(self, "block", block)

        async def _astream(self, messages, **kwargs):
            self._record_call(messages)
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="partial",
                    usage_metadata={
                        "input_tokens": 7,
                        "output_tokens": 1,
                        "total_tokens": 8,
                    },
                )
            )
            self.started.set()
            if self.block:
                await asyncio.Event().wait()
            raise RuntimeError("503 interrupted stream")

    primary = Partial()
    fallback = ScriptedChatModel(
        [
            AIMessage(
                content="done",
                usage_metadata={
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
            )
        ]
    )
    result = await _runner(primary, fallback).run("scout", "inspect")
    assert result.status == "completed"
    assert [fact.total_tokens for fact in result.usage_facts] == [8, 7]
    assert result.total_tokens == 15

    blocked = Partial(block=True)
    runner = _runner(blocked)
    updates = []
    child = asyncio.create_task(
        runner.run_background(runner.prepare("scout", "inspect"), updates.append)
    )
    await asyncio.wait_for(blocked.started.wait(), 5)
    child.cancel()
    with pytest.raises(asyncio.CancelledError):
        await child
    facts = next(update["usage_facts"] for update in reversed(updates) if "usage_facts" in update)
    assert facts and facts[-1]["total_tokens"] == 8
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "case": "partial_usage",
                "pass": True,
                "status": "cancelled",
                "fallback_tokens": result.total_tokens,
                "cancelled_tokens": facts[-1]["total_tokens"],
                "residual_tasks": 0,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.anyio
async def test_smallest_candidate_window_trims_only_old_complete_groups(tmp_path):
    primary, fallback = ScriptedChatModel([AIMessage(content="done")]), ScriptedChatModel()
    runner = _runner(primary, fallback)
    runtime = SubagentRuntime(
        primary,
        "primary",
        candidates=(
            SubagentModelCandidate(primary, "primary", "fake", 64000, 128),
            SubagentModelCandidate(fallback, "fallback", "fake", 1800, 128),
        ),
    )
    history = [
        SystemMessage(content="PARENT_SYSTEM_MUST_NOT_LEAK"),
        HumanMessage(content="old " * 900),
        tool_call_ai("old-pair", "read", {"path": "old"}),
        ToolMessage(content="old result", tool_call_id="old-pair"),
        HumanMessage(content="RECENT_USER"),
        AIMessage(content="<think>HIDDEN</think>visible"),
        AIMessage(
            content="pending",
            tool_calls=[
                {"id": "a", "name": "task", "args": {}},
                {"id": "b", "name": "read", "args": {}},
            ],
        ),
        ToolMessage(content="partial batch", tool_call_id="b"),
    ]
    prepared = runner.prepare(
        "scout", "child", runtime=runtime, parent_messages=history, context="fork"
    )
    assert prepared.context_truncated
    assert prepared.context_error is None
    assert prepared.inherited_message_count == len(prepared.messages) - 1
    submitted = repr(prepared.messages)
    assert "RECENT_USER" in submitted
    assert all(
        text not in submitted for text in ("old result", "HIDDEN", "partial batch", "PARENT_SYSTEM")
    )
    history[-3].content = "MUTATED"
    result = await runner.run_prepared(prepared)
    assert result.status == "completed"
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "case": "trim_snapshot",
                "pass": True,
                "status": result.status,
                "context_truncated": True,
                "inherited_messages": prepared.inherited_message_count,
                "paired_calls": 0,
                "residual_tasks": 0,
            }
        ),
        encoding="utf-8",
    )
