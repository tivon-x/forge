"""Unit coverage for the generic Forge subagent runner."""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException, tool
from pydantic import PrivateAttr

from conftest import make_native_tool
from forge_agent import (
    AgentToolResult,
    SubagentRunner,
    SubagentRunResult,
    SubagentRuntime,
    SubagentSpec,
    SubagentTrace,
    SubagentTraceItem,
    TokenUsage,
    aggregate_usage,
    project_subagent_trace,
)
from forge_agent.context import ForgeRuntimeContext


def _runner(model: FakeListChatModel, **spec_kwargs: object) -> SubagentRunner:
    runtime = SubagentRuntime(
        provider=model,
        model="fake",
        runtime_context=ForgeRuntimeContext(workspace_root="."),
    )
    spec = SubagentSpec(
        name="scout",
        description="inspect",
        system_prompt="You inspect code.",
        **spec_kwargs,
    )
    return SubagentRunner(lambda: runtime, [spec])


def test_runner_rejects_usage_projection_overflow() -> None:
    with pytest.raises(ValueError, match="max_model_calls must be at most 8"):
        _runner(FakeListChatModel(responses=["unused"]), max_model_calls=9)


@pytest.mark.anyio
async def test_runner_uses_fresh_instruction_and_returns_artifact() -> None:
    model = FakeListChatModel(responses=["answer"])
    runner = _runner(model)

    result = await runner.run("scout", "  inspect this  ")

    assert result.status == "completed"
    assert result.final_output == "answer"
    assert result.content == "answer"
    assert result.model_calls == 1
    artifact = result.artifact()
    assert artifact["kind"] == "subagent_run"
    assert artifact["version"] == 2
    assert artifact["instruction"] == "inspect this"
    assert artifact["error"] is None


def test_extreme_usage_values_are_dropped_before_serializing_bounded_payloads() -> None:
    huge = 1 << 20_000
    result = SubagentRunResult(
        agent="scout",
        status="completed",
        instruction="inspect",
        final_output="done",
        model_calls=1,
        tool_calls=0,
        queued_ms=0,
        duration_ms=1,
        input_tokens=huge,
        output_tokens=huge,
        total_tokens=huge,
    )
    object.__setattr__(result, "_max_result_bytes", 1024)

    artifact = result.to_artifact()
    encoded = json.dumps(artifact, ensure_ascii=False).encode("utf-8")

    assert len(encoded) <= 1024
    assert "input_tokens" not in artifact
    assert "output_tokens" not in artifact
    assert "total_tokens" not in artifact
    trace = project_subagent_trace(
        [HumanMessage(content="inspect"), AIMessage(content="done")],
        usage=TokenUsage(input_tokens=huge, output_tokens=huge, total_tokens=huge),
    )
    assert trace.input_tokens is None
    assert trace.output_tokens is None
    assert trace.total_tokens is None


def test_trace_projection_drops_thinking_raw_arguments_and_results() -> None:
    messages = [
        HumanMessage(content="Review the plan", response_metadata={"secret": "metadata"}),
        AIMessage(
            content=[
                {"type": "reasoning", "text": "do not persist this thought"},
                {"type": "text", "text": "I will inspect the files."},
            ],
            tool_calls=[
                {
                    "id": "child-call-1",
                    "name": "read",
                    "args": {"path": "C:/private/secret.txt", "token": "secret"},
                    "type": "tool_call",
                }
            ],
            usage_metadata={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            additional_kwargs={"reasoning_content": "hidden provider reasoning"},
        ),
        ToolMessage(
            content="raw file contents and secret",
            name="read",
            tool_call_id="child-call-1",
            status="error",
            artifact={"path": "C:/private/secret.txt", "content": "secret"},
        ),
        AIMessage(
            content="The plan needs one more boundary check.",
            usage_metadata={"input_tokens": 2, "output_tokens": 5, "total_tokens": 7},
        ),
    ]

    trace = project_subagent_trace(messages, agent="oracle")
    payload = json.dumps(trace.to_dict(), ensure_ascii=False)

    assert [item.kind for item in trace.items] == [
        "human",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
    ]
    assert trace.items[2].text == "Calling read"
    assert trace.items[3].status == "error"
    assert trace.input_tokens == 6
    assert trace.output_tokens == 8
    assert trace.total_tokens == 14
    assert "secret" not in payload
    assert "private" not in payload
    assert "reasoning" not in payload


def test_trace_projection_drops_unknown_blocks_and_inline_thinking_tags() -> None:
    trace = project_subagent_trace(
        [
            AIMessage(
                content=[
                    {"type": "redacted_thinking", "text": "hidden secret"},
                    {"type": "reasoning_content", "text": "private chain"},
                    {
                        "type": "tool_use",
                        "text": "C:/private/secret.txt",
                        "input": {"path": "C:/private/secret.txt"},
                    },
                    {"type": "text", "text": "<think>secret</think>Visible answer"},
                ]
            )
        ],
        agent="reviewer",
    )
    payload = json.dumps(trace.to_dict(), ensure_ascii=False)

    assert trace.items == (SubagentTraceItem(kind="assistant", text="Visible answer"),)
    assert "hidden secret" not in payload
    assert "private chain" not in payload
    assert "private/secret" not in payload


def test_trace_projection_drops_unclosed_inline_thinking_tail() -> None:
    trace = project_subagent_trace(
        [AIMessage(content="Visible answer<think>unfinished secret")],
        agent="reviewer",
    )

    assert trace.items == (SubagentTraceItem(kind="assistant", text="Visible answer"),)


def test_trace_projection_strips_hidden_tags_split_across_text_blocks() -> None:
    trace = project_subagent_trace(
        [
            AIMessage(
                content=[
                    {"type": "text", "text": "<think>"},
                    {"type": "text", "text": "secret"},
                    {"type": "text", "text": "</think>Visible answer"},
                ]
            )
        ],
        agent="reviewer",
    )

    assert trace.items == (SubagentTraceItem(kind="assistant", text="Visible answer"),)


@pytest.mark.parametrize(
    "hidden",
    [
        "<think>secret<analysis>nested</analysis>still</think>Visible",
        "<think>secret</analysis>still</think>Visible",
    ],
)
def test_trace_projection_strips_nested_and_mismatched_hidden_tags(hidden: str) -> None:
    trace = project_subagent_trace([AIMessage(content=hidden)], agent="reviewer")

    assert trace.items == (SubagentTraceItem(kind="assistant", text="Visible"),)


def test_trace_projection_preserves_thinking_like_tags_in_user_instruction() -> None:
    trace = project_subagent_trace(
        [HumanMessage(content="Review <analysis>this literal section</analysis>")],
        agent="reviewer",
    )

    assert trace.items == (
        SubagentTraceItem(
            kind="human",
            text="Review <analysis>this literal section</analysis>",
        ),
    )


def test_trace_projection_applies_item_count_and_utf8_budgets() -> None:
    messages = [HumanMessage(content="instruction")]
    messages.extend(AIMessage(content=f"step-{index}") for index in range(100))
    messages.append(AIMessage(content="final answer"))

    trace = project_subagent_trace(messages, agent="worker")
    serialized = json.dumps(trace.to_dict(), ensure_ascii=False).encode("utf-8")

    assert trace.truncated is True
    assert len(trace.items) <= 64
    assert len(serialized) <= 64 * 1024
    assert trace.items[0] == SubagentTraceItem(kind="human", text="instruction")
    assert trace.items[-1] == SubagentTraceItem(kind="assistant", text="final answer")
    omitted = [item for item in trace.items if item.kind == "omitted"]
    assert len(omitted) == 1
    assert omitted[0].omitted >= 38
    assert all(
        item.text is None or len(item.text.encode("utf-8")) <= 8 * 1024 for item in trace.items
    )


def test_trace_projection_marks_single_item_utf8_truncation() -> None:
    trace = project_subagent_trace(
        [HumanMessage(content="界" * 4000), AIMessage(content="done")],
        agent="worker",
    )

    assert trace.truncated is True
    assert len((trace.items[0].text or "").encode("utf-8")) <= 8 * 1024


def test_trace_projection_trims_aggregate_before_constructing_dto() -> None:
    messages = [HumanMessage(content="instruction")]
    messages.extend(AIMessage(content=str(index) + "x" * 8190) for index in range(20))
    messages.append(AIMessage(content="final answer"))

    trace = project_subagent_trace(messages, agent="worker")
    serialized = json.dumps(trace.to_dict(), ensure_ascii=False).encode("utf-8")

    assert trace.truncated is True
    assert len(serialized) <= 64 * 1024
    assert trace.items[0] == SubagentTraceItem(kind="human", text="instruction")
    assert trace.items[-1] == SubagentTraceItem(kind="assistant", text="final answer")
    assert any(item.kind == "omitted" for item in trace.items)


def test_trace_dto_rejects_unknown_or_unsafe_fields_and_round_trips() -> None:
    trace = SubagentTrace(
        agent="oracle",
        items=(SubagentTraceItem(kind="assistant", text="done"),),
        input_tokens=1,
    )
    assert SubagentTrace.from_dict(trace.to_dict()) == trace

    with pytest.raises(ValueError, match="unknown subagent trace field"):
        SubagentTrace.from_dict({**trace.to_dict(), "thinking": "secret"})
    with pytest.raises(ValueError, match="unknown subagent trace item field"):
        SubagentTraceItem.from_dict({"kind": "assistant", "text": "ok", "args": "secret"})
    with pytest.raises(TypeError, match="non-negative"):
        SubagentTrace.from_dict({"agent": "x", "items": [], "input_tokens": -1})


def test_usage_aggregation_ignores_missing_negative_and_provider_fields() -> None:
    usage = aggregate_usage(
        [
            AIMessage(
                content="one",
                usage_metadata={
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "provider_extra": 99,
                },
            ),
            AIMessage.model_construct(
                content="two",
                usage_metadata={"input_tokens": -4, "output_tokens": "bad"},
            ),
            HumanMessage(content="not a model call"),
        ]
    )
    assert usage.to_dict() == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}


@pytest.mark.anyio
async def test_runner_truncates_on_utf8_boundary() -> None:
    model = FakeListChatModel(responses=["甲乙丙"])
    runner = _runner(model, max_result_bytes=7)

    result = await runner.run("scout", "read")

    assert result.final_output == "甲乙"
    assert result.truncated is True
    assert len(result.final_output.encode("utf-8")) <= 7


@pytest.mark.anyio
async def test_long_provider_exception_is_bounded_across_public_result_and_artifact() -> None:
    budget = 256
    long_error = "provider failure: " + ("错误" * 2_000)

    class FailingModel(FakeListChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            raise RuntimeError(long_error)

    result = await _runner(FailingModel(responses=["unused"]), max_result_bytes=budget).run(
        "scout", "inspect"
    )
    artifact = result.to_artifact()

    assert result.error is not None
    assert len(result.error.encode("utf-8")) <= budget
    assert len(result.content.encode("utf-8")) <= budget
    assert len(json.dumps(artifact, ensure_ascii=False).encode("utf-8")) <= budget
    assert all(
        not isinstance(value, str) or len(value.encode("utf-8")) <= budget
        for value in artifact.values()
    )
    assert result.truncated is True


@pytest.mark.anyio
async def test_long_instruction_is_bounded_and_not_repeated_in_artifact() -> None:
    budget = 256
    long_instruction = "检查：" + ("说明" * 2_000)
    result = await _runner(FakeListChatModel(responses=["answer"]), max_result_bytes=budget).run(
        "scout", long_instruction
    )
    artifact = result.to_artifact()
    serialized_artifact = json.dumps(artifact, ensure_ascii=False)

    assert len(result.instruction.encode("utf-8")) <= budget
    assert len(result.content.encode("utf-8")) <= budget
    assert len(serialized_artifact.encode("utf-8")) <= budget
    assert serialized_artifact.count(long_instruction) == 0
    assert artifact["instruction"] != long_instruction
    assert artifact["final_output"] == "answer"
    assert result.truncated is True


@pytest.mark.anyio
async def test_empty_final_ai_message_is_a_completed_result() -> None:
    runner = _runner(FakeListChatModel(responses=[""]))

    result = await runner.run("scout", "inspect")

    assert result.status == "completed"
    assert result.final_output == ""
    assert result.model_calls == 1
    assert result.error is None


class _BlockingModel(BaseChatModel):
    """A cancellable model used to exercise concurrent child runs."""

    _started: asyncio.Event = PrivateAttr()
    _both_started: asyncio.Event = PrivateAttr()
    _release: asyncio.Event = PrivateAttr()
    _call_count: int = PrivateAttr(default=0)

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_started", asyncio.Event())
        object.__setattr__(self, "_both_started", asyncio.Event())
        object.__setattr__(self, "_release", asyncio.Event())

    @property
    def started(self) -> asyncio.Event:
        return self._started

    @property
    def release(self) -> asyncio.Event:
        return self._release

    @property
    def both_started(self) -> asyncio.Event:
        return self._both_started

    @property
    def _llm_type(self) -> str:
        return "forge-blocking-subagent"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        self._call_count += 1
        self.started.set()
        if self._call_count >= 2:
            self.both_started.set()
        await self.release.wait()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        raise AssertionError("blocking test model must use async generation")


@pytest.mark.anyio
async def test_running_child_cancellation_is_rethrown_and_allows_next_run() -> None:
    model = _BlockingModel()
    runner = _runner(model)
    active = asyncio.create_task(runner.run("scout", "first"))
    await model.started.wait()

    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active

    model.release.set()
    assert (await runner.run("scout", "second")).final_output == "done"


@pytest.mark.anyio
async def test_concurrent_child_cancellation_does_not_affect_other_run() -> None:
    model = _BlockingModel()
    runner = _runner(model)
    first = asyncio.create_task(runner.run("scout", "first"))
    await model.started.wait()
    waiting = asyncio.create_task(runner.run("scout", "second"))
    await asyncio.sleep(0)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    model.release.set()
    assert (await first).final_output == "done"
    assert (await runner.run("scout", "third")).final_output == "done"


@pytest.mark.anyio
async def test_runtime_reader_is_called_each_time_with_latest_provider_and_context() -> None:
    seen_contexts: list[ForgeRuntimeContext | None] = []

    async def capture(
        arguments: dict[str, object],
        signal: object | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        del arguments, signal
        seen_contexts.append(context)
        return AgentToolResult(tool_call_id="", name="capture", ok=True, content="captured")

    capture_tool = make_native_tool(
        name="capture",
        description="Capture runtime context.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        executor=capture,
    )
    from fake_models import ScriptedChatModel, tool_call_ai

    provider_one = ScriptedChatModel(
        [tool_call_ai("one", "capture", {"value": "one"}), AIMessage(content="one")]
    )
    provider_two = ScriptedChatModel(
        [tool_call_ai("two", "capture", {"value": "two"}), AIMessage(content="two")]
    )
    active = [
        SubagentRuntime(
            provider=provider_one,
            model="model-one",
            runtime_context=ForgeRuntimeContext(workspace_root="one"),
        )
    ]
    runner = SubagentRunner(
        lambda: active[0],
        [SubagentSpec("scout", "inspect", "Inspect.", tools=[capture_tool])],
    )

    first = await runner.run("scout", "first")
    active[0] = SubagentRuntime(
        provider=provider_two,
        model="model-two",
        runtime_context=ForgeRuntimeContext(workspace_root="two"),
    )
    second = await runner.run("scout", "second")

    assert first.final_output == "one"
    assert second.final_output == "two"
    assert [context.workspace_root for context in seen_contexts if context] == ["one", "two"]
    assert len(provider_one.calls) == 2
    assert len(provider_two.calls) == 2


@pytest.mark.anyio
async def test_tool_calls_are_counted_in_result() -> None:
    from fake_models import ScriptedChatModel, tool_call_ai

    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    model = ScriptedChatModel(
        [tool_call_ai("echo-1", "echo", {"value": "ok"}), AIMessage(content="done")]
    )
    result = await _runner(model, tools=[echo]).run("scout", "echo")

    assert result.status == "completed"
    assert result.tool_calls == 1


@pytest.mark.anyio
async def test_runner_aggregates_native_usage_into_v2_artifact() -> None:
    from fake_models import ScriptedChatModel

    model = ScriptedChatModel(
        [
            AIMessage(
                content="done",
                usage_metadata={"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
            )
        ]
    )
    result = await _runner(model).run("scout", "inspect")

    assert result.input_tokens == 7
    assert result.output_tokens == 4
    assert result.total_tokens == 11
    assert result.to_artifact()["version"] == 2
    assert result.to_artifact()["total_tokens"] == 11


def test_v1_subagent_artifact_remains_loadable() -> None:
    artifact = {
        "kind": "subagent_run",
        "version": 1,
        "agent": "scout",
        "status": "completed",
        "instruction": "inspect",
        "final_output": "done",
        "model_calls": 1,
        "tool_calls": 0,
        "queued_ms": 0,
        "duration_ms": 2,
        "truncated": False,
        "error": None,
    }
    result = SubagentRunResult.from_artifact(artifact)
    assert result.final_output == "done"
    assert result.input_tokens is None


@pytest.mark.anyio
async def test_model_call_limit_is_a_failed_result() -> None:
    from fake_models import ScriptedChatModel, tool_call_ai

    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    model = ScriptedChatModel([tool_call_ai("echo-1", "echo", {"value": "ok"})])
    result = await _runner(model, tools=[echo], max_model_calls=1).run("scout", "echo")

    assert result.status == "failed"
    assert result.error == "Subagent reached max_model_calls=1"
    assert result.artifact()["status"] == "failed"


@pytest.mark.anyio
async def test_failed_subagent_retains_completed_model_usage_fact() -> None:
    from fake_models import ScriptedChatModel

    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    response = AIMessage(
        content="",
        tool_calls=[{"id": "echo-1", "name": "echo", "args": {"value": "ok"}, "type": "tool_call"}],
        usage_metadata={"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
    )
    result = await _runner(
        ScriptedChatModel([response]),
        tools=[echo],
        max_model_calls=1,
    ).run("scout", "echo")

    assert result.status == "failed"
    assert len(result.usage_facts) == 1
    assert result.usage_facts[0].input_tokens == 9
    assert result.usage_facts[0].output_tokens == 4
    assert "final_output" not in result.usage_facts[0].to_dict()


@pytest.mark.anyio
async def test_runner_converts_child_failure_to_failed_result() -> None:
    class FailingModel(FakeListChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            raise RuntimeError("provider down")

    result = await _runner(FailingModel(responses=["unused"])).run("scout", "inspect")

    assert result.status == "failed"
    assert result.error == "provider down"
    assert "provider down" in result.content
    assert result.artifact()["status"] == "failed"


@pytest.mark.anyio
async def test_runtime_reader_failure_is_a_failed_result() -> None:
    def read_runtime() -> SubagentRuntime:
        raise RuntimeError("runtime unavailable")

    runner = SubagentRunner(
        read_runtime,
        [SubagentSpec("scout", "inspect", "You inspect code.")],
    )

    result = await runner.run("scout", "inspect")

    assert result.status == "failed"
    assert result.error == "runtime unavailable"
    assert "runtime unavailable" in result.content
    assert result.artifact()["status"] == "failed"


@pytest.mark.anyio
async def test_runner_rejects_invalid_input_with_tool_exception() -> None:
    runner = _runner(FakeListChatModel(responses=["answer"]))

    with pytest.raises(ToolException, match="Unknown subagent role"):
        await runner.run("unknown", "inspect")
    with pytest.raises(ToolException, match="must not be empty"):
        await runner.run("scout", " \t\n")


@pytest.mark.anyio
async def test_runner_allows_concurrent_child_calls() -> None:
    model = _BlockingModel()
    runner = _runner(model)
    first = asyncio.create_task(runner.run("scout", "first"))
    await model.started.wait()
    second = asyncio.create_task(runner.run("scout", "second"))
    await model.both_started.wait()
    model.release.set()
    assert (await first).final_output == "done"
    assert (await second).final_output == "done"


@pytest.mark.anyio
async def test_runner_replaces_specs_for_next_call() -> None:
    model = FakeListChatModel(responses=["answer"])
    runner = _runner(model)
    runner.replace_specs(
        [
            SubagentSpec(
                name="reviewer",
                description="review",
                system_prompt="Review.",
            )
        ]
    )

    with pytest.raises(ToolException, match="Unknown subagent role"):
        await runner.run("scout", "inspect")
    assert (await runner.run("reviewer", "inspect")).final_output == "answer"
