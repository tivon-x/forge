"""Unit coverage for the generic Forge subagent runner."""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException, tool
from pydantic import PrivateAttr

from forge_agent import AgentToolResult, SubagentRunner, SubagentRuntime, SubagentSpec
from forge_agent.context import ForgeRuntimeContext
from forge_coding.tools import ToolDefinition


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
    assert artifact["version"] == 1
    assert artifact["instruction"] == "inspect this"
    assert artifact["error"] is None


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
    """A cancellable model used to exercise the runner lock."""

    _started: asyncio.Event = PrivateAttr()
    _release: asyncio.Event = PrivateAttr()

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_started", asyncio.Event())
        object.__setattr__(self, "_release", asyncio.Event())

    @property
    def started(self) -> asyncio.Event:
        return self._started

    @property
    def release(self) -> asyncio.Event:
        return self._release

    @property
    def _llm_type(self) -> str:
        return "forge-blocking-subagent"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        self.started.set()
        await self.release.wait()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        raise AssertionError("blocking test model must use async generation")


@pytest.mark.anyio
async def test_running_child_cancellation_is_rethrown_and_releases_lock() -> None:
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
async def test_waiting_for_lock_cancellation_does_not_leak_lock() -> None:
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

    capture_tool = ToolDefinition(
        name="capture",
        description="Capture runtime context.",
        prompt_snippet="Capture runtime context.",
        prompt_guidelines=(),
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        executor=capture,
    ).to_langchain_tool()
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
async def test_runner_rejects_invalid_input_with_tool_exception() -> None:
    runner = _runner(FakeListChatModel(responses=["answer"]))

    with pytest.raises(ToolException, match="Unknown subagent role"):
        await runner.run("unknown", "inspect")
    with pytest.raises(ToolException, match="must not be empty"):
        await runner.run("scout", " \t\n")


@pytest.mark.anyio
async def test_runner_serializes_calls_with_session_lock() -> None:
    entered: list[str] = []
    release = asyncio.Event()

    @tool
    async def wait_tool() -> str:
        """Wait until the test releases the child tool."""
        entered.append("tool")
        await release.wait()
        return "ok"

    # The model emits a tool call first, then a final answer.  A second call
    # uses the same scripted model and must wait for the first lock holder.
    from fake_models import ScriptedChatModel, tool_call_ai

    model = ScriptedChatModel(
        [
            tool_call_ai("call-1", "wait_tool", {}),
            AIMessage(content="done"),
            AIMessage(content="second"),
        ]
    )
    runner = _runner(model, tools=[wait_tool])
    first = asyncio.create_task(runner.run("scout", "first"))
    for _ in range(100):
        if entered:
            break
        await asyncio.sleep(0.001)
    second = asyncio.create_task(runner.run("scout", "second"))
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    assert (await first).final_output == "done"
    assert (await second).final_output == "second"


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
