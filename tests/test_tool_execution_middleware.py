"""Contracts for Forge's strict, fail-fast tool execution middleware."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.types import Command

from fake_models import ScriptedChatModel
from forge_agent.langchain_runtime import _agent_middleware
from forge_agent.tool_execution import (
    SEQUENTIAL_TOOL_EXECUTION_MODE,
    TOOL_EXECUTION_MODE_METADATA_KEY,
    ToolCallBatchMiddleware,
)


def _state(*call_ids: str) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": call_id, "name": "probe", "args": {}, "type": "tool_call"}
                    for call_id in call_ids
                ],
            )
        ]
    }


def _request(call_id: str, state: object, tool: BaseTool | None = None) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"id": call_id, "name": "probe", "args": {}, "type": "tool_call"},
        tool=tool,
        state=state,
        runtime=None,
    )


def _sequential_probe() -> StructuredTool:
    return StructuredTool.from_function(
        name="probe",
        description="Run a sequential probe.",
        func=lambda: "probe",
        metadata={TOOL_EXECUTION_MODE_METADATA_KEY: SEQUENTIAL_TOOL_EXECUTION_MODE},
    )


def _success(call_id: str) -> ToolMessage:
    return ToolMessage(content=f"done:{call_id}", name="probe", tool_call_id=call_id)


Handler = Callable[[ToolCallRequest], Awaitable[ToolMessage]]


@pytest.mark.anyio
async def test_same_batch_runs_in_model_order_and_preserves_pairing() -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("first", "second")
    seq_tool = _sequential_probe()
    events: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        call_id = str(request.tool_call["id"])
        events.append(f"start:{call_id}")
        await asyncio.sleep(0.005)
        events.append(f"end:{call_id}")
        return _success(call_id)

    results = await asyncio.gather(
        middleware.awrap_tool_call(_request("first", state, seq_tool), handler),
        middleware.awrap_tool_call(_request("second", state, seq_tool), handler),
    )

    assert events == ["start:first", "end:first", "start:second", "end:second"]
    assert [result.tool_call_id for result in results if isinstance(result, ToolMessage)] == [
        "first",
        "second",
    ]
    assert all(isinstance(result, ToolMessage) and result.status == "success" for result in results)


@pytest.mark.anyio
async def test_sequential_tool_error_does_not_skip_later_handlers() -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("first", "second", "third")
    seq_tool = _sequential_probe()
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        call_id = str(request.tool_call["id"])
        called.append(call_id)
        if call_id == "first":
            raise RuntimeError("C:/private/secret.txt and raw arguments")
        return _success(call_id)

    results = await asyncio.gather(
        middleware.awrap_tool_call(_request("first", state, seq_tool), handler),
        middleware.awrap_tool_call(_request("second", state, seq_tool), handler),
        middleware.awrap_tool_call(_request("third", state, seq_tool), handler),
    )

    assert called == ["first", "second", "third"]
    assert [result.status for result in results if isinstance(result, ToolMessage)] == [
        "error",
        "success",
        "success",
    ]
    assert [result.tool_call_id for result in results if isinstance(result, ToolMessage)] == [
        "first",
        "second",
        "third",
    ]
    assert all("secret.txt" not in str(result.content) for result in results)
    assert all(len(str(result.content).encode("utf-8")) <= 256 for result in results)


@pytest.mark.anyio
async def test_sequential_tool_error_message_does_not_stop_batch() -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("first", "second")
    seq_tool = _sequential_probe()
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        call_id = str(request.tool_call["id"])
        called.append(call_id)
        return ToolMessage(
            content="tool reported failure",
            name="probe",
            tool_call_id=call_id,
            status="error",
        )

    results = await asyncio.gather(
        middleware.awrap_tool_call(_request("first", state, seq_tool), handler),
        middleware.awrap_tool_call(_request("second", state, seq_tool), handler),
    )

    assert called == ["first", "second"]
    assert [result.tool_call_id for result in results if isinstance(result, ToolMessage)] == [
        "first",
        "second",
    ]
    assert all(isinstance(result, ToolMessage) and result.status == "error" for result in results)


@pytest.mark.anyio
async def test_command_with_nested_error_message_does_not_stop_batch() -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("first", "second")
    seq_tool = _sequential_probe()
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage | Command[None]:
        call_id = str(request.tool_call["id"])
        called.append(call_id)
        if call_id == "first":
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="tool reported failure",
                            name="probe",
                            tool_call_id=call_id,
                            status="error",
                        )
                    ]
                }
            )
        return _success(call_id)

    results = await asyncio.gather(
        middleware.awrap_tool_call(_request("first", state, seq_tool), handler),
        middleware.awrap_tool_call(_request("second", state, seq_tool), handler),
    )

    assert called == ["first", "second"]
    assert isinstance(results[0], Command)
    assert isinstance(results[1], ToolMessage)
    assert results[1].status == "success"
    assert results[1].tool_call_id == "second"


@pytest.mark.anyio
async def test_command_with_serialized_error_message_does_not_stop_batch() -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("first", "second")
    seq_tool = _sequential_probe()
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage | Command[None]:
        call_id = str(request.tool_call["id"])
        called.append(call_id)
        if call_id == "first":
            return Command(
                update={
                    "messages": [
                        {
                            "type": "tool",
                            "content": "tool reported failure",
                            "tool_call_id": call_id,
                            "status": "error",
                        }
                    ]
                }
            )
        return _success(call_id)

    results = await asyncio.gather(
        middleware.awrap_tool_call(_request("first", state, seq_tool), handler),
        middleware.awrap_tool_call(_request("second", state, seq_tool), handler),
    )

    assert called == ["first", "second"]
    assert isinstance(results[1], ToolMessage)
    assert results[1].status == "success"


@pytest.mark.anyio
async def test_command_ignores_error_for_an_unrelated_tool_call() -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("first", "second")
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage | Command[None]:
        call_id = str(request.tool_call["id"])
        called.append(call_id)
        if call_id == "first":
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="old failure",
                            tool_call_id="older-call",
                            status="error",
                        ),
                        _success(call_id),
                    ]
                }
            )
        return _success(call_id)

    results = await asyncio.gather(
        middleware.awrap_tool_call(_request("first", state), handler),
        middleware.awrap_tool_call(_request("second", state), handler),
    )

    assert called == ["first", "second"]
    assert isinstance(results[1], ToolMessage)
    assert results[1].status == "success"


@pytest.mark.anyio
@pytest.mark.parametrize("requested", ["unknown", "second"])
async def test_invalid_or_out_of_order_call_fails_closed(requested: str) -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("first", "second")
    seq_tool = _sequential_probe()
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        called.append(str(request.tool_call["id"]))
        return _success(str(request.tool_call["id"]))

    first = await middleware.awrap_tool_call(_request(requested, state, seq_tool), handler)

    assert isinstance(first, ToolMessage)
    assert first.status == "error"
    assert called == []


@pytest.mark.anyio
async def test_duplicate_model_call_ids_fail_closed() -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("same", "same")
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        called.append(str(request.tool_call["id"]))
        return _success(str(request.tool_call["id"]))

    result = await middleware.awrap_tool_call(_request("same", state), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert called == []


@pytest.mark.anyio
async def test_cancellation_releases_lock_and_aborts_batch() -> None:
    middleware = ToolCallBatchMiddleware()
    state = _state("first", "second")
    seq_tool = _sequential_probe()
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_handler(request: ToolCallRequest) -> ToolMessage:
        del request
        started.set()
        await release.wait()
        return _success("first")

    task = asyncio.create_task(
        middleware.awrap_tool_call(_request("first", state, seq_tool), blocking_handler)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not middleware._lock.locked()  # noqa: SLF001 - cancellation contract
    called = False

    async def should_not_run(request: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _success(str(request.tool_call["id"]))

    skipped = await middleware.awrap_tool_call(_request("second", state, seq_tool), should_not_run)
    assert isinstance(skipped, ToolMessage)
    assert skipped.status == "error"
    assert not called


def test_runtime_places_sequential_middleware_outermost() -> None:
    middleware = _agent_middleware(None, None)

    assert isinstance(middleware[0], ToolCallBatchMiddleware)


@pytest.mark.anyio
async def test_create_agent_with_middleware_keeps_native_tool_parallelism() -> None:
    events: list[str] = []
    both_started = asyncio.Event()
    release = asyncio.Event()

    @tool
    async def first_tool() -> str:
        """Run the first probe."""
        events.append("first:start")
        if len(events) == 2:
            both_started.set()
        await release.wait()
        events.append("first:end")
        return "first"

    @tool
    async def second_tool() -> str:
        """Run the second probe."""
        events.append("second:start")
        if len(events) == 2:
            both_started.set()
        await release.wait()
        events.append("second:end")
        return "second"

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "first-call",
                        "name": "first_tool",
                        "args": {},
                        "type": "tool_call",
                    },
                    {
                        "id": "second-call",
                        "name": "second_tool",
                        "args": {},
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_agent(
        model,
        tools=[first_tool, second_tool],
        middleware=[ToolCallBatchMiddleware()],
    )

    task = asyncio.create_task(agent.ainvoke({"messages": [{"role": "user", "content": "go"}]}))
    await both_started.wait()
    assert events[:2] == ["first:start", "second:start"]
    release.set()
    result = await task

    assert set(events[2:]) == {"first:end", "second:end"}
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_messages] == ["first-call", "second-call"]


@pytest.mark.anyio
async def test_sequential_tool_metadata_serializes_the_complete_batch() -> None:
    events: list[str] = []

    @tool
    async def first_tool() -> str:
        """Run the first sequential probe."""
        events.append("first:start")
        await asyncio.sleep(0)
        events.append("first:end")
        return "first"

    @tool
    async def second_tool() -> str:
        """Run the second probe."""
        events.append("second:start")
        await asyncio.sleep(0)
        events.append("second:end")
        return "second"

    second_tool.metadata = {TOOL_EXECUTION_MODE_METADATA_KEY: SEQUENTIAL_TOOL_EXECUTION_MODE}

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "first-call", "name": "first_tool", "args": {}, "type": "tool_call"},
                    {
                        "id": "second-call",
                        "name": "second_tool",
                        "args": {},
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_agent(
        model,
        tools=[first_tool, second_tool],
        middleware=[ToolCallBatchMiddleware()],
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})

    assert events == ["first:start", "first:end", "second:start", "second:end"]
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_messages] == ["first-call", "second-call"]
