"""Contracts for Forge's strict, fail-fast tool execution middleware."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from fake_models import ScriptedChatModel
from forge_agent.langchain_runtime import _agent_middleware
from forge_agent.tool_execution import SequentialToolCallMiddleware


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


def _request(call_id: str, state: object) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"id": call_id, "name": "probe", "args": {}, "type": "tool_call"},
        tool=None,
        state=state,
        runtime=None,
    )


def _success(call_id: str) -> ToolMessage:
    return ToolMessage(content=f"done:{call_id}", name="probe", tool_call_id=call_id)


Handler = Callable[[ToolCallRequest], Awaitable[ToolMessage]]


@pytest.mark.anyio
async def test_same_batch_runs_in_model_order_and_preserves_pairing() -> None:
    middleware = SequentialToolCallMiddleware()
    state = _state("first", "second")
    events: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        call_id = str(request.tool_call["id"])
        events.append(f"start:{call_id}")
        await asyncio.sleep(0.005)
        events.append(f"end:{call_id}")
        return _success(call_id)

    results = await asyncio.gather(
        middleware.awrap_tool_call(_request("first", state), handler),
        middleware.awrap_tool_call(_request("second", state), handler),
    )

    assert events == ["start:first", "end:first", "start:second", "end:second"]
    assert [result.tool_call_id for result in results if isinstance(result, ToolMessage)] == [
        "first",
        "second",
    ]
    assert all(isinstance(result, ToolMessage) and result.status == "success" for result in results)


@pytest.mark.anyio
async def test_first_error_stops_later_handlers_and_bounds_exception() -> None:
    middleware = SequentialToolCallMiddleware()
    state = _state("first", "second", "third")
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        call_id = str(request.tool_call["id"])
        called.append(call_id)
        if call_id == "first":
            raise RuntimeError("C:/private/secret.txt and raw arguments")
        return _success(call_id)

    results = await asyncio.gather(
        middleware.awrap_tool_call(_request("first", state), handler),
        middleware.awrap_tool_call(_request("second", state), handler),
        middleware.awrap_tool_call(_request("third", state), handler),
    )

    assert called == ["first"]
    assert all(isinstance(result, ToolMessage) and result.status == "error" for result in results)
    assert [result.tool_call_id for result in results if isinstance(result, ToolMessage)] == [
        "first",
        "second",
        "third",
    ]
    assert all("secret.txt" not in str(result.content) for result in results)
    assert all(len(str(result.content).encode("utf-8")) <= 256 for result in results)


@pytest.mark.anyio
async def test_handler_error_message_marks_batch_failed() -> None:
    middleware = SequentialToolCallMiddleware()
    state = _state("first", "second")
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
        middleware.awrap_tool_call(_request("first", state), handler),
        middleware.awrap_tool_call(_request("second", state), handler),
    )

    assert called == ["first"]
    assert [result.tool_call_id for result in results if isinstance(result, ToolMessage)] == [
        "first",
        "second",
    ]
    assert all(isinstance(result, ToolMessage) and result.status == "error" for result in results)


@pytest.mark.anyio
async def test_command_with_nested_error_message_stops_later_handlers() -> None:
    middleware = SequentialToolCallMiddleware()
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
        middleware.awrap_tool_call(_request("first", state), handler),
        middleware.awrap_tool_call(_request("second", state), handler),
    )

    assert called == ["first"]
    assert isinstance(results[0], Command)
    assert isinstance(results[1], ToolMessage)
    assert results[1].status == "error"
    assert results[1].tool_call_id == "second"


@pytest.mark.anyio
async def test_command_with_serialized_error_message_stops_later_handlers() -> None:
    middleware = SequentialToolCallMiddleware()
    state = _state("first", "second")
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
        middleware.awrap_tool_call(_request("first", state), handler),
        middleware.awrap_tool_call(_request("second", state), handler),
    )

    assert called == ["first"]
    assert isinstance(results[1], ToolMessage)
    assert results[1].status == "error"


@pytest.mark.anyio
async def test_command_ignores_error_for_an_unrelated_tool_call() -> None:
    middleware = SequentialToolCallMiddleware()
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
    middleware = SequentialToolCallMiddleware()
    state = _state("first", "second")
    called: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        called.append(str(request.tool_call["id"]))
        return _success(str(request.tool_call["id"]))

    first = await middleware.awrap_tool_call(_request(requested, state), handler)

    assert isinstance(first, ToolMessage)
    assert first.status == "error"
    assert called == []


@pytest.mark.anyio
async def test_duplicate_model_call_ids_fail_closed() -> None:
    middleware = SequentialToolCallMiddleware()
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
    middleware = SequentialToolCallMiddleware()
    state = _state("first", "second")
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_handler(request: ToolCallRequest) -> ToolMessage:
        del request
        started.set()
        await release.wait()
        return _success("first")

    task = asyncio.create_task(
        middleware.awrap_tool_call(_request("first", state), blocking_handler)
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

    skipped = await middleware.awrap_tool_call(_request("second", state), should_not_run)
    assert isinstance(skipped, ToolMessage)
    assert skipped.status == "error"
    assert not called


def test_runtime_places_sequential_middleware_outermost() -> None:
    middleware = _agent_middleware(None, None)

    assert isinstance(middleware[0], SequentialToolCallMiddleware)


@pytest.mark.anyio
async def test_create_agent_with_middleware_executes_async_tools_sequentially() -> None:
    events: list[str] = []

    @tool
    async def first_tool() -> str:
        """Run the first probe."""
        events.append("first:start")
        await asyncio.sleep(0.005)
        events.append("first:end")
        return "first"

    @tool
    async def second_tool() -> str:
        """Run the second probe."""
        events.append("second:start")
        await asyncio.sleep(0.005)
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
        middleware=[SequentialToolCallMiddleware()],
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})

    assert events == ["first:start", "first:end", "second:start", "second:end"]
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_messages] == ["first-call", "second-call"]
