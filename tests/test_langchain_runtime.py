import inspect
from collections import deque

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from fake_models import ScriptedChatModel, StreamingScriptedChatModel, tool_call_ai
from forge_agent import AgentHarness, AgentHarnessConfig
from forge_agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    ToolExecutionUpdateEvent,
)
from forge_agent.langchain_runtime import (
    _project_v3_message_event,
    _ProjectionState,
    run_langchain_agent,
)
from forge_agent.steering import SteeringMiddleware


class _ToolAgentChatModel(BaseChatModel):
    """Preset-response model that can play tool calls (supports ``bind_tools``)."""

    responses: list[AIMessage] = Field(default_factory=list)

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "_calls", 0)

    @property
    def _llm_type(self) -> str:
        return "fake-forge-tool-agent"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        calls = int(getattr(self, "_calls", 0))
        response = self.responses[min(calls, len(self.responses) - 1)]
        object.__setattr__(self, "_calls", calls + 1)
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
def echo_tool(value: str) -> str:
    """Echo a value."""
    return f"echo:{value}"


@tool
def fail_tool() -> str:
    """Fail deterministically every time it runs."""
    raise ValueError("boom")


def _tool_call_ai(tool_call_id: str, name: str, args: dict[str, object]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": name, "args": args, "type": "tool_call"}],
    )


def message_content(message: object) -> str:
    """Extract plain text from a message for assertions."""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


@pytest.mark.anyio
async def test_runtime_uses_create_agent_and_streams_messages() -> None:
    messages: list = []
    events = [
        event
        async for event in run_langchain_agent(
            provider=FakeListChatModel(responses=["hello"]),
            model="fake",
            system="You are Forge.",
            messages=messages,
            tools=[],
        )
    ]
    assert isinstance(events[0], AgentStartEvent)
    assert any(isinstance(event, MessageEndEvent) for event in events)
    assert isinstance(events[-1], AgentEndEvent)
    # The native runtime keeps the real LangChain message in the transcript.
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert message_content(messages[0]) == "hello"


@pytest.mark.anyio
async def test_harness_uses_langchain_runtime_with_fake_model() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=FakeListChatModel(responses=["hello"]),
            model="fake",
            system="You are Forge.",
        )
    )

    events = [event async for event in harness.prompt("hi")]

    assert any(event.type == "message_delta" for event in events)
    # The native runtime keeps the real LangChain AIMessage in the transcript.
    assert isinstance(harness.messages[-1], AIMessage)


@pytest.mark.anyio
async def test_langchain_runtime_owns_tool_call_loop() -> None:
    model = _ToolAgentChatModel(
        responses=[
            _tool_call_ai("call-1", "echo", {"value": "ok"}),
            AIMessage(content="done"),
        ]
    )
    messages: list = []
    events = [
        event
        async for event in run_langchain_agent(
            provider=model,
            model="fake",
            system="You are Forge.",
            messages=messages,
            tools=[echo_tool],
        )
    ]

    assert [event.type for event in events].count("tool_execution_start") == 1
    assert [event.type for event in events].count("tool_execution_end") == 1
    assert any(isinstance(message, ToolMessage) for message in messages)
    assert any(isinstance(message, AIMessage) and message.content == "done" for message in messages)


@pytest.mark.anyio
async def test_langchain_runtime_preserves_failed_tool_result() -> None:
    model = _ToolAgentChatModel(
        responses=[
            _tool_call_ai("call-1", "fail", {}),
            AIMessage(content="done"),
        ]
    )
    messages: list = []
    events = [
        event
        async for event in run_langchain_agent(
            provider=model,
            model="fake",
            system="You are Forge.",
            messages=messages,
            tools=[fail_tool],
        )
    ]

    tool_result = next(message for message in messages if isinstance(message, ToolMessage))
    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert tool_result.status == "error"
    assert tool_result.content
    assert tool_end.result.ok is False


# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_non_chunk_model_emits_final_text_as_delta() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=ScriptedChatModel([AIMessage(content="plain answer")]),
            model="fake",
            system="You are Forge.",
        )
    )

    events = [event async for event in harness.prompt("Hi")]

    deltas = [event for event in events if isinstance(event, MessageDeltaEvent)]
    assert [event.delta for event in deltas] == ["plain answer"]


@pytest.mark.anyio
async def test_streaming_model_does_not_duplicate_text() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=StreamingScriptedChatModel([AIMessage(content="streamed answer")]),
            model="fake",
            system="You are Forge.",
        )
    )

    events = [event async for event in harness.prompt("Hi")]

    deltas = [event for event in events if isinstance(event, MessageDeltaEvent)]
    assert [event.delta for event in deltas] == ["streamed answer"]


@pytest.mark.anyio
async def test_tool_flow_produces_two_closed_model_call_lifecycles() -> None:
    model = _ToolAgentChatModel(
        [
            _tool_call_ai("call-1", "echo", {"value": "ok"}),
            AIMessage(content="done"),
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=model,
            model="fake",
            system="You are Forge.",
            tools=[echo_tool],
        )
    )

    events = [event async for event in harness.prompt("Go")]
    types = [event.type for event in events]

    assert types.count("turn_start") == 2
    assert types.count("turn_end") == 2
    # One user pair plus one assistant pair per model call.
    assert types.count("message_start") == 3
    assert types.count("message_end") == 3
    # No assistant MessageStart may stay open across another assistant start.
    for index, event in enumerate(events):
        if event.type == "message_start" and event.message_role == "assistant":
            assert "message_end" in types[index + 1 :]
    # Tool execution sits between the two model-call lifecycles.
    first_turn_end = types.index("turn_end")
    tool_start = types.index("tool_execution_start")
    tool_end = types.index("tool_execution_end")
    second_turn_start = types.index("turn_start", first_turn_end + 1)
    assert first_turn_end < tool_start < tool_end < second_turn_start


@pytest.mark.anyio
async def test_tool_flow_with_streaming_model_keeps_lifecycles_closed() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=StreamingScriptedChatModel(
                [
                    tool_call_ai("call-1", "echo", {"value": "ok"}),
                    AIMessage(content="final answer"),
                ]
            ),
            model="fake",
            system="You are Forge.",
            tools=[echo_tool],
        )
    )

    events = [event async for event in harness.prompt("Go")]

    assert [event.type for event in events].count("turn_start") == 2
    assert [event.type for event in events].count("turn_end") == 2
    final_deltas = [event.delta for event in events if isinstance(event, MessageDeltaEvent)]
    assert "final answer" in final_deltas


@pytest.mark.anyio
async def test_non_chunk_model_emits_reasoning_and_text_deltas() -> None:
    from forge_agent.events import ThinkingDeltaEvent

    harness = AgentHarness(
        AgentHarnessConfig(
            provider=ScriptedChatModel(
                [
                    AIMessage(
                        content=[
                            {"type": "reasoning", "reasoning": "thinking out loud"},
                            {"type": "text", "text": "the answer"},
                        ]
                    )
                ]
            ),
            model="fake",
            system="You are Forge.",
        )
    )

    events = [event async for event in harness.prompt("Hi")]

    thinking = [event.delta for event in events if isinstance(event, ThinkingDeltaEvent)]
    text = [event.delta for event in events if isinstance(event, MessageDeltaEvent)]
    assert thinking == ["thinking out loud"]
    assert text == ["the answer"]


@pytest.mark.anyio
async def test_streaming_model_reasoning_deltas_are_projected() -> None:
    from forge_agent.events import ThinkingDeltaEvent

    harness = AgentHarness(
        AgentHarnessConfig(
            provider=StreamingScriptedChatModel(
                [
                    AIMessage(
                        content=[
                            {"type": "reasoning", "reasoning": "streamed thinking"},
                            {"type": "text", "text": "streamed answer"},
                        ]
                    )
                ]
            ),
            model="fake",
            system="You are Forge.",
        )
    )

    events = [event async for event in harness.prompt("Hi")]

    thinking = [event.delta for event in events if isinstance(event, ThinkingDeltaEvent)]
    text = [event.delta for event in events if isinstance(event, MessageDeltaEvent)]
    assert thinking == ["streamed thinking"]
    assert text == ["streamed answer"]


# --------------------------------------------------------------------------- #
# Steering middleware contract: the drained HumanMessage must reach the next
# model call through the official before_model hook, and the reducer must
# assign it a stable UUID that shows up in the v3 values projection.
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_steering_middleware_message_reaches_values_projection_with_uuid() -> None:
    queue: deque = deque([HumanMessage(content="steer")])
    middleware = SteeringMiddleware(queue)
    graph = create_agent(
        ScriptedChatModel([AIMessage(content="ok")]),
        middleware=[middleware],
    )

    seen: list[HumanMessage] = []
    stream = graph.astream_events({"messages": [HumanMessage(content="hi")]}, version="v3")
    if inspect.isawaitable(stream):  # type: ignore[arg-type]
        stream = await stream  # type: ignore[assignment]
    async for event in stream:  # type: ignore[attr-defined]
        method = event.get("method")
        payload = (event.get("params") or {}).get("data")
        if method == "values":
            for message in payload.get("messages") or []:
                if (
                    isinstance(message, HumanMessage)
                    and message.content == "steer"
                    and all(message is not seen_message for seen_message in seen)
                ):
                    seen.append(message)

    assert len(seen) == 1
    assert seen[0].id  # reducer assigned a missing-id UUID
    assert queue == deque()


def test_steering_middleware_drains_one_at_a_time() -> None:
    queue: deque = deque([HumanMessage(content="a"), HumanMessage(content="b")])
    middleware = SteeringMiddleware(queue)

    assert middleware.before_model({}, object()) == {"messages": [HumanMessage(content="a")]}
    assert list(queue) == [HumanMessage(content="b")]


def test_steering_middleware_drains_all_in_all_mode() -> None:
    queue: deque = deque([HumanMessage(content="a"), HumanMessage(content="b")])
    middleware = SteeringMiddleware(queue, queue_mode="all")

    assert middleware.before_model({}, object()) == {
        "messages": [HumanMessage(content="a"), HumanMessage(content="b")]
    }
    assert list(queue) == []


# --------------------------------------------------------------------------- #
# Tool argument streaming: partial chunks accumulate into update events; odd
# shapes (missing id, non-string args, malformed JSON) never crash.
# --------------------------------------------------------------------------- #
def _project_chunk(chunk: dict[str, object]) -> list[ToolExecutionUpdateEvent]:
    state = _ProjectionState([])
    partial_arguments: dict[str, str] = {}
    partial_tool_names: dict[str, str] = {}
    projected = _project_v3_message_event(
        ({"event": "content-block-delta", "delta": chunk}, None),
        state=state,
        partial_arguments=partial_arguments,
        partial_tool_names=partial_tool_names,
        queue_update=None,
    )
    return [event for event in projected if isinstance(event, ToolExecutionUpdateEvent)]


def test_tool_argument_chunks_accumulate_into_update_events() -> None:
    chunk_a = {
        "type": "block-delta",
        "fields": {
            "type": "tool_call_chunk",
            "id": "call-1",
            "name": "echo",
            "args": '{"value": "o',
        },
    }
    chunk_b = {
        "type": "block-delta",
        "fields": {"type": "tool_call_chunk", "id": "call-1", "name": None, "args": 'k"}'},
    }

    state = _ProjectionState([])
    partial_arguments: dict[str, str] = {}
    partial_tool_names: dict[str, str] = {}

    def project(chunk: dict[str, object]) -> list[ToolExecutionUpdateEvent]:
        projected = _project_v3_message_event(
            ({"event": "content-block-delta", "delta": chunk}, None),
            state=state,
            partial_arguments=partial_arguments,
            partial_tool_names=partial_tool_names,
            queue_update=None,
        )
        return [event for event in projected if isinstance(event, ToolExecutionUpdateEvent)]

    events_a = project(chunk_a)
    events_b = project(chunk_b)

    assert len(events_a) == 1
    assert isinstance(events_a[0], ToolExecutionUpdateEvent)
    assert events_a[0].tool_call_id == "call-1"
    assert events_a[0].data == {
        "arguments_delta": '{"value": "o',
        "tool_name": "echo",
    }
    assert len(events_b) == 1
    assert isinstance(events_b[0], ToolExecutionUpdateEvent)
    assert events_b[0].data == {
        "arguments_delta": '{"value": "ok"}',
        "tool_name": "echo",  # name sticks from the first chunk
    }


def test_tool_argument_chunk_without_id_is_ignored() -> None:
    chunk = {
        "type": "block-delta",
        "fields": {"type": "tool_call_chunk", "args": '{"value": "x"}'},
    }
    assert _project_chunk(chunk) == []


def test_tool_argument_chunk_with_non_string_args_is_ignored() -> None:
    chunk = {
        "type": "block-delta",
        "fields": {"type": "tool_call_chunk", "id": "call-1", "args": {"value": "x"}},
    }
    assert _project_chunk(chunk) == []


def test_malformed_partial_json_chunk_does_not_crash() -> None:
    chunk = {
        "type": "legacy-block-delta",
        "fields": {"type": "tool_call_chunk", "id": "call-1", "name": "echo", "args": '{"value": '},
    }
    events = _project_chunk(chunk)
    assert len(events) == 1
    assert isinstance(events[0], ToolExecutionUpdateEvent)
    assert events[0].data["arguments_delta"] == '{"value": '  # type: ignore[index]


def test_unrelated_block_delta_is_ignored() -> None:
    chunk = {"type": "block-delta", "fields": {"type": "image_url", "url": "x"}}
    assert _project_chunk(chunk) == []


# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_max_turns_produces_exactly_one_assistant_reply_and_error() -> None:
    model = _ToolAgentChatModel(responses=[_tool_call_ai("call-1", "echo", {"value": "x"})])
    transcript: list[object] = []

    events = [
        event
        async for event in run_langchain_agent(
            provider=model,
            model="fake",
            system="You are Forge.",
            messages=transcript,  # type: ignore[arg-type]
            tools=[echo_tool],
            max_turns=1,
        )
    ]

    assistant_replies = [m for m in transcript if isinstance(m, AIMessage) and m.tool_calls]
    errors = [event for event in events if isinstance(event, ErrorEvent)]

    assert model._calls == 1
    assert len(assistant_replies) == 1
    assert errors == [
        ErrorEvent(message="Agent loop stopped after reaching max_turns=1", recoverable=True)
    ]
    assert not any("RECURSION_LIMIT" in str(e.message) for e in errors)
