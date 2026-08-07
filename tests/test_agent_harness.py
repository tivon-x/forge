import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from fake_native import ScriptedChatModel, tool_call_ai
from forge_agent import AgentHarness, AgentHarnessConfig, MessageEndEvent, MessageStartEvent


def _scripted(*responses: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(list(responses))


def _contents(messages) -> list[str]:
    return [message.content for message in messages]


@pytest.mark.anyio
async def test_prompt_appends_user_message_and_assistant_response() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=_scripted(AIMessage(content="Hello")), model="fake", system="You are Forge."
        )
    )

    events = [event async for event in harness.prompt("Hi")]

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert events[2].message_role == "user"  # type: ignore[attr-defined]
    assert events[3].message.content == "Hi"  # type: ignore[attr-defined]
    assert _contents(harness.messages) == ["Hi", "Hello"]
    assert isinstance(harness.messages[1], AIMessage)


@pytest.mark.anyio
async def test_continue_runs_without_adding_user_message() -> None:
    existing = HumanMessage(content="Previous prompt")
    model = _scripted(AIMessage(content="Continuing"))
    harness = AgentHarness(
        AgentHarnessConfig(provider=model, model="fake", system="You are Forge."),
        messages=[existing],
    )

    _events = [event async for event in harness.continue_()]

    assert _contents(harness.messages) == ["Previous prompt", "Continuing"]
    assert any(m is existing for m in model.calls[0]["messages"])


def test_messages_property_returns_immutable_snapshot() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(provider=_scripted(), model="fake", system="You are Forge."),
        messages=[HumanMessage(content="Hello")],
    )

    snapshot = harness.messages
    harness.append_message(AIMessage(content="Hi"))

    assert _contents(snapshot) == ["Hello"]
    assert _contents(harness.messages) == ["Hello", "Hi"]


def test_harness_can_replace_messages() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(provider=_scripted(), model="fake", system="You are Forge."),
        messages=[HumanMessage(content="Old")],
    )

    harness.replace_messages([HumanMessage(content="Summary")])

    assert _contents(harness.messages) == ["Summary"]


def test_harness_can_clear_queued_messages() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(provider=_scripted(), model="fake", system="You are Forge.")
    )

    harness.steer("Adjust")
    harness.follow_up("Later")
    cleared = harness.clear_queues()

    assert _contents(cleared.steering) == ["Adjust"]
    assert _contents(cleared.follow_up) == ["Later"]
    assert harness.pending_message_count == 0
    assert harness.queue_update_event().steering == ()
    assert harness.queue_update_event().follow_up == ()


def test_harness_can_pop_latest_follow_up_message() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(provider=_scripted(), model="fake", system="You are Forge.")
    )

    harness.follow_up("First")
    harness.follow_up("Second")
    popped = harness.pop_latest_follow_up()

    assert popped == HumanMessage(content="Second")
    assert harness.queue_update_event().follow_up == ("First",)
    assert harness.pop_latest_follow_up() == HumanMessage(content="First")
    assert harness.pop_latest_follow_up() is None


def test_harness_can_pop_latest_steering_message() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(provider=_scripted(), model="fake", system="You are Forge.")
    )

    harness.steer("First")
    harness.steer("Second")
    popped = harness.pop_latest_steering()

    assert popped == HumanMessage(content="Second")
    assert harness.queue_update_event().steering == ("First",)
    assert harness.pop_latest_steering() == HumanMessage(content="First")
    assert harness.pop_latest_steering() is None


@pytest.mark.anyio
async def test_subscribed_listeners_receive_events_and_can_unsubscribe() -> None:
    model = _scripted(AIMessage(content="Hello"), AIMessage(content="Hello"))
    harness = AgentHarness(
        AgentHarnessConfig(provider=model, model="fake", system="You are Forge.")
    )
    seen: list[str] = []

    async def listener(event: object) -> None:
        seen.append(event.type)  # type: ignore[attr-defined]

    unsubscribe = harness.subscribe(listener)

    _events = [event async for event in harness.prompt("Hi")]
    unsubscribe()
    _more_events = [event async for event in harness.continue_()]

    assert seen == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]


@pytest.mark.anyio
async def test_cancel_requests_cancellation_for_current_run() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=_scripted(AIMessage(content="hello")), model="fake", system="You are Forge."
        )
    )

    events: list = []
    try:
        async for event in harness.prompt("Hi"):
            events.append(event)
            if isinstance(event, MessageEndEvent) and event.message.content == "hello":
                harness.cancel()
    except asyncio.CancelledError:
        pass

    assert harness.is_running is False
    assert harness.was_last_run_interrupted is True
    assert events[0].type == "agent_start"
    assert any(event.type == "message_end" for event in events)


@tool
def read_tool(path: str) -> str:
    """Read a file deterministically for harness tests."""
    return f"read:{path}"


@pytest.mark.anyio
async def test_cancelled_tool_run_repairs_transcript_before_next_prompt() -> None:
    model = _scripted(
        tool_call_ai("call-1", "read_tool", {"path": "README.md"}),
        AIMessage(content="Recovered."),
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=model,
            model="fake",
            system="You are Forge.",
            tools=[read_tool],
        )
    )

    stream = harness.prompt("Read README.md")
    try:
        async for event in stream:
            if event.type == "tool_execution_start":
                harness.cancel()
                await stream.aclose()
                break
    except asyncio.CancelledError:
        pass

    assert _contents(harness.messages) == [
        "Read README.md",
        "",
        "Tool call interrupted by user",
    ]
    assert any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == "call-1"
        and message.content == "Tool call interrupted by user"
        for message in harness.messages
    )


@pytest.mark.anyio
async def test_harness_rejects_overlapping_prompt_runs() -> None:
    model = _scripted(AIMessage(content="Hello"), AIMessage(content="Queued answer"))
    harness = AgentHarness(
        AgentHarnessConfig(provider=model, model="fake", system="You are Forge.")
    )

    queued = False
    async for event in harness.prompt("Hi"):
        if (
            isinstance(event, MessageStartEvent)
            and event.message_role == "assistant"
            and not queued
        ):
            with pytest.raises(RuntimeError, match="already running"):
                harness.prompt("Overlapping")
            queue_event = harness.steer("Queued instead")
            assert queue_event.steering == ("Queued instead",)
            queued = True

    assert harness.is_running is False
    assert harness.pending_message_count == 0
    assert _contents(harness.messages) == [
        "Hi",
        "Hello",
        "Queued instead",
        "Queued answer",
    ]


@pytest.mark.anyio
async def test_harness_drains_follow_up_messages_one_at_a_time_by_default() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=_scripted(
                AIMessage(content="First"), AIMessage(content="Second"), AIMessage(content="Third")
            ),
            model="fake",
            system="You are Forge.",
        )
    )

    async for event in harness.prompt("Hi"):
        if isinstance(event, MessageEndEvent) and event.message.content == "First":
            harness.follow_up("Second prompt")
            harness.follow_up("Third prompt")

    assert _contents(harness.messages) == [
        "Hi",
        "First",
        "Second prompt",
        "Second",
        "Third prompt",
        "Third",
    ]
    assert harness.pending_message_count == 0


@pytest.mark.anyio
async def test_harness_can_drain_all_queued_messages_together() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=_scripted(AIMessage(content="First"), AIMessage(content="Second")),
            model="fake",
            system="You are Forge.",
            queue_mode="all",
        )
    )

    async for event in harness.prompt("Hi"):
        if isinstance(event, MessageEndEvent) and event.message.content == "First":
            harness.follow_up("Second prompt")
            harness.follow_up("Third prompt")

    assert _contents(harness.messages) == [
        "Hi",
        "First",
        "Second prompt",
        "Third prompt",
        "Second",
    ]


@tool
def echo_tool(value: str) -> str:
    """Echo text back."""
    return value


@pytest.mark.anyio
async def test_harness_passes_tools_to_loop() -> None:
    model = _scripted(AIMessage(content=""))
    harness = AgentHarness(
        AgentHarnessConfig(provider=model, model="fake", system="You are Forge.", tools=[echo_tool])
    )

    _events = [event async for event in harness.prompt("Hi")]

    assert [getattr(tool, "name", None) for tool in model.calls[0]["tools"]] == ["echo_tool"]
