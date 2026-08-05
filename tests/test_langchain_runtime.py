import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from forge_agent import AgentHarness, AgentHarnessConfig, AssistantMessage, ToolResultMessage
from forge_agent.events import AgentEndEvent, AgentStartEvent, MessageEndEvent
from forge_agent.langchain_runtime import run_langchain_agent
from forge_agent.tools import AgentTool, AgentToolResult, ToolCall
from forge_ai import FakeProvider, ProviderResponseEndEvent, ProviderResponseStartEvent


@pytest.mark.anyio
async def test_runtime_uses_create_agent_and_streams_messages() -> None:
    messages = []
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
    assert messages == [AssistantMessage(content="hello")]


@pytest.mark.anyio
async def test_harness_uses_langchain_runtime_with_fake_chat_model() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=FakeListChatModel(responses=["hello"]),
            model="fake",
            system="You are Forge.",
        )
    )

    events = [event async for event in harness.prompt("hi")]

    assert any(event.type == "message_delta" for event in events)
    assert harness.messages[-1] == AssistantMessage(content="hello")


@pytest.mark.anyio
async def test_langchain_runtime_owns_tool_call_loop() -> None:
    async def execute(arguments: dict[str, object], signal: object = None) -> AgentToolResult:
        del signal
        return AgentToolResult(
            tool_call_id="call-1",
            name="echo",
            ok=True,
            content=str(arguments["value"]),
        )

    tool = AgentTool(
        name="echo",
        description="Echo a value.",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        executor=execute,
    )
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        tool_calls=[ToolCall(id="call-1", name="echo", arguments={"value": "ok"})]
                    )
                ),
            ],
            [
                ProviderResponseStartEvent(model="fake"),
                ProviderResponseEndEvent(message=AssistantMessage(content="done")),
            ],
        ]
    )
    messages = []
    events = [
        event
        async for event in run_langchain_agent(
            provider=provider,
            model="fake",
            system="You are Forge.",
            messages=messages,
            tools=[tool],
        )
    ]

    assert [event.type for event in events].count("tool_execution_start") == 1
    assert [event.type for event in events].count("tool_execution_end") == 1
    assert any(
        isinstance(message, AssistantMessage) and message.content == "done" for message in messages
    )


@pytest.mark.anyio
async def test_langchain_runtime_preserves_failed_tool_result_and_signal() -> None:
    class Token:
        def is_cancelled(self) -> bool:
            return False

    token = Token()
    seen_signal: object | None = None

    async def execute(arguments: dict[str, object], signal: object = None) -> AgentToolResult:
        nonlocal seen_signal
        del arguments
        seen_signal = signal
        return AgentToolResult(
            tool_call_id="call-1",
            name="fail",
            ok=False,
            content="failed",
            data={"exit_code": 1},
            details={"diagnostic": "boom"},
            error="boom",
        )

    tool = AgentTool(
        name="fail",
        description="Fail deterministically.",
        input_schema={"type": "object", "properties": {}},
        executor=execute,
    )
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="fake"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        tool_calls=[ToolCall(id="call-1", name="fail", arguments={})]
                    )
                ),
            ],
            [
                ProviderResponseStartEvent(model="fake"),
                ProviderResponseEndEvent(message=AssistantMessage(content="done")),
            ],
        ]
    )
    messages = []
    events = [
        event
        async for event in run_langchain_agent(
            provider=provider,
            model="fake",
            system="You are Forge.",
            messages=messages,
            tools=[tool],
            signal=token,
        )
    ]

    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert seen_signal is token
    assert result.ok is False
    assert result.error == "boom"
    assert result.data == {"exit_code": 1}
    assert result.details == {"diagnostic": "boom"}
    assert tool_end.result.ok is False
