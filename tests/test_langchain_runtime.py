import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from forge_agent import AgentHarness, AgentHarnessConfig
from forge_agent.events import AgentEndEvent, AgentStartEvent, MessageEndEvent
from forge_agent.langchain_runtime import run_langchain_agent


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
