from collections.abc import Mapping

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from forge_agent import (
    AgentToolResult,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    QueueUpdateEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from forge_agent.types import JSONValue
from forge_coding.tools import ToolDefinition


def test_human_message_has_native_role() -> None:
    message = HumanMessage(content="hello")

    assert message.type == "human"
    assert message.content == "hello"


def test_assistant_message_can_include_tool_calls() -> None:
    message = AIMessage(
        content="I'll read that.",
        tool_calls=[
            {"id": "call-1", "name": "read", "args": {"path": "README.md"}, "type": "tool_call"}
        ],
    )

    assert message.type == "ai"
    assert message.tool_calls[0]["name"] == "read"
    assert message.tool_calls[0]["args"] == {"path": "README.md"}


def test_tool_message_records_tool_output() -> None:
    message = ToolMessage(
        content="file contents",
        tool_call_id="call-1",
        name="read",
        artifact={"data": {"path": "README.md"}, "details": {"bytes": 13}},
    )

    assert message.type == "tool"
    assert message.tool_call_id == "call-1"
    assert message.artifact["data"] == {"path": "README.md"}
    assert message.artifact["details"] == {"bytes": 13}


def test_tool_call_model_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ToolCall(id="call-1", name="read", unexpected=True)  # type: ignore[call-arg]


@pytest.mark.anyio
async def test_agent_tool_executes_with_json_arguments() -> None:
    class FakeCancellationToken:
        def is_cancelled(self) -> bool:
            return False

    observed_signal: list[object | None] = []

    async def executor(
        arguments: Mapping[str, JSONValue],
        *,
        signal: object | None = None,
    ) -> AgentToolResult:
        observed_signal.append(signal)
        return AgentToolResult(
            tool_call_id="call-1",
            name="echo",
            ok=True,
            content=str(arguments["text"]),
        )

    tool = ToolDefinition(
        name="echo",
        description="Echo text.",
        prompt_snippet="Echo text.",
        prompt_guidelines=(),
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        executor=executor,
    ).to_langchain_tool()

    signal = FakeCancellationToken()
    result = await tool.execute({"text": "hi"}, signal=signal)

    assert result.ok is True
    assert result.content == "hi"
    assert observed_signal == [signal]


def test_events_have_stable_type_names() -> None:
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    result = AgentToolResult(tool_call_id="call-1", name="read", ok=True, content="contents")
    message = AIMessage(content="Done")

    events = [
        MessageDeltaEvent(delta="hello"),
        QueueUpdateEvent(steering=("adjust",), follow_up=()),
        ThinkingDeltaEvent(delta="reasoning"),
        MessageEndEvent(message=message),
        ToolExecutionStartEvent(tool_call=tool_call),
        ToolExecutionEndEvent(result=result),
        ErrorEvent(message="boom", recoverable=True),
    ]

    assert [event.type for event in events] == [
        "message_delta",
        "queue_update",
        "thinking_delta",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "error",
    ]
