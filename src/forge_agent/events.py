"""Events emitted by Forge's portable agent layer."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge_agent.tools import AgentToolResult, ToolCall
from forge_agent.types import JSONValue

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    """Validated projection of one LangChain todo item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    status: TodoStatus


HumanDecisionType = Literal["respond", "approve", "edit", "reject"]


class HumanInputRequest(BaseModel):
    """One JSON-safe action request surfaced by HITL middleware."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interrupt_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)
    allowed_decisions: tuple[HumanDecisionType, ...] = ("respond",)
    description: str | None = None


class TodoUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["todo_update"] = "todo_update"
    todos: tuple[TodoItem, ...] = ()


class HumanInputRequestedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["human_input_requested"] = "human_input_requested"
    interrupt_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)
    allowed_decisions: tuple[HumanDecisionType, ...] = ("respond",)
    description: str | None = None
    requests: tuple[HumanInputRequest, ...] = ()

    @model_validator(mode="after")
    def _first_request_matches_flattened_fields(self) -> HumanInputRequestedEvent:
        if not self.requests:
            return self
        first = self.requests[0]
        flattened = (
            self.interrupt_id,
            self.tool_call_id,
            self.tool_name,
            self.arguments,
            self.allowed_decisions,
            self.description,
        )
        canonical = (
            first.interrupt_id,
            first.tool_call_id,
            first.tool_name,
            first.arguments,
            first.allowed_decisions,
            first.description,
        )
        if flattened != canonical:
            raise ValueError("flattened human-input fields must match requests[0]")
        return self


class AgentStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_end"] = "agent_end"


class TurnStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["turn_start"] = "turn_start"
    turn: int


class TurnEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["turn_end"] = "turn_end"
    turn: int


class RetryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str
    data: dict[str, JSONValue] | None = None


class QueueUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["queue_update"] = "queue_update"
    steering: tuple[str, ...] = ()
    follow_up: tuple[str, ...] = ()


class MessageStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message_start"] = "message_start"
    message_role: Literal["user", "assistant", "tool"] = "assistant"


class MessageDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message_delta"] = "message_delta"
    delta: str


class ThinkingDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class MessageEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message_end"] = "message_end"
    message: AnyMessage


class ToolExecutionStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call: ToolCall


class ToolExecutionUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    message: str
    data: dict[str, JSONValue] | None = None


class ToolExecutionEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_execution_end"] = "tool_execution_end"
    result: AgentToolResult


class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    message: str
    recoverable: bool = False
    data: dict[str, JSONValue] | None = None


type AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | QueueUpdateEvent
    | RetryEvent
    | MessageStartEvent
    | MessageDeltaEvent
    | ThinkingDeltaEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | TodoUpdateEvent
    | HumanInputRequestedEvent
    | ErrorEvent
)
