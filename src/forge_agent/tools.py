"""Provider-neutral tool definitions and tool execution results."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from forge_agent.context import ForgeRuntimeContext
from forge_agent.types import JSONValue


class ToolCancellationToken(Protocol):
    """Minimal cancellation interface accepted by tools."""

    def is_cancelled(self) -> bool:
        """Return whether tool execution should stop."""
        ...


class ToolExecutor(Protocol):
    """Async callable used to execute a tool.

    The optional ``context`` carries the session-owned ``ForgeRuntimeContext``
    (workspace root, session id, shell prefix).  Native LangChain tool
    wrappers inject it from ``ToolRuntime.context``; legacy callers that do
    not pass a context keep the factory-captured defaults.
    """

    def __call__(
        self,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> Awaitable[AgentToolResult]:
        """Execute the tool with optional cancellation and runtime context."""
        ...


class ToolCall(BaseModel):
    """A request from the assistant to execute a named tool."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)
    # Opaque signature some providers (e.g. Gemini) require echoed back next
    # turn; ignored by providers that don't use it.
    thought_signature: str | None = None


class AgentToolResult(BaseModel):
    """Structured result returned by a tool execution."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    ok: bool
    content: str
    data: dict[str, JSONValue] | None = None
    details: dict[str, JSONValue] | None = None
    error: str | None = None
