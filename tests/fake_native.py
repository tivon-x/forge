"""Shared native fake chat models for offline Forge harness/session tests.

Replaces the removed legacy ``forge_ai.fake.FakeProvider`` with LangChain
``BaseChatModel`` subclasses that play scripted ``AIMessage`` responses (one
per model call) and record every invocation so tests can assert what was
submitted. ``ScriptedChatModel`` is non-streaming (single-shot, no chunk
deltas); ``StreamingScriptedChatModel`` exposes token-like deltas for tests
that exercise live text projection.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Preset-response chat model that records per-call inputs and bound tools.

    Each model call plays the next ``responses`` entry (repeating the last one
    once the script is exhausted), replacing ``FakeProvider``'s round-based
    scripts. ``calls`` records one dict per invocation with the submitted
    ``messages`` and the tools bound via ``bind_tools``.
    """

    responses: list[AIMessage] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)
    closed: bool = False

    def __init__(self, responses: list[AIMessage] | None = None) -> None:
        super().__init__()
        object.__setattr__(self, "responses", list(responses or []))
        object.__setattr__(self, "_calls", 0)
        object.__setattr__(self, "_tools", [])
        self.calls = []

    async def aclose(self) -> None:
        self.closed = True

    @property
    def _llm_type(self) -> str:
        return "forge-scripted-chat"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tool_choice, kwargs
        object.__setattr__(self, "_tools", list(tools or []))
        return self

    def _record_call(self, messages: list) -> None:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(getattr(self, "_tools", []) or []),
            }
        )

    def _next_response(self) -> AIMessage:
        count = int(getattr(self, "_calls", 0))
        object.__setattr__(self, "_calls", count + 1)
        if not self.responses:
            return AIMessage(content="")
        return self.responses[min(count, len(self.responses) - 1)]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del stop, run_manager, kwargs
        self._record_call(messages)
        return ChatResult(generations=[ChatGeneration(message=self._next_response())])


class StreamingScriptedChatModel(ScriptedChatModel):
    """Scripted model that also streams the response text as a single chunk."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del stop, run_manager, kwargs
        self._record_call(messages)
        response = self._next_response()
        mid = f"fake-run-{len(self.calls)}"
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=response.content,
                tool_calls=response.tool_calls or [],
                id=mid,
            )
        )

    @property
    def _llm_type(self) -> str:
        return "forge-scripted-chat-stream"


class ScriptedErrorChatModel(ScriptedChatModel):
    """Scripted model that raises a themed error on one specific call.

    Replaces the legacy ``ProviderErrorEvent`` mid-script: call ``error_on_call``
    (1-indexed) raises ``RuntimeError(error_message)``; every other call plays
    the next scripted ``responses`` entry.
    """

    def __init__(
        self,
        responses: list[AIMessage] | None = None,
        *,
        error_on_call: int = 1,
        error_message: str = "fake error",
    ) -> None:
        super().__init__(responses)
        object.__setattr__(self, "error_on_call", error_on_call)
        object.__setattr__(self, "error_message", error_message)
        object.__setattr__(self, "_call_no", 0)
        object.__setattr__(self, "_resp_idx", 0)

    def _next_response(self) -> AIMessage:
        call = int(getattr(self, "_call_no", 0))
        object.__setattr__(self, "_call_no", call + 1)
        if call + 1 == getattr(self, "error_on_call", 1):
            # The failing model call consumes no response slot; the caller
            # continues from the same playback position after recovering.
            raise RuntimeError(getattr(self, "error_message", "fake error"))
        resp_idx = int(getattr(self, "_resp_idx", 0))
        object.__setattr__(self, "_resp_idx", resp_idx + 1)
        if not self.responses:
            return AIMessage(content="")
        return self.responses[min(resp_idx, len(self.responses) - 1)]


class ThrowingChatModel(BaseChatModel):
    """A chat model that raises immediately, replacing ``ProviderErrorEvent`` scripts."""

    error: str = Field(default="fake provider failure")

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        raise RuntimeError(self.error)

    @property
    def _llm_type(self) -> str:
        return "forge-throwing-chat"


def tool_call_ai(tool_call_id: str, name: str, args: dict[str, object]) -> AIMessage:
    """Build an AIMessage that requests a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": name, "args": args, "type": "tool_call"}],
    )


def message_texts(messages: list[BaseMessage]) -> list[str]:
    """Return plain ``content`` strings for assertion convenience."""
    return [str(getattr(message, "content", "")) for message in messages]


def _plain_content(message) -> str:
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


def message_signatures(messages) -> list[tuple]:
    """Return a comparable signature per message (type, content, tool-call ids, tool_call_id)."""
    sigs: list[tuple] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        call_ids = tuple(
            str(call.get("id")) for call in tool_calls if isinstance(call, dict) and call.get("id")
        )
        sigs.append(
            (
                getattr(message, "type", ""),
                _plain_content(message),
                call_ids,
                getattr(message, "tool_call_id", None),
            )
        )
    return sigs
