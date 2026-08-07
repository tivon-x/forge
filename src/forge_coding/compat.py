"""Legacy Forge provider streaming helpers (offline-fixture compatibility seam).

Production sessions construct LangChain ``BaseChatModel`` objects and never
import ``forge_ai``.  These helpers exist only for historical callers that
pass a legacy ``ModelProvider`` into a coding session (offline fixtures, old
JSONL resumes, and the TUI's pre-login placeholder).  Keeping the
``forge_ai`` provider stream protocol here confines it to a single
compatibility module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from forge_agent.messages import AgentMessage, AssistantMessage
from forge_agent.provider import CancellationToken, ModelProvider, ProviderEvent
from forge_agent.tools import AgentTool
from forge_ai.events import ProviderErrorEvent, ProviderResponseEndEvent, ProviderTextDeltaEvent


async def stream_legacy_provider_text(
    provider: ModelProvider,
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    error_label: str,
) -> str:
    """Stream a legacy provider and return its accumulated response text.

    ``error_label`` prefixes the ``RuntimeError`` raised when the provider
    emits an error event, so callers keep their historical diagnostic wording.
    """

    final_text: str | None = None
    text_parts: list[str] = []
    async for event in provider.stream_response(
        model=model,
        system=system,
        messages=messages,
        tools=[],
    ):
        if isinstance(event, ProviderTextDeltaEvent):
            text_parts.append(event.delta)
        elif isinstance(event, ProviderResponseEndEvent):
            final_text = event.message.content
        elif isinstance(event, ProviderErrorEvent):
            details = f": {event.data}" if event.data is not None else ""
            raise RuntimeError(f"{error_label}: {event.message}{details}")
    return final_text if final_text is not None else "".join(text_parts)


async def stream_legacy_provider_final(
    provider: ModelProvider,
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
) -> AssistantMessage | None:
    """Stream a legacy provider and return its final assistant message.

    Returns ``None`` on provider error or when the provider never emits a
    final response, matching the historical branch-summary contract.
    """

    response: AssistantMessage | None = None
    async for event in provider.stream_response(
        model=model,
        system=system,
        messages=messages,
        tools=[],
    ):
        if isinstance(event, ProviderErrorEvent):
            return None
        if isinstance(event, ProviderResponseEndEvent):
            response = event.message
    return response


class LoginRequiredProvider:
    """Placeholder provider used so the TUI can open before login."""

    def __init__(self, message: str) -> None:
        self.message = message

    async def aclose(self) -> None:
        """Close provider resources."""

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Surface a login-needed provider error."""
        del model, system, messages, tools, signal

        async def iterator() -> AsyncIterator[ProviderEvent]:
            yield ProviderErrorEvent(message=self.message)

        return iterator()
