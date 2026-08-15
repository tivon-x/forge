"""Pi-style final text renderer for print mode."""

from __future__ import annotations

from collections.abc import Mapping

import typer

from forge_agent import AgentEvent, ErrorEvent, MessageEndEvent
from forge_agent.message_codec import message_text


class FinalTextRenderer:
    """Render only the final assistant text after the run finishes."""

    def __init__(self) -> None:
        self._last_assistant_text = ""
        self._failed = False
        self._error_messages: list[str] = []
        self._goal_status: str | None = None

    def render(self, event: AgentEvent) -> None:
        """Record events needed for final text output."""
        # Keep this projection duck-typed so a renderer built against an older
        # forge_agent package can still consume the rest of the event stream.
        # Goal updates are status metadata, not assistant text; they must never
        # replace a final MessageEndEvent.
        if getattr(event, "type", None) == "goal_update":
            self._goal_status = _format_goal_status(getattr(event, "goal", None))
            return

        if isinstance(event, MessageEndEvent):
            self._last_assistant_text = message_text(event.message)
            return

        if isinstance(event, ErrorEvent):
            if not event.recoverable:
                self._failed = True
            self._error_messages.append(event.message)

    def finish(self) -> bool:
        """Print final text or errors and return whether the run succeeded."""
        if self._failed:
            for message in self._error_messages:
                typer.echo(f"Error: {message}", err=True)
            return False

        if self._last_assistant_text:
            typer.echo(self._last_assistant_text)
        elif self._goal_status is not None:
            typer.echo(self._goal_status)
        return True


def _format_goal_status(goal: object) -> str:
    """Render the bounded, user-facing status projection for a Goal update."""
    if goal is None:
        return "Goal: none"

    if isinstance(goal, Mapping):
        status = goal.get("status")
        objective = goal.get("objective")
    else:
        status = getattr(goal, "status", None)
        objective = getattr(goal, "objective", None)
    if not isinstance(status, str) or not status:
        status = "unknown"
    if isinstance(objective, str) and objective.strip():
        return f"Goal: {status} — {objective.strip()}"
    return f"Goal: {status}"
