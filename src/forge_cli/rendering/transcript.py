"""Human-readable streaming transcript renderer."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.text import Text

from forge_agent import (
    AgentEndEvent,
    AgentEvent,
    CommittedSnapshotEvent,
    ErrorEvent,
    HumanInputRequestedEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    RetryEvent,
    TodoUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from forge_agent.message_codec import message_text
from forge_cli.formatting import format_tool_call_block, format_tool_result_block
from forge_cli.rendering.plain import _snapshot_aborted, _snapshot_assistant_text, _snapshot_error


class TranscriptRenderer:
    """Render assistant deltas live and tool activity to stderr."""

    def __init__(self) -> None:
        self._assistant_started = False
        self._assistant_ended = False
        self._failed = False
        self._console = Console(stderr=True, highlight=False)
        self._last_snapshot_sequence: int | None = None
        self._last_snapshot_text = ""

    def render(self, event: AgentEvent) -> None:
        """Render one agent event."""
        if isinstance(event, CommittedSnapshotEvent):
            sequence = event.snapshot.get("sequence")
            if (
                type(sequence) is int
                and self._last_snapshot_sequence is not None
                and sequence <= self._last_snapshot_sequence
            ):
                return
            self._last_snapshot_sequence = sequence if type(sequence) is int else None
            snapshot_error = _snapshot_error(event.snapshot)
            if snapshot_error is not None:
                self._failed = True
                self._ensure_assistant_newline()
                self._console.print(Text(f"Error: {snapshot_error}", style="red"))
                return
            text = _snapshot_assistant_text(
                event.snapshot,
                include_live=not _snapshot_aborted(event.snapshot),
            )
            if text == self._last_snapshot_text:
                return
            self._last_snapshot_text = text
            if text:
                self._assistant_started = True
                self._assistant_ended = False
                typer.echo(text, nl=False)
                self._ensure_assistant_newline(final=True)
            return
        if (
            isinstance(
                event, (ToolExecutionStartEvent, ToolExecutionEndEvent, ToolExecutionUpdateEvent)
            )
            and event.parent_tool_call_id
        ):
            if isinstance(event, ToolExecutionEndEvent):
                self._ensure_assistant_newline()
                status = "success" if event.result.ok else "error"
                self._console.print(
                    Text(f"  ↳ {event.result.name}: {status}", style="bright_black")
                )
            return
        if isinstance(event, MessageStartEvent):
            self._assistant_started = False
            self._assistant_ended = False
            return

        if isinstance(event, MessageDeltaEvent):
            self._assistant_started = True
            typer.echo(event.delta, nl=False)
            return

        if isinstance(event, ToolExecutionStartEvent):
            self._ensure_assistant_newline()
            self._console.print(Text(format_tool_call_block(event.tool_call), style="cyan"))
            return

        if isinstance(event, ToolExecutionUpdateEvent):
            self._ensure_assistant_newline()
            self._console.print(Text(f"… {event.message}", style="bright_black"))
            return

        if isinstance(event, TodoUpdateEvent):
            self._ensure_assistant_newline()
            completed = sum(item.status == "completed" for item in event.todos)
            self._console.print(
                Text(f"… Todos ({completed}/{len(event.todos)})", style="bright_black")
            )
            return

        if isinstance(event, HumanInputRequestedEvent):
            self._ensure_assistant_newline()
            self._console.print(Text("Waiting for user input…", style="yellow"))
            return

        if isinstance(event, RetryEvent):
            self._ensure_assistant_newline()
            self._console.print(Text(f"… {event.message}", style="bright_black"))
            return

        if isinstance(event, ToolExecutionEndEvent):
            style = "green" if event.result.ok else "red"
            self._ensure_assistant_newline()
            self._console.print(
                Text(
                    format_tool_result_block(
                        name=event.result.name,
                        ok=event.result.ok,
                        content=event.result.content,
                        data=event.result.data,
                    ),
                    style=style,
                )
            )
            return

        if isinstance(event, ErrorEvent):
            if not event.recoverable:
                self._failed = True
            self._ensure_assistant_newline()
            self._console.print(Text(f"Error: {event.message}", style="red"))
            return

        if isinstance(event, MessageEndEvent):
            if getattr(event.message, "type", None) == "ai":
                self._last_snapshot_text = message_text(event.message)
            self._ensure_assistant_newline(final=True)
            return

        if isinstance(event, AgentEndEvent):
            self._ensure_assistant_newline(final=True)

    def finish(self) -> bool:
        """Return whether the rendered run succeeded."""
        return not self._failed

    def _ensure_assistant_newline(self, *, final: bool = False) -> None:
        if self._assistant_started and not self._assistant_ended:
            typer.echo()
            self._assistant_ended = True
        elif final and not self._assistant_started:
            self._assistant_ended = True
