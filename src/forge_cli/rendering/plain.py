"""Pi-style final text renderer for print mode."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import typer

from forge_agent import AgentEvent, CommittedSnapshotEvent, ErrorEvent, MessageEndEvent
from forge_agent.message_codec import message_text
from forge_agent.session import SessionState, entry_from_json_line


class FinalTextRenderer:
    """Render only the final assistant text after the run finishes."""

    def __init__(self) -> None:
        self._last_assistant_text = ""
        self._failed = False
        self._error_messages: list[str] = []
        self._goal_status: str | None = None
        self._last_snapshot_sequence: int | None = None

    def render(self, event: AgentEvent) -> None:
        """Record events needed for final text output."""
        if isinstance(event, CommittedSnapshotEvent):
            sequence = event.snapshot.get("sequence")
            if (
                type(sequence) is int
                and self._last_snapshot_sequence is not None
                and sequence <= self._last_snapshot_sequence
            ):
                return
            if type(sequence) is int:
                self._last_snapshot_sequence = sequence
            snapshot_error = _snapshot_error(event.snapshot)
            if snapshot_error is not None:
                self._failed = True
                self._error_messages.append(snapshot_error)
                return
            if _snapshot_aborted(event.snapshot):
                snapshot_text = _snapshot_assistant_text(event.snapshot, include_live=False)
            else:
                snapshot_text = _snapshot_assistant_text(event.snapshot)
            if snapshot_text:
                self._last_assistant_text = snapshot_text
            return
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


def _snapshot_assistant_text(snapshot: Mapping[str, Any], *, include_live: bool = True) -> str:
    """Read committed assistant content and, when allowed, its confirmed live tail."""
    latest = ""
    entries = snapshot.get("entries")
    if isinstance(entries, (list, tuple)):
        decoded_entries = []
        for raw in entries:
            if not isinstance(raw, Mapping):
                continue
            entry = raw.get("record") if isinstance(raw.get("record"), Mapping) else raw
            if not isinstance(entry, Mapping):
                continue
            try:
                decoded_entries.append(entry_from_json_line(json.dumps(dict(entry))))
            except (TypeError, ValueError):
                continue
        leaves = [entry for entry in decoded_entries if entry.type == "leaf"]
        state = (
            SessionState.from_entries(decoded_entries, leaf_id=leaves[-1].entry_id)
            if leaves
            else SessionState.from_entries(decoded_entries)
        )
        for decoded in state.messages:
            if getattr(decoded, "type", None) == "ai":
                latest = message_text(decoded)

    documents = snapshot.get("documents")
    if include_live and isinstance(documents, (list, tuple)):
        for document in documents:
            if not isinstance(document, Mapping):
                continue
            record = document.get("record")
            kind = record.get("kind") if isinstance(record, Mapping) else document.get("kind")
            if kind != "forge.live.v1":
                continue
            value = document.get("value")
            if (
                isinstance(value, Mapping)
                and isinstance(value.get("assistant"), str)
                and value["assistant"]
            ):
                latest = value["assistant"]
    return latest


def _snapshot_live_value(snapshot: Mapping[str, Any]) -> Mapping[str, Any] | None:
    documents = snapshot.get("documents")
    if not isinstance(documents, (list, tuple)):
        return None
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        record = document.get("record")
        kind = record.get("kind") if isinstance(record, Mapping) else document.get("kind")
        if kind == "forge.live.v1" and isinstance(document.get("value"), Mapping):
            return cast(Mapping[str, Any], document["value"])
    return None


def _snapshot_error(snapshot: Mapping[str, Any]) -> str | None:
    live = _snapshot_live_value(snapshot)
    error = live.get("error") if live is not None else None
    if not isinstance(error, Mapping):
        return None
    message = error.get("message")
    return message.strip() if isinstance(message, str) and message.strip() else "Agent run failed."


def _snapshot_aborted(snapshot: Mapping[str, Any]) -> bool:
    live = _snapshot_live_value(snapshot)
    return live is not None and live.get("aborted") is True
