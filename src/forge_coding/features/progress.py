"""Commit native progress as product documents before exposing public events."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import Any

from forge_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    HumanInputRequestedEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)

LIVE_NAMESPACE = "forge.live.v1"
PROGRESS_EVENTS = (MessageDeltaEvent, ThinkingDeltaEvent, ToolExecutionUpdateEvent)


def empty_live() -> dict[str, Any]:
    return {
        "assistant": "",
        "thinking": "",
        "tools": {},
        "background": [],
        "running": False,
        "generating": False,
        "aborted": False,
        "error": None,
        "retry": None,
        "question": None,
    }


def reduce_live(value: dict[str, Any], event: AgentEvent) -> None:
    """Update a product view, never a second model transcript."""
    if isinstance(event, AgentStartEvent):
        value.update(
            running=True, generating=True, assistant="", thinking="", tools={}, aborted=False
        )
        value.update(error=None, retry=None, question=None)
    elif isinstance(event, AgentEndEvent):
        value.update(running=False, generating=False)
    elif isinstance(event, MessageStartEvent) and event.message_role == "assistant":
        value.update(assistant="", thinking="", generating=True)
    elif isinstance(event, MessageDeltaEvent):
        value["assistant"] += event.delta
    elif isinstance(event, ThinkingDeltaEvent):
        value["thinking"] += event.delta
    elif isinstance(event, MessageEndEvent) and event.message.type == "ai":
        value.update(assistant="", thinking="", generating=False)
    elif isinstance(event, ToolExecutionStartEvent):
        value["tools"][event.tool_call.id] = {"start": event.model_dump(mode="json"), "updates": []}
    elif isinstance(event, ToolExecutionUpdateEvent):
        tool = value["tools"].get(event.tool_call_id)
        if tool is not None:
            tool["updates"].append(event.model_dump(mode="json"))
            # The visible tool tail is bounded just like native shell output.
            tool["updates"] = tool["updates"][-100:]
    elif isinstance(event, ToolExecutionEndEvent):
        value["tools"].pop(event.result.tool_call_id, None)
    elif isinstance(event, ErrorEvent):
        value["error"] = event.model_dump(mode="json")
    elif isinstance(event, RetryEvent):
        value["retry"] = event.model_dump(mode="json")
    elif isinstance(event, HumanInputRequestedEvent):
        value["question"] = event.model_dump(mode="json")


def live_operations(previous: dict[str, Any], value: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain append deltas for streamed text; reset metadata explicitly."""
    operations = []
    for key, new in value.items():
        old = previous.get(key)
        if old == new:
            continue
        if key == "tools" and old is not None and old.keys() == new.keys():
            for call_id, tool in new.items():
                previous_tool = old[call_id]
                old_updates, updates = previous_tool["updates"], tool["updates"]
                if previous_tool["start"] != tool["start"]:
                    operations.append({"op": "set", "path": [key, call_id], "value": tool})
                    continue
                overlap = min(len(old_updates), len(updates))
                while overlap and old_updates[-overlap:] != updates[:overlap]:
                    overlap -= 1
                path = [key, call_id, "updates"]
                for update in updates[overlap:]:
                    operations.append({"op": "append", "path": path, "value": update})
                trimmed = len(old_updates) - overlap
                if trimmed:
                    operations.append({"op": "trim", "path": path, "start": trimmed})
            continue
        if key in {"assistant", "thinking"} and isinstance(old, str) and new.startswith(old):
            operations.append({"op": "append", "path": [key], "value": new[len(old) :]})
        else:
            operations.append({"op": "set", "path": [key], "value": copy.deepcopy(new)})
    return operations


class ProgressWriter:
    """One trailing writer; flush waits through the actual publication callback."""

    def __init__(self, publish: Callable[[Sequence[AgentEvent]], Awaitable[None]]) -> None:
        self.publish = publish
        self.pending: list[AgentEvent] = []
        self.task: asyncio.Task[None] | None = None
        self.lock = asyncio.Lock()
        self.error: BaseException | None = None
        self.wake = asyncio.Event()

    def add(self, event: AgentEvent) -> None:
        if self.error is not None:
            raise RuntimeError("progress publication failed") from self.error
        self.pending.append(event)
        if self.task is None:
            self.wake.clear()
            self.task = asyncio.create_task(self._trailing(), name="forge-progress")

    async def _trailing(self) -> None:
        try:
            with suppress(TimeoutError):
                await asyncio.wait_for(self.wake.wait(), 0.1)
            await self._drain()
        except BaseException as error:
            self.error = error
        finally:
            self.task = None
            if self.pending and self.error is None:
                self.wake.clear()
                self.task = asyncio.create_task(self._trailing(), name="forge-progress")

    async def _drain(self) -> None:
        async with self.lock:
            if not self.pending:
                return
            batch, self.pending = self.pending, []
            await self.publish(batch)

    async def flush(self) -> None:
        # Do not cancel an in-flight commit. Drain accepted work before terminal.
        task = self.task
        if task is not None:
            self.wake.set()
            await asyncio.shield(task)
        if self.error is not None:
            raise RuntimeError("progress publication failed") from self.error
        await self._drain()
