"""Stateful Forge facade over the LangChain agent runtime."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from forge_agent.context import ForgeRuntimeContext
from forge_agent.events import (
    AgentEvent,
    HumanInputRequestedEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
)
from forge_agent.langchain_runtime import LangChainRuntimeState, run_langchain_agent
from forge_agent.message_codec import message_text as _message_text
from forge_agent.retry import RetryPolicy
from forge_agent.steering import SteeringMiddleware
from forge_agent.tools import ToolCall
from forge_agent.types import JSONValue

EventListener = Callable[[AgentEvent], Awaitable[None] | None]
QueueMode = Literal["one_at_a_time", "all"]


def _message_role(message: AnyMessage) -> Literal["user", "assistant", "tool"]:
    role = getattr(message, "role", None)
    if role == "user":
        return "user"
    if role == "tool":
        return "tool"
    if role == "assistant":
        return "assistant"
    message_type = str(getattr(message, "type", ""))
    if message_type == "human":
        return "user"
    if message_type == "tool":
        return "tool"
    return "assistant"


@dataclass(frozen=True, slots=True)
class QueuedMessages:
    """Snapshot of harness-owned queued user messages."""

    steering: tuple[AnyMessage, ...] = ()
    follow_up: tuple[AnyMessage, ...] = ()

    @property
    def count(self) -> int:
        """Return the total queued message count."""
        return len(self.steering) + len(self.follow_up)


@dataclass(slots=True)
class AgentHarnessConfig:
    """Configuration for an `AgentHarness`."""

    provider: BaseChatModel
    model: str = ""
    system: str = ""
    tools: Sequence[BaseTool] = field(default_factory=list)
    runtime_context: ForgeRuntimeContext | None = None
    max_turns: int | None = None
    queue_mode: QueueMode = "one_at_a_time"
    middleware: Sequence[object] = field(default_factory=tuple)
    interactive: bool = False
    retry: RetryPolicy = field(default_factory=RetryPolicy)


class SimpleCancellationToken:
    """Small cancellation token used by the harness and loop."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancelled


class AgentHarness:
    """Reusable stateful agent brain.

    The harness owns transcript/concurrency state and delegates model/tool
    execution to LangChain's ``create_agent`` runtime via
    :func:`forge_agent.langchain_runtime.run_langchain_agent`.
    It remains independent of CLI, Rich, Textual, session files, and coding-agent
    resource loading.
    """

    def __init__(
        self,
        config: AgentHarnessConfig,
        *,
        messages: Sequence[AnyMessage] = (),
    ) -> None:
        self._config = config
        self._messages = list(messages)
        self._listeners: list[EventListener] = []
        self._current_signal: SimpleCancellationToken | None = None
        self._current_task: asyncio.Task[object] | None = None
        self._running = False
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._last_run_interrupted = False
        self._steering_queue: deque[AnyMessage] = deque()
        self._follow_up_queue: deque[AnyMessage] = deque()
        # HITL/checkpoint support is an explicit frontend capability.  Do not
        # infer it from middleware implementation names: wrappers and subclasses
        # are valid middleware too.
        needs_runtime_state = config.interactive
        self._runtime_state: LangChainRuntimeState | None = (
            LangChainRuntimeState() if needs_runtime_state else None
        )
        self._waiting_for_input = False

    @property
    def messages(self) -> tuple[AnyMessage, ...]:
        """Return an immutable snapshot of the current transcript."""
        return tuple(self._messages)

    @property
    def was_last_run_interrupted(self) -> bool:
        """Whether the most recent turn was cancelled mid-run.

        Only a genuine interrupt (``cancel()`` on a running turn) marks this
        true. A consumer closing the event stream early without cancelling is
        not an interruption and leaves it false.
        """
        return self._last_run_interrupted

    @property
    def config(self) -> AgentHarnessConfig:
        """Return the harness configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Return whether a prompt or continuation is currently active."""
        return self._running

    async def wait_until_idle(self) -> None:
        """Wait until the current LangChain graph has fully unwound."""

        await self._idle_event.wait()

    @property
    def is_waiting_for_input(self) -> bool:
        """Return whether the agent is paused at a human-input interrupt."""

        return self._waiting_for_input

    @property
    def pending_human_input(self) -> tuple[HumanInputRequestedEvent, ...]:
        """Return the current request in a stable tuple for UI adapters."""

        state = self._runtime_state
        if state is None or not state.pending_requests:
            return ()
        first = state.pending_requests[0]
        return (
            HumanInputRequestedEvent(
                interrupt_id=first.interrupt_id,
                tool_call_id=first.tool_call_id,
                tool_name=first.tool_name,
                arguments=first.arguments,
                allowed_decisions=first.allowed_decisions,
                description=first.description,
                requests=state.pending_requests,
            ),
        )

    @property
    def queued_messages(self) -> QueuedMessages:
        """Return a snapshot of queued steering and follow-up messages."""
        return QueuedMessages(
            steering=tuple(self._steering_queue),
            follow_up=tuple(self._follow_up_queue),
        )

    @property
    def pending_message_count(self) -> int:
        """Return the total queued message count."""
        return self.queued_messages.count

    def has_queued_messages(self) -> bool:
        """Return whether either queue has pending messages."""
        return bool(self._steering_queue or self._follow_up_queue)

    def append_message(self, message: AnyMessage) -> None:
        """Append an existing message, useful for restoring session state."""
        self._messages.append(message)

    def replace_messages(self, messages: Sequence[AnyMessage]) -> None:
        """Replace the transcript, useful after durable context reconstruction."""
        self._messages = list(messages)

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        """Subscribe to streamed events and return an unsubscribe callback."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    def cancel(self) -> None:
        """Request cancellation for the currently running prompt, if any."""
        self.request_cancel()
        if self._current_task is not None:
            self._current_task.cancel()

    def request_cancel(self) -> None:
        """Request a cooperative graph stop without cancelling the consumer task.

        Session coordinators use this when a terminal product event ends the
        current LangChain graph.  User cancellation keeps the historical
        ``cancel()`` behavior and interrupts the consuming task as well.
        """
        if self._current_signal is not None:
            self._current_signal.cancel()

    def cancel_pending_input(self) -> int:
        """Close a pending HITL turn with synthetic paired tool results.

        This is used only while tearing down a session.  Interactive cancellation
        normally resumes the graph with a ``respond`` decision so the model can
        continue naturally.
        """

        if not self._waiting_for_input or self._runtime_state is None:
            return 0
        pending = self._runtime_state.pending_requests
        returned_ids = {
            message.tool_call_id for message in self._messages if isinstance(message, ToolMessage)
        }
        added = 0
        for request in pending:
            if request.tool_call_id in returned_ids:
                continue
            self._messages.append(
                ToolMessage(
                    tool_call_id=request.tool_call_id,
                    name=request.tool_name,
                    content='{"cancelled":true}',
                    status="error",
                )
            )
            returned_ids.add(request.tool_call_id)
            added += 1
        self._runtime_state.clear()
        self._waiting_for_input = False
        return added

    def respond_to_human_input(
        self,
        response: str | Mapping[str, JSONValue] | Sequence[Mapping[str, JSONValue]],
    ) -> AsyncIterator[AgentEvent]:
        """Resume the paused graph with one or more HITL decisions."""

        if self._running:
            raise RuntimeError("AgentHarness is already running")
        if not self._waiting_for_input or self._runtime_state is None:
            raise RuntimeError("AgentHarness is not waiting for human input")
        if isinstance(response, str):
            # The TUI uses a string for the common one-request case.  A
            # multi-request questionnaire serializes its ordered decisions as
            # a JSON object so the public API can remain backwards compatible.
            try:
                decoded = json.loads(response)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, Mapping) and isinstance(decoded.get("decisions"), Sequence):
                response = cast(Any, decoded)
        if isinstance(response, str):
            decisions: tuple[Mapping[str, JSONValue], ...] = (
                {"type": "respond", "message": response},
            )
        elif isinstance(response, Mapping):
            raw_decisions = response.get("decisions")
            if not isinstance(raw_decisions, Sequence) or isinstance(
                raw_decisions, (str, bytes, bytearray)
            ):
                raise ValueError("human input response must contain decisions")
            decisions = tuple(item for item in raw_decisions if isinstance(item, Mapping))
        else:
            decisions = tuple(response)
        if len(decisions) != len(self._runtime_state.pending_requests):
            raise ValueError("human input response count does not match pending requests")
        self._running = True
        return self._run(resume_decisions=decisions)

    def steer(self, content: str) -> QueueUpdateEvent:
        """Queue a steering message for the active or next run."""
        message: AnyMessage = HumanMessage(content=content)
        return self.steer_message(message)

    def steer_message(self, message: AnyMessage) -> QueueUpdateEvent:
        """Queue a message to inject after the current turn/tool batch."""
        self._steering_queue.append(message)
        return self.queue_update_event()

    def follow_up(self, content: str) -> QueueUpdateEvent:
        """Queue a follow-up message for when the active run would stop."""
        message: AnyMessage = HumanMessage(content=content)
        return self.follow_up_message(message)

    def follow_up_message(self, message: AnyMessage) -> QueueUpdateEvent:
        """Queue a message to inject when the current run would otherwise stop."""
        self._follow_up_queue.append(message)
        return self.queue_update_event()

    def clear_queues(self) -> QueuedMessages:
        """Clear all queued messages and return the cleared snapshot."""
        snapshot = self.queued_messages
        self._steering_queue.clear()
        self._follow_up_queue.clear()
        return snapshot

    def pop_latest_follow_up(self) -> AnyMessage | None:
        """Remove and return the most recently queued follow-up message."""
        if not self._follow_up_queue:
            return None
        return self._follow_up_queue.pop()

    def pop_latest_steering(self) -> AnyMessage | None:
        """Remove and return the most recently queued steering message."""
        if not self._steering_queue:
            return None
        return self._steering_queue.pop()

    def queue_update_event(self) -> QueueUpdateEvent:
        """Return the current queue state as a portable agent event."""
        return QueueUpdateEvent(
            steering=tuple(_message_text(message) for message in self._steering_queue),
            follow_up=tuple(_message_text(message) for message in self._follow_up_queue),
        )

    def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """Append a user message and run the agent loop."""
        self._ensure_not_running()
        self._append_interrupted_tool_results()
        self._running = True
        message: AnyMessage = HumanMessage(content=content)
        self._messages.append(message)
        return self._run(prompt_message=message)

    def continue_(self) -> AsyncIterator[AgentEvent]:
        """Continue the agent loop without appending a new user message."""
        self._ensure_not_running()
        self._append_interrupted_tool_results()
        self._running = True
        return self._run()

    async def _run(
        self,
        *,
        prompt_message: AnyMessage | None = None,
        resume_decisions: Sequence[Mapping[str, JSONValue]] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        # Each turn starts with a clean interrupt state; only the *current*
        # turn may mark ``was_last_run_interrupted``.  Without this reset a
        # cancelled turn would keep the flag set, and the next normally
        # completed turn would flush a stale interruption in the session's
        # ``finally`` block and re-persist messages (duplicating JSONL rows
        # when compaction/overflow retries had rebuilt the transcript).
        self._last_run_interrupted = False
        self._idle_event.clear()
        signal = SimpleCancellationToken()
        self._current_signal = signal
        self._current_task = asyncio.current_task()
        pending_prompt_event = prompt_message
        try:
            while True:
                events = run_langchain_agent(
                    provider=self._config.provider,
                    model=self._config.model,
                    system=self._config.system,
                    messages=self._messages,
                    tools=self._config.tools,
                    max_turns=self._config.max_turns,
                    signal=signal,
                    runtime_context=self._config.runtime_context,
                    steering=SteeringMiddleware(
                        self._steering_queue,
                        queue_mode=self._config.queue_mode,
                    ),
                    queue_update=self.queue_update_event,
                    middleware=self._config.middleware,
                    retry_policy=self._config.retry,
                    runtime_state=self._runtime_state,
                    resume_decisions=resume_decisions,
                )
                resume_decisions = None
                async for event in events:
                    await self._notify(event)
                    yield event
                    if isinstance(event, HumanInputRequestedEvent):
                        self._waiting_for_input = True
                    if pending_prompt_event is not None and event.type == "turn_start":
                        start = MessageStartEvent(message_role="user")
                        end = MessageEndEvent(message=pending_prompt_event)
                        for prompt_event in (start, end):
                            await self._notify(prompt_event)
                            yield prompt_event
                        pending_prompt_event = None

                if self._waiting_for_input:
                    break
                queued = self._drain_steering_messages()
                if not queued:
                    queued = self._drain_follow_up_messages()
                if not queued:
                    break
                for message in queued:
                    self._messages.append(message)
                    for queued_event in (
                        MessageStartEvent(message_role=_message_role(message)),
                        MessageEndEvent(message=message),
                    ):
                        await self._notify(queued_event)
                        yield queued_event
                queue_event = self.queue_update_event()
                await self._notify(queue_event)
                yield queue_event
        finally:
            if signal.is_cancelled():
                self._last_run_interrupted = True
                self._append_interrupted_tool_results()
            if self._current_signal is signal:
                self._current_signal = None
            if self._current_task is asyncio.current_task():
                self._current_task = None
            self._running = False
            self._idle_event.set()
            if self._runtime_state is not None and not self._runtime_state.waiting:
                self._waiting_for_input = False

    async def _notify(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            result = listener(event)
            if isawaitable(result):
                await result

    def _ensure_not_running(self) -> None:
        if self._waiting_for_input:
            raise RuntimeError(
                "AgentHarness is waiting for human input; answer or cancel the questionnaire."
            )
        if self._running:
            raise RuntimeError(
                "AgentHarness is already running; use steer() or follow_up() to queue messages."
            )

    def _drain_steering_messages(self) -> tuple[AnyMessage, ...]:
        return self._drain_queue(self._steering_queue)

    def _drain_follow_up_messages(self) -> tuple[AnyMessage, ...]:
        return self._drain_queue(self._follow_up_queue)

    def _drain_queue(self, queue: deque[AnyMessage]) -> tuple[AnyMessage, ...]:
        if not queue:
            return ()
        if self._config.queue_mode == "all":
            messages = tuple(queue)
            queue.clear()
            return messages
        return (queue.popleft(),)

    def append_interrupted_tool_results(self) -> int:
        """Repair a transcript left mid-tool-call by an interrupted run.

        Returns the number of synthetic tool results that were appended.
        """
        before_count = len(self._messages)
        self._append_interrupted_tool_results()
        return len(self._messages) - before_count

    def _append_interrupted_tool_results(self) -> None:
        """Repair a transcript left mid-tool-call by an interrupted run.

        OpenAI-compatible providers reject a transcript where an assistant tool
        call has no matching tool result anywhere in the submitted history. If
        the UI cancels the worker while a tool is still running, the normal loop
        may not get a chance to append the cancellation result, so repair that
        gap before the next model request.
        """
        returned_ids = {
            message.tool_call_id for message in self._messages if isinstance(message, ToolMessage)
        }
        for message in tuple(self._messages):
            if isinstance(message, AIMessage):
                calls = [
                    ToolCall(
                        id=str(call.get("id") or ""),
                        name=str(call.get("name") or "tool"),
                        arguments=call.get("args", {}),
                    )
                    for call in message.tool_calls
                ]
            else:
                continue
            for tool_call in calls:
                if tool_call.id in returned_ids:
                    continue
                returned_ids.add(tool_call.id)
                content = "Tool call interrupted by user"
                self._messages.append(
                    ToolMessage(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=content,
                        status="error",
                    )
                )
