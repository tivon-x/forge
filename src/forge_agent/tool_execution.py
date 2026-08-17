"""Conservative tool-call execution policy for Forge agents.

LangChain's ``ToolNode`` gathers a model's tool calls concurrently.  Forge
keeps LangChain's agent loop, but places this middleware around the native
tool handlers so one model-produced batch has deterministic side effects.
The middleware deliberately keeps only the currently active batch: the
model's latest ``AIMessage`` is the source of truth, while ``ToolMessage``
remains the result/pairing fact.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage, convert_to_messages
from langgraph.types import Command

_ERROR_MAX_BYTES = 256
_BATCH_INVALID_ERROR = "Tool batch rejected; no tool was executed."
_ORDER_ERROR = "Tool batch order was invalid; no tool was executed."
_EXECUTION_ERROR = "Tool execution failed; remaining calls were skipped."
_SKIPPED_ERROR = "Skipped because an earlier tool call failed."


ToolCallHandler = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]


@dataclass(frozen=True, slots=True)
class _BatchInfo:
    """The validated call ids for the latest model-produced tool batch."""

    key: tuple[object, ...]
    ids: tuple[str, ...] | None


def _bounded_error(value: str) -> str:
    """Keep middleware-generated error content within a small UTF-8 budget."""

    encoded = value.encode("utf-8")
    if len(encoded) <= _ERROR_MAX_BYTES:
        return value
    return encoded[:_ERROR_MAX_BYTES].decode("utf-8", errors="ignore")


def _state_messages(state: object) -> Sequence[object] | None:
    """Read messages from the public state shape without depending on a schema."""

    if isinstance(state, Mapping):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
        return messages
    return None


def _message_identity(message: AIMessage) -> tuple[str, str | int]:
    """Return a stable-enough identity for one in-memory model message.

    Provider messages normally have an id.  Fake/local models often leave it
    empty, so the object identity is the fallback.  Only this one identity is
    retained; the middleware never keeps a transcript or a batch registry.
    """

    message_id = getattr(message, "id", None)
    if isinstance(message_id, str) and message_id:
        return ("id", message_id)
    return ("object", id(message))


def _batch_info(state: object) -> _BatchInfo:
    """Derive a fail-closed batch description from the latest ``AIMessage``."""

    messages = _state_messages(state)
    if messages is None:
        return _BatchInfo(("missing-messages", id(state)), None)

    latest: AIMessage | None = None
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            latest = message
            break
    if latest is None:
        return _BatchInfo(("missing-ai-message", id(state)), None)

    identity = _message_identity(latest)
    try:
        calls = latest.tool_calls
    except Exception:  # noqa: BLE001 - malformed provider state must fail closed
        return _BatchInfo(("malformed-ai-message", *identity), None)

    ids: list[str] = []
    valid = bool(calls)
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
        valid = False
        calls_key: tuple[object, ...] = ("non-sequence",)
    else:
        calls_key_list: list[object] = []
        for call in calls:
            if not isinstance(call, Mapping):
                valid = False
                calls_key_list.append("invalid-call")
                continue
            raw_id = call.get("id")
            if isinstance(raw_id, str) and raw_id:
                ids.append(raw_id)
                calls_key_list.append(raw_id)
            else:
                valid = False
                # The concrete missing value is intentionally not retained.
                calls_key_list.append("missing-id")
        calls_key = tuple(calls_key_list)

    if len(ids) != len(set(ids)):
        valid = False
    if not ids:
        valid = False

    # Include the message identity and raw id shape so a new model batch
    # resets the small state even when a provider reuses one call id.
    key = ("tool-batch", *identity, calls_key)
    return _BatchInfo(key, tuple(ids) if valid else None)


def _request_id(request: ToolCallRequest) -> str | None:
    raw_id = request.tool_call.get("id")
    return raw_id if isinstance(raw_id, str) and raw_id else None


def _request_name(request: ToolCallRequest) -> str:
    raw_name = request.tool_call.get("name")
    return raw_name if isinstance(raw_name, str) and raw_name else "tool"


def _error_message(request: ToolCallRequest, content: str) -> ToolMessage:
    """Create a bounded error that always pairs with the current call id."""

    return ToolMessage(
        content=_bounded_error(content),
        name=_request_name(request),
        tool_call_id=_request_id(request) or "",
        status="error",
    )


def _command_has_error_tool_message(command: Command[Any], tool_call_id: str) -> bool:
    """Return whether a command carries this call's error ``ToolMessage``.

    LangGraph accepts either a state mapping with a ``messages`` update or a
    direct message sequence. Normalize the same public shapes, then apply
    Forge's fail-fast policy only to the result paired with this call. Other
    historical tool errors in the update must not poison the active batch.
    """

    update = command.update
    if isinstance(update, Mapping):
        raw_messages = update.get("messages", ())
    elif isinstance(update, Sequence) and not isinstance(update, (str, bytes, bytearray)):
        raw_messages = update
    else:
        return False
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes, bytearray)):
        return False
    try:
        messages = convert_to_messages(raw_messages)
    except (TypeError, ValueError):
        # LangGraph owns validation of malformed Command updates. Do not turn
        # that separate validation error into a Forge batch-status decision.
        return False
    return any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == tool_call_id
        and message.status == "error"
        for message in messages
    )


class SequentialToolCallMiddleware(AgentMiddleware[Any, Any, Any]):
    """Execute each model tool-call batch strictly in model order.

    One middleware instance belongs to one compiled agent graph.  LangChain
    invokes one wrapper per call and normally schedules those wrappers with
    ``asyncio.gather``; the fair lock below turns that into FIFO execution.
    The active batch state is reset when the latest AI message/call-id tuple
    changes.  Invalid or out-of-order calls fail closed before reaching any
    inner middleware or tool.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()
        self._batch_key: tuple[object, ...] | None = None
        self._batch_ids: tuple[str, ...] | None = None
        self._next_index = 0
        self._failed = False

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> ToolMessage | Command[Any]:
        """Serialize one batch, short-circuit failures, and preserve pairing."""

        batch = _batch_info(request.state)
        await self._lock.acquire()
        try:
            if batch.key != self._batch_key:
                self._batch_key = batch.key
                self._batch_ids = batch.ids
                self._next_index = 0
                self._failed = False

            if self._failed:
                return _error_message(request, _SKIPPED_ERROR)
            if self._batch_ids is None:
                self._failed = True
                return _error_message(request, _BATCH_INVALID_ERROR)

            current_id = _request_id(request)
            if current_id is None or current_id not in self._batch_ids:
                self._failed = True
                return _error_message(request, _ORDER_ERROR)
            if self._next_index >= len(self._batch_ids):
                self._failed = True
                return _error_message(request, _ORDER_ERROR)
            if current_id != self._batch_ids[self._next_index]:
                self._failed = True
                return _error_message(request, _ORDER_ERROR)

            try:
                result = await handler(request)
            except asyncio.CancelledError:
                # A cancellation aborts the active batch.  Keep the original
                # exception so the harness can perform its normal settlement.
                self._failed = True
                raise
            except Exception:  # noqa: BLE001 - convert provider/tool failure safely
                self._failed = True
                return _error_message(request, _EXECUTION_ERROR)

            if not isinstance(result, (ToolMessage, Command)):
                self._failed = True
                return _error_message(request, _ORDER_ERROR)

            if isinstance(result, ToolMessage):
                # An inner layer must not silently break ToolMessage pairing.
                # Do not expose the malformed message content in the repair row.
                if result.tool_call_id != current_id:
                    self._failed = True
                    return _error_message(request, _ORDER_ERROR)
                if result.status == "error":
                    self._failed = True
                else:
                    self._next_index += 1
            else:
                # A Command represents a valid graph control result (for
                # example HITL/task routing) and consumes this call position.
                # Its update may still contain the paired error ToolMessage;
                # that error must stop the remainder of the model batch just
                # like a direct ToolMessage result.
                if _command_has_error_tool_message(result, current_id):
                    self._failed = True
                else:
                    self._next_index += 1
            return result
        finally:
            self._lock.release()


__all__ = ["SequentialToolCallMiddleware"]
