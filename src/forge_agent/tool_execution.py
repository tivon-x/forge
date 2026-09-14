"""Conservative tool-call execution policy for Forge agents.

LangChain's ``ToolNode`` gathers a model's tool calls concurrently. Forge
keeps LangChain's agent loop and adds Pi's batch-level sequential escape hatch
around native tool handlers. The middleware deliberately keeps only the
currently active batch: the model's latest ``AIMessage`` is the source of
truth, while ``ToolMessage`` remains the result/pairing fact.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

_ERROR_MAX_BYTES = 256
_BATCH_INVALID_ERROR = "Tool batch rejected; no tool was executed."
_ORDER_ERROR = "Tool batch order was invalid; no tool was executed."
_EXECUTION_ERROR = "Tool execution failed."
_SKIPPED_ERROR = "Skipped because the tool-call batch was invalid or cancelled."
TOOL_EXECUTION_MODE_METADATA_KEY = "forge.execution_mode"
SEQUENTIAL_TOOL_EXECUTION_MODE = "sequential"


ToolCallHandler = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]


@dataclass(frozen=True, slots=True)
class _BatchInfo:
    """The validated call ids for the latest model-produced tool batch."""

    key: tuple[object, ...]
    ids: tuple[str, ...] | None
    sequential: bool


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


def _tools_by_name(request: ToolCallRequest) -> dict[str, BaseTool]:
    """Return the native tools visible to this ToolNode invocation."""

    result: dict[str, BaseTool] = {}
    runtime_tools = getattr(request.runtime, "tools", ())
    if isinstance(runtime_tools, Sequence):
        for tool in runtime_tools:
            if isinstance(tool, BaseTool) and tool.name:
                result[tool.name] = tool
    if isinstance(request.tool, BaseTool) and request.tool.name:
        # A middleware may override the current request tool (for example the
        # dynamic Goal tools), so it wins over the runtime snapshot.
        result[request.tool.name] = request.tool
    return result


def _is_sequential(tool: BaseTool | None) -> bool:
    """Read the optional Forge execution mode from native tool metadata."""

    metadata = getattr(tool, "metadata", None)
    return (
        isinstance(metadata, Mapping)
        and metadata.get(TOOL_EXECUTION_MODE_METADATA_KEY) == SEQUENTIAL_TOOL_EXECUTION_MODE
    )


def _batch_info(request: ToolCallRequest) -> _BatchInfo:
    """Derive a fail-closed batch description from the latest ``AIMessage``."""

    messages = _state_messages(request.state)
    if messages is None:
        return _BatchInfo(("missing-messages", id(request.state)), None, False)

    latest: AIMessage | None = None
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            latest = message
            break
    if latest is None:
        return _BatchInfo(("missing-ai-message", id(request.state)), None, False)

    identity = _message_identity(latest)
    try:
        calls = latest.tool_calls
    except Exception:  # noqa: BLE001 - malformed provider state must fail closed
        return _BatchInfo(("malformed-ai-message", *identity), None, False)

    ids: list[str] = []
    names: list[str] = []
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
            raw_name = call.get("name")
            call_key: list[str] = []
            if isinstance(raw_id, str) and raw_id:
                ids.append(raw_id)
                call_key.append(raw_id)
            else:
                valid = False
                # The concrete missing value is intentionally not retained.
                call_key.append("missing-id")
            if isinstance(raw_name, str) and raw_name:
                names.append(raw_name)
                call_key.append(raw_name)
            else:
                valid = False
                call_key.append("missing-name")
            calls_key_list.append(tuple(call_key))
        calls_key = tuple(calls_key_list)

    if len(ids) != len(set(ids)):
        valid = False
    if not ids:
        valid = False

    # Include the message identity and raw id shape so a new model batch
    # resets the small state even when a provider reuses one call id.
    key = ("tool-batch", *identity, calls_key)
    sequential = any(_is_sequential(_tools_by_name(request).get(name)) for name in names)
    return _BatchInfo(key, tuple(ids) if valid else None, sequential)


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


class ToolCallBatchMiddleware(AgentMiddleware[Any, Any, Any]):
    """Apply Pi's batch-level execution policy around native ToolNode calls.

    LangChain's ToolNode owns parallel scheduling and result ordering.  Forge
    only adds the Pi rule that one ``sequential`` tool makes the complete model
    batch run in AIMessage order.  Ordinary tool failures stay paired to their
    own call and do not stop sibling calls; malformed batches and pairings fail
    closed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()
        self._batch_key: tuple[object, ...] | None = None
        self._batch_ids: tuple[str, ...] | None = None
        self._next_index = 0
        self._failed = False

    async def _run_handler(
        self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> tuple[ToolMessage | Command[Any], bool]:
        """Run one handler and return whether its pairing remained valid."""

        current_id = _request_id(request)
        try:
            result = await handler(request)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - convert provider/tool failure safely
            # Ordinary tool errors are isolated to their own call.  ToolNode
            # continues gathering sibling calls and LangChain receives a
            # bounded paired error row.
            return _error_message(request, _EXECUTION_ERROR), True

        if not isinstance(result, (ToolMessage, Command)):
            return _error_message(request, _ORDER_ERROR), False
        if isinstance(result, ToolMessage) and result.tool_call_id != current_id:
            # An inner layer must not silently break ToolMessage pairing.
            return _error_message(request, _ORDER_ERROR), False
        return result, True

    async def _run_parallel(
        self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
        batch: _BatchInfo,
    ) -> ToolMessage | Command[Any]:
        """Validate one call, then let LangChain keep native concurrency."""

        if batch.ids is None:
            return _error_message(request, _BATCH_INVALID_ERROR)
        current_id = _request_id(request)
        if current_id is None or current_id not in batch.ids:
            return _error_message(request, _ORDER_ERROR)
        result, _pairing_valid = await self._run_handler(request, handler)
        return result

    async def _run_sequential(
        self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
        batch: _BatchInfo,
    ) -> ToolMessage | Command[Any]:
        """Serialize a sequential batch in model output order."""

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
                result, pairing_valid = await self._run_handler(request, handler)
            except asyncio.CancelledError:
                # A cancellation aborts the active batch.  Keep the original
                # exception so the harness can perform its normal settlement.
                self._failed = True
                raise
            if not pairing_valid:
                self._failed = True
            else:
                # Ordinary ToolMessage(status="error") and Command updates
                # still consume this call position; only malformed pairing
                # stops the remaining batch.
                self._next_index += 1
            return result
        finally:
            self._lock.release()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> ToolMessage | Command[Any]:
        batch = _batch_info(request)
        if batch.sequential:
            return await self._run_sequential(request, handler, batch)
        return await self._run_parallel(request, handler, batch)


# Keep the old public name as a compatibility alias for existing integrations.
SequentialToolCallMiddleware = ToolCallBatchMiddleware


__all__ = [
    "SEQUENTIAL_TOOL_EXECUTION_MODE",
    "SequentialToolCallMiddleware",
    "TOOL_EXECUTION_MODE_METADATA_KEY",
    "ToolCallBatchMiddleware",
]
