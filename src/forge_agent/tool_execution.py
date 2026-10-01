"""Conservative tool-call execution policy for Forge agents.

LangChain's ``ToolNode`` gathers a model's tool calls concurrently. Forge
keeps LangChain's agent loop and adds Pi's batch-level sequential escape hatch
around native tool handlers. The middleware deliberately keeps only the
currently active batch: the model's latest ``AIMessage`` is the source of
truth, while ``ToolMessage`` remains the result/pairing fact.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from time import monotonic
from typing import Any, Literal, cast

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command

from forge_agent.events import ToolExecutionEndEvent, ToolExecutionStartEvent
from forge_agent.retry import redact_model_error
from forge_agent.tools import AgentToolResult, ToolCall
from forge_agent.types import JSONValue

_ERROR_MAX_BYTES = 256
_BATCH_INVALID_ERROR = "Tool batch rejected; no tool was executed."
_ORDER_ERROR = "Tool batch order was invalid; no tool was executed."
_EXECUTION_ERROR = "Tool execution failed."
_SKIPPED_ERROR = "Skipped because the tool-call batch was invalid or cancelled."
TOOL_EXECUTION_MODE_METADATA_KEY = "forge.execution_mode"
SEQUENTIAL_TOOL_EXECUTION_MODE = "sequential"
TOOL_EXPOSURE_METADATA_KEY = "forge.exposure"
NESTED_CALLS_METADATA_KEY = "forge.nested_calls.v1"
type ToolExposure = Literal["direct", "model-only", "codemode", "deferred", "hidden"]
_EXPOSURES = {"direct", "model-only", "codemode", "deferred", "hidden"}
_MODEL_ONLY_NAMES = {
    "codemode",
    "tool_search",
    "task",
    "ask_user_question",
    "write_todos",
    "goal_complete",
    "goal_blocked",
}
_nested_executor: ContextVar[NestedToolExecutor | None] = ContextVar(
    "forge_nested_tool_executor", default=None
)


ToolCallHandler = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]


def tool_exposure(tool: BaseTool) -> ToolExposure:
    """Read the shared policy; control tools can never become callable children."""
    value = (tool.metadata or {}).get(
        TOOL_EXPOSURE_METADATA_KEY,
        "model-only" if tool.name in _MODEL_ONLY_NAMES else "direct",
    )
    if not isinstance(value, str) or value not in _EXPOSURES:
        raise ValueError("Invalid Forge tool exposure")
    if tool.name in _MODEL_ONLY_NAMES and value not in {"model-only", "hidden"}:
        raise ValueError("Control tools must remain model-only or hidden")
    return cast(ToolExposure, value)


def get_nested_tool_executor() -> NestedToolExecutor:
    """Obtain the current parent-owned native execution boundary."""
    executor = _nested_executor.get()
    if executor is None:
        raise RuntimeError("Nested execution requires an active native tool call")
    return executor


class ToolExposureMiddleware(AgentMiddleware[Any, Any, Any]):
    """Declare the initial view while ToolNode keeps the registered native tools."""

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        for tool in request.tools:
            if isinstance(tool, BaseTool):
                metadata = dict(tool.metadata or {})
                metadata["forge.declared"] = (
                    (
                        tool_exposure(tool) in {"direct", "model-only"}
                        or (
                            tool_exposure(tool) == "deferred"
                            and metadata.get("forge.loaded") is True
                        )
                    )
                    and metadata.get("forge.codemode_only") is not True
                    and metadata.get("forge.available") is not False
                )
                tool.metadata = metadata
        tools = [
            tool
            for tool in request.tools
            if not isinstance(tool, BaseTool) or (tool.metadata or {}).get("forge.declared") is True
        ]
        return await handler(request.override(tools=tools))


async def execute_tool_call(
    request: ToolCallRequest,
    handler: ToolCallHandler,
    *,
    nested: bool = False,
) -> tuple[ToolMessage | Command[Any], bool]:
    """Validate one native result, independently of model batch ordering."""
    if not _request_id(request) or not isinstance(request.tool_call.get("args"), dict):
        return _error_message(request, _ORDER_ERROR), False
    if isinstance(request.tool, BaseTool):
        try:
            exposure = tool_exposure(request.tool)
        except ValueError:
            return _error_message(request, "Invalid tool execution policy."), False
        forbidden = exposure in ({"hidden", "model-only"} if nested else {"hidden", "codemode"})
        if not nested:
            metadata = request.tool.metadata or {}
            forbidden = (
                forbidden
                or metadata.get("forge.codemode_only") is True
                or (exposure == "deferred" and metadata.get("forge.declared") is not True)
            )
        if (
            forbidden
            or (request.tool.metadata or {}).get("forge.available") is False
            or (nested and request.tool.return_direct)
        ):
            return _error_message(request, "Tool is not allowed in this execution context."), True
    try:
        result = await handler(request)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - do not expose raw tool failures
        return _error_message(request, _EXECUTION_ERROR), True
    if isinstance(result, Command):
        if nested:
            return _error_message(request, "State-changing tools require a model call."), False
        return result, True
    if not isinstance(result, ToolMessage) or result.tool_call_id != _request_id(request):
        return _error_message(request, _ORDER_ERROR), False
    if isinstance(result.artifact, Mapping) and result.artifact.get("forge.status") == "error":
        result = result.model_copy(update={"status": "error"})
    if isinstance(result.artifact, Mapping) and result.artifact.get("ok") is False:
        try:
            AgentToolResult.model_validate(result.artifact)
        except ValueError:
            pass
        else:
            result = result.model_copy(update={"status": "error"})
    return result, True


def nested_call_records(artifact: object) -> list[dict[str, Any]]:
    """Read only the bounded display contract from potentially imported artifacts."""
    if not isinstance(artifact, Mapping):
        return []
    envelope = artifact.get(NESTED_CALLS_METADATA_KEY)
    if not isinstance(envelope, Mapping) or not isinstance(envelope.get("calls"), list):
        return []
    records: list[dict[str, Any]] = []
    remaining = 32768
    for raw in envelope["calls"][:256]:
        if not isinstance(raw, Mapping):
            continue
        if not isinstance(raw.get("name"), str) or not isinstance(raw.get("id"), str):
            continue
        args = raw.get("args")
        preview = _argument_preview(args, min(8192, remaining)) if isinstance(args, Mapping) else {}
        remaining -= len(json.dumps(preview, ensure_ascii=False).encode("utf-8")) if preview else 0
        duration = raw.get("duration_ms", 0)
        records.append(
            {
                "id": raw["id"][:256],
                "name": raw["name"][:128],
                "args": preview,
                "status": "success" if raw.get("status") == "success" else "error",
                "duration_ms": max(0, duration) if isinstance(duration, int) else 0,
                "error": redact_model_error(str(raw.get("error", "")), limit=256),
            }
        )
    return records


def _argument_preview(arguments: Mapping[str, JSONValue], budget: int) -> dict[str, JSONValue]:
    """Keep a small allowlist of operational arguments; omit payloads and secrets."""
    # Content, shell commands and arbitrary MCP fields are never durable trace data.
    preview = {
        key: value
        for key, value in arguments.items()
        if key in {"path", "offset", "limit"} and isinstance(value, (str, int))
    }
    encoded = json.dumps(preview, ensure_ascii=False).encode("utf-8")
    return cast(dict[str, JSONValue], preview) if len(encoded) <= budget else {}


class NestedToolExecutor:
    """Parent-scoped calls through public ToolNode injection, with bounded traces.

    This is a tool dispatcher, not an agent loop. It does not run models, append
    messages or reacquire the outer batch lock. Close drains all owned children.
    """

    def __init__(self, runtime: ToolRuntime, parent_id: str) -> None:
        self.runtime = runtime
        self.parent_id = parent_id
        self._tools = {tool.name: tool for tool in runtime.tools}
        self._node: ToolNode | None = None
        self._count = 0
        self._closed = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._slots = asyncio.Semaphore(32)
        self._preview_bytes = 0
        self._complete = True
        self._records: list[dict[str, JSONValue]] = []

    async def _wrap(
        self, request: ToolCallRequest, handler: ToolCallHandler
    ) -> ToolMessage | Command[Any]:
        # ToolNode owns injection; only trusted parent state replaces the empty
        # standalone-call state. No private ToolNode method is called here.
        request = ToolCallRequest(
            tool_call=request.tool_call,
            tool=request.tool,
            state=self.runtime.state,
            runtime=replace(request.runtime, state=self.runtime.state),
        )
        result, _ = await execute_tool_call(request, handler, nested=True)
        return result

    def _emit(self, event: ToolExecutionStartEvent | ToolExecutionEndEvent) -> None:
        self.runtime.stream_writer(
            {"kind": "forge_nested_tool", "event": event.model_dump(mode="json")}
        )

    async def call(self, name: str, arguments: Mapping[str, JSONValue]) -> ToolMessage:
        """Execute one child; ordinary failures remain paired and isolated."""
        if self._closed:
            raise RuntimeError("Nested execution is closed")
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError("Nested tool name is invalid")
        if self._count >= 256:
            self._complete = False
            raise ValueError("Nested tool-call budget exhausted")
        self._count += 1
        call_id = f"{self.parent_id}/{self._count}"
        started = monotonic()
        preview = _argument_preview(arguments, min(8192, 32768 - self._preview_bytes))
        self._preview_bytes += (
            len(json.dumps(preview, ensure_ascii=False).encode("utf-8")) if preview else 0
        )
        if not preview and any(key in arguments for key in {"path", "offset", "limit"}):
            self._complete = False
        record: dict[str, JSONValue] = {
            "id": call_id,
            "name": name,
            "args": preview,
            "status": "running",
        }
        self._records.append(record)
        self._emit(
            ToolExecutionStartEvent(
                tool_call=ToolCall(id=call_id, name=name), parent_tool_call_id=self.parent_id
            )
        )
        current = asyncio.current_task()
        if current is not None:
            self._tasks.add(current)
        result = ToolMessage(
            content=_EXECUTION_ERROR, tool_call_id=call_id, name=name, status="error"
        )
        try:
            encoded = json.dumps(dict(arguments), allow_nan=False).encode("utf-8")
            if len(encoded) > 65536:
                raise ValueError("Nested arguments exceeded budget")
            if name not in self._tools:
                result = result.model_copy(update={"content": "Tool is not registered."})
            else:
                if self._node is None:
                    self._node = ToolNode(
                        list(self._tools.values()),
                        handle_tool_errors=False,
                        awrap_tool_call=self._wrap,
                    )
                config = cast(RunnableConfig, {**self.runtime.config, "callbacks": []})
                runtime = Runtime(
                    context=self.runtime.context,
                    store=self.runtime.store,
                    stream_writer=self.runtime.stream_writer,
                    execution_info=self.runtime.execution_info,
                    server_info=self.runtime.server_info,
                )
                async with self._slots:
                    if self._closed:
                        raise asyncio.CancelledError
                    output = await self._node.ainvoke(
                        [
                            {
                                "id": call_id,
                                "name": name,
                                "args": dict(arguments),
                                "type": "tool_call",
                            }
                        ],
                        config=config,
                        runtime=runtime,
                    )
                if isinstance(output, Mapping) and isinstance(output.get("messages"), list):
                    items = output["messages"]
                    if len(items) == 1 and isinstance(items[0], ToolMessage):
                        result = items[0]
        except asyncio.CancelledError:
            result = result.model_copy(update={"content": "Nested tool cancelled."})
            raise
        except Exception:  # noqa: BLE001 - bounded model-facing error
            pass
        finally:
            if current is not None:
                self._tasks.discard(current)
            record["status"] = result.status
            record["duration_ms"] = round((monotonic() - started) * 1000)
            if result.status == "error":
                record["error"] = redact_model_error(str(result.content), limit=256)
            self._emit(
                ToolExecutionEndEvent(
                    result=AgentToolResult(
                        tool_call_id=call_id,
                        name=name,
                        ok=result.status != "error",
                        content="",
                        details={"duration_ms": record["duration_ms"]},
                    ),
                    parent_tool_call_id=self.parent_id,
                )
            )
        return result

    async def close(self) -> None:
        """Reject new calls and drain every task owned by this parent."""
        self._closed = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def artifact(self) -> dict[str, JSONValue]:
        """Return metadata only; child result payloads remain ephemeral."""
        return {
            "calls": cast(list[JSONValue], self._records),
            "total": self._count,
            "complete": self._complete,
        }


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
        nested = (
            NestedToolExecutor(request.runtime, current_id)
            if isinstance(request.runtime, ToolRuntime) and current_id
            else None
        )
        token = _nested_executor.set(nested)
        try:
            result, valid = await execute_tool_call(request, handler)
        finally:
            try:
                if nested is not None:
                    await nested.close()
            finally:
                _nested_executor.reset(token)
        if nested is not None and nested._count and isinstance(result, ToolMessage):
            artifact = (
                dict(result.artifact)
                if isinstance(result.artifact, Mapping)
                else {"forge.original_artifact": result.artifact}
                if result.artifact is not None
                else {}
            )
            artifact[NESTED_CALLS_METADATA_KEY] = nested.artifact()
            result = result.model_copy(update={"artifact": artifact})
        return result, valid

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
    "NestedToolExecutor",
    "NESTED_CALLS_METADATA_KEY",
    "TOOL_EXPOSURE_METADATA_KEY",
    "ToolExposure",
    "ToolExposureMiddleware",
    "execute_tool_call",
    "get_nested_tool_executor",
    "tool_exposure",
    "nested_call_records",
]
