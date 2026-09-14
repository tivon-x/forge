"""Small, generic subagent execution primitives.

The parent Forge agent owns the conversation transcript.  A subagent is a
fresh LangChain agent invocation with one new ``HumanMessage``; no parent
messages or checkpointer are passed to the child.  This module deliberately
contains no coding-session or UI concerns so it can be reused by those
layers.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
    ModelCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException

from forge_agent.context import ForgeRuntimeContext
from forge_agent.message_codec import message_text
from forge_agent.tool_execution import ToolCallBatchMiddleware
from forge_agent.types import JSONValue

DEFAULT_MAX_MODEL_CALLS = 8
DEFAULT_MAX_RESULT_BYTES = 50 * 1024

# Trace budgets are deliberately independent from the task result budget.  A
# trace is a display projection, not a provider/tool transcript and therefore
# gets one small, fixed envelope regardless of the role's result limit.
TRACE_ITEM_MAX_BYTES = 8 * 1024
TRACE_MAX_ITEMS = 64
TRACE_MAX_BYTES = 64 * 1024

SubagentStatus = Literal["completed", "failed"]
TraceKind = Literal["human", "assistant", "tool_call", "tool_result", "omitted"]
TraceStatus = Literal["ok", "error"]


@dataclass(frozen=True, slots=True)
class SubagentUsageFact:
    """Private per-AI-message usage snapshot for the parent ledger.

    This DTO intentionally contains no child text, prompts, tool arguments,
    tool results, or trace metadata.  It is transported only through the
    nested runtime projection and is never included in the task artifact.
    """

    response_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "response_model": self.response_model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
        }


def _usage_token(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def project_subagent_usage(messages: Iterable[object]) -> tuple[SubagentUsageFact, ...]:
    """Project one bounded fact per child AI response."""

    facts: list[SubagentUsageFact] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = getattr(message, "usage_metadata", None)
        usage_map = usage if isinstance(usage, Mapping) else {}
        details = usage_map.get("input_token_details")
        detail_map = details if isinstance(details, Mapping) else {}
        response_metadata = getattr(message, "response_metadata", None)
        response_map = response_metadata if isinstance(response_metadata, Mapping) else {}
        response_model = next(
            (
                value.strip()
                for key in ("response_model", "model_name", "model", "model_id")
                if isinstance(value := response_map.get(key), str) and value.strip()
            ),
            None,
        )
        facts.append(
            SubagentUsageFact(
                response_model=response_model,
                input_tokens=_usage_token(usage_map.get("input_tokens")),
                output_tokens=_usage_token(usage_map.get("output_tokens")),
                cache_read_tokens=_usage_token(
                    detail_map.get("cache_read")
                    if "cache_read" in detail_map
                    else detail_map.get("cacheRead")
                ),
                cache_write_tokens=_usage_token(
                    detail_map.get("cache_creation")
                    if "cache_creation" in detail_map
                    else detail_map.get("cache_write", detail_map.get("cacheWrite"))
                ),
                total_tokens=_usage_token(usage_map.get("total_tokens")),
            )
        )
        if len(facts) >= DEFAULT_MAX_MODEL_CALLS:
            break
    return tuple(facts)


_HIDDEN_TRACE_TOKEN = re.compile(
    r"<(?P<close>/)?(?P<name>think|thinking|reasoning|analysis)(?:\s[^>]*)?>",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SubagentTraceItem:
    """Safe, JSON-friendly display item for one child execution.

    The DTO intentionally has no message metadata, tool-call ids, arguments,
    result content, or provider fields.  Construction and deserialisation are
    strict so a malformed persisted item cannot silently become a different
    kind of activity.
    """

    kind: TraceKind
    text: str | None = None
    tool: str | None = None
    status: TraceStatus | None = None
    omitted: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"human", "assistant", "tool_call", "tool_result", "omitted"}:
            raise ValueError(f"unknown subagent trace item kind: {self.kind!r}")
        if self.omitted < 0:
            raise ValueError("trace omitted count must be non-negative")
        if type(self.omitted) is not int:
            raise TypeError("trace omitted count must be an integer")
        if self.kind == "omitted":
            if self.omitted <= 0:
                raise ValueError("omitted trace item must omit at least one item")
            if self.text is not None or self.tool is not None or self.status is not None:
                raise ValueError("omitted trace item cannot carry text, tool, or status")
            return

        for name, value in (("text", self.text), ("tool", self.tool)):
            if value is not None and len(value.encode("utf-8")) > TRACE_ITEM_MAX_BYTES:
                raise ValueError(f"trace item {name} exceeds UTF-8 byte budget")
        if self.omitted:
            raise ValueError("only omitted trace items may carry an omitted count")
        if self.kind in {"human", "assistant"}:
            if not isinstance(self.text, str):
                raise TypeError(f"{self.kind} trace item requires text")
            if self.tool is not None or self.status is not None:
                raise ValueError(f"{self.kind} trace item cannot carry tool or status")
            return
        if not isinstance(self.tool, str) or not self.tool:
            raise TypeError(f"{self.kind} trace item requires a tool name")
        if self.kind == "tool_call":
            if not isinstance(self.text, str) or not self.text:
                raise TypeError("tool_call trace item requires a fixed summary")
            if self.status is not None:
                raise ValueError("tool_call trace item cannot carry status")
        else:
            if self.text is not None:
                raise ValueError("tool_result trace item cannot carry text")
            if self.status not in {"ok", "error"}:
                raise TypeError("tool_result trace item requires ok/error status")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialise this item while omitting ``None`` and zero fields."""

        value: dict[str, JSONValue] = {"kind": self.kind}
        if self.text is not None:
            value["text"] = self.text
        if self.tool is not None:
            value["tool"] = self.tool
        if self.status is not None:
            value["status"] = self.status
        if self.omitted:
            value["omitted"] = self.omitted
        return value

    @classmethod
    def from_dict(cls, value: object) -> SubagentTraceItem:
        """Strictly load one JSON trace item.

        Unknown fields are rejected rather than ignored: silently accepting a
        field such as ``args`` or ``artifact`` would make it too easy for an
        unsafe producer to persist raw child data by accident.
        """

        if not isinstance(value, Mapping):
            raise TypeError("subagent trace item must be an object")
        allowed = {"kind", "text", "tool", "status", "omitted"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown subagent trace item field(s): {', '.join(sorted(map(str, unknown)))}"
            )
        kind = value.get("kind")
        if not isinstance(kind, str):
            raise TypeError("trace item kind must be a string")
        text = value.get("text")
        if text is not None and not isinstance(text, str):
            raise TypeError("trace item text must be a string or null")
        tool = value.get("tool")
        if tool is not None and not isinstance(tool, str):
            raise TypeError("trace item tool must be a string or null")
        status = value.get("status")
        if status is not None and not isinstance(status, str):
            raise TypeError("trace item status must be a string or null")
        omitted = value.get("omitted", 0)
        if type(omitted) is not int:
            raise TypeError("trace item omitted must be an integer")
        return cls(
            kind=cast(TraceKind, kind),
            text=text,
            tool=tool,
            status=cast(TraceStatus | None, status),
            omitted=omitted,
        )


@dataclass(frozen=True, slots=True)
class SubagentTrace:
    """Bounded child trace envelope used by live events and CustomEntry."""

    agent: str
    items: tuple[SubagentTraceItem, ...] = ()
    truncated: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent, str):
            raise TypeError("trace agent must be a string")
        if len(self.agent.encode("utf-8")) > TRACE_ITEM_MAX_BYTES:
            raise ValueError("trace agent exceeds UTF-8 byte budget")
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items))
        for value in self.items:
            if not isinstance(value, SubagentTraceItem):
                raise TypeError("trace items must be SubagentTraceItem values")
        if type(self.truncated) is not bool:
            raise TypeError("trace truncated must be a boolean")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            token_count = getattr(self, name)
            if token_count is not None and (type(token_count) is not int or token_count < 0):
                raise TypeError(f"trace {name} must be a non-negative integer or null")
        if len(self.items) > TRACE_MAX_ITEMS:
            raise ValueError("subagent trace exceeds item budget")
        if _trace_json_bytes(self) > TRACE_MAX_BYTES:
            raise ValueError("subagent trace exceeds serialized byte budget")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialise the core trace fields (without event kind/version/id)."""

        return {
            "agent": self.agent,
            "items": [item.to_dict() for item in self.items],
            "truncated": self.truncated,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, value: object) -> SubagentTrace:
        """Strictly load a trace core payload.

        The caller should remove the event-only ``kind``, ``version`` and
        ``tool_call_id`` fields first.  This keeps the DTO reusable for both
        live updates and persisted CustomEntry payloads.
        """

        if not isinstance(value, Mapping):
            raise TypeError("subagent trace must be an object")
        allowed = {
            "agent",
            "items",
            "truncated",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown subagent trace field(s): {', '.join(sorted(map(str, unknown)))}"
            )
        agent = value.get("agent")
        if not isinstance(agent, str):
            raise TypeError("trace agent must be a string")
        raw_items = value.get("items")
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
            raise TypeError("trace items must be an array")
        items = tuple(SubagentTraceItem.from_dict(item) for item in raw_items)
        truncated = value.get("truncated", False)
        if type(truncated) is not bool:
            raise TypeError("trace truncated must be a boolean")

        def token(name: str) -> int | None:
            raw = value.get(name)
            if raw is None:
                return None
            if type(raw) is not int or raw < 0:
                raise TypeError(f"trace {name} must be a non-negative integer or null")
            return raw

        return cls(
            agent=agent,
            items=items,
            truncated=truncated,
            input_tokens=token("input_tokens"),
            output_tokens=token("output_tokens"),
            total_tokens=token("total_tokens"),
        )

    # A small alias is useful at call sites that distinguish JSON projection
    # from ordinary dictionary conversion.
    to_json = to_dict


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-neutral aggregate of standard LangChain usage fields."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def aggregate_usage(messages: Iterable[object]) -> TokenUsage:
    """Sum non-negative standard usage fields from native ``AIMessage`` rows.

    Provider-specific response metadata and malformed/negative values are
    ignored.  A field remains ``None`` when no message supplied a valid value;
    no synthetic total is calculated from input/output fields.
    """

    totals: dict[str, int] = {}
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = message.usage_metadata
        if not isinstance(usage, Mapping):
            continue
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                totals[key] = totals.get(key, 0) + value
                seen.add(key)
    return TokenUsage(
        input_tokens=totals.get("input_tokens") if "input_tokens" in seen else None,
        output_tokens=totals.get("output_tokens") if "output_tokens" in seen else None,
        total_tokens=totals.get("total_tokens") if "total_tokens" in seen else None,
    )


def _trace_json_bytes(trace: SubagentTrace) -> int:
    """Return the UTF-8 size of the core trace JSON envelope."""

    return _json_value_utf8_bytes(trace.to_dict())


def _visible_message_text(message: BaseMessage) -> str:
    """Extract visible text while dropping reasoning/thinking blocks."""

    strip_hidden = isinstance(message, AIMessage)

    def visible(value: str) -> str:
        return _strip_hidden_trace_tags(value) if strip_hidden else value

    content = message.content
    if isinstance(content, str):
        return visible(content)
    parts: list[str] = []
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return ""
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, Mapping):
            continue
        block_type = str(block.get("type", "")).lower()
        if block_type != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return visible("".join(parts))


def _strip_hidden_trace_tags(value: str) -> str:
    """Drop common inline reasoning containers from otherwise visible text."""
    visible: list[str] = []
    hidden_stack: list[str] = []
    cursor = 0
    for match in _HIDDEN_TRACE_TOKEN.finditer(value):
        if not hidden_stack:
            visible.append(value[cursor : match.start()])
        name = match.group("name").lower()
        if match.group("close"):
            if hidden_stack and hidden_stack[-1] == name:
                hidden_stack.pop()
        else:
            hidden_stack.append(name)
        cursor = match.end()
    if not hidden_stack:
        visible.append(value[cursor:])
    return "".join(visible)


def _bounded_trace_text(value: str) -> tuple[str, bool]:
    return _truncate_utf8(value, TRACE_ITEM_MAX_BYTES)


def _trace_tool_summary(tool: str) -> str:
    """Return a fixed summary that never interpolates tool arguments."""

    bounded_tool, _ = _bounded_trace_text(tool or "tool")
    if not bounded_tool:
        bounded_tool = "tool"
    if bounded_tool == "bash":
        return "Running bash"
    summary, _ = _bounded_trace_text(f"Calling {bounded_tool}")
    return summary or "Calling tool"


def _trace_item_from_message(
    message: object,
    tool_names: dict[str, str],
) -> tuple[list[SubagentTraceItem], bool]:
    """Project one native child message into zero or more safe items."""

    if isinstance(message, HumanMessage):
        text, truncated = _bounded_trace_text(_visible_message_text(message))
        return [SubagentTraceItem(kind="human", text=text)], truncated
    if isinstance(message, AIMessage):
        result: list[SubagentTraceItem] = []
        truncated = False
        visible = _visible_message_text(message)
        if visible:
            text, text_truncated = _bounded_trace_text(visible)
            truncated = truncated or text_truncated
            if text:
                result.append(SubagentTraceItem(kind="assistant", text=text))
        for call in message.tool_calls:
            if not isinstance(call, Mapping):
                continue
            raw_id = call.get("id")
            raw_name = call.get("name")
            call_id = raw_id if isinstance(raw_id, str) and raw_id else None
            tool = raw_name if isinstance(raw_name, str) and raw_name else "tool"
            bounded_tool, tool_truncated = _bounded_trace_text(tool)
            truncated = truncated or tool_truncated
            if not bounded_tool:
                bounded_tool = "tool"
            if call_id is not None:
                # The id is only an in-memory key.  It never reaches the DTO.
                tool_names[call_id] = bounded_tool
            result.append(
                SubagentTraceItem(
                    kind="tool_call",
                    tool=bounded_tool,
                    text=_trace_tool_summary(bounded_tool),
                )
            )
        return result, truncated
    if isinstance(message, ToolMessage):
        raw_id = message.tool_call_id
        result_tool: str | None = tool_names.get(raw_id) if isinstance(raw_id, str) else None
        if not result_tool:
            result_name = message.name
            result_tool = result_name if isinstance(result_name, str) and result_name else "tool"
        bounded_result_tool, truncated = _bounded_trace_text(result_tool)
        result_tool = bounded_result_tool or "tool"
        status: TraceStatus = "error" if message.status == "error" else "ok"
        return [SubagentTraceItem(kind="tool_result", tool=result_tool, status=status)], truncated
    return [], False


def _trace_payload_bytes(
    agent: str,
    items: Sequence[SubagentTraceItem],
    truncated: bool,
    usage: TokenUsage,
) -> int:
    payload: dict[str, JSONValue] = {
        "agent": agent,
        "items": [item.to_dict() for item in items],
        "truncated": truncated,
        **usage.to_dict(),
    }
    return _json_value_utf8_bytes(payload)


def _fit_trace_usage(agent: str, usage: TokenUsage) -> TokenUsage:
    """Drop optional usage when its JSON representation cannot fit safely."""
    if _trace_payload_bytes(agent, (), False, usage) <= TRACE_MAX_BYTES:
        return usage
    return TokenUsage()


def _fit_trace_items(
    *,
    agent: str,
    items: list[SubagentTraceItem],
    usage: TokenUsage,
    already_truncated: bool = False,
) -> tuple[tuple[SubagentTraceItem, ...], bool]:
    """Apply item-count and aggregate UTF-8 budgets to projected items."""

    if not items:
        return (), False
    truncated = already_truncated
    first_human = next((index for index, item in enumerate(items) if item.kind == "human"), None)
    last_assistant = next(
        (
            index
            for index in range(len(items) - 1, -1, -1)
            if items[index].kind == "assistant" and bool(items[index].text)
        ),
        None,
    )
    selected: set[int] = set(range(len(items)))
    protected = {index for index in (first_human, last_assistant) if index is not None}

    # Keep the first instruction, final visible answer, and the newest
    # activity.  One slot is reserved for the omitted marker whenever rows
    # need to be removed.
    if len(selected) > TRACE_MAX_ITEMS:
        selected = set(protected)
        for index in range(len(items) - 1, -1, -1):
            if index in selected:
                continue
            if len(selected) >= TRACE_MAX_ITEMS - 1:
                break
            selected.add(index)
        truncated = True

    def build(current: set[int]) -> list[SubagentTraceItem]:
        omitted = len(items) - len(current)
        ordered = [items[index] for index in sorted(current)]
        if omitted <= 0:
            return ordered
        marker = SubagentTraceItem(kind="omitted", omitted=omitted)
        first_position = next(
            (position for position, index in enumerate(sorted(current)) if index == first_human),
            None,
        )
        if first_position is None:
            return [marker, *ordered]
        return [*ordered[: first_position + 1], marker, *ordered[first_position + 1 :]]

    candidate = build(selected)
    # The aggregate budget is measured on the same core envelope emitted by
    # ``SubagentTrace.to_dict``.  Remove oldest non-protected rows first so the
    # tail remains the most useful progressive activity.
    while _trace_payload_bytes(agent, candidate, True, usage) > TRACE_MAX_BYTES:
        removable = [index for index in sorted(selected) if index not in protected]
        if not removable:
            break
        selected.remove(removable[0])
        truncated = True
        candidate = build(selected)

    # A candidate with only the protected rows is comfortably below 64 KiB in
    # normal operation.  If an unusually large role name or metadata still
    # leaves no room, shrink visible text without ever exposing another field.
    if _trace_payload_bytes(agent, candidate, True, usage) > TRACE_MAX_BYTES:
        mutable_positions = [
            index
            for index, item in enumerate(candidate)
            if item.kind in {"human", "assistant", "tool_call"} and item.text
        ]
        for position in mutable_positions:
            item = candidate[position]
            assert item.text is not None
            low, high, best = 0, len(item.text.encode("utf-8")), ""
            while low <= high:
                mid = (low + high) // 2
                text, _ = _truncate_utf8(item.text, mid)
                trial = list(candidate)
                trial[position] = SubagentTraceItem(
                    kind=item.kind,
                    text=text,
                    tool=item.tool,
                    status=item.status,
                )
                if _trace_payload_bytes(agent, trial, True, usage) <= TRACE_MAX_BYTES:
                    best = text
                    low = mid + 1
                else:
                    high = mid - 1
            if best != item.text:
                candidate[position] = SubagentTraceItem(
                    kind=item.kind,
                    text=best,
                    tool=item.tool,
                    status=item.status,
                )
                truncated = True
            if _trace_payload_bytes(agent, candidate, True, usage) <= TRACE_MAX_BYTES:
                break

    return tuple(candidate[:TRACE_MAX_ITEMS]), truncated


def project_subagent_trace(
    messages: Iterable[object],
    *,
    agent: str = "subagent",
    usage: TokenUsage | None = None,
) -> SubagentTrace:
    """Project a child native message snapshot into a bounded safe trace."""

    snapshot = tuple(messages)
    raw_items: list[SubagentTraceItem] = []
    item_truncated = False
    tool_names: dict[str, str] = {}
    for message in snapshot:
        projected, truncated = _trace_item_from_message(message, tool_names)
        raw_items.extend(projected)
        item_truncated = item_truncated or truncated
    effective_usage = usage or aggregate_usage(snapshot)
    bounded_agent, _ = _bounded_trace_text(agent or "subagent")
    if not bounded_agent:
        bounded_agent = "subagent"
    effective_usage = _fit_trace_usage(bounded_agent, effective_usage)
    bounded_items, truncated = _fit_trace_items(
        agent=bounded_agent,
        items=raw_items,
        usage=effective_usage,
        already_truncated=item_truncated,
    )
    return SubagentTrace(
        agent=bounded_agent,
        items=bounded_items,
        truncated=truncated,
        input_tokens=effective_usage.input_tokens,
        output_tokens=effective_usage.output_tokens,
        total_tokens=effective_usage.total_tokens,
    )


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    """Definition of one role available to a :class:`SubagentRunner`."""

    name: str
    description: str
    system_prompt: str
    tools: Sequence[BaseTool] = ()
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES


@dataclass(frozen=True, slots=True)
class SubagentRuntime:
    """The provider and Forge context used for one child invocation."""

    provider: BaseChatModel
    model: str = ""
    runtime_context: ForgeRuntimeContext | None = None


@dataclass(frozen=True, slots=True)
class SubagentRunResult:
    """Stable result returned by a subagent task tool.

    The parent ToolMessage artifact is version 2.  ``from_artifact`` accepts
    both v1 and v2 rows so old sessions remain readable.
    """

    agent: str
    status: SubagentStatus
    instruction: str
    final_output: str
    model_calls: int
    tool_calls: int
    queued_ms: int
    duration_ms: int
    truncated: bool = False
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # Private runtime facts; intentionally omitted from ``to_artifact``.
    usage_facts: tuple[SubagentUsageFact, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _max_result_bytes: int = field(
        default=DEFAULT_MAX_RESULT_BYTES,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def content(self) -> str:
        """Return the compact text that should be shown to the parent model."""

        if self.status == "failed":
            if self.error:
                content = f"Subagent {self.agent} failed: {self.error}"
            else:
                content = f"Subagent {self.agent} failed."
        elif self.final_output:
            content = self.final_output
        else:
            content = "Subagent completed without a final response."
        return _truncate_utf8(content, self._max_result_bytes)[0]

    def to_artifact(self) -> dict[str, JSONValue]:
        """Return the JSON-safe v2 artifact persisted in the parent ToolMessage."""

        artifact: dict[str, JSONValue] = {
            "kind": "subagent_run",
            "version": 2,
            "agent": self.agent,
            "status": self.status,
            "instruction": self.instruction,
            "final_output": self.final_output,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "queued_ms": self.queued_ms,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "error": self.error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        bounded = _fit_artifact_budget(artifact, self._max_result_bytes)
        if bounded != artifact:
            bounded["truncated"] = True
        return bounded

    def artifact(self) -> dict[str, JSONValue]:
        """Alias for :meth:`to_artifact` used by tool adapters."""

        return self.to_artifact()

    @classmethod
    def from_artifact(cls, value: object) -> SubagentRunResult:
        """Load a v1 or v2 artifact without accepting unknown versions."""

        if not isinstance(value, Mapping):
            raise TypeError("subagent artifact must be an object")
        if value.get("kind") != "subagent_run":
            raise ValueError("not a subagent_run artifact")
        version = value.get("version")
        if type(version) is not int or version not in {1, 2}:
            raise ValueError("unsupported subagent artifact version")

        def text(name: str, *, required: bool = False) -> str:
            raw = value.get(name)
            if isinstance(raw, str):
                return raw
            if required:
                raise TypeError(f"subagent artifact {name} must be a string")
            return ""

        def integer(name: str) -> int:
            raw = value.get(name)
            if type(raw) is not int or raw < 0:
                raise TypeError(f"subagent artifact {name} must be a non-negative integer")
            return raw

        status = value.get("status")
        if status not in {"completed", "failed"}:
            raise TypeError("subagent artifact status is invalid")
        error = value.get("error")
        if error is not None and not isinstance(error, str):
            raise TypeError("subagent artifact error must be a string or null")
        truncated = value.get("truncated", False)
        if type(truncated) is not bool:
            raise TypeError("subagent artifact truncated must be a boolean")

        def optional_token(name: str) -> int | None:
            raw = value.get(name)
            if raw is None:
                return None
            if type(raw) is not int or raw < 0:
                raise TypeError(f"subagent artifact {name} must be a non-negative integer or null")
            return raw

        return cls(
            agent=text("agent", required=True),
            status=cast(SubagentStatus, status),
            instruction=text("instruction", required=True),
            final_output=text("final_output"),
            model_calls=integer("model_calls"),
            tool_calls=integer("tool_calls"),
            queued_ms=integer("queued_ms"),
            duration_ms=integer("duration_ms"),
            truncated=truncated,
            error=error,
            input_tokens=None if version == 1 else optional_token("input_tokens"),
            output_tokens=None if version == 1 else optional_token("output_tokens"),
            total_tokens=None if version == 1 else optional_token("total_tokens"),
        )


RuntimeReader = Callable[[], SubagentRuntime]


def _elapsed_ms(start: float) -> int:
    """Return a monotonic elapsed duration in whole milliseconds."""

    return max(0, int((monotonic() - start) * 1000))


def _truncate_utf8(value: str, max_bytes: int) -> tuple[str, bool]:
    """Trim ``value`` to ``max_bytes`` without splitting UTF-8 code points."""

    if max_bytes < 0:
        raise ValueError("max_result_bytes must be non-negative")
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False
    # ``errors='ignore'`` is safe here because the slice can only end in the
    # middle of a UTF-8 sequence; all complete code points are retained.
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _json_value_utf8_bytes(value: Mapping[str, JSONValue]) -> int:
    """Return JSON size, treating encoder integer limits as over budget."""
    try:
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    except (ValueError, OverflowError):
        return sys.maxsize


def _json_utf8_bytes(value: Mapping[str, JSONValue]) -> int:
    """Return the default JSON UTF-8 size of one artifact mapping."""

    return _json_value_utf8_bytes(value)


def _fit_artifact_string(
    artifact: dict[str, JSONValue],
    key: str,
    max_bytes: int,
) -> bool:
    """Keep the largest UTF-8-safe value for ``key`` within an artifact budget."""

    value = artifact.get(key)
    if not isinstance(value, str):
        return False
    encoded_length = len(value.encode("utf-8"))
    if encoded_length == 0:
        return False

    low = 0
    high = encoded_length
    best = ""
    while low <= high:
        candidate_bytes = (low + high) // 2
        candidate, _ = _truncate_utf8(value, candidate_bytes)
        trial = dict(artifact)
        trial[key] = candidate
        if _json_utf8_bytes(trial) <= max_bytes:
            best = candidate
            low = candidate_bytes + 1
        else:
            high = candidate_bytes - 1
    if best == value:
        return False
    artifact[key] = best
    return True


def _fit_artifact_budget(
    artifact: dict[str, JSONValue],
    max_bytes: int,
) -> dict[str, JSONValue]:
    """Bound user-controlled artifact strings while preserving the v1 shape.

    ``max_result_bytes`` is a byte budget for returned text.  For budgets large
    enough to hold the fixed v1 metadata, the serialized artifact is fitted to
    the same budget, sacrificing the duplicated instruction before preserving
    final output or errors, then the role name.  A very small budget cannot hold
    the fixed metadata at all; in that case the stable artifact shape wins and
    its user strings remain
    field-bounded rather than silently dropping the v1 fields.
    """

    bounded = dict(artifact)
    # The instruction is already visible in the parent task call, so sacrifice
    # its duplicate first and preserve the final answer/error content.
    string_keys = ("instruction", "final_output", "error", "agent")
    for key in string_keys:
        value = bounded.get(key)
        if isinstance(value, str):
            bounded[key] = _truncate_utf8(value, max_bytes)[0]

    if _json_utf8_bytes(bounded) <= max_bytes:
        return bounded

    baseline = dict(bounded)
    # Usage is optional. Sacrifice it before user-visible strings whenever the
    # v2 envelope is over budget or an extreme integer cannot be serialized.
    usage_keys = ("input_tokens", "output_tokens", "total_tokens")
    for key in usage_keys:
        baseline.pop(key, None)
        bounded.pop(key, None)
    for key in string_keys:
        if isinstance(baseline.get(key), str):
            baseline[key] = ""
    if _json_utf8_bytes(baseline) > max_bytes:
        return bounded

    for key in string_keys:
        _fit_artifact_string(bounded, key, max_bytes)
        if _json_utf8_bytes(bounded) <= max_bytes:
            break
    return bounded


def _new_run_result(
    *,
    agent: str,
    status: SubagentStatus,
    instruction: str,
    final_output: str,
    model_calls: int,
    tool_calls: int,
    queued_ms: int,
    duration_ms: int,
    truncated: bool,
    error: str | None,
    max_result_bytes: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    usage_facts: tuple[SubagentUsageFact, ...] = (),
) -> SubagentRunResult:
    """Build a result whose public strings share one UTF-8 budget."""

    bounded_agent, agent_truncated = _truncate_utf8(agent, max_result_bytes)
    bounded_instruction, instruction_truncated = _truncate_utf8(instruction, max_result_bytes)
    bounded_output, output_truncated = _truncate_utf8(final_output, max_result_bytes)
    bounded_error: str | None = None
    error_truncated = False
    if error is not None:
        bounded_error, error_truncated = _truncate_utf8(error, max_result_bytes)

    result = SubagentRunResult(
        agent=bounded_agent,
        status=status,
        instruction=bounded_instruction,
        final_output=bounded_output,
        model_calls=model_calls,
        tool_calls=tool_calls,
        queued_ms=queued_ms,
        duration_ms=duration_ms,
        truncated=(
            truncated
            or agent_truncated
            or instruction_truncated
            or output_truncated
            or error_truncated
        ),
        error=bounded_error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        usage_facts=usage_facts,
    )
    object.__setattr__(result, "_max_result_bytes", max_result_bytes)

    # Fit the serialized artifact as a whole when the fixed v1 metadata can
    # fit in the requested budget.  Pull the fitted user strings back into the
    # result so content, fields, and artifact stay semantically aligned.
    bounded_artifact = result.to_artifact()
    changed = False
    for key in ("agent", "instruction", "final_output", "error"):
        value = bounded_artifact.get(key)
        if key == "error":
            if value != result.error:
                object.__setattr__(result, "error", value if isinstance(value, str) else None)
                changed = True
        elif isinstance(value, str) and value != getattr(result, key):
            object.__setattr__(result, key, value)
            changed = True
    if changed or bounded_artifact.get("truncated") is True:
        object.__setattr__(result, "truncated", True)
    return result


def _validate_spec(spec: SubagentSpec) -> None:
    if not spec.name.strip():
        raise ValueError("subagent name must not be empty")
    if spec.max_model_calls < 1:
        raise ValueError("max_model_calls must be at least 1")
    if spec.max_model_calls > DEFAULT_MAX_MODEL_CALLS:
        raise ValueError(f"max_model_calls must be at most {DEFAULT_MAX_MODEL_CALLS}")
    if spec.max_result_bytes < 0:
        raise ValueError("max_result_bytes must be non-negative")


def _messages_from_output(output: object) -> list[object]:
    """Extract the native child message list from an ``ainvoke`` result."""

    if not isinstance(output, Mapping):
        return []
    messages = output.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return []
    return list(messages)


class _SubagentUsageCollectorMiddleware(AgentMiddleware):
    """Capture allowlisted per-model usage before child execution can fail."""

    def __init__(self) -> None:
        self._facts: list[SubagentUsageFact] = []
        self._seen: set[str] = set()

    @property
    def facts(self) -> tuple[SubagentUsageFact, ...]:
        return tuple(self._facts)

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del runtime
        messages = state.get("messages", ()) if isinstance(state, Mapping) else ()
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
            return None
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            raw_id = getattr(message, "id", None)
            message_id = raw_id if isinstance(raw_id, str) and raw_id else f"object:{id(message)}"
            if message_id in self._seen:
                return None
            self._seen.add(message_id)
            facts = project_subagent_usage((message,))
            if facts:
                self._facts.extend(facts)
            return None
        return None


class SubagentRunner:
    """Registry and concurrent runner for stateless child LangChain agents."""

    def __init__(
        self,
        runtime_reader: RuntimeReader,
        specs: Sequence[SubagentSpec] = (),
    ) -> None:
        self._runtime_reader = runtime_reader
        self._specs: dict[str, SubagentSpec] = {}
        self.replace_specs(specs)

    @property
    def specs(self) -> tuple[SubagentSpec, ...]:
        """Return the current role registry in registration order."""

        return tuple(self._specs.values())

    def replace_specs(self, specs: Sequence[SubagentSpec]) -> None:
        """Replace the role registry atomically for future runs.

        Existing child invocations keep their already-selected spec.  The
        method is intentionally synchronous. A caller that exposes the runner
        through ``create_task_tool`` must recreate that tool after replacement
        so its model-facing registry description stays in sync.
        """

        replacement: dict[str, SubagentSpec] = {}
        for spec in specs:
            _validate_spec(spec)
            if spec.name in replacement:
                raise ValueError(f"duplicate subagent role: {spec.name}")
            replacement[spec.name] = spec
        self._specs = replacement

    async def run(self, agent: str, instruction: str) -> SubagentRunResult:
        """Run one fresh child agent and return its compact result.

        Invalid task input raises ``ToolException`` so the parent LangChain
        tool call follows the normal tool-error path.  Child/provider errors
        are ordinary failed results that the parent model can inspect.  A
        cancellation is never converted into a result and is re-raised.
        """

        spec = self._specs.get(agent) if isinstance(agent, str) else None
        if spec is None:
            raise ToolException(f"Unknown subagent role: {agent}")
        if not isinstance(instruction, str):
            raise ToolException("Subagent instruction must be a string")
        normalized_instruction = instruction.strip()
        if not normalized_instruction:
            raise ToolException("Subagent instruction must not be empty")

        queued_ms = 0
        run_start = monotonic()
        usage_collector = _SubagentUsageCollectorMiddleware()

        try:
            runtime = self._runtime_reader()
            child = cast(
                Any,
                create_agent(
                    runtime.provider,
                    tools=list(spec.tools),
                    system_prompt=spec.system_prompt,
                    middleware=[
                        ToolCallBatchMiddleware(),
                        usage_collector,
                        ModelCallLimitMiddleware(
                            run_limit=spec.max_model_calls,
                            exit_behavior="error",
                        ),
                    ],
                    context_schema=ForgeRuntimeContext,
                    name=spec.name,
                ),
            )
            output = await child.ainvoke(
                {"messages": [HumanMessage(content=normalized_instruction)]},
                context=runtime.runtime_context,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - child failures are task results
            if isinstance(exc, ModelCallLimitExceededError):
                error = f"Subagent reached max_model_calls={spec.max_model_calls}"
                model_calls = exc.run_count
            else:
                error = str(exc) or exc.__class__.__name__
                model_calls = 0
            return _new_run_result(
                agent=spec.name,
                status="failed",
                instruction=normalized_instruction,
                final_output="",
                model_calls=model_calls,
                tool_calls=0,
                queued_ms=queued_ms,
                duration_ms=_elapsed_ms(run_start),
                error=error,
                truncated=False,
                max_result_bytes=spec.max_result_bytes,
                usage_facts=usage_collector.facts,
            )

        messages = _messages_from_output(output)
        model_calls = sum(isinstance(message, AIMessage) for message in messages)
        tool_calls = sum(isinstance(message, ToolMessage) for message in messages)
        usage = aggregate_usage(messages)
        usage_facts = usage_collector.facts or project_subagent_usage(messages)
        final_output = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                candidate = message_text(message)
                if candidate.strip():
                    final_output = candidate
                    break
        final_output, truncated = _truncate_utf8(final_output, spec.max_result_bytes)
        return _new_run_result(
            agent=spec.name,
            status="completed",
            instruction=normalized_instruction,
            final_output=final_output,
            model_calls=model_calls,
            tool_calls=tool_calls,
            queued_ms=queued_ms,
            duration_ms=_elapsed_ms(run_start),
            truncated=truncated,
            error=None,
            max_result_bytes=spec.max_result_bytes,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            usage_facts=usage_facts,
        )


__all__ = [
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_MAX_RESULT_BYTES",
    "TRACE_ITEM_MAX_BYTES",
    "TRACE_MAX_BYTES",
    "TRACE_MAX_ITEMS",
    "SubagentTrace",
    "SubagentTraceItem",
    "SubagentUsageFact",
    "TokenUsage",
    "SubagentRunResult",
    "SubagentRunner",
    "SubagentRuntime",
    "SubagentSpec",
    "aggregate_usage",
    "project_subagent_trace",
    "project_subagent_usage",
]
