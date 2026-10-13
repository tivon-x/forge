"""Small, generic subagent execution primitives.

The parent Forge agent owns the conversation transcript. A child receives a
portable input snapshot and runs the sole LangChain loop without a checkpointer.
This module deliberately
contains no coding-session or UI concerns so it can be reused by those
layers.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal, cast
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
    ModelCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import add_usage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from forge_agent.context import ForgeRuntimeContext
from forge_agent.retry import classify_model_error, redact_model_error
from forge_agent.tool_execution import (
    ToolCallBatchMiddleware,
    ToolExposureMiddleware,
    ToolIntentMiddleware,
)
from forge_agent.types import JSONValue

DEFAULT_MAX_MODEL_CALLS = 8
DEFAULT_MAX_RESULT_BYTES = 50 * 1024
MAX_FALLBACK_MODELS = 2
MAX_MODEL_ATTEMPTS = DEFAULT_MAX_MODEL_CALLS * (MAX_FALLBACK_MODELS + 1)
DEFAULT_OUTPUT_TOKEN_RESERVE = 1_024
CONTEXT_MESSAGE_OVERHEAD_TOKENS = 16

# Trace budgets are deliberately independent from the task result budget.  A
# trace is a display projection, not a provider/tool transcript and therefore
# gets one small, fixed envelope regardless of the role's result limit.
TRACE_ITEM_MAX_BYTES = 8 * 1024
TRACE_MAX_ITEMS = 64
TRACE_MAX_BYTES = 64 * 1024

SubagentStatus = Literal["completed", "failed"]
TraceKind = Literal["human", "assistant", "tool_call", "tool_result", "omitted"]
TraceStatus = Literal["ok", "error"]
ContextMode = Literal["fresh", "fork", "summary"]
ResultFormat = Literal["text", "task-report"]
AttemptStatus = Literal["completed", "failed"]
UsageCallback = Callable[["SubagentUsageFact"], Awaitable[None] | None]

TASK_REPORT_MAX_ITEMS = 16
TASK_REPORT_SUMMARY_MAX_BYTES = 4 * 1024
TASK_REPORT_ITEM_MAX_BYTES = 2 * 1024
TASK_REPORT_QUESTION_MAX_BYTES = 2 * 1024


class TaskReport(BaseModel):
    """Fixed, bounded handoff returned by a task-report subagent."""

    model_config = ConfigDict(extra="forbid", strict=True)

    outcome: Literal["done", "blocked"]
    summary: StrictStr = Field(min_length=1, max_length=TASK_REPORT_SUMMARY_MAX_BYTES)
    findings: list[StrictStr] = Field(max_length=TASK_REPORT_MAX_ITEMS)
    evidence: list[StrictStr] = Field(max_length=TASK_REPORT_MAX_ITEMS)
    question: StrictStr | None = Field(max_length=TASK_REPORT_QUESTION_MAX_BYTES)

    @field_validator("summary")
    @classmethod
    def _validate_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be blank")
        if len(value.encode("utf-8")) > TASK_REPORT_SUMMARY_MAX_BYTES:
            raise ValueError("summary exceeds UTF-8 byte budget")
        return value

    @field_validator("findings", "evidence")
    @classmethod
    def _validate_items(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.strip():
                raise ValueError("report items must not be blank")
            if len(value.encode("utf-8")) > TASK_REPORT_ITEM_MAX_BYTES:
                raise ValueError("report item exceeds UTF-8 byte budget")
        return values

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str | None) -> str | None:
        if value is not None:
            if not value.strip():
                raise ValueError("question must not be blank")
            if len(value.encode("utf-8")) > TASK_REPORT_QUESTION_MAX_BYTES:
                raise ValueError("question exceeds UTF-8 byte budget")
        return value

    @model_validator(mode="after")
    def _validate_outcome(self) -> TaskReport:
        if self.outcome == "blocked" and self.question is None:
            raise ValueError("blocked reports must include a question describing the blocker")
        return self


class TaskReportBudgetError(ValueError):
    """Raised when a validated task report cannot fit the result envelope."""


def _task_report_json(report: TaskReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_task_report_tool_name(name: object) -> bool:
    return name == TaskReport.__name__


@dataclass(frozen=True, slots=True)
class SubagentModelCandidate:
    """One already-authorized model/provider candidate for a child run."""

    provider: BaseChatModel
    model: str = ""
    provider_name: str = ""
    context_window: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class SubagentAttempt:
    """Bounded fact for one actual model attempt."""

    provider: str | None
    model: str
    status: AttemptStatus
    reason: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "reason": self.reason,
        }


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
    provider: str | None = None
    requested_model: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "response_model": self.response_model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "provider": self.provider,
            "requested_model": self.requested_model,
        }


def _usage_token(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def project_subagent_usage(
    messages: Iterable[object],
    *,
    provider: str | None = None,
    requested_model: str | None = None,
    max_facts: int = MAX_MODEL_ATTEMPTS,
) -> tuple[SubagentUsageFact, ...]:
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
                provider=provider,
                requested_model=requested_model,
            )
        )
        if len(facts) >= max_facts:
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


def _usage_from_facts(facts: Sequence[SubagentUsageFact]) -> TokenUsage:
    def total(name: str) -> int | None:
        values = [value for fact in facts if (value := getattr(fact, name)) is not None]
        return sum(values) if values else None

    return TokenUsage(total("input_tokens"), total("output_tokens"), total("total_tokens"))


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


def visible_subagent_text(value: object) -> str:
    """Project persisted text while preserving untagged legacy strings."""

    if isinstance(value, str):
        return _strip_hidden_trace_tags(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ""
    parts: list[str] = []
    for block in value:
        if not isinstance(block, Mapping) or str(block.get("type", "")).lower() != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(_strip_hidden_trace_tags(text))
    return "".join(parts)


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
            if _is_task_report_tool_name(tool):
                continue
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
        if _is_task_report_tool_name(message.name):
            return [], False
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
    model: str = "inherit"
    fallback_models: tuple[str, ...] = ()
    context: ContextMode = "fresh"
    result_format: ResultFormat = "text"


@dataclass(frozen=True, slots=True)
class SubagentRuntime:
    """The provider and Forge context used for one child invocation."""

    provider: BaseChatModel
    model: str = ""
    runtime_context: ForgeRuntimeContext | None = None
    candidates: tuple[SubagentModelCandidate, ...] = ()
    cleanup: Callable[[], Awaitable[None]] | None = field(default=None, repr=False, compare=False)

    async def aclose(self) -> None:
        """Release invocation-owned clients; borrowed models have no cleanup."""
        if self.cleanup is not None:
            await self.cleanup()


@dataclass(frozen=True, slots=True)
class SubagentInvocation:
    """Frozen inputs for one child run, including its provider snapshot."""

    spec: SubagentSpec
    runtime: SubagentRuntime
    instruction: str
    messages: tuple[BaseMessage, ...] = ()
    context_mode: ContextMode = "fresh"
    context_truncated: bool = False
    inherited_message_count: int = 0
    context_error: str | None = None
    candidates: tuple[SubagentModelCandidate, ...] = ()


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
    context_mode: ContextMode = "fresh"
    inherited_message_count: int = 0
    context_truncated: bool = False
    attempts: tuple[SubagentAttempt, ...] = ()
    result_format: ResultFormat = "text"
    report: TaskReport | None = field(default=None, repr=False, compare=False)
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
        if self.report is not None:
            return content
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
            "context_mode": self.context_mode,
            "inherited_message_count": self.inherited_message_count,
            "context_truncated": self.context_truncated,
            "attempts": [attempt.to_dict() for attempt in self.attempts[:MAX_MODEL_ATTEMPTS]],
        }
        if self.result_format != "text":
            artifact["result_format"] = self.result_format
        if self.report is not None:
            artifact["report"] = cast(dict[str, JSONValue], self.report.model_dump(mode="json"))
        if self.result_format == "task-report" and self.report is not None:
            return artifact
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

        context_mode = value.get("context_mode", "fresh")
        if context_mode not in {"fresh", "fork", "summary"}:
            raise TypeError("subagent artifact context_mode is invalid")
        inherited_message_count = value.get("inherited_message_count", 0)
        if type(inherited_message_count) is not int or inherited_message_count < 0:
            raise TypeError("subagent artifact inherited_message_count is invalid")
        context_truncated = value.get("context_truncated", False)
        if type(context_truncated) is not bool:
            raise TypeError("subagent artifact context_truncated is invalid")
        result_format = value.get("result_format", "text")
        if result_format not in {"text", "task-report"}:
            raise TypeError("subagent artifact result_format is invalid")
        report: TaskReport | None = None
        raw_report = value.get("report")
        if raw_report is not None:
            try:
                report = TaskReport.model_validate(raw_report)
            except ValueError as exc:
                raise TypeError("subagent artifact report is invalid") from exc
        if result_format == "task-report" and status == "completed" and report is None:
            raise TypeError("completed task-report artifact is missing report")
        raw_attempts = value.get("attempts", [])
        if not isinstance(raw_attempts, Sequence) or isinstance(
            raw_attempts, (str, bytes, bytearray)
        ):
            raise TypeError("subagent artifact attempts must be an array")
        attempts: list[SubagentAttempt] = []
        for raw_attempt in raw_attempts[:MAX_MODEL_ATTEMPTS]:
            if not isinstance(raw_attempt, Mapping):
                raise TypeError("subagent artifact attempt must be an object")
            provider = raw_attempt.get("provider")
            model = raw_attempt.get("model")
            attempt_status = raw_attempt.get("status")
            reason = raw_attempt.get("reason")
            if provider is not None and not isinstance(provider, str):
                raise TypeError("subagent artifact attempt provider is invalid")
            if not isinstance(model, str) or attempt_status not in {"completed", "failed"}:
                raise TypeError("subagent artifact attempt is invalid")
            if reason is not None and not isinstance(reason, str):
                raise TypeError("subagent artifact attempt reason is invalid")
            attempts.append(
                SubagentAttempt(
                    provider=provider,
                    model=model,
                    status=cast(AttemptStatus, attempt_status),
                    reason=reason,
                )
            )

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
            context_mode=cast(ContextMode, context_mode),
            inherited_message_count=inherited_message_count,
            context_truncated=context_truncated,
            attempts=tuple(attempts),
            result_format=cast(ResultFormat, result_format),
            report=report,
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
    # Old embedder budgets can be smaller than the R2 envelope. Optional facts
    # remain on the runtime result/Fleet even when the tool artifact cannot fit.
    for key in ("attempts", "context_mode", "inherited_message_count", "context_truncated"):
        if _json_utf8_bytes(baseline) <= max_bytes:
            break
        baseline.pop(key, None)
        bounded.pop(key, None)
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
    context_mode: ContextMode = "fresh",
    inherited_message_count: int = 0,
    context_truncated: bool = False,
    attempts: tuple[SubagentAttempt, ...] = (),
    usage_facts: tuple[SubagentUsageFact, ...] = (),
    result_format: ResultFormat = "text",
    report: TaskReport | None = None,
) -> SubagentRunResult:
    """Build a result whose public strings share one UTF-8 budget."""

    bounded_agent, agent_truncated = _truncate_utf8(agent, max_result_bytes)
    bounded_instruction, instruction_truncated = _truncate_utf8(instruction, max_result_bytes)
    if result_format == "task-report" and report is not None:
        bounded_output = _task_report_json(report)
        if len(bounded_output.encode("utf-8")) > max_result_bytes:
            raise TaskReportBudgetError("Structured TaskReport exceeds the subagent result budget")
        output_truncated = False
    else:
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
        context_mode=context_mode,
        inherited_message_count=inherited_message_count,
        context_truncated=context_truncated,
        attempts=attempts,
        usage_facts=usage_facts,
        result_format=result_format,
        report=report,
    )
    object.__setattr__(result, "_max_result_bytes", max_result_bytes)

    # Fit the serialized artifact as a whole when the fixed v1 metadata can
    # fit in the requested budget.  Pull the fitted user strings back into the
    # result so content, fields, and artifact stay semantically aligned.
    bounded_artifact = result.to_artifact()
    if result_format == "task-report" and report is not None:
        if _json_utf8_bytes(bounded_artifact) > max_result_bytes:
            raise TaskReportBudgetError(
                "Structured TaskReport artifact exceeds the subagent result budget"
            )
        return result
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
    if not isinstance(spec.model, str) or not spec.model.strip():
        raise ValueError("subagent model must be inherit or a configured model")
    if len(spec.fallback_models) > MAX_FALLBACK_MODELS:
        raise ValueError(f"subagent fallback_models must contain at most {MAX_FALLBACK_MODELS}")
    if any(not isinstance(model, str) or not model.strip() for model in spec.fallback_models):
        raise ValueError("subagent fallback model names must be non-empty strings")
    if "inherit" in spec.fallback_models:
        raise ValueError("subagent fallback_models cannot contain inherit")
    if len(set(spec.fallback_models)) != len(spec.fallback_models):
        raise ValueError("subagent fallback_models must not contain duplicates")
    if spec.model != "inherit" and spec.model in spec.fallback_models:
        raise ValueError("subagent model must not also be a fallback model")
    if spec.context not in {"fresh", "fork", "summary"}:
        raise ValueError("subagent context must be fresh, fork, or summary")
    if spec.result_format not in {"text", "task-report"}:
        raise ValueError("subagent result_format must be text or task-report")


def _portable_content(value: object, *, ai: bool = False) -> str | list[str | dict[str, Any]]:
    """Copy portable content and omit reasoning/provider-only blocks."""

    if isinstance(value, str):
        return _strip_hidden_trace_tags(value) if ai else value
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ""
    blocks: list[str | dict[str, Any]] = []
    for block in value:
        if isinstance(block, str):
            blocks.append(_strip_hidden_trace_tags(block) if ai else block)
            continue
        if not isinstance(block, Mapping):
            continue
        block_type = str(block.get("type", "")).lower()
        if ai and block_type not in {"text", "tool_use", "tool_call"}:
            continue
        if block_type in {
            "reasoning",
            "thinking",
            "analysis",
            "redacted_thinking",
            "reasoning_content",
        }:
            continue
        text = block.get("text")
        if isinstance(text, str):
            blocks.append({"type": "text", "text": _strip_hidden_trace_tags(text) if ai else text})
    return copy.deepcopy(blocks)


def _portable_message(message: object) -> BaseMessage | None:
    """Create a deep-copied message without provider metadata or artifacts."""

    if isinstance(message, HumanMessage):
        return HumanMessage(content=_portable_content(message.content))
    if isinstance(message, SystemMessage):
        return SystemMessage(content=_portable_content(message.content))
    if isinstance(message, AIMessage):
        calls: list[dict[str, object]] = []
        for call in message.tool_calls:
            if not isinstance(call, Mapping):
                continue
            call_id, name, args = call.get("id"), call.get("name"), call.get("args")
            if not isinstance(call_id, str) or not call_id:
                continue
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(args, Mapping):
                continue
            calls.append(
                {
                    "id": call_id,
                    "name": name,
                    "args": copy.deepcopy(dict(args)),
                    "type": "tool_call",
                }
            )
        return AIMessage(
            content=_portable_content(message.content, ai=True),
            tool_calls=calls,
        )
    if isinstance(message, ToolMessage):
        call_id = message.tool_call_id
        if not isinstance(call_id, str) or not call_id:
            return None
        return ToolMessage(
            content=_portable_content(message.content),
            tool_call_id=call_id,
            name=message.name,
            status=message.status,
        )
    return None


def portable_parent_messages(parent_messages: Sequence[object]) -> tuple[BaseMessage, ...]:
    """Return complete, portable parent message groups.

    The current AI batch is identified by an AI message with an unpaired tool
    call.  Its whole suffix is excluded so a child never inherits a partial
    batch or an orphaned ``ToolMessage``.
    """

    copied: list[BaseMessage] = []
    for raw in parent_messages:
        if isinstance(raw, SystemMessage):
            continue
        message = _portable_message(raw)
        if isinstance(raw, AIMessage) and (
            raw.invalid_tool_calls
            or (
                raw.tool_calls
                and (
                    not isinstance(message, AIMessage)
                    or len(message.tool_calls) != len(raw.tool_calls)
                )
            )
        ):
            break
        if message is not None:
            copied.append(message)
    selected: list[BaseMessage] = []
    index = 0
    while index < len(copied):
        message = copied[index]
        if isinstance(message, ToolMessage):
            index += 1
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            ids = [call["id"] for call in message.tool_calls]
            end = index + 1
            while end < len(copied) and isinstance(copied[end], ToolMessage):
                end += 1
            results = [cast(ToolMessage, item).tool_call_id for item in copied[index + 1 : end]]
            if (
                None in ids
                or len(set(ids)) != len(ids)
                or set(ids) != set(results)
                or len(ids) != len(results)
            ):
                break
            selected.extend(copied[index:end])
            index = end
        else:
            selected.append(message)
            index += 1
    return tuple(selected)


def _estimate_message_tokens(messages: Sequence[BaseMessage]) -> int:
    total = CONTEXT_MESSAGE_OVERHEAD_TOKENS * len(messages)
    for message in messages:
        # UTF-8 bytes conservatively cover tokenization differences across
        # providers; this is a capacity guard, not billable usage estimation.
        content = (
            message.content
            if isinstance(message.content, str)
            else json.dumps(message.content, ensure_ascii=False)
        )
        total += max(1, len(content.encode("utf-8")))
        if message.additional_kwargs:
            total += len(json.dumps(message.additional_kwargs, ensure_ascii=False).encode("utf-8"))
        if isinstance(message, AIMessage):
            total += sum(
                len(json.dumps(call, ensure_ascii=False).encode("utf-8"))
                for call in message.tool_calls
                if isinstance(call, Mapping)
            )
    return total


def _history_groups(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    groups: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage) and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return groups


def _fit_context_messages(
    messages: tuple[BaseMessage, ...],
    *,
    spec: SubagentSpec,
    candidates: Sequence[SubagentModelCandidate],
) -> tuple[tuple[BaseMessage, ...], bool, str | None]:
    """Trim complete oldest turns while preserving the newest user turn."""

    windows = [candidate.context_window for candidate in candidates if candidate.context_window]
    if not windows:
        return messages, False, None
    window = min(windows)
    system_tokens = max(1, len(spec.system_prompt.encode("utf-8")))
    tool_tokens = sum(
        len(json.dumps(convert_to_openai_tool(tool), ensure_ascii=False).encode("utf-8"))
        for tool in spec.tools
    )
    if spec.result_format == "task-report":
        tool_tokens += len(
            json.dumps(convert_to_openai_tool(TaskReport), ensure_ascii=False).encode("utf-8")
        )
    output_reserve = (
        max((candidate.max_output_tokens or 0) for candidate in candidates)
        or DEFAULT_OUTPUT_TOKEN_RESERVE
    )

    def fits(value: Sequence[BaseMessage]) -> bool:
        return (
            _estimate_message_tokens(value) + system_tokens + tool_tokens + output_reserve <= window
        )

    if fits(messages):
        return messages, False, None
    groups = _history_groups(messages[:-1]) if messages else []
    protected_group = next(
        (
            index
            for index, group in enumerate(groups)
            if any(isinstance(message, HumanMessage) for message in group)
        ),
        None,
    )
    # The newest inherited user turn is always protected; delete only older
    # complete turns.  This is intentionally a small O(n) scan.
    for index in range(len(groups) - 1, -1, -1):
        if any(isinstance(message, HumanMessage) for message in groups[index]):
            protected_group = index
            break
    removed = False
    while groups and not fits(
        tuple(message for group in groups for message in group) + messages[-1:]
    ):
        removable = next((index for index in range(len(groups)) if index != protected_group), None)
        if removable is None:
            break
        groups.pop(removable)
        if protected_group is not None and removable < protected_group:
            protected_group -= 1
        removed = True
    candidate = tuple(message for group in groups for message in group) + messages[-1:]
    if not fits(candidate):
        return (
            candidate,
            removed,
            "Inherited context exceeds the smallest model window; retry with context=summary.",
        )
    return candidate, removed, None


def _messages_from_output(output: object) -> list[object]:
    """Extract the native child message list from an ``ainvoke`` result."""

    if not isinstance(output, Mapping):
        return []
    messages = output.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return []
    return list(messages)


def _background_tool_name(data: Mapping[str, Any]) -> str:
    """Extract only a bounded tool name from one v3 tool event."""

    raw_name = data.get("tool_name")
    if not isinstance(raw_name, str) or not raw_name:
        output = data.get("output")
        raw_name = getattr(output, "name", None)
    name = raw_name if isinstance(raw_name, str) and raw_name else "tool"
    bounded, _ = _bounded_trace_text(name)
    return bounded or "tool"


def _background_trace_update(
    invocation: SubagentInvocation,
    messages: Sequence[object],
    usage_facts: tuple[SubagentUsageFact, ...] = (),
    attempts: tuple[SubagentAttempt, ...] = (),
) -> dict[str, JSONValue]:
    """Build the bounded callback payload for a terminal child snapshot."""

    messages = messages[max(0, len(invocation.messages) - 1) :]
    usage = _usage_from_facts(usage_facts) if usage_facts else aggregate_usage(messages)
    trace = project_subagent_trace(messages, agent=invocation.spec.name, usage=usage)
    effective_facts = usage_facts or project_subagent_usage(messages)
    return {
        "trace": trace.to_dict(),
        "tool_calls": sum(
            isinstance(message, ToolMessage) and not _is_task_report_tool_name(message.name)
            for message in messages
        ),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "usage_facts": [fact.to_dict() for fact in effective_facts],
        "attempts": [attempt.to_dict() for attempt in attempts],
        "model_attempts": len(attempts),
    }


class _SubagentUsageCollectorMiddleware:
    """Capture usage for each actual provider attempt before child failure."""

    def __init__(self, on_usage: UsageCallback | None = None) -> None:
        self._facts: list[SubagentUsageFact] = []
        self._on_usage = on_usage
        self.model_calls = 0
        self.messages: list[object] = []

    @property
    def facts(self) -> tuple[SubagentUsageFact, ...]:
        return tuple(self._facts[:MAX_MODEL_ATTEMPTS])

    async def record_messages(
        self,
        messages: Iterable[object],
        candidate: SubagentModelCandidate,
    ) -> None:
        facts = project_subagent_usage(
            messages,
            provider=candidate.provider_name or None,
            requested_model=candidate.model or None,
            max_facts=MAX_MODEL_ATTEMPTS,
        )
        for fact in facts:
            if len(self._facts) >= MAX_MODEL_ATTEMPTS:
                return
            self._facts.append(fact)
            await self._emit(fact)

    async def record_exception(
        self,
        exc: BaseException,
        candidate: SubagentModelCandidate,
    ) -> None:
        """Record provider usage attached to an exception, when available."""

        payloads: list[object] = [exc]
        for name in ("usage_metadata", "usage", "response", "body"):
            value = getattr(exc, name, None)
            if value is not None:
                payloads.append(value)
        for payload in payloads:
            if isinstance(payload, AIMessage):
                await self.record_messages((payload,), candidate)
                return
            if isinstance(payload, Mapping):
                usage = payload.get("usage_metadata", payload.get("usage", payload))
                if isinstance(usage, Mapping):
                    message = AIMessage.model_construct(
                        content="",
                        usage_metadata=dict(usage),
                        response_metadata={},
                    )
                    await self.record_messages((message,), candidate)
                    return

    async def _emit(self, fact: SubagentUsageFact) -> None:
        callback = self._on_usage
        if callback is None:
            return
        result = callback(fact)
        if inspect.isawaitable(result):
            await result


class _FailedAttemptUsage:
    """Retain only usage/model metadata observed before a stream fails."""

    def __init__(self) -> None:
        self.usage: Any = None
        self.model: str | None = None

    def observe(self, chunk: BaseMessage) -> None:
        if isinstance(chunk, AIMessage) and chunk.usage_metadata:
            self.usage = add_usage(self.usage, chunk.usage_metadata)
        for key in ("response_model", "model_name", "model", "model_id"):
            value = chunk.response_metadata.get(key)
            if isinstance(value, str) and value:
                self.model = value
                break

    @property
    def messages(self) -> list[AIMessage]:
        return (
            [
                AIMessage.model_construct(
                    content="",
                    usage_metadata=self.usage,
                    response_metadata={"model_name": self.model} if self.model else {},
                )
            ]
            if self.usage is not None
            else []
        )


class _ObservedChildModel(BaseChatModel):
    """Forward the native model stream while observing reported usage only.

    The v3 chunk bridge delays usage until message-finish. Observing the public
    model stream avoids losing earlier usage on failure without owning a model
    or tool loop, copying SDK configuration, or retaining another transcript.
    """

    wrapped: Any = Field(exclude=True, repr=False)
    usage_observer: _FailedAttemptUsage = Field(exclude=True, repr=False)
    streaming: bool | None = None

    @property
    def _llm_type(self) -> str:
        return "forge-observed-child"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, AIMessage]:
        return _ObservedChildModel(
            wrapped=self.wrapped.bind_tools(tools, **kwargs),
            usage_observer=self.usage_observer,
            streaming=self.streaming,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Child models run asynchronously")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = await self.wrapped.ainvoke(
            messages, config={"callbacks": []}, stop=stop, **kwargs
        )
        self.usage_observer.observe(message)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # The outer LangChain model publishes events once; the delegated stream
        # supplies standard messages without duplicate parent/child callbacks.
        stream = self.wrapped.astream(messages, config={"callbacks": []}, stop=stop, **kwargs)
        try:
            async for chunk in stream:
                self.usage_observer.observe(chunk)
                if isinstance(chunk, AIMessageChunk):
                    yield ChatGenerationChunk(message=chunk)
                elif isinstance(chunk, AIMessage):
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(**chunk.model_dump(exclude={"type"}))
                    )
        finally:
            await stream.aclose()


class _SubagentFallbackMiddleware(AgentMiddleware):
    """Try the fixed candidate chain for one logical model request."""

    def __init__(
        self,
        candidates: Sequence[SubagentModelCandidate],
        *,
        usage: _SubagentUsageCollectorMiddleware,
        attempts: list[SubagentAttempt],
        result_format: ResultFormat,
    ) -> None:
        self.candidates = tuple(candidates)
        self.usage = usage
        self.attempts = attempts
        self.result_format = result_format
        self._message_owners: dict[str, tuple[str, str]] = {}

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self.usage.model_calls += 1
        self.usage.messages = list(request.messages)
        last_error: BaseException | None = None
        for candidate in self.candidates[: MAX_FALLBACK_MODELS + 1]:
            if len(self.attempts) >= MAX_MODEL_ATTEMPTS:
                raise RuntimeError("Subagent reached max_model_attempts=24")
            owner = (candidate.provider_name, candidate.model)
            candidate_messages = [
                copy.deepcopy(raw)
                if isinstance(raw, AIMessage) and self._message_owners.get(raw.id or "") == owner
                else message
                for raw in request.messages
                if (message := _portable_message(raw)) is not None
            ]
            # New tool results can grow a later request. Never silently trim the
            # running graph or change its already executed tool history.
            fitted, removed, capacity_error = _fit_context_messages(
                tuple(candidate_messages),
                spec=SubagentSpec(
                    "request",
                    "",
                    request.system_prompt or "",
                    tools=request.tools,
                    result_format=self.result_format,
                ),
                candidates=self.candidates,
            )
            if capacity_error or removed or fitted != tuple(candidate_messages):
                raise ValueError("Child context exceeds the smallest model window; use summary")
            partial = _FailedAttemptUsage()
            model_request = request.override(
                model=_ObservedChildModel(
                    wrapped=candidate.provider,
                    usage_observer=partial,
                    streaming=(
                        getattr(candidate.provider, "streaming", None)
                        if "streaming" in candidate.provider.model_fields_set
                        else None
                    ),
                ),
                messages=candidate_messages,
                system_message=(
                    _portable_message(request.system_message)
                    if request.system_message is not None
                    else None
                ),
                model_settings={
                    "max_tokens": candidate.max_output_tokens or DEFAULT_OUTPUT_TOKEN_RESERVE
                },
            )
            try:
                response = await handler(model_request)
            except GraphBubbleUp:
                raise
            except BaseException as exc:
                reason = (
                    "abort"
                    if isinstance(exc, asyncio.CancelledError)
                    else classify_model_error(exc).kind
                )
                self.attempts.append(
                    SubagentAttempt(
                        candidate.provider_name or None,
                        candidate.model,
                        "failed",
                        reason,
                    )
                )
                last_error = exc
                if partial.messages:
                    await self.usage.record_messages(partial.messages[-1:], candidate)
                else:
                    before = len(self.usage.facts)
                    await self.usage.record_exception(exc, candidate)
                    if len(self.usage.facts) == before:
                        await self.usage.record_messages([AIMessage(content="")], candidate)
                if reason != "transient" or not isinstance(exc, Exception):
                    raise
                continue
            self.attempts.append(
                SubagentAttempt(
                    candidate.provider_name or None,
                    candidate.model,
                    "completed",
                )
            )
            result_messages = getattr(response, "result", None)
            if isinstance(result_messages, Sequence):
                for message in result_messages:
                    if isinstance(message, AIMessage):
                        if message.id is None:
                            message.id = uuid4().hex
                        self._message_owners[message.id] = owner
                await self.usage.record_messages(result_messages, candidate)
                self.usage.messages.extend(result_messages)
            return response
        if last_error is not None:
            raise last_error
        raise RuntimeError("subagent has no model candidate")


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

    def _validated_input(self, agent: str, instruction: str) -> tuple[SubagentSpec, str]:
        spec = self._specs.get(agent) if isinstance(agent, str) else None
        if spec is None:
            raise ToolException(f"Unknown subagent role: {agent}")
        if not isinstance(instruction, str):
            raise ToolException("Subagent instruction must be a string")
        normalized_instruction = instruction.strip()
        if not normalized_instruction:
            raise ToolException("Subagent instruction must not be empty")
        return spec, normalized_instruction

    @staticmethod
    def _runtime_candidates(runtime: SubagentRuntime) -> tuple[SubagentModelCandidate, ...]:
        if runtime.candidates:
            return tuple(runtime.candidates)
        return (
            SubagentModelCandidate(
                provider=runtime.provider,
                model=runtime.model,
            ),
        )

    @classmethod
    def _select_candidates(
        cls,
        spec: SubagentSpec,
        runtime: SubagentRuntime,
    ) -> tuple[SubagentModelCandidate, ...]:
        available = cls._runtime_candidates(runtime)
        if not available:
            raise ValueError(f"No configured model candidate for subagent {spec.name}")
        primary: SubagentModelCandidate | None
        if spec.model == "inherit":
            primary = available[0]
        else:
            primary = next(
                (
                    candidate
                    for candidate in available
                    if spec.model
                    in (candidate.model, f"{candidate.provider_name}/{candidate.model}")
                ),
                None,
            )
            if primary is None:
                raise ValueError(
                    f"Subagent model {spec.model!r} is not configured for role {spec.name!r}"
                )
        selected = [primary]
        for model in spec.fallback_models:
            candidate = next(
                (
                    item
                    for item in available
                    if model in (item.model, f"{item.provider_name}/{item.model}")
                ),
                None,
            )
            if candidate is None:
                raise ValueError(
                    f"Subagent fallback model {model!r} is not configured for role {spec.name!r}"
                )
            if candidate not in selected:
                selected.append(candidate)
        return tuple(selected[: MAX_FALLBACK_MODELS + 1])

    @classmethod
    def _prepare_messages(
        cls,
        spec: SubagentSpec,
        instruction: str,
        *,
        parent_messages: Sequence[object],
        context: ContextMode | None,
        summary: str | None,
        candidates: Sequence[SubagentModelCandidate],
    ) -> tuple[tuple[BaseMessage, ...], ContextMode, bool, int, str | None]:
        mode: ContextMode = spec.context
        if context is not None:
            if context not in {"fresh", "fork", "summary"}:
                raise ValueError("Invalid subagent context mode")
            mode = context

        inherited: tuple[BaseMessage, ...] = ()
        if mode == "fork":
            inherited = portable_parent_messages(parent_messages)
            messages = inherited + (HumanMessage(content=instruction),)
        elif mode == "summary":
            if not isinstance(summary, str) or not summary.strip():
                return (
                    (HumanMessage(content=instruction),),
                    mode,
                    False,
                    0,
                    "Summary context was unavailable; retry with context=fresh.",
                )
            summary_message = HumanMessage(content=f"Parent summary:\n{summary.strip()}")
            inherited = portable_parent_messages(parent_messages)
            messages = (summary_message, HumanMessage(content=instruction))
        else:
            messages = (HumanMessage(content=instruction),)
        fitted, truncated, error = _fit_context_messages(
            messages,
            spec=spec,
            candidates=candidates,
        )
        return fitted, mode, truncated, len(fitted) - 1 if mode != "fresh" else 0, error

    def prepare(
        self,
        agent: str,
        instruction: str,
        *,
        runtime: SubagentRuntime | None = None,
        parent_messages: Sequence[object] = (),
        context: ContextMode | None = None,
        summary: str | None = None,
    ) -> SubagentInvocation:
        """Freeze role, candidate chain, and a portable input snapshot."""

        spec, normalized_instruction = self._validated_input(agent, instruction)
        selected_runtime = runtime if runtime is not None else self._runtime_reader()
        candidates = self._select_candidates(spec, selected_runtime)
        messages, mode, truncated, inherited_count, context_error = self._prepare_messages(
            spec,
            normalized_instruction,
            parent_messages=parent_messages,
            context=context,
            summary=summary,
            candidates=candidates,
        )
        return SubagentInvocation(
            spec=spec,
            runtime=selected_runtime,
            instruction=normalized_instruction,
            messages=messages,
            context_mode=mode,
            context_truncated=truncated,
            inherited_message_count=inherited_count,
            context_error=context_error,
            candidates=candidates,
        )

    @staticmethod
    def _create_child(
        invocation: SubagentInvocation,
        usage_collector: _SubagentUsageCollectorMiddleware,
        attempts: list[SubagentAttempt],
    ) -> Any:
        spec = invocation.spec
        fallback = _SubagentFallbackMiddleware(
            invocation.candidates,
            usage=usage_collector,
            attempts=attempts,
            result_format=spec.result_format,
        )
        return cast(
            Any,
            create_agent(
                invocation.candidates[0].provider,
                tools=list(spec.tools),
                system_prompt=spec.system_prompt,
                response_format=(
                    ToolStrategy(TaskReport) if spec.result_format == "task-report" else None
                ),
                middleware=[
                    ToolCallBatchMiddleware(),
                    ToolExposureMiddleware(),
                    ToolIntentMiddleware(),
                    fallback,
                    ModelCallLimitMiddleware(
                        run_limit=spec.max_model_calls,
                        exit_behavior="error",
                    ),
                ],
                context_schema=ForgeRuntimeContext,
                # D5 owns persistent child conversations; None would inherit the root saver.
                checkpointer=False,
                name=spec.name,
            ),
        )

    @staticmethod
    def _failed_result(
        invocation: SubagentInvocation,
        *,
        error: str,
        model_calls: int,
        started: float,
        usage_facts: tuple[SubagentUsageFact, ...],
        attempts: tuple[SubagentAttempt, ...] = (),
    ) -> SubagentRunResult:
        usage = _usage_from_facts(usage_facts)
        return _new_run_result(
            agent=invocation.spec.name,
            status="failed",
            instruction=invocation.instruction,
            final_output="",
            model_calls=model_calls,
            tool_calls=0,
            queued_ms=0,
            duration_ms=_elapsed_ms(started),
            error=redact_model_error(error),
            truncated=False,
            max_result_bytes=invocation.spec.max_result_bytes,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            context_mode=invocation.context_mode,
            inherited_message_count=invocation.inherited_message_count,
            context_truncated=invocation.context_truncated,
            attempts=attempts,
            usage_facts=usage_facts,
            result_format=invocation.spec.result_format,
        )

    @staticmethod
    def _result_from_messages(
        invocation: SubagentInvocation,
        messages: Sequence[object],
        *,
        started: float,
        usage_facts: tuple[SubagentUsageFact, ...],
        attempts: tuple[SubagentAttempt, ...] = (),
        structured_response: object | None = None,
    ) -> SubagentRunResult:
        runtime_messages = list(messages[len(invocation.messages) :])
        model_calls = sum(isinstance(message, AIMessage) for message in runtime_messages)
        tool_calls = sum(
            isinstance(message, ToolMessage) and not _is_task_report_tool_name(message.name)
            for message in runtime_messages
        )
        usage = _usage_from_facts(usage_facts) if usage_facts else aggregate_usage(runtime_messages)
        effective_usage_facts = usage_facts or project_subagent_usage(runtime_messages)
        report: TaskReport | None = None
        if invocation.spec.result_format == "task-report":
            if structured_response is None:
                raise TaskReportBudgetError("structured_response TaskReport was missing")
            try:
                report = (
                    structured_response
                    if isinstance(structured_response, TaskReport)
                    else TaskReport.model_validate(structured_response)
                )
            except ValueError as exc:
                raise TaskReportBudgetError("structured_response TaskReport was invalid") from exc
            final_output = _task_report_json(report)
            truncated = False
        else:
            final_output = ""
            for message in reversed(runtime_messages):
                if isinstance(message, AIMessage):
                    candidate = _visible_message_text(message)
                    if candidate.strip():
                        final_output = candidate
                        break
            final_output, truncated = _truncate_utf8(
                final_output,
                invocation.spec.max_result_bytes,
            )
        return _new_run_result(
            agent=invocation.spec.name,
            status="completed",
            instruction=invocation.instruction,
            final_output=final_output,
            model_calls=model_calls,
            tool_calls=tool_calls,
            queued_ms=0,
            duration_ms=_elapsed_ms(started),
            truncated=truncated,
            error=None,
            max_result_bytes=invocation.spec.max_result_bytes,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            context_mode=invocation.context_mode,
            inherited_message_count=invocation.inherited_message_count,
            context_truncated=invocation.context_truncated,
            attempts=attempts,
            usage_facts=effective_usage_facts,
            result_format=invocation.spec.result_format,
            report=report,
        )

    @staticmethod
    def _emit_update(
        callback: Callable[[dict[str, JSONValue]], None] | None,
        update: dict[str, JSONValue],
    ) -> None:
        if callback is None:
            return
        try:
            callback(update)
        except Exception:  # noqa: BLE001 - observers must not break child work
            return

    @staticmethod
    def _progress_update(messages: Sequence[object]) -> dict[str, JSONValue]:
        usage = aggregate_usage(messages)
        update: dict[str, JSONValue] = {
            "tool_calls": sum(
                isinstance(message, ToolMessage) and not _is_task_report_tool_name(message.name)
                for message in messages
            ),
        }
        for name, value in (
            ("input_tokens", usage.input_tokens),
            ("output_tokens", usage.output_tokens),
            ("total_tokens", usage.total_tokens),
        ):
            if value is not None:
                update[name] = value
        return update if len(update) > 1 or update["tool_calls"] else {}

    async def run(
        self,
        agent: str,
        instruction: str,
        *,
        prepared: SubagentInvocation | None = None,
        parent_messages: Sequence[object] = (),
        context: ContextMode | None = None,
        on_usage: UsageCallback | None = None,
    ) -> SubagentRunResult:
        """Run one child agent from a fixed snapshot and return its compact result.

        Invalid task input raises ``ToolException`` so the parent LangChain
        tool call follows the normal tool-error path.  Child/provider errors
        are ordinary failed results that the parent model can inspect.  A
        cancellation is never converted into a result and is re-raised.
        """

        if prepared is None:
            spec, normalized_instruction = self._validated_input(agent, instruction)
        else:
            spec, normalized_instruction = prepared.spec, prepared.instruction

        run_start = monotonic()
        usage_collector = _SubagentUsageCollectorMiddleware(on_usage)
        attempts: list[SubagentAttempt] = []
        invocation = prepared

        try:
            if invocation is None:
                invocation = self.prepare(
                    agent, instruction, parent_messages=parent_messages, context=context
                )
            if invocation.context_error:
                raise ValueError(invocation.context_error)
            child = self._create_child(invocation, usage_collector, attempts)
            # Stateless children must not inherit the parent's sync checkpoint policy.
            output = await child.ainvoke(
                {"messages": list(invocation.messages)},
                context=invocation.runtime.runtime_context,
                durability="async",
            )
        except asyncio.CancelledError:
            raise
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001 - child failures are task results
            if isinstance(exc, ModelCallLimitExceededError):
                error = f"Subagent reached max_model_calls={spec.max_model_calls}"
                model_calls = usage_collector.model_calls
            else:
                error = redact_model_error(str(exc) or exc.__class__.__name__)
                model_calls = usage_collector.model_calls
            result = _new_run_result(
                agent=spec.name,
                status="failed",
                instruction=normalized_instruction,
                final_output="",
                model_calls=model_calls,
                tool_calls=sum(
                    isinstance(m, ToolMessage) and not _is_task_report_tool_name(m.name)
                    for m in usage_collector.messages[
                        len(invocation.messages) if invocation else 0 :
                    ]
                ),
                queued_ms=0,
                duration_ms=_elapsed_ms(run_start),
                error=error,
                truncated=False,
                max_result_bytes=spec.max_result_bytes,
                input_tokens=_usage_from_facts(usage_collector.facts).input_tokens,
                output_tokens=_usage_from_facts(usage_collector.facts).output_tokens,
                total_tokens=_usage_from_facts(usage_collector.facts).total_tokens,
                context_mode=invocation.context_mode if invocation is not None else spec.context,
                inherited_message_count=invocation.inherited_message_count if invocation else 0,
                context_truncated=invocation.context_truncated if invocation else False,
                attempts=tuple(attempts),
                usage_facts=usage_collector.facts,
            )
            return result

        messages = _messages_from_output(output)
        structured_response = (
            output.get("structured_response") if isinstance(output, Mapping) else None
        )
        try:
            return self._result_from_messages(
                invocation,
                messages,
                started=run_start,
                usage_facts=usage_collector.facts,
                attempts=tuple(attempts),
                structured_response=structured_response,
            )
        except TaskReportBudgetError as exc:
            return self._failed_result(
                invocation,
                error=str(exc),
                model_calls=usage_collector.model_calls,
                started=run_start,
                usage_facts=usage_collector.facts,
                attempts=tuple(attempts),
            )

    async def run_prepared(self, prepared: SubagentInvocation) -> SubagentRunResult:
        """Execute the frozen role rather than the current registry entry."""
        return await self.run(prepared.spec.name, prepared.instruction, prepared=prepared)

    async def run_background(
        self,
        prepared: SubagentInvocation,
        on_update: Callable[[dict[str, JSONValue]], None] | None = None,
    ) -> SubagentRunResult:
        """Run a prepared child on an isolated v3 event stream."""

        if not isinstance(prepared, SubagentInvocation):
            raise TypeError("prepared must be a SubagentInvocation")

        started = monotonic()
        usage_collector = _SubagentUsageCollectorMiddleware()
        attempts: list[SubagentAttempt] = []
        messages: list[object] = list(prepared.messages)
        structured_response: object | None = None
        start_activity, _ = _bounded_trace_text(f"{prepared.spec.name} started")
        self._emit_update(on_update, {"activity": start_activity})
        try:
            if prepared.context_error:
                raise ValueError(prepared.context_error)
            child = self._create_child(prepared, usage_collector, attempts)
            config = cast(RunnableConfig, {"callbacks": []})
            # Stateless children must not inherit the parent's sync checkpoint policy.
            event_stream = child.astream_events(
                {"messages": messages},
                config=config,
                context=prepared.runtime.runtime_context,
                durability="async",
                version="v3",
            )
            if inspect.isawaitable(event_stream):
                event_stream = await event_stream
            partial: dict[str, str] = {"output": "", "thinking": ""}
            from forge_agent.langchain_runtime import managed_event_stream

            async with managed_event_stream(event_stream) as event_stream:
                async for raw_event in cast(AsyncIterator[object], event_stream):
                    if not isinstance(raw_event, Mapping):
                        continue
                    method = raw_event.get("method")
                    params = raw_event.get("params")
                    if not isinstance(params, Mapping):
                        continue
                    if method == "messages":
                        # Reuse the native v3 text decoding; these are product
                        # progress values, never child messages in the root history.
                        from forge_agent.langchain_runtime import (
                            _content_deltas,
                            _mapping_content_delta,
                        )

                        payload = params.get("data")
                        if isinstance(payload, tuple) and payload:
                            payload = payload[0]
                        deltas: list[tuple[str, str]] = []
                        if isinstance(payload, AIMessageChunk):
                            deltas = _content_deltas(payload)
                        elif isinstance(payload, Mapping):
                            if payload.get("event") == "message-start":
                                partial = {"output": "", "thinking": ""}
                            delta = payload.get("delta")
                            if isinstance(delta, Mapping):
                                decoded = _mapping_content_delta(delta)
                                if decoded is not None:
                                    deltas = [decoded]
                        for kind, text in deltas:
                            key = "thinking" if kind == "reasoning" else "output"
                            partial[key] = (partial[key] + text)[-50_000:]
                        if deltas:
                            self._emit_update(on_update, dict(partial))
                        continue
                    if method == "values":
                        raw_values = params.get("data")
                        if isinstance(raw_values, Mapping) and "structured_response" in raw_values:
                            structured_response = raw_values.get("structured_response")
                        snapshot = _messages_from_output(raw_values)
                        if snapshot:
                            messages = snapshot
                            progress = self._progress_update(messages[len(prepared.messages) :])
                            if attempts:
                                progress.update(
                                    model_attempts=len(attempts),
                                    attempts=[a.to_dict() for a in attempts],
                                )
                            if progress:
                                self._emit_update(on_update, progress)
                        continue
                    if method != "tools":
                        continue
                    data = params.get("data")
                    if not isinstance(data, Mapping):
                        continue
                    event_name = data.get("event")
                    if event_name not in {"tool-started", "tool-finished"}:
                        continue
                    tool_name = _background_tool_name(data)
                    if _is_task_report_tool_name(tool_name):
                        continue
                    if event_name == "tool-started":
                        activity = _trace_tool_summary(tool_name)
                    else:
                        activity, _ = _bounded_trace_text(f"Finished {tool_name}")
                    self._emit_update(on_update, {"activity": activity})
        except asyncio.CancelledError:
            self._emit_update(on_update, {"activity": "Subagent cancelled"})
            self._emit_update(
                on_update,
                _background_trace_update(
                    prepared, messages, usage_collector.facts, tuple(attempts)
                ),
            )
            raise
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001 - child failures are task results
            if isinstance(exc, ModelCallLimitExceededError):
                error = f"Subagent reached max_model_calls={prepared.spec.max_model_calls}"
                model_calls = usage_collector.model_calls
            else:
                error = redact_model_error(str(exc) or exc.__class__.__name__)
                model_calls = usage_collector.model_calls
            result = self._failed_result(
                prepared,
                error=error,
                model_calls=model_calls,
                started=started,
                usage_facts=usage_collector.facts or project_subagent_usage(messages),
                attempts=tuple(attempts),
            )
            self._emit_update(on_update, {"activity": "Subagent failed"})
            self._emit_update(
                on_update,
                _background_trace_update(
                    prepared, messages, usage_collector.facts, tuple(attempts)
                ),
            )
            return result

        try:
            result = self._result_from_messages(
                prepared,
                messages,
                started=started,
                usage_facts=usage_collector.facts,
                attempts=tuple(attempts),
                structured_response=structured_response,
            )
        except TaskReportBudgetError as exc:
            result = self._failed_result(
                prepared,
                error=str(exc),
                model_calls=usage_collector.model_calls,
                started=started,
                usage_facts=usage_collector.facts,
                attempts=tuple(attempts),
            )
            self._emit_update(on_update, {"activity": "Subagent failed"})
        self._emit_update(
            on_update,
            _background_trace_update(prepared, messages, usage_collector.facts, tuple(attempts)),
        )
        return result


__all__ = [
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_MAX_RESULT_BYTES",
    "TRACE_ITEM_MAX_BYTES",
    "TRACE_MAX_BYTES",
    "TRACE_MAX_ITEMS",
    "TaskReport",
    "ResultFormat",
    "SubagentTrace",
    "SubagentTraceItem",
    "SubagentUsageFact",
    "TokenUsage",
    "SubagentInvocation",
    "SubagentRunResult",
    "SubagentRunner",
    "SubagentRuntime",
    "SubagentSpec",
    "aggregate_usage",
    "project_subagent_trace",
    "project_subagent_usage",
    "visible_subagent_text",
]
