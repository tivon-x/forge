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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
    ModelCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException

from forge_agent.context import ForgeRuntimeContext
from forge_agent.message_codec import message_text
from forge_agent.types import JSONValue

DEFAULT_MAX_MODEL_CALLS = 8
DEFAULT_MAX_RESULT_BYTES = 50 * 1024

SubagentStatus = Literal["completed", "failed"]


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
    """Stable v1 result returned by a subagent task tool."""

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
        """Return the JSON-safe v1 artifact persisted in the parent ToolMessage."""

        artifact: dict[str, JSONValue] = {
            "kind": "subagent_run",
            "version": 1,
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
        }
        bounded = _fit_artifact_budget(artifact, self._max_result_bytes)
        if bounded != artifact:
            bounded["truncated"] = True
        return bounded

    def artifact(self) -> dict[str, JSONValue]:
        """Alias for :meth:`to_artifact` used by tool adapters."""

        return self.to_artifact()


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


def _json_utf8_bytes(value: Mapping[str, JSONValue]) -> int:
    """Return the default JSON UTF-8 size of one artifact mapping."""

    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


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


class SubagentRunner:
    """Registry and serial runner for stateless child LangChain agents."""

    def __init__(
        self,
        runtime_reader: RuntimeReader,
        specs: Sequence[SubagentSpec] = (),
    ) -> None:
        self._runtime_reader = runtime_reader
        self._lock = asyncio.Lock()
        self._specs: dict[str, SubagentSpec] = {}
        self.replace_specs(specs)

    @property
    def specs(self) -> tuple[SubagentSpec, ...]:
        """Return the current role registry in registration order."""

        return tuple(self._specs.values())

    def replace_specs(self, specs: Sequence[SubagentSpec]) -> None:
        """Replace the role registry atomically for future runs.

        Existing child invocations keep their already-selected spec.  The
        method is intentionally synchronous so a coding session can refresh
        role prompts during ``reload`` without replacing its task tool.
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

        queued_start = monotonic()
        async with self._lock:
            queued_ms = _elapsed_ms(queued_start)
            run_start = monotonic()
            runtime = self._runtime_reader()

            try:
                child = cast(
                    Any,
                    create_agent(
                        runtime.provider,
                        tools=list(spec.tools),
                        system_prompt=spec.system_prompt,
                        middleware=[
                            ModelCallLimitMiddleware(
                                run_limit=spec.max_model_calls,
                                exit_behavior="error",
                            )
                        ],
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
                )

            messages = _messages_from_output(output)
            model_calls = sum(isinstance(message, AIMessage) for message in messages)
            tool_calls = sum(isinstance(message, ToolMessage) for message in messages)
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
            )


__all__ = [
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_MAX_RESULT_BYTES",
    "SubagentRunResult",
    "SubagentRunner",
    "SubagentRuntime",
    "SubagentSpec",
]
