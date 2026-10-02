"""Session-owned background children; persistence remains with CodingSession."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage

from forge_agent import SubagentInvocation, SubagentRunner, SubagentRunResult, SubagentTrace
from forge_agent.session import CustomEntry
from forge_agent.types import JSONValue

RUN_NAMESPACE = "forge.subagent_run.v1"
TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
MAX_RETAINED_RUNS = 64
_ID = re.compile(r"^[a-f0-9]{32}$")
_CONTEXTS = frozenset({"fresh", "fork", "summary"})
_FALLBACK_KINDS = frozenset(
    {"transient", "overflow", "auth", "quota", "invalid_request", "unknown"}
)


def _context_mode(value: object, default: str = "fresh") -> str:
    return value if isinstance(value, str) and value in _CONTEXTS else default


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _fallback_kinds(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    for item in value[:8]:
        if isinstance(item, Mapping):
            item = item.get("kind")
        if not isinstance(item, str):
            continue
        kind = item.strip().lower()
        if kind not in _FALLBACK_KINDS:
            kind = "unknown"
        if kind not in result:
            result.append(kind)
    return tuple(result)


def _attempt_rows(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rows: list[Mapping[str, object]] = []
    for item in value[:24]:
        if isinstance(item, Mapping):
            rows.append(item)
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            with suppress(Exception):
                row = to_dict()
                if isinstance(row, Mapping):
                    rows.append(row)
    return tuple(rows)


def _attempt_fallback_kinds(rows: tuple[Mapping[str, object], ...]) -> tuple[str, ...]:
    return _fallback_kinds(
        [
            previous.get("reason")
            for previous, following in zip(rows, rows[1:], strict=False)
            if previous.get("reason") == "transient"
            and (previous.get("provider"), previous.get("model"))
            != (following.get("provider"), following.get("model"))
        ]
    )


def _runtime_provider(invocation: SubagentInvocation) -> str:
    candidates = invocation.candidates or invocation.runtime.candidates
    if candidates:
        provider = candidates[0].provider_name
        if provider:
            return provider[:128]
    return ""


async def _close_invocation(invocation: SubagentInvocation | None) -> None:
    if invocation is None:
        return
    with suppress(Exception):
        await invocation.runtime.aclose()


@dataclass(slots=True)
class BackgroundRun:
    run_id: str
    agent: str
    model: str
    generation: str
    origin: str | None
    prepared: SubagentInvocation | None = None
    status: str = "queued"
    accepted: bool = False
    persisted: bool = False
    announced: bool = False
    historical: bool = False
    activity: str = ""
    duration_ms: int = 0
    started: float | None = None
    tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    baseline: str | None = None
    result: dict[str, JSONValue] | None = None
    trace: dict[str, JSONValue] | None = None
    usage_facts: tuple[dict[str, JSONValue], ...] = ()
    usage_persisted: int = 0
    actual_provider: str = ""
    actual_model: str = ""
    context: str = "fresh"
    inherited_messages: int = 0
    context_truncated: bool = False
    model_attempts: int = 0
    fallback_reasons: tuple[str, ...] = ()
    task: asyncio.Task[None] | None = None
    settled: asyncio.Event = field(default_factory=asyncio.Event)

    def snapshot(self, *, detail: bool = False) -> dict[str, JSONValue]:
        data: dict[str, JSONValue] = {
            "version": 1,
            "run_id": self.run_id,
            "agent": self.agent,
            "model": self.model,
            "actual_provider": self.actual_provider,
            "actual_model": self.actual_model or self.model,
            "generation": self.generation,
            "origin": self.origin,
            "status": self.status,
            "activity": self.activity,
            "duration_ms": (
                max(0, int((monotonic() - self.started) * 1000))
                if self.status == "running" and self.started is not None
                else self.duration_ms
            ),
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "baseline": self.baseline,
            "context": self.context,
            "inherited_messages": self.inherited_messages,
            "context_truncated": self.context_truncated,
            "model_attempts": self.model_attempts,
            "fallback_reasons": list(self.fallback_reasons),
            "notification": self.persisted and self.status in TERMINAL and not self.historical,
        }
        if detail:
            data.update(result=self.result, trace=self.trace)
        # Callers and renderers cannot mutate the manager's private result.
        return cast(dict[str, JSONValue], json.loads(json.dumps(data)))


class SubagentRuns:
    def __init__(
        self,
        *,
        runner: SubagentRunner | None,
        changed: Callable[[], None],
        origin: Callable[[], str | None],
        concurrency: int = 4,
        queue_limit: int = 16,
        timeout_seconds: float = 900,
    ) -> None:
        if type(concurrency) is not int or not 1 <= concurrency <= 32:
            raise ValueError("background concurrency must be between 1 and 32")
        if type(queue_limit) is not int or not 0 <= queue_limit <= 128:
            raise ValueError("background queue limit must be between 0 and 128")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 3600
        ):
            raise ValueError("background timeout must be between 0 and 3600 seconds")
        self.runner = runner
        self.changed = changed
        self.origin = origin
        self.concurrency = concurrency
        self.queue_limit = queue_limit
        self.timeout_seconds = timeout_seconds
        self.generation = uuid4().hex
        self.runs: dict[str, BackgroundRun] = {}
        self.closed = False

    @property
    def active_count(self) -> int:
        return sum(run.status not in TERMINAL for run in self.runs.values())

    @property
    def dirty(self) -> tuple[BackgroundRun, ...]:
        return tuple(
            r for r in self.runs.values() if r.accepted and r.status in TERMINAL and not r.persisted
        )

    @property
    def snapshots(self) -> tuple[dict[str, JSONValue], ...]:
        return tuple(r.snapshot() for r in self.runs.values() if r.accepted)

    def submit(
        self,
        agent: str | SubagentInvocation,
        instruction: str | None = None,
        prepared: SubagentInvocation | None = None,
    ) -> dict[str, JSONValue]:
        self.check_submission_capacity()
        if isinstance(agent, SubagentInvocation):
            if prepared is not None or instruction is not None:
                raise ValueError("prepared subagent invocation cannot be combined with agent input")
            prepared = agent
            agent_name = prepared.spec.name
        else:
            if not isinstance(agent, str):
                raise ValueError("Subagent agent must be a string")
            if prepared is not None:
                if not isinstance(prepared, SubagentInvocation):
                    raise ValueError("prepared must be a SubagentInvocation")
                if prepared.spec.name != agent:
                    raise ValueError("prepared subagent role does not match agent")
                if instruction is not None and instruction.strip() != prepared.instruction:
                    raise ValueError("prepared subagent instruction does not match instruction")
                agent_name = agent
            else:
                if not isinstance(instruction, str):
                    raise ValueError("Subagent instruction must be a string")
                assert self.runner is not None
                prepared = self.runner.prepare(agent, instruction)
                agent_name = agent
        run = BackgroundRun(
            uuid4().hex,
            agent_name,
            prepared.runtime.model,
            self.generation,
            self.origin(),
            prepared=prepared,
            actual_provider=_runtime_provider(prepared),
            actual_model=prepared.runtime.model,
            context=prepared.context_mode,
            inherited_messages=prepared.inherited_message_count,
            context_truncated=prepared.context_truncated,
        )
        self.runs[run.run_id] = run
        self._trim()
        return {
            "kind": "subagent_receipt",
            "version": 1,
            "run_id": run.run_id,
            "agent": agent_name,
            "status": "accepted",
        }

    def get(self, run_id: str) -> BackgroundRun:
        run = self.runs.get(run_id)
        if run is None or not run.accepted:
            raise ValueError("Unknown background subagent run in this session branch")
        return run

    def check_submission_capacity(self) -> None:
        if self.closed or self.runner is None:
            raise ValueError("Background subagents are unavailable")
        if self.active_count >= self.concurrency + self.queue_limit:
            raise ValueError("Background subagent queue is full")

    def accept(self, run_id: str) -> None:
        run = self.runs[run_id]
        run.accepted = True
        self._pump()

    async def discard_provisional(self) -> None:
        provisional = tuple(run.prepared for run in self.runs.values() if not run.accepted)
        close_error: BaseException | None = None
        for invocation in provisional:
            try:
                await _close_invocation(invocation)
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
        self.runs = {key: run for key, run in self.runs.items() if run.accepted}
        if close_error is not None:
            raise close_error

    def _pump(self) -> None:
        if self.closed:
            return
        running = sum(r.task is not None and not r.task.done() for r in self.runs.values())
        for run in self.runs.values():
            if running >= self.concurrency:
                break
            if run.accepted and run.status == "queued" and run.task is None:
                run.status = "running"
                run.task = asyncio.create_task(self._execute(run), name=f"forge-child-{run.run_id}")
                running += 1

    def _update(self, run: BackgroundRun, data: dict[str, JSONValue]) -> None:
        if run.generation != self.generation:
            return
        activity = data.get("activity")
        if isinstance(activity, str):
            run.activity = activity[:128]
        for name in (
            "tool_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "inherited_messages",
            "model_attempts",
        ):
            value = data.get(name)
            if type(value) is int and value >= 0:
                setattr(run, name, value)
        inherited_count = data.get("inherited_message_count")
        if type(inherited_count) is int and inherited_count >= 0:
            run.inherited_messages = inherited_count
        attempts = _attempt_rows(data.get("attempts"))
        if attempts:
            run.model_attempts = max(run.model_attempts, len(attempts))
            if not run.fallback_reasons:
                run.fallback_reasons = _attempt_fallback_kinds(attempts)
            if isinstance(attempts[-1], Mapping):
                last_provider = attempts[-1].get("provider")
                last_model = attempts[-1].get("model")
                if isinstance(last_provider, str) and last_provider.strip():
                    run.actual_provider = last_provider.strip()[:128]
                if isinstance(last_model, str) and last_model.strip():
                    run.actual_model = last_model.strip()[:128]
        context = data.get("context", data.get("context_mode"))
        if isinstance(context, str) and context in _CONTEXTS:
            run.context = context
        context_truncated = data.get("context_truncated")
        if isinstance(context_truncated, bool):
            run.context_truncated = context_truncated
        actual_model = data.get("actual_model")
        if isinstance(actual_model, str) and actual_model.strip():
            run.actual_model = actual_model.strip()[:128]
        actual_provider = data.get("actual_provider")
        if isinstance(actual_provider, str) and actual_provider.strip():
            run.actual_provider = actual_provider.strip()[:128]
        if "fallback_reasons" in data:
            run.fallback_reasons = _fallback_kinds(data.get("fallback_reasons"))
        facts = data.get("usage_facts")
        if isinstance(facts, list) and all(isinstance(fact, dict) for fact in facts):
            run.usage_facts = tuple(cast(dict[str, JSONValue], fact) for fact in facts[:128])
        trace = data.get("trace")
        if isinstance(trace, dict):
            with suppress(ValueError, TypeError):
                run.trace = SubagentTrace.from_dict(trace).to_dict()

    def _update_result_facts(self, run: BackgroundRun, result: SubagentRunResult) -> None:
        self._update(
            run,
            {
                "context_mode": result.context_mode,
                "inherited_message_count": result.inherited_message_count,
                "context_truncated": result.context_truncated,
                "attempts": [attempt.to_dict() for attempt in result.attempts],
                "model_attempts": len(result.attempts),
            },
        )

    async def _execute(self, run: BackgroundRun) -> None:
        started = monotonic()
        run.started = started
        try:
            runner = self.runner
            if runner is None or run.prepared is None:
                raise RuntimeError("No child runner")
            async with asyncio.timeout(self.timeout_seconds):
                context = run.prepared.runtime.runtime_context
                if context is not None:
                    run.baseline = await _git_baseline(context.workspace_root)
                result = await runner.run_background(run.prepared, lambda d: self._update(run, d))
            self._update_result_facts(run, result)
            run.status = result.status
            run.result = result.to_artifact()
            if result.status == "failed":
                run.result["error"] = "Background subagent failed"
            result_usage_facts = tuple(fact.to_dict() for fact in result.usage_facts)
            if result_usage_facts:
                run.usage_facts = result_usage_facts
            run.tool_calls = max(run.tool_calls, result.tool_calls)
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                value = getattr(result, name)
                if value is not None:
                    setattr(run, name, value)
        except asyncio.CancelledError:
            run.status = "cancelled"
        except TimeoutError:
            run.status = "failed"
            run.result = {"final_output": "", "error": "Background subagent timed out"}
        except Exception:
            run.status = "failed"
            run.result = {"final_output": "", "error": "Background subagent failed"}
        finally:
            run.duration_ms = max(0, int((monotonic() - started) * 1000))
            run.activity = ""
            try:
                await _close_invocation(run.prepared)
            finally:
                run.prepared = None
                run.settled.set()
                if run.generation == self.generation:
                    self.changed()
                    # Schedule after this task exits so the released slot is visible.
                    asyncio.get_running_loop().call_soon(self._pump)

    async def control(
        self, action: str, run_id: str | None = None, timeout_seconds: float = 10
    ) -> dict[str, JSONValue]:
        if action == "list":
            return {"runs": list(self.snapshots)}
        if not isinstance(run_id, str):
            raise ValueError("An exact background run id is required")
        run = self.get(run_id)
        if action == "cancel":
            if run.status not in TERMINAL:
                if run.task is not None:
                    run.task.cancel()
                    await asyncio.gather(run.task, return_exceptions=True)
                    try:
                        await _close_invocation(run.prepared)
                    finally:
                        run.prepared = None
                        if run.status not in TERMINAL:
                            run.status = "cancelled"
                        run.settled.set()
                        self.changed()
                else:
                    try:
                        await _close_invocation(run.prepared)
                    finally:
                        run.status = "cancelled"
                        run.prepared = None
                        run.settled.set()
                        self.changed()
        elif action == "wait":
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds)
                or not 0 <= timeout_seconds <= 60
            ):
                raise ValueError("wait timeout must be between 0 and 60 seconds")
            with suppress(TimeoutError):
                await asyncio.wait_for(run.settled.wait(), timeout_seconds)
        elif action != "status":
            raise ValueError("Unknown background control action")
        return run.snapshot(detail=True)

    async def stop(self) -> None:
        self.closed = True
        for run in tuple(self.runs.values()):
            if run.accepted:
                await self.control("cancel", run.run_id)
        await self.discard_provisional()

    def restore(self, entries: Iterable[CustomEntry], *, preserve_running: bool = False) -> None:
        retained = {key: run for key, run in self.runs.items() if run.accepted}
        self.runs.clear()
        self.closed = False
        if not preserve_running:
            self.generation = uuid4().hex
        for entry in entries:
            if entry.namespace != RUN_NAMESPACE:
                continue
            data = entry.data
            run_id, agent = data.get("run_id"), data.get("agent")
            status = data.get("status")
            if (
                type(data.get("version")) is not int
                or data.get("version") != 1
                or not isinstance(run_id, str)
                or not _ID.fullmatch(run_id)
                or not isinstance(agent, str)
                or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", agent)
                or not isinstance(status, str)
                or status not in TERMINAL | {"queued", "running"}
            ):
                continue
            if len(json.dumps(data, ensure_ascii=False).encode()) > 128 * 1024:
                continue
            model = str(data.get("model", ""))[:128]
            actual_provider = data.get("actual_provider")
            if not isinstance(actual_provider, str):
                actual_provider = ""
            actual_model = data.get("actual_model")
            if not isinstance(actual_model, str) or not actual_model.strip():
                actual_model = model
            context = _context_mode(data.get("context", data.get("context_mode")))
            inherited_messages = _nonnegative_int(data.get("inherited_messages")) or 0
            context_truncated = data.get("context_truncated") is True
            model_attempts = _nonnegative_int(data.get("model_attempts")) or 0
            fallback_reasons = _fallback_kinds(data.get("fallback_reasons"))
            run = BackgroundRun(
                run_id,
                agent,
                model,
                self.generation,
                None,
                accepted=True,
                persisted=True,
                historical=True,
                announced=True,
                actual_provider=actual_provider[:128],
                actual_model=actual_model[:128],
                context=context,
                inherited_messages=inherited_messages,
                context_truncated=context_truncated,
                model_attempts=model_attempts,
                fallback_reasons=fallback_reasons,
            )
            run.status = str(data["status"])
            if run.status not in TERMINAL:
                run.status = "interrupted"
                run.announced = False
                run.persisted = False
            for name in (
                "duration_ms",
                "tool_calls",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            ):
                value = data.get(name)
                if type(value) is int and value >= 0:
                    setattr(run, name, value)
            baseline = data.get("baseline")
            if isinstance(baseline, str) and re.fullmatch(r"[a-f0-9]{40,64}", baseline):
                run.baseline = baseline
            result = data.get("result")
            if isinstance(result, Mapping):
                # Persisted result display has its own fixed allowlist.
                run.result = {
                    key: value
                    for key in ("final_output", "error")
                    if isinstance(value := result.get(key), str)
                    and len(value.encode()) <= 50 * 1024
                }
            trace = data.get("trace")
            if isinstance(trace, Mapping):
                with suppress(ValueError, TypeError):
                    run.trace = SubagentTrace.from_dict(trace).to_dict()
            run.settled.set()
            self.runs[run_id] = run
            self._trim()
        if preserve_running:
            self.runs.update(retained)

    def _trim(self) -> None:
        for key, run in tuple(self.runs.items()):
            if len(self.runs) <= MAX_RETAINED_RUNS:
                break
            if run.status in TERMINAL and run.persisted:
                del self.runs[key]


class BackgroundRunMiddleware(AgentMiddleware[Any, Any, Any]):
    """Project completion ids into the next model request, never the transcript."""

    def __init__(self, runs: SubagentRuns) -> None:
        self.runs = runs

    async def awrap_model_call(self, request: ModelRequest[Any], handler: Any) -> Any:
        completed = [
            r
            for r in self.runs.runs.values()
            if r.accepted and r.persisted and r.status in TERMINAL and not r.announced
        ][:16]
        if not completed:
            return await handler(request)
        base = request.system_message.content if request.system_message else ""
        notices = "\n".join(f"{r.run_id}: {r.agent} {r.status}" for r in completed)
        system = SystemMessage(
            content=f"{base}\n\nBackground runs settled:\n{notices}\n"
            "Use task_control to retrieve results when relevant."
        )
        response = await handler(request.override(system_message=system))
        for run in completed:
            run.announced = True
        return response


async def _git_baseline(cwd: str) -> str | None:
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "HEAD",
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await asyncio.wait_for(process.communicate(), 2)
        value = output.decode().strip()
        return (
            value if process.returncode == 0 and re.fullmatch(r"[a-f0-9]{40,64}", value) else None
        )
    except (OSError, TimeoutError):
        return None
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
