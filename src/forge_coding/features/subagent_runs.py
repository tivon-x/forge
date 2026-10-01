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

from forge_agent import SubagentInvocation, SubagentRunner, SubagentTrace
from forge_agent.session import CustomEntry
from forge_agent.types import JSONValue

RUN_NAMESPACE = "forge.subagent_run.v1"
TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
MAX_RETAINED_RUNS = 64
_ID = re.compile(r"^[a-f0-9]{32}$")


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
    task: asyncio.Task[None] | None = None
    settled: asyncio.Event = field(default_factory=asyncio.Event)

    def snapshot(self, *, detail: bool = False) -> dict[str, JSONValue]:
        data: dict[str, JSONValue] = {
            "version": 1,
            "run_id": self.run_id,
            "agent": self.agent,
            "model": self.model,
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

    def submit(self, agent: str, instruction: str) -> dict[str, JSONValue]:
        if self.closed or self.runner is None:
            raise ValueError("Background subagents are unavailable")
        if self.active_count >= self.concurrency + self.queue_limit:
            raise ValueError("Background subagent queue is full")
        prepared = self.runner.prepare(agent, instruction)
        run = BackgroundRun(
            uuid4().hex,
            agent,
            prepared.runtime.model,
            self.generation,
            self.origin(),
            prepared=prepared,
        )
        self.runs[run.run_id] = run
        self._trim()
        return {
            "kind": "subagent_receipt",
            "version": 1,
            "run_id": run.run_id,
            "agent": agent,
            "status": "accepted",
        }

    def get(self, run_id: str) -> BackgroundRun:
        run = self.runs.get(run_id)
        if run is None or not run.accepted:
            raise ValueError("Unknown background subagent run in this session branch")
        return run

    def accept(self, run_id: str) -> None:
        run = self.runs[run_id]
        run.accepted = True
        self._pump()

    def discard_provisional(self) -> None:
        self.runs = {key: run for key, run in self.runs.items() if run.accepted}

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
        for name in ("tool_calls", "input_tokens", "output_tokens", "total_tokens"):
            value = data.get(name)
            if type(value) is int and value >= 0:
                setattr(run, name, value)
        facts = data.get("usage_facts")
        if isinstance(facts, list) and all(isinstance(fact, dict) for fact in facts):
            run.usage_facts = tuple(cast(dict[str, JSONValue], fact) for fact in facts[:128])
        trace = data.get("trace")
        if isinstance(trace, dict):
            with suppress(ValueError, TypeError):
                run.trace = SubagentTrace.from_dict(trace).to_dict()

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
                    if run.status not in TERMINAL:
                        run.status = "cancelled"
                        run.prepared = None
                        run.settled.set()
                        self.changed()
                else:
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
        self.discard_provisional()

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
            run = BackgroundRun(
                run_id,
                agent,
                str(data.get("model", ""))[:128],
                self.generation,
                None,
                accepted=True,
                persisted=True,
                historical=True,
                announced=True,
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
