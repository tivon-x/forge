"""Saved project scripts compose session-owned runs through scoped IPC."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from langchain.tools import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from forge_agent import AgentToolResult, JSONValue, SubagentRunner
from forge_agent.subagents import SubagentInvocation
from forge_coding.codemode.runtime import CodemodeRuntime
from forge_coding.features.subagent_runs import RUN_NAMESPACE, SubagentRuns
from forge_coding.resources import (
    ForgeResourcePaths,
    ResourceDiagnostic,
    ResourceError,
    read_resource_text,
)
from forge_coding.resources.trust import find_project_root, project_path_is_safe
from forge_coding.tools import ToolDefinition
from forge_coding.tools.base import _create_native_tool

WORKFLOW_NAMESPACE = "forge.workflow.v1"
REPORT_BYTES = 64 * 1024
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Workflow:
    name: str
    path: Path
    code: str


def load_workflows(
    paths: ForgeResourcePaths,
) -> tuple[tuple[Workflow, ...], tuple[ResourceDiagnostic, ...]]:
    if paths.cwd is None or not paths.project_resources_allowed:
        return (), ()
    folder = paths._paths().project_forge_dir(paths.cwd) / "workflows"
    project_root = find_project_root(paths.cwd)
    rows: list[Workflow] = []
    diagnostics: list[ResourceDiagnostic] = []
    try:
        if not project_path_is_safe(folder, project_root=project_root):
            raise ResourceError("project workflow directory escapes its boundary")
        for path in sorted(folder.glob("*.js")):
            if not _NAME.fullmatch(path.stem) or len(rows) >= 32:
                raise ResourceError("workflow name or resource count exceeds limits")
            code = read_resource_text(path, project_root=project_root)
            if not code.strip() or len(code.encode()) > REPORT_BYTES:
                raise ResourceError("workflow script must be nonempty and at most 64 KiB")
            rows.append(Workflow(path.stem, path, code))
    except (OSError, ValueError, UnicodeError) as exc:
        diagnostics.append(ResourceDiagnostic(kind="workflow", message=str(exc), severity="error"))
        # A bad catalog revokes the whole scope rather than publishing a partial reload.
        return (), tuple(diagnostics)
    return tuple(rows), tuple(diagnostics)


class WorkflowInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: StrictStr = Field(min_length=1, max_length=64)
    args: dict[str, JSONValue] = Field(default_factory=dict)


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    step_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    agent: str = Field(min_length=1, max_length=32)
    instruction: str = Field(min_length=1, max_length=32768)
    mode: str = "foreground"
    context: str | None = None


class WorkflowScope:
    def __init__(
        self,
        runs: SubagentRuns,
        prepare: Callable[
            [SubagentRunner, str, str, str | None, ToolRuntime | None],
            Awaitable[SubagentInvocation],
        ],
        persist: Callable[[str, dict[str, JSONValue]], Awaitable[None]],
        runtime: ToolRuntime | None,
    ) -> None:
        self.workflow_id = uuid4().hex
        self.runs = runs
        self.prepare = prepare
        self.persist = persist
        self.runtime = runtime
        self.steps: dict[str, str] = {}
        self.contracts: dict[str, RunInput] = {}
        self.uncollected: set[str] = set()
        self.lock = asyncio.Lock()
        self.closed = False

    async def call(self, name: str, arguments: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        name = name.removeprefix("runs.")
        if self.closed:
            raise ValueError("Workflow scope closed")
        if name == "control":
            if set(arguments) - {"action", "run_id", "timeout_seconds"}:
                raise ValueError("Invalid workflow control arguments")
            action = arguments.get("action")
            if action == "list":
                return {"runs": self.snapshots()}
            ident = arguments.get("run_id")
            if not isinstance(ident, str) or ident not in self.steps.values():
                raise ValueError("Run is not owned by this workflow")
            if action not in {"status", "wait", "cancel"}:
                raise ValueError("Invalid workflow control action")
            timeout = arguments.get("timeout_seconds", 10)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise ValueError("Invalid workflow wait timeout")
            result = await self.runs.control(action, ident, timeout)
            if action in {"wait", "cancel"} and self.runs.get(ident).settled.is_set():
                self.uncollected.discard(ident)
            return result
        if name != "run":
            raise ValueError("Unknown workflow operation")
        options = RunInput.model_validate(arguments)
        if options.mode not in {"foreground", "background"} or options.context not in {
            None,
            "fresh",
            "fork",
            "summary",
        }:
            raise ValueError("Invalid workflow run mode or context")
        async with self.lock:
            existing = self.steps.get(options.step_key)
            if existing is not None:
                if self.contracts[options.step_key] != options:
                    raise ValueError("Step key already has a different assignment")
                run = self.runs.get(existing)
            else:
                if len(self.steps) >= 16:
                    raise ValueError("Workflow launch limit is 16")
                self.runs.check_submission_capacity()
                runner = self.runs.runner
                assert runner is not None
                prepared = await self.prepare(
                    runner, options.agent, options.instruction, options.context, self.runtime
                )
                transferred = False
                submitted_id: str | None = None
                try:
                    receipt = self.runs.submit(prepared)
                    submitted_id = cast(str, receipt["run_id"])
                    run = self.runs.runs[submitted_id]
                    run.workflow_id = self.workflow_id
                    run.step_key = options.step_key
                    # Session's existing writer persists the parent call first.
                    await self.persist(RUN_NAMESPACE, run.snapshot(detail=True))
                    self.steps[options.step_key] = submitted_id
                    self.contracts[options.step_key] = options
                    self.runs.accept(submitted_id)
                    transferred = True
                finally:
                    if not transferred:
                        await prepared.runtime.aclose()
                        if submitted_id is not None:
                            self.runs.runs.pop(submitted_id, None)
        if options.mode == "foreground":
            await run.settled.wait()
            return run.snapshot(detail=True)
        self.uncollected.add(run.run_id)
        return {
            "kind": "subagent_receipt",
            "run_id": run.run_id,
            "workflow_id": self.workflow_id,
            "step_key": options.step_key,
            "status": "accepted",
        }

    def snapshots(self) -> list[JSONValue]:
        return [self.runs.get(ident).snapshot(detail=True) for ident in self.steps.values()]

    async def close(self) -> None:
        self.closed = True
        for ident in self.steps.values():
            await self.runs.control("cancel", ident)


def create_workflow_tool(
    resources: tuple[Workflow, ...],
    codemode: CodemodeRuntime,
    runs: SubagentRuns,
    prepare: Callable[
        [SubagentRunner, str, str, str | None, ToolRuntime | None], Awaitable[SubagentInvocation]
    ],
    persist: Callable[[str, dict[str, JSONValue]], Awaitable[None]],
) -> ToolDefinition:
    async def execute_runtime(
        arguments: Mapping[str, JSONValue], runtime: ToolRuntime | None
    ) -> AgentToolResult:
        options = WorkflowInput.model_validate(arguments)
        if (
            len(json.dumps(options.args, ensure_ascii=False, allow_nan=False).encode())
            > REPORT_BYTES
        ):
            raise ValueError("Workflow args exceed 64 KiB")
        resource = next((row for row in resources if row.name == options.name), None)
        if resource is None:
            raise ValueError("Unknown loaded workflow")
        scope = WorkflowScope(runs, prepare, persist, runtime)
        cancelled = False
        payload: dict[str, JSONValue] = {
            "version": 1,
            "workflow_id": scope.workflow_id,
            "name": options.name,
            "status": "failed",
            "value": None,
        }
        await persist(WORKFLOW_NAMESPACE, {**payload, "status": "running", "steps": []})
        try:
            _, artifact = await codemode.execute(
                resource.code, scope="workflow", workflow_args=options.args, runs_handler=scope.call
            )
            result = artifact["forge.codemode.v1"]
            payload.update(
                status="completed" if result.get("ok") else "failed",
                value=result.get("value"),
                output=result.get("output"),
                error=result.get("error"),
            )
            value = result.get("value")
            if isinstance(value, dict) and value.get("status") == "failed":
                payload.update(status="failed", error="Workflow script reported failure")
        except asyncio.CancelledError:
            cancelled = True
            payload.update(status="cancelled", error="Workflow cancelled")
        except Exception:
            payload.update(status="failed", error="Workflow failed before script completion")
        finally:
            cleanup = asyncio.create_task(scope.close())
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            await cleanup
            if cancelled:
                payload.update(status="cancelled", error="Workflow cancelled")
            payload["steps"] = scope.snapshots()
            if payload["status"] == "completed" and (
                scope.uncollected
                or any(runs.get(ident).status != "completed" for ident in scope.steps.values())
            ):
                payload.update(status="failed", error="Workflow has unsettled or failed steps")
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode()) > REPORT_BYTES:
                payload.update(
                    status="failed", value=None, output=[], error="Workflow report exceeds 64 KiB"
                )
                # Omit whole oversized payloads, never slice serialized JSON.
                for row in cast(list[dict[str, JSONValue]], payload["steps"]):
                    row.update(result=None, trace=None, result_omitted=True)
                encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            await persist(WORKFLOW_NAMESPACE, payload)
        if cancelled:
            raise asyncio.CancelledError
        ok = payload["status"] == "completed"
        return AgentToolResult(
            tool_call_id="",
            name="workflow",
            ok=ok,
            content=encoded,
            data=payload,
            error=None if ok else cast(str, payload.get("error") or "Workflow failed"),
        )

    async def execute(
        arguments: Mapping[str, JSONValue], signal: Any = None, context: Any = None
    ) -> AgentToolResult:
        return await execute_runtime(arguments, None)

    tool = _create_native_tool(
        name="workflow",
        description="Run a saved bounded project workflow. Loaded: "
        + ", ".join(r.name for r in resources),
        args_schema=WorkflowInput,
        executor=execute,
        runtime_executor=execute_runtime,
    )
    tool.metadata = {
        **(tool.metadata or {}),
        "forge.exposure": "model-only",
        "forge.execution_mode": "sequential",
    }
    return ToolDefinition(
        tool=tool,
        label="workflow",
        prompt_snippet="Compose isolated tasks with saved workflow scripts.",
    )
