"""Built-in coding subagents and the model-facing ``task`` tool."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from forge_agent import AgentToolResult, JSONValue, SubagentRunner, SubagentSpec
from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import ToolCancellationToken
from forge_coding.resources.skills import Skill
from forge_coding.resources.subagent_profiles import (
    PROFILE_MAX_COUNT,
    TASK_REGISTRY_MAX_DESCRIPTION_BYTES,
    CodingSubagentProfile,
    builtin_subagent_profiles,
)
from forge_coding.resources.system_prompt import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    build_system_prompt,
)
from forge_coding.tools import ForgeStructuredTool, ToolDefinition, ToolSet
from forge_coding.tools.base import _create_native_tool

if TYPE_CHECKING:
    from forge_coding.features.subagent_runs import SubagentRuns

TASK_TOOL_DESCRIPTION = (
    "Delegate one bounded task to a fresh, isolated coding subagent. "
    "Each call accepts exactly one task and returns only the subagent's final result."
)
TASK_PROMPT_GUIDELINES = (
    "Use task only for a clearly bounded assignment that benefits from an isolated context",
    "Give each task enough concrete scope and acceptance criteria to work independently",
)
TASK_CONTROL_DESCRIPTION = (
    "Inspect or control accepted background subagent runs. Actions: list, status, wait, cancel."
)
TASK_CONTROL_ACTIONS = frozenset({"list", "status", "wait", "cancel"})
TASK_RUN_ID_MAX_LENGTH = 128
TASK_CONTROL_TIMEOUT_MAX_SECONDS = 60.0
TaskMode = Literal["foreground", "background"]


def coding_subagent_tool_set(tools: ToolSet) -> ToolSet:
    """Grant native tools independently of the parent's declaration/loadout state.

    Parent control closures and script-only tools are never child capabilities.
    Deferred native tools become direct tools within the selected role.
    """
    return ToolSet(
        ToolDefinition(
            tool=d.tool.model_copy(
                update={
                    "metadata": {
                        **(d.tool.metadata or {}),
                        "forge.exposure": "direct",
                        "forge.codemode_only": False,
                        "forge.loaded": False,
                        "forge.declared": True,
                    }
                }
            ),
            label=d.label,
            prompt_snippet=d.prompt_snippet,
            prompt_guidelines=d.prompt_guidelines,
        )
        for d in tools
        if d.name
        not in {
            "task",
            "task_control",
            "tool_search",
            "codemode",
            "ask_user_question",
        }
        and d.exposure in {"direct", "deferred"}
        and (d.tool.metadata or {}).get("forge.available") is not False
    )


class TaskToolInput(BaseModel):
    """Input schema for Forge's built-in task tool.

    Extra fields stay allowed so LangChain can inject ``ToolRuntime`` and the
    executor can produce the stable user-facing unexpected-argument error.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    agent: StrictStr = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[a-z][a-z0-9_-]*$",
        description="Subagent role name",
    )
    instruction: StrictStr = Field(
        min_length=1,
        description="One self-contained, bounded task",
    )
    mode: TaskMode | None = Field(
        default=None,
        description="Run in the foreground or submit to the background fleet",
    )


class TaskControlInput(BaseModel):
    """Input schema for the parent-facing background-run control tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    action: StrictStr = Field(
        min_length=1,
        max_length=16,
        description="One of list, status, wait, or cancel",
    )
    run_id: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=TASK_RUN_ID_MAX_LENGTH,
        description="Exact background run id for status, wait, or cancel",
    )
    timeout_seconds: StrictInt | StrictFloat = Field(
        default=10,
        ge=0,
        le=TASK_CONTROL_TIMEOUT_MAX_SECONDS,
        description="Maximum wait time for the wait action",
    )


def create_coding_subagent_specs(
    *,
    cwd: Path,
    tools: Sequence[BaseTool | ToolDefinition] | ToolSet,
    skills: Sequence[Skill],
    context_files: Sequence[ProjectContextFile],
    system: str | None,
    custom_system_prompt: str | None,
    append_system_prompt: str | None,
    profiles: Sequence[CodingSubagentProfile] | None = None,
) -> tuple[SubagentSpec, ...]:
    """Compile validated coding profiles into generic Forge agent specs."""
    active_profiles = tuple(profiles) if profiles is not None else builtin_subagent_profiles()
    if len(active_profiles) > PROFILE_MAX_COUNT:
        raise ValueError(f"subagent registry may contain at most {PROFILE_MAX_COUNT} roles")
    catalog = tools if isinstance(tools, ToolSet) else ToolSet.from_tools(tools)
    base_catalog = coding_subagent_tool_set(catalog)
    tools_by_name = base_catalog.by_name

    specs: list[SubagentSpec] = []
    for profile in active_profiles:
        if profile.name in {"task", "task_control"}:
            raise ValueError(f"subagent profile name is reserved: {profile.name}")
        if not 1 <= profile.max_model_calls <= 8:
            raise ValueError(
                f"subagent profile {profile.name} max_model_calls must be between 1 and 8"
            )
        if not 1024 <= profile.max_result_bytes <= 50 * 1024:
            raise ValueError(
                f"subagent profile {profile.name} max_result_bytes must be between 1024 and 51200"
            )
        if profile.tool_names is None:
            role_catalog = base_catalog
        else:
            if len(set(profile.tool_names)) != len(profile.tool_names):
                raise ValueError(f"duplicate tools in subagent profile: {profile.name}")
            reserved = sorted(set(profile.tool_names) & {"task", "task_control"})
            if reserved:
                raise ValueError(
                    f"subagent profile {profile.name} may not use {', '.join(reserved)}"
                )
            unknown = [name for name in profile.tool_names if name not in tools_by_name]
            if unknown and profile.source != "builtin":
                raise ValueError(
                    f"subagent profile {profile.name} requests unavailable tool(s): "
                    f"{', '.join(unknown)}"
                )
            selected_names = tuple(name for name in profile.tool_names if name in tools_by_name)
            role_catalog = base_catalog.select(selected_names)
        role_tools = role_catalog.registered_tools
        base_prompt = (
            system
            if system is not None
            else build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=cwd,
                    tools=role_catalog,
                    skills=skills,
                    custom_prompt=custom_system_prompt,
                    append_system_prompt=append_system_prompt,
                    context_files=context_files,
                )
            )
        )
        contract = (
            f'<subagent_role source="{profile.source}">\n{profile.prompt.strip()}\n</subagent_role>'
        )
        specs.append(
            SubagentSpec(
                name=profile.name,
                description=profile.description,
                system_prompt=f"{base_prompt}\n\n{contract}",
                tools=role_tools,
                max_model_calls=profile.max_model_calls,
                max_result_bytes=profile.max_result_bytes,
            )
        )
    return tuple(specs)


def create_task_tool_definition(
    runner: SubagentRunner,
    *,
    background: SubagentRuns | None = None,
    interactive: bool = False,
    profile_modes: Mapping[str, str] | None = None,
) -> ToolDefinition:
    """Create the parent-facing ``task(agent, instruction[, mode])`` definition."""

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        del signal, context
        unexpected = sorted(set(arguments) - {"agent", "instruction", "mode"})
        if unexpected:
            raise ValueError(f"Unexpected task argument(s): {', '.join(unexpected)}")
        agent = arguments.get("agent")
        instruction = arguments.get("instruction")
        if not isinstance(agent, str):
            raise ValueError("Subagent agent must be a string")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Subagent instruction must not be empty")
        requested_mode = arguments.get("mode")
        if requested_mode is not None and requested_mode not in {"foreground", "background"}:
            raise ValueError("Subagent mode must be foreground or background")
        configured_mode = profile_modes.get(agent, "foreground") if profile_modes else "foreground"
        if configured_mode not in {"foreground", "background"}:
            raise ValueError("Invalid configured subagent mode")
        mode = cast(TaskMode, requested_mode or configured_mode)
        if mode == "background":
            if not interactive:
                raise ValueError("Background subagents require an interactive session")
            if background is None:
                raise ValueError("Background subagents are unavailable")
            receipt = background.submit(agent=agent, instruction=instruction)
            if not isinstance(receipt, Mapping):
                raise ValueError("Background manager returned an invalid receipt")
            payload = _json_payload(receipt)
            accepted = payload.get("status") == "accepted"
            return AgentToolResult(
                tool_call_id="",
                name="task",
                ok=accepted,
                content=_json_content(payload),
                data=payload,
                error=None if accepted else "Background subagent was not accepted",
            )
        result = await runner.run(
            agent=agent,
            instruction=instruction,
        )
        ok = result.status != "failed"
        return AgentToolResult(
            tool_call_id="",
            name="task",
            ok=ok,
            content=result.content,
            data=result.to_artifact(),
            error=None if ok else result.error,
        )

    specs = runner.specs
    if len(specs) > PROFILE_MAX_COUNT:
        raise ValueError(f"subagent registry may contain at most {PROFILE_MAX_COUNT} roles")
    available = "\n".join(
        f"- {spec.name}: {spec.description}"
        + (
            f" (default mode: {profile_modes[spec.name]})"
            if profile_modes is not None and spec.name in profile_modes
            else ""
        )
        for spec in specs
    )
    task_description = (
        f"{TASK_TOOL_DESCRIPTION}\n\nAvailable subagents:\n{available}"
        if available
        else f"{TASK_TOOL_DESCRIPTION}\n\nAvailable subagents: none"
    )
    if len(task_description.encode("utf-8")) > TASK_REGISTRY_MAX_DESCRIPTION_BYTES:
        raise ValueError(
            "subagent registry description exceeds "
            f"{TASK_REGISTRY_MAX_DESCRIPTION_BYTES} UTF-8 bytes"
        )
    prompt_snippet = (
        "Delegate one bounded task to " + ", ".join(spec.name for spec in specs) + "."
        if specs
        else "No subagents are currently available."
    )
    tool = _create_native_tool(
        name="task",
        description=task_description,
        args_schema=TaskToolInput,
        executor=execute,
    )
    return ToolDefinition(
        tool=tool,
        label="task",
        prompt_snippet=prompt_snippet,
        prompt_guidelines=TASK_PROMPT_GUIDELINES,
    )


def create_task_tool(
    runner: SubagentRunner,
    *,
    background: SubagentRuns | None = None,
    interactive: bool = False,
    profile_modes: Mapping[str, str] | None = None,
) -> ForgeStructuredTool:
    """Create the parent-facing native ``task(agent, instruction)`` tool."""

    return create_task_tool_definition(
        runner,
        background=background,
        interactive=interactive,
        profile_modes=profile_modes,
    ).tool  # type: ignore[return-value]


def create_task_control_tool_definition(manager: SubagentRuns) -> ToolDefinition:
    """Create the parent-facing native background-run control tool."""

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        del signal, context
        unexpected = sorted(set(arguments) - {"action", "run_id", "timeout_seconds"})
        if unexpected:
            raise ValueError(f"Unexpected task_control argument(s): {', '.join(unexpected)}")
        action = arguments.get("action")
        if not isinstance(action, str) or action not in TASK_CONTROL_ACTIONS:
            raise ValueError("task_control action must be list, status, wait, or cancel")
        run_id = arguments.get("run_id")
        if run_id is not None and (
            not isinstance(run_id, str) or not run_id or len(run_id) > TASK_RUN_ID_MAX_LENGTH
        ):
            raise ValueError(f"run_id must be 1-{TASK_RUN_ID_MAX_LENGTH} characters")
        if action == "list" and run_id is not None:
            raise ValueError("run_id is only valid for status, wait, or cancel")
        if action != "list" and run_id is None:
            raise ValueError("run_id is required for status, wait, or cancel")
        timeout = arguments.get("timeout_seconds", 10)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout_seconds must be a number")
        if (
            not math.isfinite(float(timeout))
            or timeout < 0
            or timeout > TASK_CONTROL_TIMEOUT_MAX_SECONDS
        ):
            raise ValueError(
                f"timeout_seconds must be between 0 and {int(TASK_CONTROL_TIMEOUT_MAX_SECONDS)}"
            )
        result = await manager.control(
            action=action,
            run_id=run_id if isinstance(run_id, str) else None,
            timeout_seconds=float(timeout),
        )
        if not isinstance(result, Mapping):
            raise ValueError("Background manager returned an invalid control result")
        payload = _json_payload(result)
        status = payload.get("status")
        failed = status in {"failed", "cancelled", "interrupted", "error"}
        error = payload.get("error")
        error_text = (
            str(error) if failed and error else "Background subagent failed" if failed else None
        )
        return AgentToolResult(
            tool_call_id="",
            name="task_control",
            ok=not failed,
            content=_json_content(payload),
            data=payload,
            error=error_text,
        )

    tool = _create_native_tool(
        name="task_control",
        description=TASK_CONTROL_DESCRIPTION,
        args_schema=TaskControlInput,
        executor=execute,
    )
    return ToolDefinition(tool=tool, label="task_control", prompt_snippet=TASK_CONTROL_DESCRIPTION)


def _json_payload(value: Mapping[str, object]) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], dict(value))


def _json_content(value: Mapping[str, JSONValue]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def ensure_task_name_available(tools: Sequence[BaseTool]) -> None:
    """Reject custom tools that collide with Forge's built-in subagent tools."""
    for reserved in ("task", "task_control"):
        if any(tool.name == reserved for tool in tools):
            raise ValueError(
                f"CodingSession tool name '{reserved}' is reserved when subagents are enabled; "
                "rename the custom tool or set enable_subagents=False"
            )
