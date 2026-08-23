"""Built-in coding subagents and the model-facing ``task`` tool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, StrictStr

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

TASK_TOOL_DESCRIPTION = (
    "Delegate one bounded task to a fresh, isolated coding subagent. "
    "Each call accepts exactly one task and returns only the subagent's final result."
)
TASK_PROMPT_GUIDELINES = (
    "Use task only for a clearly bounded assignment that benefits from an isolated context",
    "Give each task enough concrete scope and acceptance criteria to work independently",
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
    base_catalog = ToolSet(tuple(definition for definition in catalog if definition.name != "task"))
    tools_by_name = base_catalog.by_name

    specs: list[SubagentSpec] = []
    for profile in active_profiles:
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
            if "task" in profile.tool_names:
                raise ValueError(f"subagent profile {profile.name} may not use task")
            unknown = [name for name in profile.tool_names if name not in tools_by_name]
            if unknown and profile.source != "builtin":
                raise ValueError(
                    f"subagent profile {profile.name} requests unavailable tool(s): "
                    f"{', '.join(unknown)}"
                )
            selected_names = tuple(name for name in profile.tool_names if name in tools_by_name)
            role_catalog = base_catalog.select(selected_names)
        role_tools = role_catalog.tools
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


def create_task_tool_definition(runner: SubagentRunner) -> ToolDefinition:
    """Create the parent-facing ``task(agent, instruction)`` definition."""

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        del signal, context
        unexpected = sorted(set(arguments) - {"agent", "instruction"})
        if unexpected:
            raise ValueError(f"Unexpected task argument(s): {', '.join(unexpected)}")
        agent = arguments.get("agent")
        instruction = arguments.get("instruction")
        result = await runner.run(
            agent=agent if isinstance(agent, str) else "",
            instruction=instruction if isinstance(instruction, str) else "",
        )
        return AgentToolResult(
            tool_call_id="",
            name="task",
            ok=True,
            content=result.content,
            data=result.to_artifact(),
        )

    specs = runner.specs
    if len(specs) > PROFILE_MAX_COUNT:
        raise ValueError(f"subagent registry may contain at most {PROFILE_MAX_COUNT} roles")
    available = "\n".join(f"- {spec.name}: {spec.description}" for spec in specs)
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


def create_task_tool(runner: SubagentRunner) -> ForgeStructuredTool:
    """Create the parent-facing native ``task(agent, instruction)`` tool."""

    return create_task_tool_definition(runner).tool  # type: ignore[return-value]


def ensure_task_name_available(tools: Sequence[BaseTool]) -> None:
    """Reject a custom tool that collides with Forge's built-in task tool."""
    if any(tool.name == "task" for tool in tools):
        raise ValueError(
            "CodingSession tool name 'task' is reserved when subagents are enabled; "
            "rename the custom tool or set enable_subagents=False"
        )
