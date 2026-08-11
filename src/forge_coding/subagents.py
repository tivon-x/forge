"""Built-in coding subagents and the model-facing ``task`` tool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from forge_agent import AgentToolResult, JSONValue, SubagentRunner, SubagentSpec
from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import ToolCancellationToken
from forge_coding.skills import Skill
from forge_coding.system_prompt import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    build_system_prompt,
)
from forge_coding.tools import ForgeStructuredTool, ToolDefinition

TASK_TOOL_DESCRIPTION = (
    "Delegate one bounded task to a fresh, isolated coding subagent. "
    "Use scout to investigate code, worker to implement a scoped change, or reviewer to "
    "independently review existing work. Each call accepts exactly one task and returns only "
    "the subagent's final result."
)
TASK_PROMPT_SNIPPET = (
    "Delegate one bounded task to scout (investigation), worker (implementation), or reviewer "
    "(independent review)"
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

    agent: Literal["scout", "worker", "reviewer"] = Field(description="Subagent role")
    instruction: str = Field(
        min_length=1,
        description="One self-contained, bounded task",
    )


_ROLE_DEFINITIONS = (
    (
        "scout",
        "Investigate code and collect evidence without editing files.",
        frozenset({"read", "bash"}),
        """You are Forge's scout subagent. Investigate only; do not modify files. Locate the
relevant files, symbols, and call paths, support conclusions with concrete evidence, and report
remaining uncertainty. Your bash access is a convenience, not a read-only security sandbox, so
do not run commands that change the workspace.""",
    ),
    (
        "worker",
        "Implement one clearly bounded coding change and verify it.",
        None,
        """You are Forge's worker subagent. Implement only the explicitly assigned scope. Reuse
the existing design, make the smallest coherent change, run targeted validation, and report the
files changed, checks run, and any remaining risk. Do not broaden the task.""",
    ),
    (
        "reviewer",
        "Independently review existing code or a proposed change without editing files.",
        frozenset({"read", "bash"}),
        """You are Forge's reviewer subagent. Review only; do not modify files. Report actionable
findings first with file and symbol locations, then state remaining risks or missing validation.
Your bash access is a convenience, not a read-only security sandbox, so do not run commands that
change the workspace.""",
    ),
)


def create_coding_subagent_specs(
    *,
    cwd: Path,
    tools: Sequence[BaseTool],
    skills: Sequence[Skill],
    context_files: Sequence[ProjectContextFile],
    system: str | None,
    custom_system_prompt: str | None,
    append_system_prompt: str | None,
) -> tuple[SubagentSpec, ...]:
    """Build the three built-in role specs from the current session resources."""
    specs: list[SubagentSpec] = []
    for name, description, allowed_names, contract in _ROLE_DEFINITIONS:
        role_tools = tuple(
            tool
            for tool in tools
            if tool.name != "task" and (allowed_names is None or tool.name in allowed_names)
        )
        base_prompt = (
            system
            if system is not None
            else build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=cwd,
                    tools=role_tools,
                    skills=skills,
                    custom_prompt=custom_system_prompt,
                    append_system_prompt=append_system_prompt,
                    context_files=context_files,
                )
            )
        )
        specs.append(
            SubagentSpec(
                name=name,
                description=description,
                system_prompt=f"{base_prompt}\n\n<subagent_role>\n{contract}\n</subagent_role>",
                tools=role_tools,
                max_model_calls=8,
                max_result_bytes=50 * 1024,
            )
        )
    return tuple(specs)


def create_task_tool(runner: SubagentRunner) -> ForgeStructuredTool:
    """Create the parent-facing ``task(agent, instruction)`` tool."""

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

    definition = ToolDefinition(
        name="task",
        description=TASK_TOOL_DESCRIPTION,
        prompt_snippet=TASK_PROMPT_SNIPPET,
        prompt_guidelines=TASK_PROMPT_GUIDELINES,
        input_schema={
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": ["scout", "worker", "reviewer"],
                    "description": "Built-in subagent role",
                },
                "instruction": {
                    "type": "string",
                    "description": "One self-contained, bounded task",
                },
            },
            "required": ["agent", "instruction"],
            "additionalProperties": False,
        },
        executor=execute,
        args_schema=TaskToolInput,
    )
    return definition.to_langchain_tool()


def ensure_task_name_available(tools: Sequence[BaseTool]) -> None:
    """Reject a custom tool that collides with Forge's built-in task tool."""
    if any(tool.name == "task" for tool in tools):
        raise ValueError(
            "CodingSession tool name 'task' is reserved when subagents are enabled; "
            "rename the custom tool or set enable_subagents=False"
        )
