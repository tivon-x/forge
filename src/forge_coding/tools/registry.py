"""Factories for Forge's default coding-tool set."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from forge_coding.tools.base import ForgeStructuredTool
from forge_coding.tools.bash import create_bash_tool_definition
from forge_coding.tools.edit import create_edit_tool_definition
from forge_coding.tools.read import create_read_tool_definition
from forge_coding.tools.tool_set import ToolSet
from forge_coding.tools.write import create_write_tool_definition


def create_coding_tool_set(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
) -> ToolSet:
    """Create the built-in coding tools in stable product order."""

    root = Path.cwd() if cwd is None else Path(cwd)
    return ToolSet(
        (
            create_read_tool_definition(cwd=root),
            create_write_tool_definition(cwd=root),
            create_edit_tool_definition(cwd=root),
            create_bash_tool_definition(
                cwd=root,
                shell_command_prefix=shell_command_prefix,
            ),
        )
    )


def create_coding_tools(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
) -> list[ForgeStructuredTool]:
    """Create native tools in stable prompt and execution order.

    This compatibility factory intentionally returns only ``BaseTool``
    instances; callers that need Forge prompt metadata should use
    :func:`create_coding_tool_set`.
    """

    return [
        cast(ForgeStructuredTool, tool)
        for tool in create_coding_tool_set(
            cwd=cwd,
            shell_command_prefix=shell_command_prefix,
        ).tools
    ]
