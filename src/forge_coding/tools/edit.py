"""LangChain-native exact text edit tool."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken
from forge_agent.types import JSONValue
from forge_coding.tools.base import (
    ForgeStructuredTool,
    ToolInputError,
    _create_native_tool,
)
from forge_coding.tools.common import (
    _edits_arg,
    _path_arg,
    _prepare_edit_arguments,
    _workspace_root,
)
from forge_coding.tools.definition import ToolDefinition
from forge_coding.tools.edit_diff import (
    _strip_bom,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
)
from forge_coding.tools.file_operation_queue import file_operation_queue


class EditItemInput(BaseModel):
    """One exact replacement in an ``edit`` request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    oldText: str = Field(description="Exact text to replace", coerce_numbers_to_str=True)
    newText: str = Field(description="Replacement text", coerce_numbers_to_str=True)


class EditToolInput(BaseModel):
    """Validated model-facing arguments for the ``edit`` tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: StrictStr = Field(description="Path to the file to edit")
    edits: list[EditItemInput] = Field(
        min_length=1,
        description="One or more targeted replacements",
    )


def create_edit_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create a definition for the `edit` tool.

    The tool applies one or more exact text replacements to a single UTF-8 file
    resolved relative to `cwd`. Each edit item contains `oldText` and `newText`.
    Every `oldText` must be non-empty, must occur exactly once in the original
    file, and must not overlap another edit span. All replacements are validated
    before writing, so the file is left unchanged if any edit fails.

    File content and edit text are normalized to LF for matching, then the
    original file's dominant line ending is restored after replacement. UTF-8
    byte-order marks are preserved. The executor also accepts top-level
    `oldText`/`newText` arguments and JSON-string `edits` values by normalizing
    them into the canonical edits list.

    Successful results include the resolved path, edit count, an ndiff-style
    diff, a unified patch, and the first changed line in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        context: ForgeRuntimeContext | None = None,
    ) -> AgentToolResult:
        del signal
        prepared = _prepare_edit_arguments(arguments)
        workspace = _workspace_root(context, root)
        path = _path_arg(prepared, "path", cwd=workspace, for_write=True)
        edits = _edits_arg(prepared)

        async with file_operation_queue.operation(path):
            if not path.exists():
                raise ToolInputError(f"Could not edit file: {path}. File not found.")
            if path.is_dir():
                raise ToolInputError(f"Could not edit file: {path}. Path is a directory.")

            raw_content = path.read_text(encoding="utf-8")
            bom, content = _strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized = normalize_to_lf(content)
            base_content, new_content = apply_edits_to_normalized_content(
                normalized, edits, str(path)
            )
            final_content = bom + restore_line_endings(new_content, original_ending)
            path.write_text(final_content, encoding="utf-8")

        diff_text, first_changed_line = generate_diff_string(base_content, new_content)
        patch = generate_unified_patch(str(path), base_content, new_content)
        return AgentToolResult(
            tool_call_id="",
            name="edit",
            ok=True,
            content=f"Successfully replaced {len(edits)} block(s) in {path}.",
            data={
                "path": str(path),
                "edits": len(edits),
                "diff": diff_text,
                "patch": patch,
                "first_changed_line": first_changed_line,
            },
        )

    tool = _create_native_tool(
        name="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match "
            "a unique, non-overlapping region of the original file. If two changes affect the "
            "same block or nearby lines, merge them into one edit instead of emitting overlapping "
            "edits. Do not include large unchanged regions just to connect distant changes."
        ),
        executor=execute,
        args_schema=EditToolInput,
    )
    return ToolDefinition(
        tool=tool,
        label="edit",
        prompt_snippet=(
            "Make precise file edits with exact text replacement, including multiple disjoint "
            "edits in one call"
        ),
        prompt_guidelines=(
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with "
            "multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier "
            "edits are applied. Do not emit overlapping or nested edits. Merge nearby "
            "changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. "
            "Do not pad with large unchanged regions.",
        ),
    )


def create_edit_tool(*, cwd: str | Path | None = None) -> ForgeStructuredTool:
    """Create a native LangChain tool for exact validated text replacements."""
    return create_edit_tool_definition(cwd=cwd).tool  # type: ignore[return-value]
