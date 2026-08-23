"""Forge's product-level metadata for native LangChain tools.

The native ``BaseTool`` remains the only source of provider-visible name,
description, input schema and execution.  ``ToolDefinition`` intentionally
keeps only the product label and prompt metadata needed by coding-session
composition.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.tools import BaseTool

from forge_agent.tools import ToolExecutor
from forge_agent.types import JSONValue


class ToolDefinition:
    """A native tool plus Forge-only display and prompt metadata.

    ``tool`` is deliberately stored by identity.  The compatibility
    properties below derive all runtime facts from it instead of maintaining a
    second schema/description/executor copy.
    """

    __slots__ = ("tool", "label", "prompt_snippet", "prompt_guidelines")

    tool: BaseTool
    label: str
    prompt_snippet: str | None
    prompt_guidelines: tuple[str, ...]

    def __init__(
        self,
        tool: BaseTool,
        label: str | None = None,
        prompt_snippet: str | None = None,
        prompt_guidelines: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("ToolDefinition tool must be a LangChain BaseTool")
        if label is None:
            raise TypeError("ToolDefinition requires a non-empty label")
        if not label.strip():
            raise ValueError("ToolDefinition label must not be empty")

        object.__setattr__(self, "tool", tool)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "prompt_snippet", prompt_snippet)
        object.__setattr__(
            self,
            "prompt_guidelines",
            tuple(prompt_guidelines),
        )

    def __repr__(self) -> str:
        return (
            f"ToolDefinition(tool={self.tool!r}, label={self.label!r}, "
            f"prompt_snippet={self.prompt_snippet!r}, "
            f"prompt_guidelines={self.prompt_guidelines!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolDefinition):
            return NotImplemented
        return (
            self.tool is other.tool
            and self.label == other.label
            and self.prompt_snippet == other.prompt_snippet
            and self.prompt_guidelines == other.prompt_guidelines
        )

    def __hash__(self) -> int:
        return hash((id(self.tool), self.label, self.prompt_snippet, self.prompt_guidelines))

    @property
    def name(self) -> str:
        """Compatibility alias derived from the native tool."""

        return self.tool.name

    @property
    def description(self) -> str:
        """Compatibility alias derived from the native tool."""

        return self.tool.description

    @property
    def args_schema(self) -> Any:
        """Compatibility alias derived from the native tool."""

        return self.tool.args_schema

    @property
    def input_schema(self) -> Mapping[str, JSONValue]:
        """Return the current model-visible JSON schema from the native tool."""

        schema = self.tool.tool_call_schema
        if isinstance(schema, Mapping):
            raw = cast(dict[str, JSONValue], dict(schema))
        else:
            model_json_schema = getattr(schema, "model_json_schema", None)
            if not callable(model_json_schema):
                return {"type": "object", "properties": {}}
            raw = cast(dict[str, JSONValue], model_json_schema())
        # Keep the historical convenience shape for nullable primitive fields
        # while leaving the provider-visible schema untouched.
        properties = raw.get("properties")
        if isinstance(properties, dict):
            projected = dict(properties)
            for key, value in projected.items():
                if not isinstance(value, dict) or "type" in value:
                    continue
                alternatives = value.get("anyOf")
                if not isinstance(alternatives, list):
                    continue
                primitive = next(
                    (
                        option.get("type")
                        for option in alternatives
                        if isinstance(option, dict) and isinstance(option.get("type"), str)
                    ),
                    None,
                )
                if primitive is not None:
                    projected[key] = {**value, "type": primitive}
            raw["properties"] = projected
        return raw

    @property
    def executor(self) -> ToolExecutor:
        """Return a direct executor for a Forge-native tool.

        Plain custom ``BaseTool`` instances intentionally have no direct
        Forge executor and fail explicitly instead of inventing an adapter.
        """

        from forge_coding.tools.base import ForgeStructuredTool

        if not isinstance(self.tool, ForgeStructuredTool):
            raise AttributeError(f"Tool {self.name!r} does not expose Forge direct execution")
        return self.tool.executor


__all__ = ["ToolDefinition"]
