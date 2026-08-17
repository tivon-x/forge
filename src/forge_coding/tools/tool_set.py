"""Ordered product catalog for coding-session tools."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType

from langchain_core.tools import BaseTool

from forge_coding.tools.definition import ToolDefinition


class ToolSet:
    """An immutable ordered catalog of ``ToolDefinition`` objects.

    ``ToolSet`` has no execution state.  It only provides deterministic
    ordering, lookup and profile selection while ``BaseTool`` remains the
    runtime object passed to LangChain.
    """

    __slots__ = ("_definitions", "_by_name")

    def __init__(self, definitions: Iterable[ToolDefinition] = ()) -> None:
        ordered = tuple(definitions)
        by_name: dict[str, ToolDefinition] = {}
        for definition in ordered:
            if not isinstance(definition, ToolDefinition):
                raise TypeError("ToolSet entries must be ToolDefinition instances")
            name = definition.name
            if not name or not name.strip():
                raise ValueError("ToolSet tool names must not be empty")
            if name in by_name:
                raise ValueError(f"duplicate tool name: {name}")
            by_name[name] = definition
        self._definitions = ordered
        self._by_name = MappingProxyType(by_name)

    @classmethod
    def from_tools(cls, tools: Iterable[BaseTool | ToolDefinition]) -> ToolSet:
        """Build a catalog from native tools and/or existing definitions."""

        definitions: list[ToolDefinition] = []
        for item in tools:
            if isinstance(item, ToolDefinition):
                definitions.append(item)
            elif isinstance(item, BaseTool):
                definitions.append(
                    ToolDefinition(
                        tool=item,
                        label=item.name,
                        prompt_snippet=getattr(item, "prompt_snippet", None),
                        prompt_guidelines=tuple(getattr(item, "prompt_guidelines", ())),
                    )
                )
            else:
                raise TypeError("ToolSet tools must be BaseTool or ToolDefinition instances")
        return cls(definitions)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return definitions in their stable catalog order."""

        return self._definitions

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        """Return native tools in the same order as ``definitions``."""

        return tuple(definition.tool for definition in self._definitions)

    @property
    def by_name(self) -> Mapping[str, ToolDefinition]:
        """Return an immutable name-to-definition mapping."""

        return self._by_name

    def select(self, names: Sequence[str] | Iterable[str]) -> ToolSet:
        """Select an ordered subset, rejecting duplicate/unknown names."""

        requested = tuple(names)
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate tool name in selection")
        unknown = [name for name in requested if name not in self._by_name]
        if unknown:
            raise ValueError(f"unknown tool name(s): {', '.join(unknown)}")
        requested_set = set(requested)
        return ToolSet(
            definition for definition in self._definitions if definition.name in requested_set
        )

    def with_tools(self, *tools: BaseTool | ToolDefinition) -> ToolSet:
        """Append plain native tools (or definitions) to this catalog."""

        additions: Iterable[BaseTool | ToolDefinition]
        if len(tools) == 1 and not isinstance(tools[0], (BaseTool, ToolDefinition)):
            candidate = tools[0]
            additions = candidate if isinstance(candidate, Iterable) else tools
        else:
            additions = tools
        added = ToolSet.from_tools(additions)
        return ToolSet((*self._definitions, *added.definitions))

    def __iter__(self) -> Iterator[ToolDefinition]:
        return iter(self._definitions)

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name


__all__ = ["ToolSet"]
