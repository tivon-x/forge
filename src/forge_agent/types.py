"""Shared low-level types for Forge's portable agent layer."""

from __future__ import annotations

from typing import Protocol

# Pydantic needs PEP 695 named recursive aliases for JSON-like values.
type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]


class CancellationToken(Protocol):
    """Minimal cooperative cancellation interface accepted by the agent loop."""

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        ...
