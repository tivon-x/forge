"""Shared low-level types for Forge's portable agent layer."""

from __future__ import annotations

from typing import Protocol

# Pydantic needs PEP 695 named recursive aliases for JSON-like values.
type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]


def stripped_text(value: str | None, *, field: str = "text") -> str | None:
    """Strip surrounding whitespace; ``None`` passes through; empty raises."""

    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


class CancellationToken(Protocol):
    """Minimal cooperative cancellation interface accepted by the agent loop."""

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        ...
