"""Todo planning helpers built around LangChain's official middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import SystemMessage
from pydantic import ValidationError

from forge_agent import TodoItem
from forge_agent.session import CustomEntry
from forge_agent.types import JSONValue

TODO_NAMESPACE = "forge.todo.v1"


class _ForgeTodoListMiddleware(TodoListMiddleware):
    """Use the official tool while keeping simple system prompts as strings."""

    def __init__(self, *, include_system_prompt: bool = True) -> None:
        super().__init__()
        self._include_system_prompt = include_system_prompt

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._with_system_prompt(request))

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        return await handler(self._with_system_prompt(request))

    def _with_system_prompt(self, request: Any) -> Any:
        if not self._include_system_prompt:
            return request
        system_message = request.system_message
        if system_message is None:
            content: Any = self.system_prompt
        elif isinstance(system_message.content, str):
            content = f"{system_message.content}\n\n{self.system_prompt}"
        else:
            content = [
                *system_message.content,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        return request.override(system_message=SystemMessage(content=content))


def validate_todos(value: object) -> tuple[TodoItem, ...]:
    """Validate a complete LangChain todo snapshot.

    The middleware deliberately replaces the whole list.  Forge therefore keeps
    a small strict projection: each item must be an object with non-empty text
    and one of LangChain's three statuses.  Unknown fields are rejected so a
    future/foreign state cannot silently become durable product data.
    """

    if (
        value is None
        or isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
    ):
        raise ValueError("todos must be a list")
    result: list[TodoItem] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("todo items must be objects")
        try:
            item = TodoItem.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("invalid todo item") from exc
        if not item.content.strip():
            raise ValueError("todo content must not be empty")
        result.append(item)
    return tuple(result)


def todos_to_json(todos: Sequence[TodoItem]) -> list[dict[str, str]]:
    """Return a stable JSON-safe todo representation."""

    return [{"content": item.content, "status": item.status} for item in todos]


def todo_entry_data(todos: Sequence[TodoItem]) -> dict[str, JSONValue]:
    """Build the versioned CustomEntry payload for a todo snapshot."""

    return {"todos": cast(list[JSONValue], todos_to_json(todos))}


def todos_from_custom_entry(entry: CustomEntry) -> tuple[TodoItem, ...] | None:
    """Decode one versioned todo CustomEntry, ignoring malformed data."""

    if entry.namespace != TODO_NAMESPACE or set(entry.data) != {"todos"}:
        return None
    try:
        return validate_todos(entry.data.get("todos"))
    except ValueError:
        return None


def latest_todo_snapshot(entries: Sequence[CustomEntry]) -> tuple[TodoItem, ...]:
    """Return the last valid todo snapshot on a replayed branch."""

    latest: tuple[TodoItem, ...] = ()
    for entry in entries:
        snapshot = todos_from_custom_entry(entry)
        if snapshot is not None:
            latest = snapshot
    return latest


def create_todo_middleware(*, include_system_prompt: bool = True) -> TodoListMiddleware:
    """Create Forge's default official LangChain todo middleware."""

    return _ForgeTodoListMiddleware(include_system_prompt=include_system_prompt)


def format_todos(todos: Sequence[TodoItem]) -> str:
    """Format a complete todo list for ``/todos`` and plain renderers."""

    if not todos:
        return "No active todos."
    completed = sum(item.status == "completed" for item in todos)
    lines = [f"Todos ({completed}/{len(todos)}):"]
    symbols = {"completed": "✓", "in_progress": "◐", "pending": "○"}
    for item in todos:
        lines.append(f"{symbols[item.status]} {item.content}")
    return "\n".join(lines)


__all__ = [
    "TODO_NAMESPACE",
    "TodoItem",
    "create_todo_middleware",
    "format_todos",
    "latest_todo_snapshot",
    "todo_entry_data",
    "todos_from_custom_entry",
    "todos_to_json",
    "validate_todos",
]
