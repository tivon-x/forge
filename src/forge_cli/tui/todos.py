"""Compact, keyboard-friendly Todo panel for the Forge TUI."""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual.widgets import Static

from forge_agent import TodoItem
from forge_cli.tui.config import FORGE_DARK_THEME, TuiTheme

TODO_MAX_LINES = 12


def todo_symbol(status: str) -> str:
    return {"completed": "✓", "in_progress": "◐", "pending": "○"}.get(status, "?")


def visible_todos(
    todos: Sequence[TodoItem],
    *,
    max_lines: int = TODO_MAX_LINES,
) -> tuple[tuple[TodoItem, ...], int]:
    """Apply the panel line budget, hiding completed rows before truncating."""

    if max_lines <= 1:
        return (), len(todos)
    row_budget = max_lines - 1
    if len(todos) <= row_budget:
        return tuple(todos), 0
    active = tuple(item for item in todos if item.status != "completed")
    # Reserve the final line for the overflow marker so the panel never
    # exceeds its advertised height.
    candidates = active[: max(max_lines - 2, 0)]
    omitted = len(todos) - len(candidates)
    return candidates, omitted


def render_todos(
    todos: Sequence[TodoItem],
    *,
    collapsed: bool = False,
    max_lines: int = TODO_MAX_LINES,
    theme: TuiTheme = FORGE_DARK_THEME,
) -> Text:
    """Render a complete Todo snapshot as a bounded Rich text block."""

    if not todos:
        return Text("")
    completed = sum(item.status == "completed" for item in todos)
    if max_lines <= 1:
        return Text(f"● Todos ({completed}/{len(todos)})", style=theme.accent)
    if collapsed:
        return Text(
            f"● Todos ({completed}/{len(todos)}) · Ctrl+Shift+T to expand",
            style=theme.muted_text,
        )
    rows, omitted = visible_todos(todos, max_lines=max_lines)
    output = Text(f"● Todos ({completed}/{len(todos)})", style=theme.accent)
    for item in rows:
        output.append("\n  ")
        output.append(
            todo_symbol(item.status),
            style=theme.success if item.status == "completed" else theme.accent,
        )
        output.append(f" {item.content}", style=theme.screen_text)
    if omitted:
        output.append(f"\n  +{omitted} more", style=theme.muted_text)
    return output


class TodoPanel(Static):
    """The panel shown immediately above the queued messages and composer."""

    DEFAULT_CSS = """
    TodoPanel {
        height: auto;
        max-height: 12;
        margin: 0 1 0 1;
        padding: 0 1;
        background: $forge-screen-background;
        color: $forge-screen-text;
    }
    """

    def update_from_state(
        self,
        todos: Sequence[TodoItem],
        *,
        collapsed: bool = False,
        theme: TuiTheme = FORGE_DARK_THEME,
    ) -> None:
        self.display = bool(todos)
        self.update(render_todos(todos, collapsed=collapsed, theme=theme))


__all__ = ["TODO_MAX_LINES", "TodoPanel", "render_todos", "todo_symbol", "visible_todos"]
