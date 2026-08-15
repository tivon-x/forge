"""Goal presentation and the small ``/goal`` manager used by the TUI.

The goal coordinator owns lifecycle and persistence.  This module deliberately
keeps only a presentation snapshot and emits a tiny action intent back to the
application.  In particular, the objective and all status text are rendered as
plain text so model-provided markup can never become Textual markup.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Literal, Protocol

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static, TextArea

from forge_cli.tui.config import FORGE_DARK_THEME, TuiTheme

GOAL_MAX_AUTOMATIC_RUNS = 25

GoalActionKind = Literal["start", "pause", "resume", "edit", "status", "clear"]


class GoalSnapshotLike(Protocol):
    """The immutable fields consumed by the presentation layer."""

    id: str
    objective: str
    status: str


@dataclass(frozen=True, slots=True)
class GoalAction:
    """A user intent emitted by :class:`GoalManagerScreen`.

    The coding/session layer may use its own immutable command-intent class.  A
    dataclass with the same ``kind``/``objective`` shape keeps the TUI decoupled
    from that backend type while remaining straightforward to adapt.
    """

    kind: GoalActionKind
    objective: str | None = None
    goal_id: str | None = None
    replace: bool = False

    @property
    def action(self) -> GoalActionKind:
        """Backend-compatible alias used by ``GoalCommandAction``."""

        return self.kind


GoalActionHandler = Callable[[GoalAction], object | Awaitable[object]]


def goal_value(goal: object | None, name: str, default: Any = None) -> Any:
    """Read a snapshot field from either a model or a test-friendly mapping."""

    if goal is None:
        return default
    if isinstance(goal, dict):
        return goal.get(name, default)
    return getattr(goal, name, default)


def _goal_status(goal: object | None) -> str | None:
    status = goal_value(goal, "status")
    return status if isinstance(status, str) else None


def _automatic_limit(goal: object | None) -> int:
    value = goal_value(goal, "automatic_run_limit", GOAL_MAX_AUTOMATIC_RUNS)
    if isinstance(value, int) and value > 0:
        return value
    return GOAL_MAX_AUTOMATIC_RUNS


def _automatic_runs(goal: object | None) -> int:
    value = goal_value(goal, "automatic_runs", 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _status_reason(goal: object | None) -> str | None:
    reason = goal_value(goal, "stop_reason")
    return reason if isinstance(reason, str) and reason else None


def render_goal_status(
    goal: object | None,
    *,
    theme: TuiTheme = FORGE_DARK_THEME,
) -> Text:
    """Render the single-line goal projection shown above the composer.

    The line intentionally contains no objective text.  The manager is the one
    place where the full objective and completion details are shown.
    """

    status = _goal_status(goal)
    if status is None:
        return Text("")

    runs = _automatic_runs(goal)
    limit = _automatic_limit(goal)
    reason = _status_reason(goal)
    if status == "active":
        detail = f"automatic {runs}/{limit}"
        style = theme.accent
    elif status == "paused":
        if reason in {"automatic_limit", "automatic-limit", "limit"}:
            detail = f"automatic limit {runs}/{limit}"
        elif reason in {"no_progress", "no-progress"}:
            detail = "no progress"
        elif reason:
            detail = _pretty_reason(reason)
        else:
            detail = "paused"
        style = getattr(theme, "warning", theme.accent)
    elif status == "blocked":
        detail = "blocked"
        style = theme.error
    elif status == "complete":
        detail = "complete"
        style = theme.success
    else:
        detail = status
        style = theme.muted_text

    output = Text("🎯 ", style=style)
    output.append(status, style=style)
    if detail != status:
        output.append(" · ", style=theme.muted_text)
        output.append(detail, style=style)
    return output


def _pretty_reason(reason: str) -> str:
    return reason.replace("_", " ").replace("-", " ")


# Descriptive alias for callers that prefer the widget-oriented name.
render_goal_status_line = render_goal_status


def goal_manager_actions(
    goal: object | None,
) -> tuple[tuple[GoalActionKind | Literal["help", "close"], str], ...]:
    """Return the action labels for the current snapshot."""

    status = _goal_status(goal)
    if status is None:
        return (("start", "Start a goal…"), ("help", "Help"), ("close", "Close"))
    if status == "active":
        first: tuple[GoalActionKind | Literal["help", "close"], str] = (
            "pause",
            "Pause goal",
        )
        return (
            first,
            ("edit", "Edit goal…"),
            ("status", "View full status"),
            ("clear", "Clear goal…"),
            ("help", "Help"),
            ("close", "Close"),
        )
    if status in {"paused", "blocked"}:
        first_label = (
            "Review and continue…"
            if _status_reason(goal)
            in {
                "automatic_limit",
                "automatic-limit",
                "no_progress",
                "no-progress",
            }
            else "Resume goal"
        )
        return (
            ("resume", first_label),
            ("edit", "Edit goal…"),
            ("status", "View full status"),
            ("clear", "Clear goal…"),
            ("help", "Help"),
            ("close", "Close"),
        )
    # A completed goal is retained until the next user turn.  Starting a new
    # goal is therefore safe without a replacement confirmation.
    return (
        ("start", "Start a goal…"),
        ("status", "View full status"),
        ("clear", "Clear goal…"),
        ("help", "Help"),
        ("close", "Close"),
    )


class GoalStatusLine(Static):
    """The compact, non-interactive goal status row above the prompt."""

    DEFAULT_CSS = """
    GoalStatusLine {
        height: auto;
        max-height: 1;
        margin: 0 1 0 1;
        padding: 0 1;
        background: $forge-screen-background;
        color: $forge-screen-text;
        overflow-x: hidden;
    }
    """

    def update_from_goal(
        self,
        goal: object | None,
        *,
        theme: TuiTheme = FORGE_DARK_THEME,
    ) -> None:
        self.display = goal is not None and _goal_status(goal) is not None
        self.update(render_goal_status(goal, theme=theme))


class GoalEditorScreen(ModalScreen[str | None]):
    """Multiline objective editor used by Start and Edit actions."""

    DEFAULT_CSS = """
    GoalEditorScreen {
        align: center middle;
    }

    #goal-editor {
        width: 72;
        max-width: 90%;
        height: 10;
        max-height: 50%;
        padding: 1;
        background: $forge-prompt-background;
        border: tall $forge-border;
    }

    #goal-editor-help {
        width: 72;
        max-width: 90%;
        height: 1;
        margin-top: 1;
        color: $forge-muted-text;
    }
    """

    def __init__(self, objective: str = "", *, theme: TuiTheme = FORGE_DARK_THEME) -> None:
        self.objective = objective
        self.theme = theme
        super().__init__()

    def compose(self) -> ComposeResult:
        yield TextArea(
            self.objective,
            id="goal-editor",
            placeholder="Describe the goal…",
            highlight_cursor_line=False,
        )
        yield Label(
            "Ctrl+Enter save · Enter newline · Esc cancel",
            id="goal-editor-help",
            markup=False,
        )

    def on_mount(self) -> None:
        self.query_one("#goal-editor", TextArea).focus()

    def on_key(self, event: Key) -> None:
        if event.key in {"escape", "esc"}:
            event.stop()
            self.dismiss(None)
            return
        if event.key in {"ctrl+enter", "ctrl+return"}:
            event.stop()
            text = self.query_one("#goal-editor", TextArea).text.strip()
            self.dismiss(text or None)


class GoalConfirmScreen(ModalScreen[bool]):
    """Small yes/no confirmation used for destructive/replacing actions."""

    DEFAULT_CSS = """
    GoalConfirmScreen {
        align: center middle;
    }

    #goal-confirm {
        width: 72;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        background: $forge-chrome-background;
        border: tall $forge-border;
    }

    #goal-confirm-text {
        height: auto;
        color: $forge-chrome-text;
    }

    #goal-confirm-help {
        height: 1;
        margin-top: 1;
        color: $forge-muted-text;
    }
    """

    def __init__(self, message: str, *, theme: TuiTheme = FORGE_DARK_THEME) -> None:
        self.message = message
        self.theme = theme
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="goal-confirm"):
            yield Label(self.message, id="goal-confirm-text", markup=False)
            yield Label("Enter/y confirm · n/Esc cancel", id="goal-confirm-help", markup=False)

    def on_key(self, event: Key) -> None:
        if event.key in {"escape", "esc", "n"}:
            event.stop()
            self.dismiss(False)
        elif event.key in {"enter", "return", "y"}:
            event.stop()
            self.dismiss(True)


class GoalManagerScreen(ModalScreen[GoalAction | None]):
    """Keyboard-first manager for the session's current Goal snapshot."""

    DEFAULT_CSS = """
    GoalManagerScreen {
        align: center middle;
    }

    #goal-manager {
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $forge-chrome-background;
        border: tall $forge-border;
    }

    #goal-manager-scroll {
        width: 1fr;
        height: auto;
        max-height: 16;
        overflow-y: auto;
    }

    #goal-manager-title {
        height: auto;
        color: $forge-chrome-text;
        text-style: bold;
        margin-bottom: 1;
    }

    #goal-manager-objective,
    #goal-manager-details,
    #goal-manager-feedback {
        height: auto;
        color: $forge-screen-text;
        overflow-x: hidden;
    }

    #goal-manager-details {
        margin-top: 1;
        color: $forge-muted-text;
    }

    #goal-manager-list {
        height: auto;
        max-height: 12;
        margin-top: 1;
        background: $forge-transcript-background;
        border: tall $forge-border;
    }

    #goal-manager-help {
        width: 1fr;
        height: 1;
        margin-top: 1;
        color: $forge-muted-text;
    }
    """

    def __init__(
        self,
        goal: object | None = None,
        *,
        theme: TuiTheme = FORGE_DARK_THEME,
        on_action: GoalActionHandler | None = None,
    ) -> None:
        self.goal = goal
        self.theme = theme
        self.on_action = on_action
        self._actions: tuple[tuple[GoalActionKind | Literal["help", "close"], str], ...] = ()
        super().__init__()

    def compose(self) -> ComposeResult:
        self._actions = goal_manager_actions(self.goal)
        with Vertical(id="goal-manager"):
            with VerticalScroll(id="goal-manager-scroll"):
                yield Label(self._title(), id="goal-manager-title", markup=False)
                yield Static(self._objective_text(), id="goal-manager-objective", markup=False)
                yield Static(self._details_text(), id="goal-manager-details", markup=False)
                yield ListView(
                    *(
                        ListItem(Label(label, markup=False), id=f"goal-action-{kind}")
                        for kind, label in self._actions
                    ),
                    id="goal-manager-list",
                )
                yield Static("", id="goal-manager-feedback", markup=False)
            yield Label(
                "↑↓/j k navigate · Enter select · Esc close",
                id="goal-manager-help",
                markup=False,
            )

    def on_mount(self) -> None:
        self.query_one("#goal-manager-list", ListView).focus()

    def update_goal(self, goal: object | None) -> None:
        """Refresh visible snapshot details when an event arrives while open."""

        self.goal = goal
        actions = goal_manager_actions(goal)
        actions_changed = actions != self._actions
        self._actions = actions
        # The screen can be dismissed between an event and this refresh.
        with suppress(NoMatches):
            self.query_one("#goal-manager-title", Label).update(self._title())
            self.query_one("#goal-manager-objective", Static).update(self._objective_text())
            self.query_one("#goal-manager-details", Static).update(self._details_text())
            if actions_changed:
                list_view = self.query_one("#goal-manager-list", ListView)
                if len(list_view.children) == len(actions):
                    for item, (_, label) in zip(list_view.children, actions, strict=True):
                        item.query_one(Label).update(label)
                else:
                    self.call_after_refresh(self._rebuild_action_list)

    async def _rebuild_action_list(self) -> None:
        with suppress(NoMatches):
            list_view = self.query_one("#goal-manager-list", ListView)
            await list_view.remove_children()
            await list_view.mount(
                *(
                    ListItem(Label(label, markup=False), id=f"goal-action-{kind}")
                    for kind, label in self._actions
                )
            )

    @on(ListView.Selected)
    def on_action_selected(self, event: ListView.Selected) -> None:
        event.stop()
        if event.list_view.id != "goal-manager-list":
            return
        if event.index < 0 or event.index >= len(self._actions):
            return
        self._select(self._actions[event.index][0])

    def on_key(self, event: Key) -> None:
        if event.key in {"escape", "esc"}:
            event.stop()
            self.dismiss(None)
            return
        list_view = self.query_one("#goal-manager-list", ListView)
        if event.key in {"j", "down"}:
            event.stop()
            list_view.action_cursor_down()
        elif event.key in {"k", "up"}:
            event.stop()
            list_view.action_cursor_up()

    def _select(self, kind: GoalActionKind | Literal["help", "close"]) -> None:
        if kind == "close":
            self.dismiss(None)
            return
        if kind == "help":
            self.query_one("#goal-manager-feedback", Static).update(
                "Start a goal and Forge will continue bounded work until it is complete or "
                "blocked. Pause to review external changes; resume to continue with a new "
                "Goal ID."
            )
            return
        if kind == "status":
            self.query_one("#goal-manager-feedback", Static).update(self._full_status())
            return
        if kind == "start":
            self._push_editor("start", "")
            return
        snapshot_id = goal_value(self.goal, "id")
        goal_id = snapshot_id if isinstance(snapshot_id, str) and snapshot_id else None
        objective = str(goal_value(self.goal, "objective", ""))
        if kind == "edit":
            if _goal_status(self.goal) == "active":
                self.app.push_screen(
                    GoalConfirmScreen(
                        "Editing an active goal replaces it and rotates the Goal ID. Continue?",
                        theme=self.theme,
                    ),
                    lambda confirmed: self._after_edit_confirmation(confirmed, goal_id, objective),
                )
            else:
                self._push_editor("edit", objective, goal_id=goal_id)
            return
        if kind == "clear":
            self.app.push_screen(
                GoalConfirmScreen(
                    "Clear the current goal? This cannot be undone.",
                    theme=self.theme,
                ),
                lambda confirmed: self._after_clear_confirmation(confirmed, goal_id),
            )
            return
        self._emit(GoalAction(kind))

    def _after_edit_confirmation(
        self,
        confirmed: bool | None,
        goal_id: str | None,
        objective: str,
    ) -> None:
        if confirmed:
            self._push_editor("edit", objective, goal_id=goal_id)

    def _after_clear_confirmation(self, confirmed: bool | None, goal_id: str | None) -> None:
        if confirmed:
            self._emit(GoalAction("clear", goal_id=goal_id))

    def _push_editor(
        self,
        kind: Literal["start", "edit"],
        objective: str,
        *,
        goal_id: str | None = None,
    ) -> None:
        self.app.push_screen(
            GoalEditorScreen(objective, theme=self.theme),
            lambda value: self._after_editor(kind, value, goal_id),
        )

    def _after_editor(
        self,
        kind: Literal["start", "edit"],
        objective: str | None,
        goal_id: str | None,
    ) -> None:
        if objective is None:
            return
        self._emit(GoalAction(kind, objective, goal_id))

    def _emit(self, action: GoalAction) -> None:
        if action.goal_id is None and action.kind not in {"start", "status"}:
            snapshot_id = goal_value(self.goal, "id")
            if isinstance(snapshot_id, str) and snapshot_id:
                action = GoalAction(action.kind, action.objective, snapshot_id)
        if self.on_action is None:
            self.dismiss(action)
            return
        result = self.on_action(action)
        # A callback may be async (the normal app path).  Textual owns the
        # worker's lifecycle; synchronous test callbacks remain deterministic.
        if isawaitable(result):
            self.run_worker(result, exclusive=False)
        self.dismiss(None)

    def _title(self) -> str:
        status = _goal_status(self.goal)
        return f"Goal · {_pretty_status(status)}" if status else "Goal · No goal"

    def _objective_text(self) -> str:
        objective = goal_value(self.goal, "objective", "")
        if not isinstance(objective, str) or not objective:
            return "No goal is currently set"
        return objective

    def _details_text(self) -> str:
        if self.goal is None or _goal_status(self.goal) is None:
            return f"Automatic work pauses after {GOAL_MAX_AUTOMATIC_RUNS} runs"
        runs = _automatic_runs(self.goal)
        limit = _automatic_limit(self.goal)
        status = _goal_status(self.goal)
        if status in {"active", "paused"}:
            remaining = max(limit - runs, 0)
            return f"Automatic work: {runs} of {limit} runs · {remaining} remaining"
        return self._full_status()

    def _full_status(self) -> str:
        if self.goal is None:
            return "No goal is currently set"
        status = _goal_status(self.goal) or "unknown"
        bits = [
            f"Status: {status}",
            f"Automatic runs: {_automatic_runs(self.goal)}/{_automatic_limit(self.goal)}",
        ]
        reason = _status_reason(self.goal)
        if reason:
            bits.append(f"Stop reason: {_pretty_reason(reason)}")
        summary = goal_value(self.goal, "completion_summary")
        if isinstance(summary, str) and summary:
            bits.append(f"Summary: {summary}")
        return "\n".join(bits)


def _pretty_status(status: str | None) -> str:
    if status is None:
        return "No goal"
    return status.capitalize()


__all__ = [
    "GOAL_MAX_AUTOMATIC_RUNS",
    "GoalAction",
    "GoalConfirmScreen",
    "GoalEditorScreen",
    "GoalManagerScreen",
    "GoalSnapshotLike",
    "GoalStatusLine",
    "goal_manager_actions",
    "goal_value",
    "render_goal_status",
    "render_goal_status_line",
]
