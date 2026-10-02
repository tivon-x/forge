"""Fleet status and inspector widgets for session-owned background tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, ClassVar, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click, Key
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListItem, ListView, Static

from forge_agent.subagents import visible_subagent_text
from forge_cli.tui.config import FORGE_DARK_THEME, TuiTheme
from forge_cli.tui.screens import BindingEntry

FleetAction = Literal["collect"]
FleetStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "unknown",
]

_STATUSES: tuple[str, ...] = (
    "running",
    "queued",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "unknown",
)
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
_CONTEXTS = frozenset({"fresh", "fork", "summary"})
_FALLBACK_KINDS = frozenset(
    {"transient", "overflow", "auth", "quota", "invalid_request", "unknown"}
)
_STATUS_SYMBOLS = {
    "queued": "◌",
    "running": "◌",
    "completed": "✓",
    "failed": "×",
    "cancelled": "–",
    "interrupted": "!",
    "unknown": "?",
}


def _text(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _bounded_text(value: object, *, limit: int = 64_000) -> str:
    text = _text(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… [truncated]"


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def _optional_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _context(value: object) -> str:
    return value if isinstance(value, str) and value in _CONTEXTS else "fresh"


def _fallback_kinds(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    for item in value[:8]:
        if isinstance(item, Mapping):
            item = item.get("kind", item.get("reason"))
        if not isinstance(item, str):
            continue
        kind = item.strip().lower()
        if kind not in _FALLBACK_KINDS:
            kind = "unknown"
        if kind not in result:
            result.append(kind)
    return tuple(result)


def _duration_label(duration_ms: int) -> str:
    if duration_ms < 1_000:
        return f"{duration_ms}ms"
    return f"{duration_ms / 1_000:.1f}s"


def _tokens_label(run: FleetRun) -> str:
    if run.total_tokens is not None:
        return f"{run.total_tokens} tokens"
    values = [value for value in (run.input_tokens, run.output_tokens) if value is not None]
    return f"{sum(values)} tokens" if values else "tokens –"


@dataclass(frozen=True, slots=True)
class FleetRun:
    """Allowlisted status facts used by the Fleet projection."""

    run_id: str
    agent: str
    status: FleetStatus
    model: str
    duration_ms: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    activity: str
    notification: bool
    baseline: str | None
    actual_provider: str = ""
    actual_model: str = ""
    context: str = "fresh"
    inherited_messages: int = 0
    context_truncated: bool = False
    model_attempts: int = 0
    fallback_reasons: tuple[str, ...] = ()

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> FleetRun | None:
        run_id = _text(snapshot.get("run_id")).strip()
        if not run_id:
            return None
        raw_status = _text(snapshot.get("status"), "unknown")
        status: FleetStatus = raw_status if raw_status in _STATUSES else "unknown"  # type: ignore[assignment]
        activity = _text(snapshot.get("activity"), status)
        model = _bounded_text(snapshot.get("model"), limit=128) or "unknown model"
        actual_provider = _bounded_text(snapshot.get("actual_provider"), limit=128)
        actual_model = _bounded_text(snapshot.get("actual_model"), limit=128) or model
        return cls(
            run_id=run_id,
            agent=_text(snapshot.get("agent"), "subagent"),
            status=status,
            model=model,
            duration_ms=_integer(snapshot.get("duration_ms")),
            tool_calls=_integer(snapshot.get("tool_calls")),
            input_tokens=_optional_integer(snapshot.get("input_tokens")),
            output_tokens=_optional_integer(snapshot.get("output_tokens")),
            total_tokens=_optional_integer(snapshot.get("total_tokens")),
            activity=activity,
            notification=snapshot.get("notification") is True,
            baseline=_text(snapshot.get("baseline")) or None,
            actual_provider=actual_provider,
            actual_model=actual_model,
            context=_context(snapshot.get("context", snapshot.get("context_mode"))),
            inherited_messages=_integer(
                snapshot.get("inherited_messages", snapshot.get("inherited_message_count"))
            ),
            context_truncated=snapshot.get("context_truncated") is True,
            model_attempts=_integer(snapshot.get("model_attempts")),
            fallback_reasons=_fallback_kinds(snapshot.get("fallback_reasons")),
        )


@dataclass(frozen=True, slots=True)
class FleetScreenResult:
    """Action returned when the user asks the parent to read a task result."""

    action: FleetAction
    run_id: str


def normalize_fleet_runs(
    snapshots: Sequence[Mapping[str, object]] | Sequence[object],
) -> tuple[FleetRun, ...]:
    """Keep only valid, backend-owned run snapshots in their supplied order."""
    runs: list[FleetRun] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        run = FleetRun.from_snapshot(snapshot)
        if run is not None:
            runs.append(run)
    return tuple(runs)


def _run_row(run: FleetRun) -> str:
    actual = "/".join(value for value in (run.actual_provider, run.actual_model) if value)
    model = f"{run.model} → {actual}" if actual and actual != run.model else run.model
    return (
        f"{_STATUS_SYMBOLS[run.status]} {run.run_id} · {run.agent} · {run.status} · "
        f"{model} · {_duration_label(run.duration_ms)} · "
        f"{run.tool_calls} tools · {_tokens_label(run)}"
    )


def _summary_text(runs: Sequence[FleetRun]) -> str:
    if not runs:
        return ""
    parts = [f"{len(runs)} run" if len(runs) == 1 else f"{len(runs)} runs"]
    for status in _STATUSES:
        count = sum(run.status == status for run in runs)
        if count:
            parts.append(f"{count} {status}")
    return "Fleet: " + " · ".join(parts) + " · /fleet to inspect"


class FleetSummary(Static):
    """One-line composer status for every session-owned background task."""

    def update_from_snapshots(self, snapshots: Sequence[Mapping[str, object]]) -> None:
        runs = normalize_fleet_runs(snapshots)
        self.display = bool(runs)
        self.update(_summary_text(runs))

    def on_click(self, event: Click) -> None:
        """Open Fleet when the composer summary is clicked."""
        event.stop()
        action = getattr(self.app, "action_open_fleet", None)
        if callable(action):
            action()


class FleetListView(ListView):
    """List that keeps Fleet navigation local to its screen."""

    async def _on_key(self, event: Key) -> None:
        screen = self.screen
        if not isinstance(screen, FleetScreen):
            await self.handle_key(event)
            return
        actions = {
            "up": screen.action_cursor_up,
            "down": screen.action_cursor_down,
            "enter": screen.action_inspect,
            "escape": screen.action_back,
            "c": screen.action_cancel_selected,
            "r": screen.action_collect,
            "y": screen.action_copy_visible,
        }
        action = actions.get(event.key)
        if action is None:
            await self.handle_key(event)
            return
        event.stop()
        event.prevent_default()
        action()


class FleetScreen(ModalScreen[FleetScreenResult | None]):
    """Keyboard-first Fleet list and detail inspector."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "back", "Back", priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("enter", "inspect", "Inspect", show=False, priority=True),
        Binding("c", "cancel_selected", "Cancel", priority=True),
        Binding("r", "collect", "Read", priority=True),
        Binding("y", "copy_visible", "Copy", priority=True),
    ]

    def __init__(
        self,
        snapshots: Sequence[Mapping[str, object]],
        *,
        theme: TuiTheme = FORGE_DARK_THEME,
        on_inspect: Callable[[str], Mapping[str, object] | None] | None = None,
        on_cancel: Callable[[str], object] | None = None,
    ) -> None:
        super().__init__()
        self.theme = theme
        self._runs = normalize_fleet_runs(snapshots)
        self._on_inspect = on_inspect
        self._on_cancel = on_cancel
        self._details: dict[str, Mapping[str, object]] = {}
        self._selected_run_id: str | None = None
        self._cancel_pending: set[str] = set()
        self.view: Literal["list", "detail"] = "list"

    @property
    def runs(self) -> tuple[FleetRun, ...]:
        """Return the current immutable list projection."""
        return self._runs

    def compose(self) -> ComposeResult:
        """Compose the list/detail inspector."""
        with Vertical(id="fleet-screen"):
            yield Static("Fleet · background tasks", id="fleet-title", markup=False)
            yield Static("", id="fleet-summary", markup=False)
            yield FleetListView(id="fleet-list")
            with VerticalScroll(id="fleet-detail-scroll"):
                yield Static("", id="fleet-detail", markup=False)
            with Horizontal(id="fleet-actions"):
                yield Button("Cancel selected", id="fleet-cancel")
                yield Button("Read into prompt", id="fleet-collect")
                yield Button("Copy visible", id="fleet-copy")
            yield Static(
                "↑/↓ select · Enter inspect · c cancel · r read result · y copy · Esc back",
                id="fleet-help",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the list without taking focus from the parent composer permanently."""
        self.call_after_refresh(self._finish_mount)

    def _finish_mount(self) -> None:
        """Wait for the nested panel widgets to mount before querying them."""
        if not self.is_mounted:
            return
        self._refresh_list()
        self.query_one("#fleet-list", ListView).focus()

    def on_key(self, event: Key) -> None:
        """Route navigation keys before a focused list consumes them."""
        actions = {
            "up": self.action_cursor_up,
            "down": self.action_cursor_down,
            "enter": self.action_inspect,
            "escape": self.action_back,
            "c": self.action_cancel_selected,
            "r": self.action_collect,
            "y": self.action_copy_visible,
        }
        action = actions.get(event.key)
        if action is None:
            return
        event.stop()
        event.prevent_default()
        action()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Open a row selected by Enter or a mouse click."""
        event.stop()
        self.action_inspect()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Expose the same actions as visible buttons."""
        event.stop()
        if event.button.id == "fleet-cancel":
            self.action_cancel_selected()
        elif event.button.id == "fleet-collect":
            self.action_collect()
        elif event.button.id == "fleet-copy":
            self.action_copy_visible()

    def update_runs(self, snapshots: Sequence[Mapping[str, object]]) -> None:
        """Refresh the projection while preserving the selected exact run id."""
        previous_runs = {run.run_id: run for run in self._runs}
        self._runs = normalize_fleet_runs(snapshots)
        if not self.query("#fleet-list"):
            return
        selected_id = self._selected_id()
        if selected_id is not None:
            self._selected_run_id = selected_id
        if self.view == "detail" and self._selected_run() is None:
            self._show_list()
            return
        selected = self._selected_run()
        if selected is not None and previous_runs.get(selected.run_id) != selected:
            self._details.pop(selected.run_id, None)
        self._refresh_list()
        if self.view == "detail":
            self._refresh_detail()

    def action_cursor_up(self) -> None:
        """Move the list selection or scroll the detail."""
        if self.view == "detail":
            self._scroll_detail(-1)
            return
        self.query_one("#fleet-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move the list selection or scroll the detail."""
        if self.view == "detail":
            self._scroll_detail(1)
            return
        self.query_one("#fleet-list", ListView).action_cursor_down()

    def action_inspect(self) -> None:
        """Inspect the selected run through the session-owned callback."""
        if self.view == "detail":
            return
        run = self._selected_run()
        if run is None:
            return
        self.view = "detail"
        self._refresh_visibility()
        self._refresh_detail()
        self.query_one("#fleet-detail-scroll", VerticalScroll).focus()

    def action_back(self) -> None:
        """Return from detail to list, then close the modal."""
        if self.view == "detail":
            self._show_list()
            return
        self.dismiss(None)

    def action_cancel_selected(self) -> None:
        """Ask the parent session to cancel the selected exact run id."""
        run = self._selected_run()
        if run is None or run.run_id in self._cancel_pending or self._on_cancel is None:
            return
        if run.status in _TERMINAL_STATUSES:
            return
        self._cancel_pending.add(run.run_id)
        self.query_one("#fleet-cancel", Button).disabled = True
        result = self._on_cancel(run.run_id)
        if isawaitable(result):
            self.run_worker(self._finish_cancel(run.run_id, result), exclusive=True)
        else:
            self._cancel_pending.discard(run.run_id)
            self.query_one("#fleet-cancel", Button).disabled = False

    async def _finish_cancel(self, run_id: str, operation: Any) -> None:
        try:
            await operation
        finally:
            self._cancel_pending.discard(run_id)
            if self.is_mounted:
                self.query_one("#fleet-cancel", Button).disabled = False

    def action_collect(self) -> None:
        """Return a terminal run id so the parent can ask task_control for it."""
        run = self._selected_run()
        if run is None or run.status not in _TERMINAL_STATUSES:
            return
        self.dismiss(FleetScreenResult(action="collect", run_id=run.run_id))

    def action_copy_visible(self) -> None:
        """Copy only the currently visible inspector payload."""
        if self.view != "detail":
            return
        copy_to_clipboard = getattr(self.app, "copy_to_clipboard", None)
        if not callable(copy_to_clipboard):
            return
        copy_to_clipboard(str(self.query_one("#fleet-detail", Static).render()))

    def _selected_id(self) -> str | None:
        if self.view == "detail":
            return self._selected_run_id
        list_view = self.query_one("#fleet-list", ListView)
        index = list_view.index
        if index is None or index >= len(self._runs):
            return self._selected_run_id
        self._selected_run_id = self._runs[index].run_id
        return self._selected_run_id

    def _selected_run(self) -> FleetRun | None:
        selected_id = self._selected_id()
        if selected_id is None:
            return None
        return next((run for run in self._runs if run.run_id == selected_id), None)

    def _refresh_list(self) -> None:
        list_view = self.query_one("#fleet-list", ListView)
        selected_id = self._selected_run_id
        list_view.clear()
        list_view.extend(ListItem(Label(_run_row(run), markup=False)) for run in self._runs)
        if not self._runs:
            list_view.index = None
            self._selected_run_id = None
        else:
            selected_index = next(
                (index for index, run in enumerate(self._runs) if run.run_id == selected_id),
                0,
            )
            list_view.index = selected_index
            self._selected_run_id = self._runs[selected_index].run_id
        self.query_one("#fleet-summary", Static).update(_summary_text(self._runs))
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        self.query_one("#fleet-list", ListView).display = self.view == "list"
        self.query_one("#fleet-detail-scroll", VerticalScroll).display = self.view == "detail"
        selected = self._selected_run()
        can_cancel = selected is not None and selected.status not in _TERMINAL_STATUSES
        self.query_one("#fleet-cancel", Button).display = (
            self.view == "list" or selected is not None
        )
        self.query_one("#fleet-cancel", Button).disabled = not can_cancel or (
            selected is not None and selected.run_id in self._cancel_pending
        )
        self.query_one("#fleet-collect", Button).display = (
            self.view == "detail" and selected is not None and selected.status in _TERMINAL_STATUSES
        )
        self.query_one("#fleet-copy", Button).display = self.view == "detail"

    def _refresh_detail(self) -> None:
        run = self._selected_run()
        if run is None:
            self.query_one("#fleet-detail", Static).update("No task selected.")
            return
        detail = self._details.get(run.run_id)
        if detail is None and self._on_inspect is not None:
            result = self._on_inspect(run.run_id)
            if result is not None:
                detail = result
                self._details[run.run_id] = result
        if detail is None:
            detail = {}
        self.query_one("#fleet-detail", Static).update(_detail_text(run, detail))
        self._refresh_visibility()

    def _show_list(self) -> None:
        self.view = "list"
        self._refresh_list()
        self.query_one("#fleet-list", ListView).focus()

    def _scroll_detail(self, delta: int) -> None:
        scroll = self.query_one("#fleet-detail-scroll", VerticalScroll)
        scroll.scroll_y = max(0, min(scroll.max_scroll_y, scroll.scroll_y + delta))


def _detail_text(run: FleetRun, detail: Mapping[str, object]) -> str:
    result = detail.get("result")
    result_map = result if isinstance(result, Mapping) else {}
    baseline = _text(detail.get("baseline")) or _text(result_map.get("baseline")) or run.baseline
    actual_provider = run.actual_provider
    raw_actual_provider = detail.get("actual_provider")
    if isinstance(raw_actual_provider, str) and raw_actual_provider.strip():
        actual_provider = _bounded_text(raw_actual_provider, limit=128)
    actual_model = run.actual_model
    raw_actual_model = detail.get("actual_model")
    if isinstance(raw_actual_model, str) and raw_actual_model.strip():
        actual_model = _bounded_text(raw_actual_model, limit=128)
    context = run.context
    raw_context = detail.get("context", detail.get("context_mode"))
    if isinstance(raw_context, str) and raw_context in _CONTEXTS:
        context = raw_context
    inherited_messages = run.inherited_messages
    raw_inherited = detail.get("inherited_messages", detail.get("inherited_message_count"))
    if type(raw_inherited) is int and raw_inherited >= 0:
        inherited_messages = raw_inherited
    context_truncated = (
        detail["context_truncated"]
        if isinstance(detail.get("context_truncated"), bool)
        else run.context_truncated
    )
    model_attempts = run.model_attempts
    raw_attempts = detail.get("model_attempts")
    if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool) and raw_attempts >= 0:
        model_attempts = raw_attempts
    fallback_reasons = (
        _fallback_kinds(detail["fallback_reasons"])
        if "fallback_reasons" in detail
        else run.fallback_reasons
    )
    lines = [
        f"{_STATUS_SYMBOLS[run.status]} {run.run_id}",
        f"agent: {run.agent}",
        f"model: {run.model}",
        *([f"actual provider: {actual_provider}"] if actual_provider else []),
        *([f"actual model: {actual_model}"] if actual_model else []),
        f"status: {run.status} · activity: {run.activity}",
        f"duration: {_duration_label(run.duration_ms)} · tools: {run.tool_calls}",
        (
            f"tokens: input {run.input_tokens if run.input_tokens is not None else '–'} · "
            f"output {run.output_tokens if run.output_tokens is not None else '–'} · "
            f"total {run.total_tokens if run.total_tokens is not None else '–'}"
        ),
        f"context: {context} · {inherited_messages} inherited"
        f"{' · truncated' if context_truncated else ''}",
        f"model attempts: {model_attempts}",
        *([f"fallback: {', '.join(fallback_reasons)}"] if fallback_reasons else []),
        *([f"baseline: {baseline}"] if baseline else []),
        "",
        "Result:",
    ]
    final_output = _bounded_text(visible_subagent_text(result_map.get("final_output")))
    if final_output:
        lines.append(final_output)
    else:
        lines.append("No result available.")

    trace = detail.get("trace")
    trace_map = trace if isinstance(trace, Mapping) else {}
    raw_items = trace_map.get("items")
    if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes)):
        lines.extend(("", "Trace:"))
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            raw_kind = _text(raw_item.get("kind"))
            kind = (
                raw_kind
                if raw_kind
                in {
                    "human",
                    "assistant",
                    "tool_call",
                    "tool_result",
                    "omitted",
                }
                else "item"
            )
            tool = _bounded_text(raw_item.get("tool"), limit=128)
            status = _text(raw_item.get("status"))
            if status not in {"ok", "error"}:
                status = ""
            text = _bounded_text(visible_subagent_text(raw_item.get("text")))
            suffix = " · ".join(value for value in (tool, status) if value)
            lines.append(f"{kind}{f' · {suffix}' if suffix else ''}: {text}".rstrip())
    return "\n".join(lines)
