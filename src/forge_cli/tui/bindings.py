"""Keybinding-to-Textual-Binding builders for the Forge TUI.

Every configurable action resolves to one or more Textual ``Binding`` objects
(pi-style multi-key semantics).  This module owns the mapping from
``TuiKeybindings`` to the bindings installed on the app and the prompt editor,
plus the ``/hotkeys`` help text rendered from the user's actual configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from textual.binding import Binding

from forge_cli.tui.autocomplete import CompletionState
from forge_cli.tui.config import KEYBINDING_ACTION_LABELS, KEYBINDING_ACTIONS, TuiKeybindings
from forge_cli.tui.state import TuiState


def _is_thinking_cycle_key(key: str, keybindings: TuiKeybindings) -> bool:
    keys = keybindings.keys_for("thinking_cycle")
    if key in keys:
        return True
    return "shift+tab" in keys and key == "backtab"


def _hotkeys_help_text(keybindings: TuiKeybindings) -> str:
    """Render the user's actual configured keybindings for ``/hotkeys``."""
    lines = ["Forge keyboard shortcuts (configured in ~/.forge/tui.json, /reload applies):"]
    for action in KEYBINDING_ACTIONS:
        label = KEYBINDING_ACTION_LABELS.get(action, action)
        lines.append(f"- {label}: {keybindings.key_display(action)}")
    return "\n".join(lines)


def _prompt_footer_mode(
    state: TuiState,
    completion_state: CompletionState,
) -> Literal["normal", "completion", "running"]:
    if completion_state.items:
        return "completion"
    if state.running:
        return "running"
    return "normal"


def _multi_binding(
    keybindings: TuiKeybindings,
    action: str,
    handler: str,
    description: str,
    *,
    show: bool,
    priority: bool = False,
) -> Binding | None:
    """Build one Binding for an action, accepting every configured key."""
    keys = keybindings.keys_for(action)
    if not keys:
        return None
    return Binding(
        ",".join(keys),
        handler,
        description,
        show=show,
        priority=priority,
    )


def _app_bindings(keybindings: TuiKeybindings) -> list[Binding]:
    return [
        binding
        for binding in (
            _multi_binding(keybindings, "cancel", "cancel", "Cancel", show=False),
            _multi_binding(
                keybindings, "command_palette", "open_command_palette", "Commands", show=False
            ),
            _multi_binding(
                keybindings, "session_picker", "open_session_picker", "Sessions", show=False
            ),
            _multi_binding(keybindings, "thinking_cycle", "cycle_thinking", "Thinking", show=False),
            _multi_binding(keybindings, "model_cycle", "cycle_model", "Model", show=False),
            _multi_binding(
                keybindings,
                "accept_completion",
                "accept_completion",
                "Complete",
                priority=True,
                show=False,
            ),
            _multi_binding(
                keybindings,
                "queue_follow_up",
                "submit_follow_up",
                "Follow-up",
                priority=True,
                show=False,
            ),
            _multi_binding(
                keybindings,
                "completion_next",
                "completion_next",
                "Next completion",
                priority=True,
                show=False,
            ),
            _multi_binding(
                keybindings,
                "completion_previous",
                "completion_previous",
                "Previous completion",
                priority=True,
                show=False,
            ),
            _multi_binding(
                keybindings,
                "toggle_tool_results",
                "toggle_tool_results",
                "Tool results",
                show=False,
            ),
            _multi_binding(keybindings, "toggle_todos", "toggle_todos", "Todos", show=False),
            _multi_binding(
                keybindings, "toggle_thinking", "toggle_thinking", "Thinking tokens", show=False
            ),
            _multi_binding(keybindings, "copy_message", "clear_prompt", "Clear input", show=False),
            _multi_binding(keybindings, "quit", "quit", "Quit", show=False),
            _multi_binding(
                keybindings, "dequeue_queued", "edit_queued_message", "Restore queued", show=False
            ),
            _multi_binding(
                keybindings, "transcript_search", "open_transcript_search", "Search", show=False
            ),
        )
        if binding is not None
    ]


def _prompt_bindings(
    keybindings: TuiKeybindings,
    *,
    mode: Literal["normal", "completion", "running"],
) -> list[Binding]:
    def joined(action: str) -> str:
        return ",".join(keybindings.keys_for(action))

    if mode == "completion":
        bindings = [
            Binding(
                joined("accept_completion"),
                "accept_completion",
                "Complete",
                key_display=f"{keybindings.key_display('accept_completion')}/Enter",
                priority=True,
            ),
            Binding(
                joined("completion_next"),
                "completion_next",
                "Choose",
                key_display=(
                    f"{keybindings.key_display('completion_previous')}/"
                    f"{keybindings.key_display('completion_next')}"
                ),
                priority=True,
            ),
            Binding(joined("cancel"), "cancel", "Close", priority=True),
        ]
        bindings = [binding for binding in bindings if binding.key]
        return bindings + _hidden_prompt_bindings(keybindings, visible_bindings=bindings)
    if mode == "running":
        bindings = [
            Binding("enter", "submit_prompt", "Steer", priority=True),
            Binding(
                joined("queue_follow_up"),
                "submit_follow_up",
                "Follow-up",
                priority=True,
            ),
            Binding(joined("cancel"), "cancel", "Cancel", priority=True),
            Binding(
                joined("toggle_tool_results"),
                "toggle_tool_results",
                "Tools",
                priority=True,
            ),
        ]
        bindings = [binding for binding in bindings if binding.key]
        return bindings + _hidden_prompt_bindings(keybindings, visible_bindings=bindings)
    bindings = [
        Binding("enter", "submit_prompt", "Submit", priority=True),
        Binding("shift+enter", "insert_newline", "Newline", priority=True),
        Binding(
            joined("command_palette"),
            "open_command_palette",
            "Commands",
            priority=True,
        ),
        Binding(
            joined("session_picker"),
            "open_session_picker",
            "Sessions",
            priority=True,
        ),
    ]
    bindings = [binding for binding in bindings if binding.key]
    return bindings + _hidden_prompt_bindings(keybindings, visible_bindings=bindings)


def _hidden_prompt_bindings(
    keybindings: TuiKeybindings,
    *,
    visible_bindings: Sequence[Binding],
) -> list[Binding]:
    visible_keys = {key for binding in visible_bindings for key in binding.key.split(",")}
    candidates = (
        ("command_palette", "open_command_palette", "Commands"),
        ("session_picker", "open_session_picker", "Sessions"),
        ("queue_follow_up", "submit_follow_up", "Follow-up"),
        ("thinking_cycle", "cycle_thinking", "Thinking"),
        ("model_cycle", "cycle_model", "Model"),
        ("toggle_tool_results", "toggle_tool_results", "Tools"),
        ("toggle_todos", "toggle_todos", "Todos"),
        ("toggle_thinking", "toggle_thinking", "Thinking tokens"),
        ("copy_message", "clear_prompt", "Clear"),
        ("accept_completion", "accept_completion", "Complete"),
        ("completion_next", "completion_next", "Next completion"),
        ("completion_previous", "completion_previous", "Previous completion"),
        ("dequeue_queued", "edit_queued_message", "Restore queued"),
        ("transcript_search", "open_transcript_search", "Search"),
        ("quit", "quit", "Quit"),
    )
    bindings: list[Binding] = []
    for action, handler, description in candidates:
        for key in keybindings.keys_for(action):
            if key in visible_keys:
                continue
            bindings.append(Binding(key, handler, description, show=False, priority=True))
    return bindings
