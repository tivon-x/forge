"""Prompt editor for Forge's Textual TUI.

``PromptInput`` is a Textual ``TextArea`` subclass that owns submission,
completion routing, kill-ring editing, large-paste placeholders, and the
external-editor flow.  Keeping it separate from ``app.py`` makes the editor a
reusable component instead of a private implementation detail of the app shell.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, cast

from rich.text import Text
from textual import events
from textual.binding import BindingsMap
from textual.events import Key
from textual.widgets import TextArea

from forge_cli.tui.bindings import _is_thinking_cycle_key, _prompt_bindings
from forge_cli.tui.config import TuiKeybindings
from forge_cli.tui.screens import BindingEntry


class CompletionActionTarget(Protocol):
    """App actions used by the prompt input completion bindings."""

    def action_accept_completion(self) -> None: ...

    def action_cancel(self) -> None: ...

    def action_completion_next(self) -> None: ...

    def action_completion_previous(self) -> None: ...

    def action_open_command_palette(self) -> None: ...

    def action_open_session_picker(self) -> None: ...

    def action_cycle_thinking(self) -> None: ...

    def action_cycle_model(self) -> None: ...

    def action_toggle_tool_results(self) -> None: ...

    def action_toggle_todos(self) -> None: ...

    def action_toggle_thinking(self) -> None: ...

    def action_open_transcript_search(self) -> None: ...

    def action_edit_queued_message(self) -> bool: ...

    async def action_submit_prompt(self) -> None: ...

    async def action_submit_follow_up(self) -> None: ...


PASTE_DISPLAY_THRESHOLD = 2_000


PASTE_DISPLAY_THRESHOLD = 2_000


class PromptInput(TextArea):
    """Multiline prompt input with completion key bindings."""

    BINDINGS: ClassVar[list[BindingEntry]] = []
    shell_mode_style: str = ""

    def __init__(
        self,
        *,
        tui_keybindings: TuiKeybindings | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("highlight_cursor_line", False)
        super().__init__(**kwargs)
        self.tui_keybindings = tui_keybindings or TuiKeybindings()
        self._base_bindings = self._bindings.copy()
        self._footer_mode: Literal["normal", "completion", "running"] = "normal"
        self._pending_pastes: list[tuple[str, str]] = []
        self._paste_placeholder_counter = 0
        self._kill_ring: list[str] = []
        self._yank_index = 0
        self._last_yank_span: tuple[int, int] | None = None
        self._apply_prompt_bindings()

    def set_footer_mode(self, mode: Literal["normal", "completion", "running"]) -> None:
        """Switch the prompt bindings shown by Textual's built-in footer."""
        if mode == self._footer_mode:
            return
        self._footer_mode = mode
        self._apply_prompt_bindings()
        self.refresh_bindings()

    def _apply_prompt_bindings(self) -> None:
        self._bindings = BindingsMap.merge(
            [
                self._base_bindings,
                BindingsMap(_prompt_bindings(self.tui_keybindings, mode=self._footer_mode)),
            ]
        )

    @property
    def value(self) -> str:
        """Compatibility alias for tests and code that previously used Input.value."""
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.text = text

    @property
    def cursor_position(self) -> int:
        """Return a flat cursor offset for Input compatibility."""
        row, column = self.cursor_location
        lines = self.text.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + column

    @cursor_position.setter
    def cursor_position(self, offset: int) -> None:
        text = self.text
        bounded = max(0, min(offset, len(text)))
        before = text[:bounded]
        self.move_cursor((before.count("\n"), len(before.rsplit("\n", 1)[-1])))

    def action_accept_completion(self) -> None:
        """Accept the selected app-level completion."""
        self._completion_target().action_accept_completion()

    def action_completion_next(self) -> None:
        """Select the next app-level completion or move down in the prompt."""
        if self._has_completion_options():
            self._completion_target().action_completion_next()
        else:
            self.action_cursor_down()

    def action_completion_previous(self) -> None:
        """Select the previous app-level completion or move up in the prompt."""
        if self._has_completion_options():
            self._completion_target().action_completion_previous()
        elif self._completion_target().action_edit_queued_message():
            return
        else:
            self.action_cursor_up()

    def action_cancel(self) -> None:
        """Run the app-level cancel action."""
        self._completion_target().action_cancel()

    def action_open_command_palette(self) -> None:
        """Open the app-level command palette."""
        self._completion_target().action_open_command_palette()

    def action_open_session_picker(self) -> None:
        """Open the app-level session picker."""
        self._completion_target().action_open_session_picker()

    def action_cycle_thinking(self) -> None:
        """Cycle the app-level thinking mode."""
        self._completion_target().action_cycle_thinking()

    def action_cycle_model(self) -> None:
        """Cycle the app-level scoped model."""
        self._completion_target().action_cycle_model()

    def action_toggle_tool_results(self) -> None:
        """Toggle app-level tool result display."""
        self._completion_target().action_toggle_tool_results()

    def action_toggle_todos(self) -> None:
        """Toggle app-level Todo panel visibility."""
        self._completion_target().action_toggle_todos()

    def action_toggle_thinking(self) -> None:
        """Toggle app-level thinking-token display."""
        self._completion_target().action_toggle_thinking()

    def action_clear_prompt(self) -> None:
        """Clear the current prompt."""
        if self.selected_text:
            return
        if self.text:
            self.text = ""
            self.move_cursor((0, 0))
            self._clear_pending_paste()

    def get_line(self, line_index: int) -> Text:
        """Retrieve one prompt line with shell prefixes highlighted."""
        line = super().get_line(line_index)
        if line_index != 0 or not self.shell_mode_style:
            return line
        span = _terminal_command_prefix_span(self.text)
        if span is None:
            return line
        start, end = span
        line.stylize(self.shell_mode_style, start, end)
        return line

    async def action_submit_follow_up(self) -> None:
        """Submit the prompt as an app-level follow-up."""
        await self._completion_target().action_submit_follow_up()

    async def action_submit_prompt(self) -> None:
        """Submit the prompt through the app-level action."""
        await self._completion_target().action_submit_prompt()

    def action_insert_newline(self) -> None:
        """Insert a newline in the prompt."""
        self.insert("\n")

    async def action_quit(self) -> None:
        """Quit the app through the app-level action."""
        await self.app.action_quit()

    def action_scroll_down(self) -> None:
        """Use down arrow for completion selection while focused."""
        self.action_completion_next()

    def action_scroll_up(self) -> None:
        """Use up arrow for completion selection while focused."""
        self.action_completion_previous()

    def on_paste(self, event: events.Paste) -> None:
        """Show a compact placeholder instead of rendering very large pasted text."""
        if len(event.text) <= PASTE_DISPLAY_THRESHOLD:
            return
        event.stop()
        event.prevent_default()
        self._show_large_paste_placeholder(event.text)

    def _show_large_paste_placeholder(self, content: str) -> None:
        """Store large pasted text and render a compact placeholder."""
        self._paste_placeholder_counter += 1
        placeholder = self._large_paste_placeholder(content, self._paste_placeholder_counter)
        self._pending_pastes.append((placeholder, content))
        self.insert(placeholder)

    def _large_paste_placeholder(self, content: str, paste_number: int) -> str:
        """Build the display text for a large paste."""
        char_count = len(content)
        line_count = content.count("\n") + 1
        kb = char_count / 1024
        parts: list[str] = [f"{char_count:,} characters"]
        if line_count > 1:
            parts.append(f"{line_count} lines")
        if kb >= 1:
            parts.append(f"{kb:.1f} KB")
        return f"[Pasted content #{paste_number}: {', '.join(parts)}]"

    def _clear_pending_paste(self) -> None:
        """Forget any stored large paste content."""
        self._pending_pastes.clear()

    def sync_pending_paste(self) -> None:
        """Invalidate stored paste content when its placeholder is edited away."""
        self._pending_pastes = [
            (placeholder, content)
            for placeholder, content in self._pending_pastes
            if placeholder in self.text
        ]

    def text_for_submission(self) -> str:
        """Return the prompt text, expanding intact large-paste placeholders."""
        self.sync_pending_paste()
        text = self.text
        for placeholder, content in self._pending_pastes:
            text = text.replace(placeholder, content, 1)
        return text

    def _matches(self, action: str, key: str) -> bool:
        """Return whether a key is bound to one configurable action."""
        return key in self.tui_keybindings.keys_for(action)

    def _kill_text(self, start: int, end: int) -> None:
        """Remove ``text[start:end]`` and push it onto the kill ring."""
        if end <= start:
            return
        text = self.text
        killed = text[start:end]
        self.text = text[:start] + text[end:]
        self.move_cursor(_text_location_from_offset(self.text, start))
        self._push_kill(killed)
        self._last_yank_span = None

    def _push_kill(self, killed: str) -> None:
        """Add one killed fragment to the bounded kill ring."""
        if not killed:
            return
        if self._kill_ring and self._kill_ring[-1] == killed:
            return
        self._kill_ring.append(killed)
        if len(self._kill_ring) > 20:
            self._kill_ring = self._kill_ring[-20:]
        self._yank_index = len(self._kill_ring) - 1

    def _kill_word_backward(self) -> None:
        """Delete the whitespace-delimited word before the cursor."""
        text = self.text
        cursor = self.cursor_position
        while cursor > 0 and text[cursor - 1].isspace():
            cursor -= 1
        while cursor > 0 and not text[cursor - 1].isspace():
            cursor -= 1
        self._kill_text(cursor, self.cursor_position)

    def _kill_word_forward(self) -> None:
        """Delete the whitespace-delimited word after the cursor."""
        text = self.text
        cursor = self.cursor_position
        while cursor < len(text) and not text[cursor].isspace():
            cursor += 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        self._kill_text(self.cursor_position, cursor)

    def _kill_to_line_start(self) -> None:
        """Delete from the current line start to the cursor."""
        offset = self.cursor_position
        line_start = offset - len(self.text[:offset].rsplit("\n", 1)[-1])
        self._kill_text(line_start, offset)

    def _kill_to_line_end(self) -> None:
        """Delete from the cursor to the line end, joining the next line."""
        offset = self.cursor_position
        text = self.text
        rest = text[offset:]
        line_end = rest.find("\n")
        end = len(text) if line_end == -1 else offset + line_end
        if end == offset and rest.startswith("\n"):
            end = offset + 1
        self._kill_text(offset, end)

    def _insert_yank(self, killed: str) -> None:
        """Insert killed text at the cursor and remember its span."""
        offset = self.cursor_position
        text = self.text
        self.text = text[:offset] + killed + text[offset:]
        self.move_cursor(_text_location_from_offset(self.text, offset + len(killed)))
        self._last_yank_span = (offset, offset + len(killed))

    def _yank(self) -> None:
        """Paste the most recently killed text."""
        if not self._kill_ring:
            return
        self._yank_index = len(self._kill_ring) - 1
        self._insert_yank(self._kill_ring[self._yank_index])

    def _yank_pop(self) -> None:
        """Replace the last yank with the previous killed fragment."""
        if not self._kill_ring or self._last_yank_span is None:
            return
        start, end = self._last_yank_span
        text = self.text
        previous = self._kill_ring[self._yank_index]
        if text[start:end] != previous:
            return  # The yanked text was edited away; never clobber it.
        self.text = text[:start] + text[end:]
        self.move_cursor(_text_location_from_offset(self.text, start))
        self._last_yank_span = None
        self._yank_index = max(0, self._yank_index - 1)
        self._insert_yank(self._kill_ring[self._yank_index])

    async def _open_external_editor(self) -> None:
        """Edit the prompt in ``$VISUAL``/``$EDITOR`` and load the result."""
        editor = _external_editor_command()
        if editor is None:
            self.app.notify(
                "No external editor found. Set $VISUAL or $EDITOR.",
                severity="warning",
            )
            return
        content = self.text
        temp_path = Path(tempfile.mkstemp(prefix="forge-prompt-", suffix=".txt")[1])
        try:
            temp_path.write_text(content, encoding="utf-8")
            process = await asyncio.create_subprocess_exec(
                *editor,
                str(temp_path),
                stdin=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            updated = temp_path.read_text(encoding="utf-8")
            if updated != content:
                self.text = updated
                self.move_cursor(_text_end_location(updated))
                self._clear_pending_paste()
        finally:
            with suppress(OSError):
                temp_path.unlink()

    async def on_key(self, event: Key) -> None:
        """Route completion and submission keys before default input handling."""
        keybindings = self.tui_keybindings
        if self._matches("queue_follow_up", event.key):
            event.stop()
            event.prevent_default()
            await self._completion_target().action_submit_follow_up()
        elif event.key == "enter":
            event.stop()
            event.prevent_default()
            await self._completion_target().action_submit_prompt()
        elif event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
        elif self._matches("accept_completion", event.key):
            event.stop()
            self._completion_target().action_accept_completion()
        elif self._matches("cancel", event.key):
            event.stop()
            self._completion_target().action_cancel()
        elif self._matches("command_palette", event.key):
            event.stop()
            self._completion_target().action_open_command_palette()
        elif self._matches("session_picker", event.key):
            event.stop()
            self._completion_target().action_open_session_picker()
        elif _is_thinking_cycle_key(event.key, keybindings):
            event.stop()
            self._completion_target().action_cycle_thinking()
        elif self._matches("model_cycle", event.key):
            event.stop()
            self._completion_target().action_cycle_model()
        elif self._matches("toggle_tool_results", event.key):
            event.stop()
            self._completion_target().action_toggle_tool_results()
        elif self._matches("toggle_todos", event.key):
            event.stop()
            self._completion_target().action_toggle_todos()
        elif self._matches("toggle_thinking", event.key):
            event.stop()
            self._completion_target().action_toggle_thinking()
        elif self._matches("dequeue_queued", event.key):
            event.stop()
            self._completion_target().action_edit_queued_message()
        elif self._matches("transcript_search", event.key):
            event.stop()
            self._completion_target().action_open_transcript_search()
        elif self._matches("copy_message", event.key):
            if self.selected_text:
                return
            event.stop()
            event.prevent_default()
            if self.text:
                self.text = ""
                self.move_cursor((0, 0))
        elif self._matches("delete_word_backward", event.key):
            event.stop()
            self._kill_word_backward()
        elif self._matches("delete_word_forward", event.key):
            event.stop()
            self._kill_word_forward()
        elif self._matches("delete_to_line_start", event.key):
            event.stop()
            self._kill_to_line_start()
        elif self._matches("delete_to_line_end", event.key):
            event.stop()
            self._kill_to_line_end()
        elif self._matches("yank", event.key):
            event.stop()
            self._yank()
        elif self._matches("yank_pop", event.key):
            event.stop()
            self._yank_pop()
        elif self._matches("open_external_editor", event.key):
            event.stop()
            self.run_worker(self._open_external_editor())
        elif self._matches("completion_next", event.key):
            event.stop()
            if self._has_completion_options():
                self._completion_target().action_completion_next()
            else:
                self.action_cursor_down()
        elif self._matches("completion_previous", event.key):
            event.stop()
            self.action_completion_previous()
        elif self._matches("quit", event.key):
            event.stop()
            await self.action_quit()

    def _has_completion_options(self) -> bool:
        completion_state = getattr(self.app, "_completion_state", None)
        return bool(getattr(completion_state, "items", ()))

    def _completion_target(self) -> CompletionActionTarget:
        return cast(CompletionActionTarget, self.app)


def _is_terminal_command_prompt(text: str) -> bool:
    """Return whether the prompt is currently in terminal-command mode."""
    return _terminal_command_prefix_span(text) is not None


def _terminal_command_prefix_span(text: str) -> tuple[int, int] | None:
    """Return the input span for a leading ! or !! terminal-command prefix."""
    leading_whitespace = len(text) - len(text.lstrip())
    stripped = text[leading_whitespace:]
    if stripped.startswith("!!"):
        return (leading_whitespace, leading_whitespace + 2)
    if stripped.startswith("!"):
        return (leading_whitespace, leading_whitespace + 1)
    return None


def _text_location_from_offset(text: str, offset: int) -> tuple[int, int]:
    """Return the (row, column) TextArea location for a flat text offset."""
    before = text[:offset]
    return (before.count("\n"), len(before.rsplit("\n", 1)[-1]))


def _text_end_location(text: str) -> tuple[int, int]:
    """Return the TextArea cursor location at the end of text."""
    line, _, column_text = text.rpartition("\n")
    return (line.count("\n") + 1 if line else 0, len(column_text))


def _external_editor_command() -> list[str] | None:
    """Return the external editor command for the prompt editor."""
    for env_name in ("VISUAL", "EDITOR"):
        editor = os.environ.get(env_name)
        if editor:
            return [editor]
    if sys.platform == "win32":
        return ["notepad"]
    for candidate in ("nano", "vi"):
        found = shutil.which(candidate)
        if found:
            return [found]
    return None

