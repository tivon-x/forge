"""Forge Textual application shell.

The app owns session wiring, event routing, and the transcript/chrome
behavior; the editor, keybinding builders, modal screens, presentation
helpers, stylesheet, and session startup live in sibling modules:

- ``prompt.py``       - ``PromptInput`` editor (kill ring, external editor)
- ``bindings.py``     - keybinding-to-``Binding`` builders and ``/hotkeys``
- ``screens.py``      - modal pickers and login flows
- ``presentation.py`` - pure rendering/session helpers
- ``css.py``          - the Textual stylesheet
- ``startup.py``      - ``run_tui_app`` provider/session bootstrap
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from inspect import isawaitable
from typing import Any, ClassVar, Literal, cast

from langchain_core.messages import HumanMessage
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import BindingsMap
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.events import Resize
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Static,
    TextArea,
)
from textual.worker import Worker

from forge_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    HumanInputRequestedEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from forge_agent.message_codec import message_text
from forge_cli.formatting import format_terminal_command_result_block
from forge_cli.tui.adapter import TuiEventAdapter
from forge_cli.tui.autocomplete import (
    CompletionState,
    build_completion_state,
)
from forge_cli.tui.bindings import (
    _app_bindings,
    _hotkeys_help_text,
    _prompt_footer_mode,
)
from forge_cli.tui.config import (
    BUILTIN_TUI_THEME_NAMES,
    TuiSettings,
    TuiTheme,
    TuiThemeName,
    available_theme_names,
    load_tui_settings,
    save_tui_settings,
    tui_settings_signature,
)
from forge_cli.tui.css import FORGE_TUI_CSS
from forge_cli.tui.goals import GoalAction, GoalConfirmScreen, GoalManagerScreen
from forge_cli.tui.presentation import (
    _MISSING,
    ACTIVITY_TICK_SECONDS,
    COMPLETION_INITIAL_TERMINAL_FRACTION,
    COMPLETION_MAX_VISIBLE_LINES,
    COMPLETION_MIN_TRANSCRIPT_LINES,
    COMPLETION_WIDGET_CHROME_LINES,
    NO_STORED_CREDENTIALS_MESSAGE,
    SETTINGS_RELOAD_INTERVAL_SECONDS,
    _activity_prompt_border_color,
    _api_key_login_providers,
    _attach_diagnostic_log_path_to_error,
    _command_message_uses_notification,
    _command_message_uses_transcript,
    _command_output_title,
    _completion_visible_line_limit,
    _credential_store_has_entry,
    _event_message_role,
    _format_prompt_error,
    _is_user_message_end_event,
    _render_queued_messages,
    _session_command_registry,
    _session_header_sub_title,
    _session_options,
    _session_records,
    _session_thinking_level,
    _should_optimistically_render_prompt,
    _stored_credential_providers,
    _subscription_login_providers,
    _textual_theme_for_forge_theme,
    _theme_css_variables,
    _visible_completion_state,
)
from forge_cli.tui.prompt import (
    PromptInput,
    _is_terminal_command_prompt,
    _text_end_location,
)
from forge_cli.tui.questionnaire import AskUserQuestionScreen, QuestionnaireResult
from forge_cli.tui.screens import (
    BindingEntry,
    CommandOutputScreen,
    CustomProviderLoginResult,
    CustomProviderLoginScreen,
    LoginMethodPickerScreen,
    LoginProviderPickerScreen,
    LoginScreen,
    ModelPickerScreen,
    OAuthLoginScreen,
    SessionPickerScreen,
    ThemePickerScreen,
    TranscriptSearchScreen,
    TreePickerResult,
    TreePickerScreen,
)
from forge_cli.tui.state import TuiState
from forge_cli.tui.terminal_title import TerminalTitleController
from forge_cli.tui.todos import TodoPanel
from forge_cli.tui.widgets import (
    CompactSessionInfo,
    GoalStatusLine,
    TranscriptView,
    WelcomeView,
    render_completion_suggestions,
)
from forge_coding.providers.auth.credentials import FileCredentialStore, OAuthCredential
from forge_coding.providers.catalog import (
    BUILTIN_PROVIDER_CATALOG,
    ProviderCatalogEntry,
    builtin_provider_entry,
)
from forge_coding.providers.catalog_loader import save_user_catalog_entries
from forge_coding.providers.config import (
    OpenAICompatibleProviderConfig,
    load_provider_settings,
    provider_config_from_catalog_entry,
    save_provider_settings,
    upsert_openai_compatible_provider,
    upsert_saved_provider,
)
from forge_coding.session import (
    TREE_RUNNING_MESSAGE,
    CodingSession,
    ModelChoice,
    SessionTreeBranchResult,
    parse_terminal_command,
)


class ForgeTuiApp(App[None]):
    """Interactive Textual frontend for a ``CodingSession``."""

    TITLE = "Forge"
    CSS = FORGE_TUI_CSS
    BINDINGS: ClassVar[list[BindingEntry]] = []

    def __init__(
        self,
        session: CodingSession,
        *,
        tui_settings: TuiSettings | None = None,
        startup_message: str | None = None,
        startup_notice: str | None = None,
        startup_notices: Sequence[str] = (),
        initial_prompt: str | None = None,
    ) -> None:
        self.tui_settings = tui_settings or TuiSettings()
        self.startup_message = startup_message
        legacy_notices = (startup_notice,) if startup_notice else ()
        self.startup_notices = tuple((*startup_notices, *legacy_notices))
        self.initial_prompt = initial_prompt
        super().__init__()
        self._register_forge_textual_themes()
        self.theme = self.tui_settings.theme
        self._bindings = BindingsMap(_app_bindings(self.tui_settings.keybindings))
        self.session = session
        self.state = TuiState(skills=session.skills)
        for notice in self.startup_notices:
            self.state.add_item("status", notice)
        self._prompt_history: tuple[str, ...] = ()
        self._load_session_messages_from_session()
        self.adapter = TuiEventAdapter(self.state)
        self._prompt_worker: Worker[None] | None = None
        self._compaction_worker: Worker[None] | None = None
        self._prompt_run_id = 0
        self._optimistic_user_messages: list[tuple[int, str]] = []
        self._completion_state = CompletionState()
        self._completion_visible_line_budget: int | None = None
        self._activity_frame = 0
        self._activity_timer: Timer | None = None
        self._terminal_title = TerminalTitleController()
        self._active_notification_keys: set[tuple[str, str]] = set()
        self._supports_pyperclip: bool | None = None
        self._sync_header_title()

    def _sync_header_title(self) -> None:
        """Reflect the active session name in Textual's header state."""
        self.title = "Forge"
        self.sub_title = _session_header_sub_title(self.session)
        self._sync_terminal_title()

    def _sync_terminal_title(self) -> None:
        """Reflect the active session name and running state in the terminal tab title."""
        self._terminal_title.update(
            getattr(self.session, "session_title", None),
            running=self.state.running,
            frame=self._activity_frame,
        )

    def _sync_text_selection_state(self) -> None:
        """Disable native text selection while the transcript is mutating."""
        type(self).ALLOW_SELECT = not self.state.running
        if self.state.running and self.screen_stack:
            with suppress(Exception):
                self.screen.clear_selection()

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text using pyperclip when available, then Textual's fallback."""
        if self._supports_pyperclip is None:
            try:
                import pyperclip  # type: ignore[import-untyped]
            except ImportError:
                self._supports_pyperclip = False
            else:
                self._supports_pyperclip = True
        if self._supports_pyperclip:
            import pyperclip

            with suppress(Exception):
                pyperclip.copy(text)
        super().copy_to_clipboard(text)

    def _register_forge_textual_themes(self) -> None:
        """Register Forge themes with Textual's theme system.

        Textual exposes its own theme menu and command palette entries. Registering
        Forge's built-in themes there makes those controls update the same theme as
        `/theme` instead of changing only Textual's chrome.
        """
        self._registered_themes.clear()
        for theme_name in BUILTIN_TUI_THEME_NAMES:
            self.register_theme(_textual_theme_for_forge_theme(theme_name))

    def _watch_theme(self, theme_name: str) -> None:
        """Keep Textual theme changes synchronized with Forge's durable TUI theme."""
        super()._watch_theme(theme_name)
        if theme_name not in BUILTIN_TUI_THEME_NAMES:
            return
        forge_theme: TuiThemeName = theme_name
        if self.tui_settings.theme == forge_theme:
            return
        self._replace_tui_settings(theme=forge_theme)
        save_tui_settings(self.tui_settings)

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Return Forge-specific CSS variables for the selected TUI theme."""
        variables = super().get_theme_variable_defaults()
        return {**variables, **_theme_css_variables(self.tui_settings.resolved_theme)}

    def compose(self) -> ComposeResult:
        """Compose the TUI widgets."""
        with Vertical(id="workspace"), Vertical(id="main-pane"):
            yield WelcomeView(id="welcome")
            yield TranscriptView(
                id="transcript",
                min_width=1,
                wrap=True,
                highlight=True,
                markup=False,
            )
            yield TodoPanel(id="todos")
            yield Static("", id="queued-messages")
            yield GoalStatusLine(id="goal-status")
            with Vertical(id="prompt-row"):
                yield PromptInput(
                    placeholder="Ask Forge…",
                    id="prompt",
                    tui_keybindings=self.tui_settings.keybindings,
                )
            yield CompactSessionInfo(id="compact-session-info")
            yield Static("", id="autocomplete")
        yield Footer(show_command_palette=False)

    async def on_mount(self) -> None:
        """Focus the prompt when the app starts."""
        prompt = self.query_one(PromptInput)
        prompt.shell_mode_style = self.tui_settings.resolved_theme.accent
        self._sync_prompt_shell_mode(prompt.text)
        prompt.focus()
        self._refresh()
        self._sync_text_selection_state()
        self._refresh_completions()
        self._settings_signature = tui_settings_signature()
        self._settings_timer = self.set_interval(
            SETTINGS_RELOAD_INTERVAL_SECONDS,
            self._maybe_reload_settings,
            name="settings-hot-reload",
        )
        if self.startup_message:
            self._notify(self.startup_message, severity="warning")
        if self.initial_prompt and self.initial_prompt.strip():
            await self._submit_prompt(self.initial_prompt.strip())

    def on_unmount(self) -> None:
        """Stop activity animations when the app is torn down."""
        if self._activity_timer is not None:
            self._activity_timer.stop()
            self._activity_timer = None
        settings_timer = getattr(self, "_settings_timer", None)
        if settings_timer is not None:
            settings_timer.stop()
        self._terminal_title.restore()

    def on_resize(self, event: Resize) -> None:
        """Reset completion sizing when the terminal changes size."""
        self._completion_visible_line_budget = None
        del event

    def on_click(self, event: events.Click) -> None:
        """Return keyboard focus to the prompt after clicks in the main TUI."""
        if event.button != 1:
            return
        with suppress(NoMatches):
            self.screen.query_one("#prompt", PromptInput).focus()

    @on(events.TextSelected)
    async def on_text_selected(self) -> None:
        """Optionally copy selected text automatically."""
        active_screen = self.screen
        if not (
            self.tui_settings.auto_copy_selection
            or getattr(active_screen, "auto_copy_selection", False)
        ):
            return
        selection = active_screen.get_selected_text()
        if selection:
            self.copy_to_clipboard(selection)
            self._notify("Copied selection to clipboard.")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update prompt autocomplete when the prompt text changes."""
        if event.text_area.id != "prompt":
            return
        # TextArea can emit an initial Changed while the app is still
        # composing; guard against querying the not-yet-mounted prompt.
        try:
            prompt = self.query_one("#prompt", PromptInput)
        except NoMatches:
            return
        prompt.sync_pending_paste()
        self._sync_prompt_shell_mode(event.text_area.text)
        self._completion_state = self._build_completion_state(event.text_area.text)
        self._refresh_completions()

    async def action_submit_prompt(self) -> None:
        """Submit the current prompt text or slash command."""
        await self._submit_prompt_from_editor(streaming_behavior="steer")

    async def action_submit_follow_up(self) -> None:
        """Submit the current prompt as a queued follow-up while running."""
        await self._submit_prompt_from_editor(streaming_behavior="follow_up")

    async def _submit_prompt_from_editor(
        self,
        *,
        streaming_behavior: Literal["steer", "follow_up"],
    ) -> None:
        prompt = self.query_one("#prompt", PromptInput)
        raw_text = prompt.text_for_submission()
        applied_completion = self._apply_selected_completion(raw_text)
        if applied_completion is not None and applied_completion != raw_text:
            prompt.text = applied_completion
            prompt._clear_pending_paste()
            prompt.move_cursor(_text_end_location(applied_completion))
            self._completion_state = self._build_completion_state(applied_completion)
            self._refresh_completions()
            return

        text = raw_text.strip()
        if not text:
            prompt.text = ""
            prompt._clear_pending_paste()
            self._completion_state = CompletionState()
            self._refresh_completions()
            return

        if self._is_compaction_active():
            if text.startswith("/compact"):
                self._notify("A compaction is already running.", severity="warning")
            else:
                prompt.text = raw_text
                prompt.move_cursor(_text_end_location(raw_text))
                self._notify(
                    "Compaction is still running. You can keep editing, but wait to submit.",
                    severity="warning",
                )
            return

        prompt.text = ""
        prompt._clear_pending_paste()
        self._completion_state = CompletionState()
        self._refresh_completions()

        terminal_command = parse_terminal_command(text)
        if terminal_command is not None:
            self.run_worker(
                self._run_terminal_command(
                    terminal_command.command,
                    add_to_context=terminal_command.add_to_context,
                ),
                exclusive=True,
            )
            return

        if text == "/hotkeys":
            self._show_command_message(
                "/hotkeys",
                _hotkeys_help_text(self.tui_settings.keybindings),
            )
            return

        command = self.session.handle_command(text)
        if command.handled:
            if command.clear_requested:
                self.state.clear()
            if command.new_session_requested:
                await self._new_session()
            if command.compact_summary is not None:
                if self._is_compaction_active():
                    self._notify("A compaction is already running.", severity="warning")
                elif self._is_agent_or_queue_active():
                    prompt.text = raw_text
                    prompt.move_cursor(_text_end_location(raw_text))
                    self._notify(
                        "Wait for the current agent turn and queued messages to finish "
                        "before compacting.",
                        severity="warning",
                    )
                    return
                else:
                    self._compaction_worker = self.run_worker(
                        self._run_compaction(command.compact_summary),
                        exclusive=False,
                    )
            if command.export_requested:
                try:
                    exported_path = await self.session.export(
                        command.export_destination,
                        format=command.export_format,
                    )
                    self._notify(f"Exported session to {exported_path}")
                except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
                    self._notify(f"Could not export session: {exc}", severity="error")
            if command.resume_session_id is not None:
                await self._resume_session(command.resume_session_id)
            if command.resume_picker_requested:
                self.action_open_session_picker()
            if command.tree_picker_requested:
                if self._is_agent_or_queue_active():
                    prompt.text = raw_text
                    prompt.move_cursor(_text_end_location(raw_text))
                    self._notify(TREE_RUNNING_MESSAGE, severity="warning")
                    return
                await self._open_tree_picker()
            if bool(getattr(command, "goal_manager_requested", False)):
                self.action_open_goal_manager()
            goal_action = getattr(command, "goal_action", None)
            if goal_action is not None:
                await self._run_goal_action(goal_action)
            if command.login_picker_requested:
                self._open_login_picker()
            if command.custom_provider_login_requested:
                self._open_custom_provider_login()
            if command.login_provider is not None:
                self._open_login(command.login_provider)
            if command.logout_picker_requested:
                self._open_logout_picker()
            if command.logout_provider is not None:
                self._logout(command.logout_provider)
            if command.model_picker_requested:
                self._open_model_picker()
            if command.scoped_models_picker_requested:
                self._open_scoped_models_picker()
            if command.theme_picker_requested:
                self._open_theme_picker()
            if command.thinking_level is not None:
                await self._set_thinking_level(command.thinking_level)
            if command.theme is not None:
                self._set_tui_theme(command.theme)
            self.state.set_skills(self.session.skills)
            if command.message:
                if _command_message_uses_notification(text, command.message):
                    self._notify(command.message)
                elif _command_message_uses_transcript(text):
                    self._append_command_message(text, command.message)
                else:
                    self._show_command_message(text, command.message)
            self._refresh()
            if command.exit_requested:
                self.exit()
            return

        if bool(getattr(self.session, "is_waiting_for_input", False)):
            prompt.text = raw_text
            prompt.move_cursor(_text_end_location(raw_text))
            self._notify("Answer the open questionnaire before sending another prompt.")
            return

        if self.state.running:
            self._remember_prompt(text)
            await self._queue_prompt(text, streaming_behavior=streaming_behavior)
            return

        self._remember_prompt(text)
        await self._submit_prompt(text)

    def _remember_prompt(self, text: str) -> None:
        """Remember a submitted user prompt for lightweight input recall."""
        if not text.strip():
            return
        self._prompt_history = (*self._prompt_history, text)

    def _load_session_messages_from_session(self) -> None:
        """Load visible session messages and reseed prompt history from them."""
        traces = getattr(self.session, "subagent_traces", None)
        if callable(traces):
            with suppress(Exception):
                traces = traces()
        self.state.load_messages(
            self.session.messages,
            subagent_traces=traces if isinstance(traces, Mapping) else None,
        )
        self.state.update_todos(getattr(self.session, "todos", ()))
        self.state.update_goal(getattr(self.session, "goal", None))
        self._prompt_history = tuple(
            message_text(message)
            for message in self.session.messages
            if isinstance(message, HumanMessage) and message_text(message).strip()
        )

    def _is_compaction_active(self) -> bool:
        """Return whether a manual compaction worker is still running."""
        worker = self._compaction_worker
        return worker is not None and not worker.is_finished and not worker.is_cancelled

    def _is_agent_or_queue_active(self) -> bool:
        """Return whether compaction would race an active or queued agent turn."""
        self._sync_queue_state()
        worker = self._prompt_worker
        is_worker_active = worker is not None and not worker.is_finished and not worker.is_cancelled
        is_session_running = bool(getattr(self.session, "is_running", False))
        is_waiting_for_input = bool(getattr(self.session, "is_waiting_for_input", False))
        return (
            self.state.running
            or is_session_running
            or is_waiting_for_input
            or is_worker_active
            or self.state.queued_message_count > 0
        )

    async def _run_compaction(self, summary: str) -> None:
        """Run manual compaction without disabling prompt editing."""
        self.state.clear()
        self.state.add_item("status", "Compacting session…")
        self._refresh()
        try:
            compact_message = await self.session.compact(summary)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
            return
        finally:
            self._compaction_worker = None
        self.state.clear()
        self.state.set_skills(self.session.skills)
        self._load_session_messages_from_session()
        self._notify(compact_message)
        self._refresh()

    async def _submit_prompt(self, text: str) -> None:
        """Add a prompt to the transcript and start the agent worker."""
        self._prompt_run_id += 1
        run_id = self._prompt_run_id
        if _should_optimistically_render_prompt(text):
            self._optimistic_user_messages.append((run_id, text))
            await self._append_optimistic_user_message(text)
        self._prompt_worker = self.run_worker(self._run_prompt(text, run_id), exclusive=True)

    async def _append_optimistic_user_message(self, text: str) -> None:
        """Render a submitted user message immediately without rebuilding the transcript."""
        start_index = len(self.state.items)
        self.state.add_user_message(text)
        self._follow_transcript_output()
        if not self.screen_stack:
            self._refresh()
            return
        theme = self.tui_settings.resolved_theme
        try:
            transcript = self.query_one("#transcript", TranscriptView)
        except NoMatches:
            self._refresh()
            return
        for item in self.state.items[start_index:]:
            await transcript.append_item(
                item,
                theme=theme,
                show_tool_results=self.state.show_tool_results,
                scroll_end=True,
            )
        self._refresh_chrome(theme=theme)

    def _consume_optimistic_user_event(self, event: AgentEvent, *, run_id: int) -> bool:
        """Return whether a user event confirms an already-rendered optimistic message."""
        if not isinstance(event, MessageEndEvent) or _event_message_role(event.message) != "user":
            return False
        for index, (pending_run_id, pending_text) in enumerate(self._optimistic_user_messages):
            if pending_run_id == run_id and pending_text == message_text(event.message):
                del self._optimistic_user_messages[index]
                return True
        return False

    def _clear_optimistic_user_messages(self, *, run_id: int) -> None:
        """Drop unconfirmed optimistic messages once their run is no longer active."""
        self._optimistic_user_messages = [
            pending for pending in self._optimistic_user_messages if pending[0] != run_id
        ]

    async def _append_confirmed_user_message(self, message: Any) -> None:
        """Render a non-optimistic user event incrementally when possible."""
        if _event_message_role(message) != "user":
            self._refresh()
            return
        await self._append_optimistic_user_message(message_text(message))

    def _follow_transcript_output(self) -> None:
        """Put the transcript back in follow mode for explicit user actions."""
        if not self.screen_stack:
            return
        with suppress(NoMatches):
            self.query_one("#transcript", TranscriptView).follow_output()

    async def _run_terminal_command(self, command: str, *, add_to_context: bool) -> None:
        run_terminal_command = getattr(self.session, "run_terminal_command", None)
        if not callable(run_terminal_command):
            self._notify("Terminal commands are not available.", severity="error")
            return

        item_index = len(self.state.items)
        self.state.add_item(
            "tool",
            f"$ {command.strip()}",
            always_show_tool_result=True,
        )
        self._follow_transcript_output()
        self._refresh()

        try:
            result = await run_terminal_command(command, add_to_context=add_to_context)
        except Exception as exc:  # noqa: BLE001 - surface command execution failures in the TUI
            if item_index < len(self.state.items):
                item = self.state.items[item_index]
                item.tool_result_text = format_terminal_command_result_block(
                    ok=False,
                    added_to_context=add_to_context,
                    output=str(exc),
                )
            self._notify(f"Could not run command: {exc}", severity="error")
            self._refresh()
            return

        if item_index >= len(self.state.items):
            return
        item = self.state.items[item_index]
        item.text = f"$ {result.command}"
        item.tool_result_text = format_terminal_command_result_block(
            ok=result.ok,
            added_to_context=result.added_to_context,
            output=result.output,
        )
        self._follow_transcript_output()
        self._refresh()

    def _replace_tui_settings(self, *, theme: TuiThemeName) -> None:
        """Replace the current immutable TUI settings with a new theme."""
        self.tui_settings = TuiSettings(
            keybindings=self.tui_settings.keybindings,
            theme=theme,
            auto_copy_selection=self.tui_settings.auto_copy_selection,
        )

    def _set_tui_theme(self, theme: TuiThemeName) -> None:
        self._replace_tui_settings(theme=theme)
        save_tui_settings(self.tui_settings)
        if theme in BUILTIN_TUI_THEME_NAMES:
            self.theme = theme
        self._refresh()

    async def _queue_prompt(
        self,
        text: str,
        *,
        streaming_behavior: Literal["steer", "follow_up"],
    ) -> None:
        """Queue a prompt for the active agent worker."""
        try:
            async for event in self.session.prompt(text, streaming_behavior=streaming_behavior):
                self.adapter.apply(event)
        except Exception as exc:  # noqa: BLE001 - surface queueing failures in the TUI
            self._notify(f"Could not queue message: {exc}", severity="error")
            return
        self._refresh()

    async def _run_prompt(self, text: str, run_id: int | None = None) -> None:
        """Run one prompt and stream session events into the TUI state."""
        active_run_id = self._prompt_run_id if run_id is None else run_id
        try:
            async for event in self.session.prompt(text):
                if active_run_id != self._prompt_run_id:
                    return
                if self._consume_optimistic_user_event(event, run_id=active_run_id):
                    self._sync_text_selection_state()
                    self._refresh_chrome()
                    continue
                if not (_is_user_message_end_event(event) and self.screen_stack):
                    self.adapter.apply(event)
                self._sync_text_selection_state()
                if isinstance(event, ErrorEvent) and not event.recoverable:
                    _attach_diagnostic_log_path_to_error(self.state, self.session)
                await self._apply_streaming_transcript_event(event)
        except Exception as exc:  # noqa: BLE001 - surface unexpected worker errors in the TUI
            if active_run_id != self._prompt_run_id:
                return
            message = _format_prompt_error(exc, self.session)
            self.state.error = message
            self.state.add_item("error", message)
            self.state.running = False
            self._sync_text_selection_state()
            self._refresh()
        finally:
            self._clear_optimistic_user_messages(run_id=active_run_id)
            if active_run_id == self._prompt_run_id:
                self._prompt_worker = None

    def _open_questionnaire(self, event: HumanInputRequestedEvent) -> None:
        """Show the active HITL questionnaire over the transcript."""

        if any(isinstance(screen, AskUserQuestionScreen) for screen in self.screen_stack):
            return
        self.push_screen(
            AskUserQuestionScreen(event, theme=self.tui_settings.resolved_theme),
            callback=self._handle_questionnaire_result,
        )

    def _handle_questionnaire_result(self, result: QuestionnaireResult | None) -> None:
        if result is None:
            return
        self._prompt_worker = self.run_worker(self._resume_human_input(result), exclusive=True)

    async def _resume_human_input(self, result: QuestionnaireResult) -> None:
        """Resume the paused graph and stream the resulting ToolMessage/model turn."""

        try:
            async for event in self.session.respond_to_human_input(result.message):
                self.adapter.apply(event)
                self._sync_text_selection_state()
                await self._apply_streaming_transcript_event(event)
        except Exception as exc:  # noqa: BLE001 - surface resume failures in the TUI
            self._notify(f"Could not submit answers: {exc}", severity="error")
            self.state.running = False
            self._refresh()
        finally:
            self._prompt_worker = None

    async def _apply_streaming_transcript_event(self, event: AgentEvent) -> None:
        """Apply an agent event to mounted transcript widgets without full redraws."""
        if not self.screen_stack:
            self._refresh()
            return
        theme = self.tui_settings.resolved_theme
        try:
            transcript = self.query_one("#transcript", TranscriptView)
        except NoMatches:
            self._refresh()
            return
        if isinstance(event, AgentStartEvent):
            self._refresh_chrome()
            return
        if isinstance(event, AgentEndEvent):
            await transcript.finish_assistant_message()
            self._refresh_chrome()
            return
        if isinstance(event, MessageStartEvent):
            return
        if isinstance(event, MessageDeltaEvent):
            await transcript.append_assistant_delta(event.delta, theme=theme)
            self._sync_activity_indicator()
            return
        if isinstance(event, ThinkingDeltaEvent):
            await transcript.append_thinking_delta(
                event.delta,
                theme=theme,
                show_thinking=self.state.show_thinking,
            )
            self._sync_activity_indicator()
            return
        if isinstance(event, MessageEndEvent):
            role = _event_message_role(event.message)
            if role == "user":
                await self._append_confirmed_user_message(event.message)
                self._sync_header_title()
                return
            if role == "assistant":
                await transcript.finish_assistant_message(message_text(event.message))
                self._refresh_chrome()
                return
            return
        if isinstance(event, HumanInputRequestedEvent):
            self._refresh_chrome()
            self._open_questionnaire(event)
            return
        if isinstance(event, ToolExecutionStartEvent):
            await transcript.finish_assistant_message()
            await transcript.append_item(
                self.state.items[-1],
                theme=theme,
                show_tool_results=self.state.show_tool_results,
            )
            self._refresh_chrome()
            return
        if isinstance(event, ToolExecutionUpdateEvent):
            if event.data and "arguments_delta" in event.data:
                # Tool argument streaming adds no state item; re-appending
                # the last item here would duplicate it per chunk.
                self._refresh_chrome()
                return
            if event.data and event.data.get("kind") == "subagent_trace":
                if self.state.items and self.state.has_subagent_task(event.tool_call_id):
                    item = next(
                        (
                            candidate
                            for candidate in reversed(self.state.items)
                            if candidate.tool_call_id == event.tool_call_id
                            and candidate.subagent is not None
                        ),
                        None,
                    )
                    if item is not None and await transcript.update_subagent_activity(
                        item,
                        theme=theme,
                        expanded=self.state.show_tool_results,
                    ):
                        self._refresh_chrome()
                        return
                # Malformed/unknown trace data is deliberately silent; the
                # root task block remains visible with its V1 result.
                self._refresh_chrome()
                return
            if event.data and event.data.get("kind") == "subagent_activity":
                if self.state.items and self.state.has_subagent_task(event.tool_call_id):
                    item = next(
                        (
                            candidate
                            for candidate in reversed(self.state.items)
                            if candidate.tool_call_id == event.tool_call_id
                            and candidate.subagent is not None
                        ),
                        None,
                    )
                    if item is not None and await transcript.update_subagent_activity(
                        item,
                        theme=theme,
                        expanded=self.state.show_tool_results,
                    ):
                        self._refresh_chrome()
                        return
                self._refresh_chrome()
                return
            if self.state.has_subagent_task(event.tool_call_id):
                # Keep unknown nested updates out of the parent transcript.
                self._refresh_chrome()
                return
            await transcript.finish_assistant_message()
            if self.state.items:
                await transcript.append_item(
                    self.state.items[-1],
                    theme=theme,
                    show_tool_results=self.state.show_tool_results,
                )
            self._refresh_chrome()
            return
        if isinstance(event, RetryEvent | ErrorEvent):
            await transcript.finish_assistant_message()
            if (
                isinstance(event, ErrorEvent)
                and event.recoverable
                and event.message == "Agent run cancelled"
            ):
                for item in reversed(self.state.items):
                    if item.subagent is None:
                        continue
                    await transcript.update_subagent_activity(
                        item,
                        theme=theme,
                        expanded=self.state.show_tool_results,
                    )
            if self.state.items:
                await transcript.append_item(
                    self.state.items[-1],
                    theme=theme,
                    show_tool_results=self.state.show_tool_results,
                )
            self._refresh_chrome()
            return
        if isinstance(event, ToolExecutionEndEvent):
            if event.result.name == "task" or self.state.has_subagent_task(
                event.result.tool_call_id
            ):
                item = next(
                    (
                        candidate
                        for candidate in reversed(self.state.items)
                        if candidate.tool_call_id == event.result.tool_call_id
                        and candidate.subagent is not None
                    ),
                    None,
                )
                if item is not None and await transcript.update_subagent_activity(
                    item,
                    theme=theme,
                    expanded=self.state.show_tool_results,
                ):
                    self._refresh_chrome()
                    return
            self._refresh()
            return
        if isinstance(event, QueueUpdateEvent):
            self._refresh_chrome()
            return
        self._refresh_chrome()

    def action_cancel(self) -> None:
        """Cancel the active compaction or agent turn."""
        if self._cancel_active_compaction(notify=True):
            return
        self._cancel_active_prompt(notify=True)

    def _cancel_active_compaction(self, *, notify: bool) -> bool:
        """Cancel the active manual compaction worker and restore visible session state."""
        worker = self._compaction_worker
        if worker is None or worker.is_finished or worker.is_cancelled:
            return False

        worker.cancel()
        self._compaction_worker = None
        self.state.clear()
        self.state.set_skills(self.session.skills)
        self._load_session_messages_from_session()
        self._refresh()
        if notify:
            self._notify("Cancelled compaction.")
        return True

    def _cancel_active_prompt(self, *, notify: bool, interrupt: bool = False) -> None:
        """Cancel the active prompt worker and ignore any late events from it."""
        worker = self._prompt_worker
        is_worker_active = worker is not None and not worker.is_cancelled
        is_session_running = bool(getattr(self.session, "is_running", False))
        if not (self.state.running or is_session_running or is_worker_active):
            return

        self._prompt_run_id += 1
        cancel = getattr(self.session, "cancel", None)
        if callable(cancel):
            cancel()
        if worker is not None and not worker.is_cancelled:
            worker.cancel()
        self._prompt_worker = None
        self.state.running = False
        self.state.cancel_subagent_tasks()
        self.state.assistant_buffer = ""
        restored = 0 if interrupt else self._restore_queued_messages_to_editor()
        self._sync_text_selection_state()
        self._refresh()
        if notify:
            if restored:
                self._notify(
                    f"Interrupted current operation. Restored {restored} queued "
                    "message to the editor." if restored == 1
                    else f"Interrupted current operation. Restored {restored} queued "
                    "messages to the editor."
                )
            else:
                self._notify("Interrupted current operation.")

    def _restore_queued_messages_to_editor(self) -> int:
        """Move queued follow-up and steering messages back into the prompt.

        Returns the number of restored messages.  Messages are popped newest
        first and joined oldest-first for editing.
        """
        popped: list[str] = []
        pop_follow_up = getattr(self.session, "pop_latest_follow_up_message", None)
        if callable(pop_follow_up):
            while True:
                message = pop_follow_up()
                if not isinstance(message, str) or not message:
                    break
                popped.append(message)
        pop_steering = getattr(self.session, "pop_latest_steering_message", None)
        if callable(pop_steering):
            while True:
                message = pop_steering()
                if not isinstance(message, str) or not message:
                    break
                popped.append(message)
        if not popped:
            return 0
        restored = "\n\n".join(reversed(popped))
        prompt = self.query_one("#prompt", PromptInput)
        prompt.text = restored
        prompt.move_cursor(_text_end_location(restored))
        self._sync_queue_state()
        return len(popped)

    def action_accept_completion(self) -> None:
        """Accept the currently selected prompt completion."""
        if isinstance(self.screen, ModelPickerScreen):
            self.screen.action_toggle_mode()
            return
        if isinstance(
            self.screen,
            SessionPickerScreen
            | TreePickerScreen
            | LoginMethodPickerScreen
            | LoginProviderPickerScreen
            | ThemePickerScreen,
        ):
            self.screen.action_select_cursor()
            return
        prompt = self.query_one("#prompt", PromptInput)
        applied = self._apply_selected_completion(prompt.text)
        if applied is None:
            return
        prompt.text = applied
        prompt.move_cursor(_text_end_location(applied))
        self._completion_state = self._build_completion_state(prompt.text)
        self._refresh_completions()

    def action_completion_next(self) -> None:
        """Select the next prompt completion or move down in the prompt."""
        if isinstance(self.screen, CommandOutputScreen):
            self.screen.action_scroll_down()
            return
        if isinstance(
            self.screen,
            SessionPickerScreen
            | TreePickerScreen
            | LoginMethodPickerScreen
            | LoginProviderPickerScreen
            | ThemePickerScreen
            | ModelPickerScreen,
        ):
            self.screen.action_cursor_down()
            return
        if not self._completion_state.items:
            self.query_one("#prompt", PromptInput).action_cursor_down()
            return
        self._completion_state = self._completion_state.select_next()
        self._refresh_completions()

    def action_completion_previous(self) -> None:
        """Select the previous prompt completion or move up in the prompt."""
        if isinstance(self.screen, CommandOutputScreen):
            self.screen.action_scroll_up()
            return
        if isinstance(
            self.screen,
            SessionPickerScreen
            | TreePickerScreen
            | LoginMethodPickerScreen
            | LoginProviderPickerScreen
            | ThemePickerScreen
            | ModelPickerScreen,
        ):
            self.screen.action_cursor_up()
            return
        if not self._completion_state.items:
            if self.action_edit_queued_message():
                return
            if self.action_recall_previous_prompt():
                return
            self.query_one("#prompt", PromptInput).action_cursor_up()
            return
        self._completion_state = self._completion_state.select_previous()
        self._refresh_completions()

    def action_recall_previous_prompt(self) -> bool:
        """Recall the most recent submitted prompt into an empty prompt input."""
        prompt = self.query_one("#prompt", PromptInput)
        # Only recall into an empty input so an accidental Up press does not
        # erase a prompt the user is still writing.
        if prompt.text.strip() or not self._prompt_history:
            return False
        previous_prompt = self._prompt_history[-1]
        prompt.text = previous_prompt
        prompt.move_cursor(_text_end_location(previous_prompt))
        self._completion_state = self._build_completion_state(prompt.text)
        self._refresh_completions()
        return True

    def action_edit_queued_message(self) -> bool:
        """Move the latest queued message back into the prompt for editing."""
        if not self.state.running:
            return False
        prompt = self.query_one("#prompt", PromptInput)
        if prompt.text.strip():
            return False

        message = self._pop_latest_queued_message()
        if not message:
            return False
        prompt.text = message
        prompt.move_cursor(_text_end_location(message))
        self._sync_queue_state()
        self._completion_state = self._build_completion_state(prompt.text)
        self._refresh()
        return True

    def action_edit_queued_follow_up(self) -> bool:
        """Move the latest queued message back into the prompt for editing."""
        return self.action_edit_queued_message()

    def _pop_latest_queued_message(self) -> str | None:
        """Pop the latest queued follow-up or steering message from the session."""
        pop_follow_up = getattr(self.session, "pop_latest_follow_up_message", None)
        if callable(pop_follow_up):
            message = pop_follow_up()
            if isinstance(message, str) and message:
                return message

        pop_steering = getattr(self.session, "pop_latest_steering_message", None)
        if callable(pop_steering):
            message = pop_steering()
            if isinstance(message, str) and message:
                return message

        return None

    def action_open_command_palette(self) -> None:
        """Open the slash-command palette in the prompt."""
        prompt = self.query_one("#prompt", PromptInput)
        prompt.focus()
        prompt.text = "/"
        prompt.move_cursor((0, 1))
        self._completion_state = self._build_completion_state(prompt.text)
        self._refresh_completions()

    def action_open_transcript_search(self) -> None:
        """Open the transcript search box."""
        if not self.state.items:
            self._notify("Nothing to search yet.")
            return
        self._search_matches: list[Widget] = []
        self._search_index = 0
        self.push_screen(
            TranscriptSearchScreen(
                theme=self.tui_settings.resolved_theme,
                on_query=self._search_transcript_query,
                on_next=self._search_transcript_next,
                on_previous=self._search_transcript_previous,
            ),
            callback=lambda _result: self._clear_search(),
        )

    def _clear_search(self) -> None:
        """Drop transcript search state after the search box closes."""
        self._search_matches = []
        self._search_index = 0

    def _search_transcript_query(self, needle: str) -> None:
        """Re-run the transcript search and scroll to the first match."""
        transcript = self.query_one("#transcript", TranscriptView)
        self._search_matches = transcript.search(needle)
        self._search_index = 0
        self._scroll_to_search_match()
        self._update_search_status()

    def _search_transcript_next(self) -> None:
        """Jump to the next transcript match."""
        if not self._search_matches:
            return
        self._search_index = (self._search_index + 1) % len(self._search_matches)
        self._scroll_to_search_match()
        self._update_search_status()

    def _search_transcript_previous(self) -> None:
        """Jump to the previous transcript match."""
        if not self._search_matches:
            return
        self._search_index = (self._search_index - 1) % len(self._search_matches)
        self._scroll_to_search_match()
        self._update_search_status()

    def _scroll_to_search_match(self) -> None:
        """Scroll the transcript to the currently selected match."""
        if not self._search_matches:
            return
        transcript = self.query_one("#transcript", TranscriptView)
        transcript.scroll_to_match(self._search_matches[self._search_index])

    def _update_search_status(self) -> None:
        """Refresh the search box match counter."""
        screen = self.screen
        if not isinstance(screen, TranscriptSearchScreen):
            return
        screen.update_status(len(self._search_matches), self._search_index)

    def action_open_session_picker(self) -> None:
        """Open the indexed session picker."""
        if self.state.running:
            self._notify("Forge is already working. Press Escape to cancel.")
            return
        records = _session_records(self.session)
        if not records:
            self._notify("No sessions found.")
            return
        self.push_screen(
            SessionPickerScreen(
                records,
                theme=self.tui_settings.resolved_theme,
                records_provider=lambda: _session_records(self.session),
                on_rename=self._rename_session_record,
                on_delete=self._delete_session_record,
            ),
            callback=self._handle_session_picker_result,
        )

    def _rename_session_record(self, session_id: str, title: str) -> bool:
        """Rename a session through its manager; report failures via notify."""
        manager = getattr(self.session, "session_manager", None)
        rename = getattr(manager, "rename_session", None)
        if not callable(rename):
            self._notify("Session rename is not available.", severity="warning")
            return False
        try:
            rename(session_id, title)
        except Exception as exc:  # noqa: BLE001 - surface picker failures in the TUI
            self._notify(f"Could not rename session: {exc}", severity="error")
            return False
        return True

    def _delete_session_record(self, session_id: str) -> bool:
        """Delete a session through its manager; report failures via notify."""
        manager = getattr(self.session, "session_manager", None)
        delete = getattr(manager, "delete_session", None)
        if not callable(delete):
            self._notify("Session deletion is not available.", severity="warning")
            return False
        try:
            return bool(delete(session_id))
        except Exception as exc:  # noqa: BLE001 - surface picker failures in the TUI
            self._notify(f"Could not delete session: {exc}", severity="error")
            return False

    def action_open_goal_manager(self) -> None:
        """Open the session Goal manager from a bare ``/goal`` command."""
        self.push_screen(
            GoalManagerScreen(
                self.state.goal,
                theme=self.tui_settings.resolved_theme,
                on_action=self._handle_goal_manager_action,
            )
        )

    async def _handle_goal_manager_action(self, action: GoalAction) -> None:
        await self._run_goal_action(action)

    async def _run_goal_action(self, action: object) -> None:
        """Apply one Goal intent and project resulting events into the TUI."""
        apply_action = getattr(self.session, "apply_goal_action", None)
        result: object | None = None
        try:
            expected_goal_id = getattr(action, "goal_id", None)
            current_goal = getattr(self.session, "goal", None)
            current_goal_id = getattr(current_goal, "id", None)
            if (
                isinstance(expected_goal_id, str)
                and expected_goal_id
                and current_goal_id != expected_goal_id
            ):
                self._notify(
                    "Goal changed while the manager was open. Reopen /goal and try again.",
                    severity="warning",
                )
                return
            if self._goal_action_replaces_unfinished_goal(action):
                replacement = GoalAction(
                    "start",
                    objective=getattr(action, "objective", None),
                    goal_id=current_goal_id if isinstance(current_goal_id, str) else None,
                    replace=True,
                )
                self.push_screen(
                    GoalConfirmScreen(
                        "Replace the current Goal? This clears its history and starts a new Goal.",
                        theme=self.tui_settings.resolved_theme,
                    ),
                    lambda confirmed: self._handle_goal_replacement_confirmation(
                        confirmed,
                        replacement,
                    ),
                )
                return
            if callable(apply_action):
                result = apply_action(action)
            else:
                # Keep the TUI tolerant of early session implementations that
                # expose explicit methods before the unified action API lands.
                kind = getattr(action, "kind", getattr(action, "action", None))
                method = (
                    getattr(self.session, f"{kind}_goal", None) if isinstance(kind, str) else None
                )
                if not callable(method):
                    self._notify("Goal controls are not available.", severity="warning")
                    return
                objective = getattr(action, "objective", None)
                result = method(objective) if objective is not None else method()
            if isawaitable(result):
                result = await result
            await self._consume_goal_result(result)
        except Exception as exc:  # noqa: BLE001 - surface session failures in the TUI
            self._notify(f"Could not update goal: {exc}", severity="error")
        finally:
            # Backends may expose the current immutable snapshot in addition to
            # yielding an event.  The event remains the canonical projection.
            session_goal = getattr(self.session, "goal", _MISSING)
            if session_goal is not _MISSING:
                self.state.update_goal(session_goal)
            self._refresh()

    def _goal_action_replaces_unfinished_goal(self, action: object) -> bool:
        """Return whether a shorthand start needs an interactive replacement confirm."""
        kind = getattr(action, "action", getattr(action, "kind", None))
        if kind != "start":
            return False
        if bool(getattr(action, "replace", False)):
            return False
        goal = getattr(self.session, "goal", None)
        status = getattr(goal, "status", None)
        if isinstance(goal, Mapping):
            status = goal.get("status")
        return status in {"active", "paused", "blocked"}

    def _handle_goal_replacement_confirmation(
        self,
        confirmed: bool | None,
        action: object,
    ) -> None:
        if not confirmed:
            return
        self.run_worker(
            self._replace_goal_then_start(action),
            exclusive=False,
        )

    async def _replace_goal_then_start(self, action: object) -> None:
        """Submit one backend-owned replacement intent after confirmation."""
        if bool(getattr(action, "replace", False)):
            await self._run_goal_action(action)
            return
        captured_goal_id = getattr(action, "goal_id", None)
        replacement = GoalAction(
            "start",
            objective=getattr(action, "objective", None),
            goal_id=captured_goal_id if isinstance(captured_goal_id, str) else None,
            replace=True,
        )
        await self._run_goal_action(replacement)

    async def _consume_goal_result(self, result: object | None) -> None:
        if result is None:
            return
        if isinstance(getattr(result, "type", None), str):
            self._apply_goal_result_event(result)
            return
        if hasattr(result, "__aiter__"):
            async for event in result:
                if isinstance(getattr(event, "type", None), str):
                    self._apply_goal_result_event(event)
            return
        if isinstance(result, (list, tuple)):
            for event in result:
                if isinstance(getattr(event, "type", None), str):
                    self._apply_goal_result_event(event)
            return
        message = getattr(result, "message", None)
        if isinstance(result, str):
            self._notify(result)
        elif isinstance(message, str) and message:
            self._notify(message)

    def _apply_goal_result_event(self, event: object) -> None:
        """Project one event yielded by a Goal-owned run and refresh the UI."""
        self.adapter.apply(cast(AgentEvent, event))
        self._refresh()

    def action_cycle_thinking(self) -> None:
        """Cycle the active thinking mode."""
        self.run_worker(self._cycle_thinking_level(), exclusive=False)

    def action_cycle_model(self) -> None:
        """Cycle through scoped models."""
        if self.state.running:
            self._notify("Forge is already working. Press Escape to cancel.")
            return
        self.run_worker(self._cycle_scoped_model(), exclusive=False)

    def action_toggle_tool_results(self) -> None:
        """Toggle inline tool result details in the transcript."""
        expanded = self.state.toggle_tool_results()
        self._refresh()
        self._notify("Tool results expanded." if expanded else "Tool results collapsed.")

    def action_toggle_todos(self) -> None:
        """Collapse or expand the current Todo panel."""
        collapsed = self.state.toggle_todos()
        self._refresh_chrome()
        self._notify("Todos collapsed." if collapsed else "Todos expanded.")

    def action_toggle_thinking(self) -> None:
        """Toggle thinking-token display in the transcript."""
        self.state.toggle_thinking()
        transcript = self.query_one("#transcript", TranscriptView)
        transcript.update_thinking_visibility(
            self.state,
            theme=self.tui_settings.resolved_theme,
        )

    def _handle_session_picker_result(self, session_id: str | None) -> None:
        if session_id is None:
            return
        self.run_worker(self._resume_session(session_id), exclusive=False)

    async def _resume_session(self, session_id: str) -> None:
        try:
            resume_message = await self.session.resume(session_id)
            self.state.clear()
            self.state.set_skills(self.session.skills)
            self._load_session_messages_from_session()
            self._notify(resume_message)
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
        self._refresh()

    async def _open_tree_picker(self) -> None:
        if self._is_agent_or_queue_active():
            self._notify(TREE_RUNNING_MESSAGE, severity="warning")
            return
        tree_choices = getattr(self.session, "tree_choices", None)
        if tree_choices is None:
            self._notify("Session tree is not available.", severity="warning")
            return
        try:
            choices = tuple(await tree_choices())
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
            return
        if not choices:
            self._notify("No session entries are available for branching.", severity="warning")
            return
        self.push_screen(
            TreePickerScreen(choices, theme=self.tui_settings.resolved_theme),
            callback=self._handle_tree_picker_result,
        )

    def _handle_tree_picker_result(self, result: TreePickerResult | None) -> None:
        if result is None:
            return
        self.run_worker(
            self._branch_to_tree_entry(
                result.entry_id,
                summarize=result.summarize,
                custom_instructions=result.custom_instructions,
            ),
            exclusive=False,
        )

    async def _branch_to_tree_entry(
        self,
        entry_id: str,
        *,
        summarize: bool,
        custom_instructions: str | None = None,
    ) -> None:
        if self._is_agent_or_queue_active():
            self._notify(TREE_RUNNING_MESSAGE, severity="warning")
            return
        branch_to_entry = getattr(self.session, "branch_to_entry", None)
        if branch_to_entry is None:
            self._notify("Session tree is not available.", severity="warning")
            return
        try:
            if summarize:
                self.state.clear()
                self.state.add_item("status", "Summarizing branch…")
                self._refresh()

            result = branch_to_entry(
                entry_id,
                summarize=summarize,
                custom_instructions=custom_instructions,
            )
            if isawaitable(result):
                result = await result
            self.state.clear()
            self.state.set_skills(self.session.skills)
            self._load_session_messages_from_session()
            if isinstance(result, SessionTreeBranchResult):
                if result.input_prefill is not None:
                    prompt = self.query_one("#prompt", PromptInput)
                    prompt.value = result.input_prefill
                    prompt.move_cursor(_text_end_location(result.input_prefill))
                    prompt.focus()
                self._notify(result.message)
            elif isinstance(result, str):
                self._notify(result)
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
        self._refresh()

    async def _new_session(self) -> None:
        worker = self._prompt_worker
        self._cancel_active_prompt(notify=False, interrupt=True)
        # The old prompt's ``finally`` re-persists interrupted state through
        # the session harness; swap only after the worker has fully unwound so
        # the finally never runs against the replaced session.
        if worker is not None:
            with suppress(BaseException):
                await worker.wait()
        new_session = getattr(self.session, "new_session", None)
        if new_session is None:
            self._notify("Session manager is not available.")
            return
        try:
            await new_session()
            self.state.clear()
            self.state.set_skills(self.session.skills)
            self._load_session_messages_from_session()
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
        self._refresh()

    def _apply_selected_completion(self, value: str) -> str | None:
        item = self._completion_state.selected
        if item is None:
            return None
        return item.apply(value)

    def _append_command_message(self, command_text: str, message: str) -> None:
        """Append non-persistent command output to the visible transcript."""
        self.state.add_item("status", f"{_command_output_title(command_text)}\n{message}")

    def _show_command_message(self, command_text: str, message: str) -> None:
        self.push_screen(
            CommandOutputScreen(
                _command_output_title(command_text),
                message,
                theme=self.tui_settings.resolved_theme,
                auto_copy_selection=command_text.strip().split(maxsplit=1)[0] == "/session",
            )
        )

    def _open_login_picker(self) -> None:
        self.push_screen(
            LoginMethodPickerScreen(theme=self.tui_settings.resolved_theme),
            callback=self._handle_login_method_result,
        )

    def _handle_login_method_result(self, method: str | None) -> None:
        if method is None:
            return
        if method == "subscription":
            providers = _subscription_login_providers(BUILTIN_PROVIDER_CATALOG)
        elif method == "api-key":
            providers = _api_key_login_providers(BUILTIN_PROVIDER_CATALOG)
        elif method == "custom":
            self._open_custom_provider_login()
            return
        else:
            self._notify(f"Unknown login method: {method}", severity="error")
            return
        if not providers:
            self._notify("No login providers are available for that method.", severity="warning")
            return
        self.push_screen(
            LoginProviderPickerScreen(
                providers,
                theme=self.tui_settings.resolved_theme,
            ),
            callback=self._handle_login_provider_result,
        )

    def _handle_login_provider_result(self, provider_name: str | None) -> None:
        if provider_name is None:
            return
        self._open_login(provider_name)

    def _open_custom_provider_login(self) -> None:
        self.push_screen(
            CustomProviderLoginScreen(theme=self.tui_settings.resolved_theme),
            callback=self._handle_custom_provider_login_result,
        )

    def _handle_custom_provider_login_result(
        self,
        result: CustomProviderLoginResult | None,
    ) -> None:
        if result is None:
            return
        provider = OpenAICompatibleProviderConfig(
            name=result.provider_name,
            base_url=result.base_url.rstrip("/"),
            api_key_env=result.api_key_env,
            credential_name=result.provider_name,
            models=result.models,
            default_model=result.default_model,
        )
        catalog_entry = ProviderCatalogEntry(
            name=provider.name,
            display_name=result.display_name,
            kind="openai-compatible",
            base_url=provider.base_url,
            api_key_env=provider.api_key_env,
            credential_name=provider.credential_name,
            models=provider.models,
            default_model=provider.default_model,
            docs_url=provider.base_url,
        )
        try:
            save_user_catalog_entries((catalog_entry,))
            FileCredentialStore().set(provider.credential_name or provider.name, result.api_key)
            settings = load_provider_settings()
            updated = upsert_openai_compatible_provider(settings, provider, set_default=False)
            save_provider_settings(updated)
            self.session.reload_provider_settings()
            try:
                self.session.set_provider(provider.name, persist_default=False)
            except TypeError:
                self.session.set_provider(provider.name)
        except Exception as exc:  # noqa: BLE001 - surface login failures in the TUI
            self._notify(f"Could not save custom provider: {exc}", severity="error")
            return
        self._notify(f"Saved custom provider {result.display_name}.")
        self._refresh()

    def _open_login(self, provider_name: str) -> None:
        entry = builtin_provider_entry(provider_name)
        if entry is None:
            self._notify(f"Unknown provider: {provider_name}", severity="error")
            return
        if entry.kind == "openai-codex":
            self.push_screen(
                OAuthLoginScreen(entry, theme=self.tui_settings.resolved_theme),
                callback=lambda credential: self._handle_oauth_login_result(entry, credential),
            )
            return
        self.push_screen(
            LoginScreen(entry, theme=self.tui_settings.resolved_theme),
            callback=lambda api_key: self._handle_login_result(entry, api_key),
        )

    def _handle_login_result(self, entry: ProviderCatalogEntry, api_key: str | None) -> None:
        if api_key is None:
            return
        if entry.credential_name is None:
            self._notify(
                f"Provider {entry.name} does not support saved credentials.",
                severity="error",
            )
            return
        try:
            FileCredentialStore().set(entry.credential_name, api_key)
            provider = provider_config_from_catalog_entry(entry.name)
            upsert_saved_provider(provider, set_default=False)
            self.session.reload_provider_settings()
            try:
                self.session.set_provider(entry.name, persist_default=False)
            except TypeError:
                self.session.set_provider(entry.name)
        except Exception as exc:  # noqa: BLE001 - surface login failures in the TUI
            self._notify(f"Could not save login: {exc}", severity="error")
            return
        self._notify(f"Saved login for {entry.display_name}.")
        self._refresh()

    def _handle_oauth_login_result(
        self,
        entry: ProviderCatalogEntry,
        credential: OAuthCredential | None,
    ) -> None:
        if credential is None:
            return
        if entry.credential_name is None:
            self._notify(
                f"Provider {entry.name} does not support saved credentials.",
                severity="error",
            )
            return
        try:
            FileCredentialStore().set_oauth(entry.credential_name, credential)
            provider = provider_config_from_catalog_entry(entry.name)
            upsert_saved_provider(provider, set_default=False)
            self.session.reload_provider_settings()
            try:
                self.session.set_provider(entry.name, persist_default=False)
            except TypeError:
                self.session.set_provider(entry.name)
        except Exception as exc:  # noqa: BLE001 - surface login failures in the TUI
            self._notify(f"Could not save login: {exc}", severity="error")
            return
        self._notify(f"Saved login for {entry.display_name}.")
        self._refresh()

    def _open_logout_picker(self) -> None:
        providers = _stored_credential_providers(BUILTIN_PROVIDER_CATALOG)
        if not providers:
            self._notify(NO_STORED_CREDENTIALS_MESSAGE, severity="warning")
            return
        self.push_screen(
            LoginProviderPickerScreen(
                providers,
                theme=self.tui_settings.resolved_theme,
                title="Logout",
            ),
            callback=self._handle_logout_provider_result,
        )

    def _handle_logout_provider_result(self, provider_name: str | None) -> None:
        if provider_name is None:
            return
        self._logout(provider_name)

    def _logout(self, provider_name: str) -> None:
        entry = builtin_provider_entry(provider_name)
        if entry is None:
            self._notify(f"Unknown provider: {provider_name}", severity="error")
            return

        if entry.credential_name is None:
            self._notify(NO_STORED_CREDENTIALS_MESSAGE, severity="warning")
            return
        credential_store = FileCredentialStore()
        if not _credential_store_has_entry(credential_store, entry.credential_name):
            self._notify(NO_STORED_CREDENTIALS_MESSAGE, severity="warning")
            return

        try:
            credential_store.delete(entry.credential_name)
            self.session.reload_provider_settings()
        except Exception as exc:  # noqa: BLE001 - surface logout failures in the TUI
            self._notify(f"Could not log out: {exc}", severity="error")
            return

        if entry.kind == "openai-codex":
            self._notify(f"Logged out of {entry.display_name}.")
        else:
            self._notify(
                f"Removed stored API key for {entry.display_name}. "
                "Environment variables and providers.json config are unchanged."
            )
        self._refresh()

    def _available_model_choices(self) -> tuple[ModelChoice, ...]:
        fallback_choices = (
            ModelChoice(provider_name=self.session.provider_name, model=model)
            for model in self.session.available_models
        )
        return tuple(
            getattr(
                self.session,
                "available_model_choices",
                fallback_choices,
            )
        )

    def _open_model_picker(self) -> None:
        choices = self._available_model_choices()
        if not choices:
            self._notify(
                "No configured providers are usable. Run /login to set up a provider.",
                severity="warning",
            )
            return
        self.push_screen(
            ModelPickerScreen(
                choices,
                scoped_choices=tuple(getattr(self.session, "scoped_model_choices", ())),
                current_model=self.session.model,
                provider_name=self.session.provider_name,
                theme=self.tui_settings.resolved_theme,
                on_toggle_scoped=None,
                picker_kind="model",
            ),
            callback=self._handle_model_picker_result,
        )

    def _open_scoped_models_picker(self) -> None:
        choices = self._available_model_choices()
        if not choices:
            self._notify(
                "No configured providers are usable. Run /login to set up a provider.",
                severity="warning",
            )
            return
        self.push_screen(
            ModelPickerScreen(
                choices,
                scoped_choices=tuple(getattr(self.session, "scoped_model_choices", ())),
                current_model=self.session.model,
                provider_name=self.session.provider_name,
                theme=self.tui_settings.resolved_theme,
                on_toggle_scoped=self._toggle_scoped_model,
                picker_kind="scoped",
            ),
            callback=self._handle_scoped_models_picker_result,
        )

    def _toggle_scoped_model(self, choice: ModelChoice) -> Sequence[ModelChoice]:
        toggle_scoped_model = getattr(self.session, "toggle_scoped_model", None)
        if toggle_scoped_model is None:
            self._notify("Scoped model controls are not available.", severity="warning")
            return tuple(getattr(self.session, "scoped_model_choices", ()))
        try:
            return tuple(toggle_scoped_model(choice))
        except Exception as exc:  # noqa: BLE001 - surface session state failures in the TUI
            self._notify(f"Could not update scoped models: {exc}", severity="error")
            return tuple(getattr(self.session, "scoped_model_choices", ()))

    def _handle_scoped_models_picker_result(self, choice: ModelChoice | None) -> None:
        del choice
        self._refresh_chrome()

    def _handle_model_picker_result(self, choice: ModelChoice | None) -> None:
        if choice is None:
            return
        try:
            set_model_choice = getattr(self.session, "set_model_choice", None)
            if set_model_choice is None:
                if choice.provider_name != self.session.provider_name:
                    self.session.set_provider(choice.provider_name)
                self.session.set_model(choice.model)
            else:
                set_model_choice(choice)
        except Exception as exc:  # noqa: BLE001 - surface model switch failures in the TUI
            self._notify(f"Could not switch model: {exc}", severity="error")
            return
        self._refresh_chrome()

    def _open_theme_picker(self) -> None:
        self.push_screen(
            ThemePickerScreen(
                current_theme=self.tui_settings.theme,
                theme=self.tui_settings.resolved_theme,
                theme_names=available_theme_names(),
            ),
            callback=self._handle_theme_picker_result,
        )

    def _handle_theme_picker_result(self, theme: TuiThemeName | None) -> None:
        if theme is None:
            return
        self._set_tui_theme(theme)

    async def _set_thinking_level(self, level: str) -> None:
        setter = getattr(self.session, "set_thinking_level", None)
        if setter is None:
            self._notify("Thinking controls are not available.", severity="warning")
            return
        try:
            result = setter(level)
            if isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - surface session state failures in the TUI
            self._notify(f"Could not change thinking mode: {exc}", severity="error")
            return
        self._refresh_chrome()

    async def _cycle_thinking_level(self) -> None:
        cycler = getattr(self.session, "cycle_thinking_level", None)
        if cycler is None:
            self._notify("Thinking controls are not available.", severity="warning")
            return
        try:
            result = cycler()
            if isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - surface session state failures in the TUI
            self._notify(f"Could not change thinking mode: {exc}", severity="error")
            return
        self._refresh_chrome()

    async def _cycle_scoped_model(self) -> None:
        cycler = getattr(self.session, "cycle_scoped_model", None)
        if cycler is None:
            self._notify("Scoped model controls are not available.", severity="warning")
            return
        try:
            result = cycler()
            if isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - surface session state failures in the TUI
            self._notify(f"Could not switch scoped model: {exc}", severity="error")
            return
        self._refresh_chrome()

    def _maybe_reload_settings(self) -> None:
        """Hot-reload keybindings and themes when their files change."""
        try:
            signature = tui_settings_signature()
        except OSError:
            return
        if signature == self._settings_signature:
            return
        self._settings_signature = signature
        try:
            settings = load_tui_settings()
        except Exception as exc:  # noqa: BLE001 - keep the TUI alive on bad config
            self._notify(f"Could not reload TUI settings: {exc}", severity="warning")
            return
        theme_changed = settings.theme != self.tui_settings.theme
        keybindings_changed = settings.keybindings != self.tui_settings.keybindings
        self.tui_settings = settings
        prompt = self.query_one("#prompt", PromptInput)
        if theme_changed:
            self._register_forge_textual_themes()
            if settings.theme in BUILTIN_TUI_THEME_NAMES:
                self.theme = settings.theme
            prompt.shell_mode_style = settings.resolved_theme.accent
        if keybindings_changed:
            prompt.tui_keybindings = settings.keybindings
            prompt._apply_prompt_bindings()
            prompt.refresh_bindings()
            self._bindings = BindingsMap(_app_bindings(settings.keybindings))
            self.refresh_bindings()
        self._refresh_chrome()
        self._notify("Reloaded TUI settings (keybindings/theme).")

    def _notify(
        self,
        message: str,
        *,
        severity: Literal["information", "warning", "error"] = "information",
    ) -> None:
        key = (message, severity)
        if key in self._active_notification_keys:
            return
        self._active_notification_keys.add(key)
        self.set_timer(
            self.NOTIFICATION_TIMEOUT,
            lambda: self._active_notification_keys.discard(key),
            name=f"notification-dedupe-{hash(key)}",
        )
        self.notify(message, severity=severity, markup=False)

    def _refresh(self) -> None:
        theme = self.tui_settings.resolved_theme
        self._refresh_chrome(theme=theme)
        transcript = self.query_one("#transcript", TranscriptView)
        transcript.update_from_state(self.state, theme=theme)

    def _refresh_chrome(self, *, theme: TuiTheme | None = None) -> None:
        """Refresh non-transcript chrome without remounting transcript blocks."""
        theme = theme or self.tui_settings.resolved_theme
        self._sync_header_title()
        self._sync_text_selection_state()
        self._sync_queue_state()
        welcome = self.query_one("#welcome", WelcomeView)
        welcome_visible = not self.state.items and not self.state.assistant_buffer
        welcome.display = welcome_visible
        if welcome_visible:
            welcome.update_from_session(
                self.session,
                keybindings=self.tui_settings.keybindings,
                theme=theme,
            )
        compact_info = self.query_one("#compact-session-info", CompactSessionInfo)
        compact_info.update_from_session(self.session, theme=theme)
        queued_messages = self.query_one("#queued-messages", Static)
        todos = self.query_one("#todos", TodoPanel)
        todos.update_from_state(
            self.state.todos,
            collapsed=self.state.todos_collapsed,
            theme=theme,
        )
        goal_status = self.query_one("#goal-status", GoalStatusLine)
        goal_status.update_from_goal(self.state.goal, theme=theme)
        if isinstance(self.screen, GoalManagerScreen):
            self.screen.update_goal(self.state.goal)
        queued_messages.display = self.state.queued_message_count > 0
        queued_messages.update(_render_queued_messages(self.state, theme=theme))
        self._sync_activity_indicator()
        self._refresh_footer_bindings()

    def _sync_queue_state(self) -> None:
        queue_event = getattr(self.session, "queue_update_event", None)
        if not callable(queue_event):
            return
        self.adapter.apply(queue_event())

    def _sync_activity_indicator(self) -> None:
        self._sync_terminal_title()
        if self.state.running:
            if self._activity_timer is None:
                self._activity_timer = self.set_interval(
                    ACTIVITY_TICK_SECONDS,
                    self._tick_activity,
                    name="activity-indicator",
                )
            else:
                self._activity_timer.resume()
            self._apply_activity_indicator()
            return
        self._activity_frame = 0
        if self._activity_timer is not None:
            self._activity_timer.pause()
        self._apply_activity_indicator()

    def _tick_activity(self) -> None:
        if not self.state.running:
            return
        self._activity_frame += 1
        self._apply_activity_indicator()
        self._sync_terminal_title()

    def _apply_activity_indicator(self) -> None:
        theme = self.tui_settings.resolved_theme
        try:
            prompt = self.query_one("#prompt", PromptInput)
            prompt_row = self.query_one("#prompt-row", Vertical)
        except NoMatches:
            return
        shell_mode = _is_terminal_command_prompt(prompt.text)
        prompt_row.set_class(self.state.running, "-running")
        prompt_row.set_class(shell_mode, "-shell-mode")
        border_color = _activity_prompt_border_color(
            theme,
            frame=self._activity_frame,
            running=self.state.running,
            shell_mode=shell_mode,
            thinking_level=_session_thinking_level(self.session),
        )
        prompt_row.styles.border_top = ("tall", border_color)
        prompt_row.styles.border_bottom = ("tall", border_color)

    def _refresh_completions(self) -> None:
        try:
            suggestions = self.query_one("#autocomplete", Static)
        except NoMatches:
            # The autocomplete chrome may not be mounted yet when an early
            # TextArea.Changed arrives during compose.
            return
        suggestions.display = bool(self._completion_state.items)
        if not self._completion_state.items:
            self._completion_visible_line_budget = None
            suggestions.update(
                render_completion_suggestions(
                    CompletionState(),
                    theme=self.tui_settings.resolved_theme,
                )
            )
            self._refresh_footer_bindings()
            return
        max_lines = self._completion_window_line_budget(suggestions)
        suggestions.update(
            render_completion_suggestions(
                _visible_completion_state(
                    self._completion_state,
                    max_lines=max_lines,
                    width=max(suggestions.content_size.width or suggestions.size.width, 1),
                ),
                theme=self.tui_settings.resolved_theme,
            )
        )
        self._refresh_footer_bindings()

    def _completion_window_line_budget(self, suggestions: Static) -> int:
        """Return a stable completion window size for the current suggestion box.

        The autocomplete widget has ``height: auto``. If we used its current
        rendered height as the next render limit unconditionally, selecting an
        item could render fewer rows, which would shrink the widget, which would
        then make the next render limit smaller again. Keep the largest measured
        height for the current completion session so navigation does not feed
        back into progressively smaller boxes.
        """
        measured_limit = _completion_visible_line_limit(suggestions)
        if suggestions.size.height <= 0:
            if self._completion_visible_line_budget is None:
                self._completion_visible_line_budget = self._initial_completion_line_budget()
            return self._completion_visible_line_budget
        self._completion_visible_line_budget = max(
            self._completion_visible_line_budget or measured_limit,
            measured_limit,
        )
        return self._completion_visible_line_budget

    def _initial_completion_line_budget(self) -> int:
        """Estimate the first completion window size before Textual lays it out."""
        terminal_height = self.size.height
        if terminal_height <= 0:
            return COMPLETION_MAX_VISIBLE_LINES

        reserved_rows = COMPLETION_MIN_TRANSCRIPT_LINES + COMPLETION_WIDGET_CHROME_LINES
        reserved_rows += 1  # Footer.
        for selector in (
            "#prompt-row",
            "#compact-session-info",
            "#queued-messages",
            "#goal-status",
        ):
            with suppress(NoMatches):
                widget = self.query_one(selector)
                if widget.display:
                    reserved_rows += widget.outer_size.height

        available_rows = terminal_height - reserved_rows
        terminal_fraction_rows = max(1, terminal_height // COMPLETION_INITIAL_TERMINAL_FRACTION)
        return max(
            1,
            min(COMPLETION_MAX_VISIBLE_LINES, available_rows, terminal_fraction_rows),
        )

    def _build_completion_state(self, text: str) -> CompletionState:
        registry = _session_command_registry(self.session)
        return build_completion_state(
            text,
            command_registry=registry,
            skills=self.session.skills,
            prompt_templates=self.session.prompt_templates,
            model_names=self.session.available_models,
            provider_names=self.session.available_providers,
            thinking_levels=getattr(self.session, "available_thinking_levels", ()),
            theme_names=BUILTIN_TUI_THEME_NAMES,
            session_options=_session_options(self.session),
            cwd=self.session.cwd,
        )

    def _refresh_footer_bindings(self) -> None:
        prompt = self.query_one("#prompt", PromptInput)
        prompt.set_footer_mode(_prompt_footer_mode(self.state, self._completion_state))

    def _sync_prompt_shell_mode(self, text: str) -> None:
        try:
            prompt = self.query_one("#prompt", PromptInput)
            prompt_row = self.query_one("#prompt-row", Vertical)
        except NoMatches:
            return
        shell_mode = _is_terminal_command_prompt(text)
        prompt.shell_mode_style = self.tui_settings.resolved_theme.accent
        prompt.set_class(shell_mode, "-shell-mode")
        prompt_row.set_class(shell_mode, "-shell-mode")
        prompt_row.set_class(self.state.running, "-running")
        prompt.refresh()
        self._apply_activity_indicator()





























































































