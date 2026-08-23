"""Pure presentation helpers for the Forge TUI.

Rendering math, session/providers helpers, theme mapping, and command output
classification live here so the application shell and the prompt editor stay
focused on behavior.  Nothing in this module may import ``app.py`` or
``prompt.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from langchain_core.messages import HumanMessage
from rich.console import Console, Group
from rich.text import Text
from textual.theme import Theme
from textual.widgets import Static

from forge_agent import AgentEvent, MessageEndEvent
from forge_cli.tui.autocomplete import CompletionItem, CompletionOption, CompletionState
from forge_cli.tui.config import (
    FORGE_DARK_THEME,
    TuiSettings,
    TuiTheme,
    TuiThemeName,
)
from forge_cli.tui.screens import SessionCompletionRecord, _named_session_title
from forge_cli.tui.state import TuiState
from forge_cli.tui.widgets import render_completion_suggestions
from forge_coding.commands import CommandRegistry, create_default_command_registry
from forge_coding.providers.auth.credentials import FileCredentialStore
from forge_coding.providers.catalog import ProviderCatalogEntry
from forge_coding.session import CodingSession


def _event_message_role(message: object) -> str:
    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(
        str(getattr(message, "type", "")),
        "assistant",
    )


ACTIVITY_TICK_SECONDS = 0.15
SETTINGS_RELOAD_INTERVAL_SECONDS = 2.0
COMPLETION_MAX_VISIBLE_LINES = 16
COMPLETION_INITIAL_TERMINAL_FRACTION = 3
COMPLETION_MIN_TRANSCRIPT_LINES = 4
COMPLETION_WIDGET_CHROME_LINES = 3
NO_STORED_CREDENTIALS_MESSAGE = (
    "No stored credentials to remove. /logout only removes credentials saved by /login; "
    "environment variables and providers.json config are unchanged."
)
_MISSING = object()


def _activity_prompt_border_color(
    theme: TuiTheme,
    *,
    frame: int,
    running: bool,
    shell_mode: bool,
    thinking_level: str | None = None,
) -> str:
    """Return the prompt border color for the current activity state."""
    del frame
    if shell_mode:
        return theme.shell_border or theme.accent
    if running:
        return theme.thinking_border(thinking_level)
    return theme.prompt_border


def _should_optimistically_render_prompt(text: str) -> bool:
    """Return whether submitted text can be safely shown before session expansion."""
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("/")


def _is_user_message_end_event(event: AgentEvent) -> bool:
    """Return whether an agent event closes a user message."""
    return isinstance(event, MessageEndEvent) and isinstance(event.message, HumanMessage)


def _completion_visible_line_limit(suggestions: Static) -> int:
    """Return the number of completion render lines that fit in the widget body."""
    if suggestions.size.height > 0:
        return max(min(COMPLETION_MAX_VISIBLE_LINES, suggestions.size.height), 1)
    return COMPLETION_MAX_VISIBLE_LINES


def _visible_completion_state(
    state: CompletionState,
    *,
    max_lines: int,
    width: int | None = None,
) -> CompletionState:
    """Return a completion-state window with the selected item visible."""
    if not state.items or max_lines <= 0:
        return CompletionState()

    selected_line_limit = max(max_lines - 1, 1)
    start = 0
    while start < state.selected_index:
        candidate = CompletionState(
            items=state.items[start:],
            selected_index=state.selected_index - start,
        )
        if _completion_selected_render_line(candidate, width=width) < selected_line_limit:
            break
        start += 1

    end = len(state.items)
    while end > state.selected_index + 1:
        candidate = CompletionState(
            items=state.items[start:end],
            selected_index=state.selected_index - start,
        )
        if _completion_render_line_count(candidate, width=width) <= max_lines:
            break
        end -= 1

    while start < state.selected_index:
        candidate = CompletionState(
            items=state.items[start:end],
            selected_index=state.selected_index - start,
        )
        if _completion_render_line_count(candidate, width=width) <= max_lines:
            break
        start += 1

    return CompletionState(
        items=state.items[start:end],
        selected_index=state.selected_index - start,
    )


def _completion_selected_render_line(state: CompletionState, *, width: int | None = None) -> int:
    """Return the rendered line number for the selected completion item."""
    line = 0
    has_rendered_text = False
    previous_category: str | None = None
    for index, item in enumerate(state.items):
        if item.category != previous_category:
            if has_rendered_text:
                line += 1
            if item.category:
                line += 1
                has_rendered_text = True
            previous_category = item.category
        elif has_rendered_text:
            line += 1
        if index == state.selected_index:
            return line
        line += _completion_item_extra_wrapped_lines(item, width=width)
        has_rendered_text = True
    return line


def _completion_render_line_count(state: CompletionState, *, width: int | None = None) -> int:
    """Return how many lines the completion state renders into."""
    if not state.items:
        return 0
    line_count = 0
    previous_category: str | None = None
    for index, item in enumerate(state.items):
        if item.category != previous_category:
            if index:
                line_count += 1
            if item.category:
                line_count += 1
            previous_category = item.category
        line_count += 1 + _completion_item_extra_wrapped_lines(item, width=width)
    return line_count


def _completion_item_extra_wrapped_lines(
    item: CompletionItem,
    *,
    width: int | None,
) -> int:
    """Return extra rendered lines used when a completion description wraps."""
    if width is None or width <= 0 or not item.description:
        return 0
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
        legacy_windows=False,
    )
    console.print(
        render_completion_suggestions(
            CompletionState(items=(item,), selected_index=0),
            theme=FORGE_DARK_THEME,
        ),
        end="",
    )
    line_count = len(output.getvalue().splitlines())
    return max(line_count - 1, 0)


def _session_command_registry(session: CodingSession) -> CommandRegistry:
    registry = getattr(session, "command_registry", None)
    if isinstance(registry, CommandRegistry):
        return registry
    return create_default_command_registry()


def _session_options(session: CodingSession) -> tuple[CompletionOption, ...]:
    return tuple(_session_option(record) for record in _session_records(session))


def _session_records(session: CodingSession) -> tuple[SessionCompletionRecord, ...]:
    manager = getattr(session, "session_manager", None)
    if manager is None:
        return ()
    try:
        records = manager.list_sessions(session.cwd)
    except TypeError:
        records = manager.list_sessions()
    return tuple(records)


def _session_option(record: SessionCompletionRecord) -> CompletionOption:
    description_parts = [record.title if record.title else "Untitled session"]
    if record.model:
        description_parts.append(record.model)
    description_parts.append(_short_path(record.cwd))
    return CompletionOption(value=record.id, description=" - ".join(description_parts))


def _short_path(path: Path) -> str:
    home = Path.home()
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)


def _session_header_sub_title(session: CodingSession) -> str:
    """Return the session label shown beside Forge in the TUI header."""
    title = _named_session_title(getattr(session, "session_title", None))
    return title or "Untitled session"


def _subscription_login_providers(
    providers: Sequence[ProviderCatalogEntry],
) -> tuple[ProviderCatalogEntry, ...]:
    return tuple(provider for provider in providers if provider.kind == "openai-codex")


def _api_key_login_providers(
    providers: Sequence[ProviderCatalogEntry],
) -> tuple[ProviderCatalogEntry, ...]:
    return tuple(provider for provider in providers if provider.kind != "openai-codex")


def _stored_credential_providers(
    providers: Sequence[ProviderCatalogEntry],
) -> tuple[ProviderCatalogEntry, ...]:
    credential_store = FileCredentialStore()
    return tuple(
        provider
        for provider in providers
        if provider.credential_name is not None
        and _credential_store_has_entry(credential_store, provider.credential_name)
    )


def _credential_store_has_entry(
    credential_store: FileCredentialStore,
    credential_name: str,
) -> bool:
    return (
        credential_store.get(credential_name) is not None
        or credential_store.get_oauth(credential_name) is not None
    )


def _command_message_uses_transcript(command_text: str) -> bool:
    """Return whether slash-command output should appear inline in the transcript."""
    command_name = command_text.split(maxsplit=1)[0].casefold()
    return command_name in {"/reload", "/system"}


def _command_message_uses_notification(command_text: str, message: str) -> bool:
    """Return whether slash-command output should appear as a notification."""
    command_name = command_text.split(maxsplit=1)[0].casefold()
    return command_name == "/name" and message.startswith("Session renamed: ")


def _command_output_title(command_text: str) -> str:
    command_name = command_text.split(maxsplit=1)[0].removeprefix("/")
    return f"/{command_name or 'help'}"


def _session_thinking_level(session: object) -> str | None:
    """Return the active thinking level for prompt-border coloring."""
    level = getattr(session, "thinking_level", None)
    if isinstance(level, str) and level:
        return level
    state = getattr(session, "state", None)
    level = getattr(state, "thinking_level", None)
    if isinstance(level, str) and level:
        return level
    return None


def _textual_theme_for_forge_theme(theme_name: TuiThemeName) -> Theme:
    """Map a Forge theme to Textual's native theme type."""
    theme = TuiSettings(theme=theme_name).resolved_theme
    return Theme(
        name=theme.name,
        primary=theme.accent,
        secondary=theme.prompt_border,
        warning=theme.markdown_bullet,
        error=theme.error,
        success=theme.success,
        accent=theme.accent,
        foreground=theme.screen_text,
        background=theme.screen_background,
        surface=theme.chrome_background,
        panel=theme.chrome_background,
        dark=theme.name != "forge-light",
        variables=_theme_css_variables(theme),
    )


def _theme_css_variables(theme: TuiTheme) -> dict[str, str]:
    """Return Textual CSS variables for a resolved Forge theme."""
    return {
        "forge-screen-background": theme.screen_background,
        "forge-screen-text": theme.screen_text,
        "forge-chrome-background": theme.chrome_background,
        "forge-chrome-text": theme.chrome_text,
        "forge-muted-text": theme.muted_text,
        "forge-border": theme.border,
        "forge-transcript-background": theme.transcript_background,
        "forge-prompt-background": theme.prompt_background,
        "forge-prompt-text": theme.prompt_text,
        "forge-prompt-border": theme.prompt_border,
        "forge-autocomplete-background": theme.autocomplete_background,
        "forge-accent": theme.accent,
        "forge-success": theme.success,
        "forge-error": theme.error,
        "forge-subagent": theme.role_styles["subagent"].border,
        "forge-subagent-running": theme.role_styles["subagent-running"].border,
        "forge-subagent-success": theme.role_styles["subagent-success"].border,
        "forge-subagent-error": theme.role_styles["subagent-error"].border,
        "forge-highlight-background": theme.highlight_background,
        "forge-highlight-text": theme.highlight_text,
        "forge-markdown-highlight": theme.markdown_heading,
        "forge-markdown-table-header": theme.markdown_table_header,
        "forge-markdown-table-border": theme.markdown_table_border,
        "forge-markdown-inline-code": theme.markdown_inline_code,
        "forge-markdown-code-block-background": theme.markdown_code_block_background,
        "forge-markdown-link": theme.markdown_link,
        "forge-markdown-bullet": theme.markdown_bullet,
        "footer-background": theme.chrome_background,
        "footer-foreground": theme.chrome_text,
        "footer-description-background": theme.chrome_background,
        "footer-description-foreground": theme.chrome_text,
        "footer-key-background": theme.chrome_background,
        "footer-key-foreground": theme.accent,
        "footer-item-background": theme.chrome_background,
    }


def _render_queued_messages(state: TuiState, *, theme: TuiTheme) -> Group:
    """Render queued prompts stacked above the prompt input."""
    rows: list[Text] = []
    for message in state.queued_steering:
        row = Text("↪ steering · queued: ", style=theme.muted_text)
        row.append(_queued_message_preview(message), style=theme.prompt_text)
        rows.append(row)
    for message in state.queued_follow_up:
        row = Text("↳ follow-up · queued: ", style=theme.muted_text)
        row.append(_queued_message_preview(message), style=theme.prompt_text)
        rows.append(row)
    return Group(*rows)


def _queued_message_preview(message: str) -> str:
    """Return the single-line preview shown above the prompt."""
    lines = message.splitlines()
    return lines[0] if lines else ""


def _format_prompt_error(exc: BaseException, session: CodingSession) -> str:
    detail = str(exc) or type(exc).__name__
    message = f"Error: {detail}"
    log_path = getattr(session, "last_diagnostic_log_path", None)
    if isinstance(log_path, Path):
        return f"{message}\nLog: {log_path}"
    return message


def _attach_diagnostic_log_path_to_error(state: TuiState, session: CodingSession) -> None:
    log_path = getattr(session, "last_diagnostic_log_path", None)
    if not isinstance(log_path, Path) or state.error is None:
        return
    message = f"Error: {state.error}\nLog: {log_path}"
    state.error = message
    for item in reversed(state.items):
        if item.role == "error":
            item.text = message
            return
    state.add_item("error", message)

