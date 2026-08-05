"""Textual TUI frontend for Forge coding sessions."""

from __future__ import annotations

from forge_coding.tui.adapter import TuiEventAdapter
from forge_coding.tui.app import ForgeTuiApp, run_tui_app
from forge_coding.tui.autocomplete import CompletionOption
from forge_coding.tui.config import (
    BUILTIN_TUI_THEME_NAMES,
    FORGE_DARK_THEME,
    FORGE_LIGHT_THEME,
    HIGH_CONTRAST_THEME,
    TuiConfigError,
    TuiKeybindings,
    TuiRoleStyle,
    TuiSettings,
    TuiTheme,
    TuiThemeName,
    get_tui_theme,
    load_tui_settings,
    save_tui_settings,
    tui_settings_path,
)
from forge_coding.tui.state import ChatItem, TuiState
from forge_coding.tui.widgets import (
    CompactSessionInfo,
    SessionSidebar,
    StreamingTranscriptMessageWidget,
    TranscriptMessageWidget,
    TranscriptView,
    render_chat_item,
    render_compact_session_info,
    render_session_sidebar,
    transcript_item_selection_text,
)

__all__ = [
    "BUILTIN_TUI_THEME_NAMES",
    "ChatItem",
    "CompletionOption",
    "CompactSessionInfo",
    "ForgeTuiApp",
    "SessionSidebar",
    "FORGE_DARK_THEME",
    "FORGE_LIGHT_THEME",
    "StreamingTranscriptMessageWidget",
    "TranscriptMessageWidget",
    "TranscriptView",
    "TuiEventAdapter",
    "TuiConfigError",
    "HIGH_CONTRAST_THEME",
    "TuiKeybindings",
    "TuiRoleStyle",
    "TuiSettings",
    "TuiTheme",
    "TuiThemeName",
    "TuiState",
    "get_tui_theme",
    "load_tui_settings",
    "render_chat_item",
    "render_compact_session_info",
    "render_session_sidebar",
    "run_tui_app",
    "save_tui_settings",
    "transcript_item_selection_text",
    "tui_settings_path",
]
