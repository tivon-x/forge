"""Textual TUI frontend for Forge coding sessions.

This package is the public API surface for the TUI: ``app.py`` owns the
application shell (``ForgeTuiApp``), while the editor, keybinding builders,
modal screens, presentation helpers, stylesheet, and session bootstrap live
in sibling modules.  Everything consumers (CLI, tests, extensions) may use is
re-exported here; module-internal helpers stay in their owning module.
"""

from __future__ import annotations

from forge_cli.tui.adapter import TuiEventAdapter
from forge_cli.tui.app import ForgeTuiApp
from forge_cli.tui.autocomplete import CompletionOption
from forge_cli.tui.config import (
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
from forge_cli.tui.goals import (
    GoalAction,
    GoalConfirmScreen,
    GoalEditorScreen,
    GoalManagerScreen,
    GoalStatusLine,
    render_goal_status,
    render_goal_status_line,
)
from forge_cli.tui.prompt import PASTE_DISPLAY_THRESHOLD, PromptInput
from forge_cli.tui.questionnaire import AskUserQuestionScreen, QuestionnaireResult
from forge_cli.tui.screens import (
    CommandOutputScreen,
    CustomProviderLoginResult,
    CustomProviderLoginScreen,
    LoginMethodPickerScreen,
    LoginProviderPickerScreen,
    LoginScreen,
    ModelPickerScreen,
    OAuthLoginScreen,
    SessionImportTrustScreen,
    SessionPickerScreen,
    ThemePickerScreen,
    TranscriptSearchScreen,
    TreePickerResult,
    TreePickerScreen,
)
from forge_cli.tui.startup import run_tui_app
from forge_cli.tui.state import ChatItem, SubagentDisplay, TuiState
from forge_cli.tui.todos import TodoPanel, render_todos, visible_todos
from forge_cli.tui.widgets import (
    CompactSessionInfo,
    StreamingTranscriptMessageWidget,
    SubagentTranscriptWidget,
    TranscriptMessageWidget,
    TranscriptView,
    WelcomeView,
    render_chat_item,
    render_compact_session_info,
    render_welcome,
    transcript_item_selection_text,
)

__all__ = [
    "AskUserQuestionScreen",
    "BUILTIN_TUI_THEME_NAMES",
    "ChatItem",
    "CommandOutputScreen",
    "CompactSessionInfo",
    "CompletionOption",
    "CustomProviderLoginResult",
    "CustomProviderLoginScreen",
    "FORGE_DARK_THEME",
    "FORGE_LIGHT_THEME",
    "ForgeTuiApp",
    "GoalAction",
    "GoalConfirmScreen",
    "GoalEditorScreen",
    "GoalManagerScreen",
    "GoalStatusLine",
    "HIGH_CONTRAST_THEME",
    "LoginMethodPickerScreen",
    "LoginProviderPickerScreen",
    "LoginScreen",
    "ModelPickerScreen",
    "OAuthLoginScreen",
    "PASTE_DISPLAY_THRESHOLD",
    "PromptInput",
    "QuestionnaireResult",
    "SessionPickerScreen",
    "SessionImportTrustScreen",
    "StreamingTranscriptMessageWidget",
    "SubagentDisplay",
    "SubagentTranscriptWidget",
    "ThemePickerScreen",
    "TranscriptMessageWidget",
    "TranscriptSearchScreen",
    "TranscriptView",
    "TreePickerResult",
    "TreePickerScreen",
    "TuiConfigError",
    "TuiEventAdapter",
    "TuiKeybindings",
    "TuiRoleStyle",
    "TuiSettings",
    "TuiTheme",
    "TuiThemeName",
    "TuiState",
    "TodoPanel",
    "WelcomeView",
    "get_tui_theme",
    "load_tui_settings",
    "render_chat_item",
    "render_compact_session_info",
    "render_goal_status",
    "render_goal_status_line",
    "render_todos",
    "render_welcome",
    "run_tui_app",
    "save_tui_settings",
    "transcript_item_selection_text",
    "tui_settings_path",
    "visible_todos",
]
