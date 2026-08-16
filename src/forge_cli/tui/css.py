"""Forge TUI stylesheet.

Kept as a module-level constant so the application shell stays focused on
behavior while the full Textual CSS lives in one reviewable place.
"""

from __future__ import annotations

FORGE_TUI_CSS = """
Screen {
    layout: vertical;
    align: center top;
    background: $forge-screen-background;
    color: $forge-screen-text;
}

Footer {
    background: $forge-chrome-background;
    color: $forge-muted-text;
}

Footer FooterKey {
    background: $forge-chrome-background;
    color: $forge-muted-text;
}

Footer FooterKey .footer-key--key {
    background: $forge-chrome-background;
    color: $forge-accent;
}

Footer FooterKey .footer-key--description,
Footer FooterLabel {
    background: $forge-chrome-background;
    color: $forge-muted-text;
}

Toast {
    background: $forge-chrome-background;
    color: $forge-chrome-text;
}

Toast .toast--title {
    color: $forge-accent;
}

#workspace {
    width: 100%;
    max-width: 120;
    height: 1fr;
}

#main-pane {
    width: 100%;
    height: 1fr;
    padding: 0 1 0 1;
}

#transcript {
    height: 1fr;
    border: none;
    background: $forge-transcript-background;
    padding: 0 0 0 2;
    overflow-x: auto;
    scrollbar-size-vertical: 0;
    scrollbar-size-horizontal: 1;
}

/* Top-anchored conversations get the same leading air as the welcome view
   (margin: 1 2) instead of starting flush against the terminal top.  The
   full margin is declared because Textual resolves ``margin-top`` in a
   higher-specificity rule by replacing the whole ``margin`` property, which
   would drop the base rule's right/bottom spacing (``0 1 1 0``). */
#transcript > .transcript-message:first-child {
    margin: 1 1 1 0;
}

#welcome {
    height: auto;
    max-height: 4;
    margin: 1 2;
    color: $forge-screen-text;
    content-align: left top;
    overflow-x: hidden;
}

#queued-messages {
    height: auto;
    max-height: 8;
    margin: 0 1 1 1;
    padding: 0 1;
    background: $forge-screen-background;
    color: $forge-muted-text;
}

#goal-status {
    height: auto;
    max-height: 1;
    margin: 0 1 0 1;
    padding: 0 1;
    background: $forge-screen-background;
    color: $forge-screen-text;
    overflow-x: hidden;
}

#prompt-row {
    height: auto;
    margin: 0 1 1 1;
    padding: 0 1;
    border-top: tall $forge-border;
    border-bottom: tall $forge-border;
}

#prompt {
    width: 1fr;
    height: auto;
    background: $forge-screen-background;
    color: $forge-prompt-text;
    border: none;
    margin: 0;
    padding: 0;
    max-height: 6;
}

#prompt-row.-running {
    border-top: tall $forge-accent;
    border-bottom: tall $forge-accent;
}

#prompt-row.-shell-mode {
    border-top: tall $forge-accent;
    border-bottom: tall $forge-accent;
}

#compact-session-info {
    height: auto;
    max-height: 2;
    margin: 0 1 1 1;
    padding: 0 1;
    color: $forge-muted-text;
    overflow-x: hidden;
}

#autocomplete {
    height: auto;
    max-height: 18;
    margin: 0 1 1 1;
    padding: 0 1;
    background: $forge-autocomplete-background;
    color: $forge-screen-text;
    border: tall $forge-border;
    overflow-y: auto;
}

SessionPickerScreen,
TreePickerScreen,
CommandOutputScreen {
    align: center middle;
}

#session-picker,
#tree-picker {
    width: 76;
    max-width: 90%;
    height: auto;
    max-height: 70%;
    padding: 1 2;
    background: $forge-chrome-background;
    border: tall $forge-border;
}

#session-picker-title,
#tree-picker-title {
    height: 1;
    color: $forge-chrome-text;
    text-style: bold;
    margin-bottom: 1;
}

#session-picker-search {
    height: 3;
    margin-bottom: 1;
    background: $forge-prompt-background;
    color: $forge-prompt-text;
    border: tall $forge-prompt-border;
}

#session-rename {
    width: 72;
    max-width: 92%;
    height: auto;
    padding: 1 2;
    background: $forge-chrome-background;
    color: $forge-chrome-text;
    border: tall $forge-border;
}

#session-rename-title {
    height: 1;
    color: $forge-chrome-text;
    text-style: bold;
    margin-bottom: 1;
}

#session-rename-input {
    background: $forge-prompt-background;
    color: $forge-prompt-text;
    border: tall $forge-prompt-border;
    margin-bottom: 1;
}

#session-rename-help {
    height: 1;
    color: $forge-muted-text;
}

#session-picker-list,
#tree-picker-list {
    height: auto;
    max-height: 16;
    background: $forge-transcript-background;
    border: tall $forge-border;
}

ListView > ListItem.--highlight {
    background: $forge-highlight-background;
    color: $forge-highlight-text;
}

ListView > ListItem.--highlight Label {
    background: $forge-highlight-background;
    color: $forge-highlight-text;
}

#session-picker-help,
#tree-picker-help {
    height: 1;
    margin-top: 1;
    color: $forge-muted-text;
}

#command-output {
    width: 76;
    max-width: 90%;
    height: auto;
    max-height: 70%;
    padding: 1 2;
    background: $forge-chrome-background;
    color: $forge-chrome-text;
    border: tall $forge-border;
}

#command-output-title {
    height: 1;
    color: $forge-chrome-text;
    text-style: bold;
    margin-bottom: 1;
}

#command-output-scroll {
    height: auto;
    max-height: 18;
    background: $forge-transcript-background;
    border: tall $forge-border;
}

#command-output-body {
    color: $forge-screen-text;
    padding: 1;
}

#command-output-help {
    height: 1;
    margin-top: 1;
    color: $forge-muted-text;
}

TranscriptSearchScreen {
    align: center middle;
}

#transcript-search {
    width: 76;
    max-width: 90%;
    height: auto;
    padding: 1 2;
    background: $forge-chrome-background;
    color: $forge-chrome-text;
    border: tall $forge-border;
}

#transcript-search-title {
    height: 1;
    color: $forge-chrome-text;
    text-style: bold;
    margin-bottom: 1;
}

#transcript-search-input {
    background: $forge-prompt-background;
    color: $forge-prompt-text;
    border: tall $forge-prompt-border;
    margin-bottom: 1;
}

#transcript-search-status {
    height: 1;
    color: $forge-accent;
    margin-bottom: 1;
}

#transcript-search-help {
    height: 1;
    color: $forge-muted-text;
}

AskUserQuestionScreen {
    align: center middle;
}

#ask-user-question {
    width: 84;
    max-width: 94%;
    height: auto;
    max-height: 80%;
    padding: 1 2;
    background: $forge-chrome-background;
    color: $forge-chrome-text;
    border: tall $forge-accent;
}

#ask-user-question-title,
#ask-user-question-progress {
    height: 1;
    color: $forge-accent;
    text-style: bold;
}

#ask-user-question-body {
    height: auto;
    margin: 1 0;
    color: $forge-screen-text;
}

#ask-user-question-options {
    height: auto;
    max-height: 12;
    background: $forge-transcript-background;
    border: tall $forge-border;
}

#ask-user-question-preview {
    height: auto;
    max-height: 8;
    margin-top: 1;
    color: $forge-muted-text;
    overflow-y: auto;
}

#ask-user-question-notes {
    height: 4;
    margin-top: 1;
    background: $forge-prompt-background;
    color: $forge-prompt-text;
    border: tall $forge-prompt-border;
}

#ask-user-question-help {
    height: 1;
    margin-top: 1;
    color: $forge-muted-text;
}

LoginMethodPickerScreen,
LoginProviderPickerScreen,
ThemePickerScreen,
ModelPickerScreen {
    align: center middle;
}

#login-method-picker,
#login-provider-picker,
#theme-picker,
#model-picker {
    width: 76;
    max-width: 90%;
    height: auto;
    max-height: 70%;
    padding: 1 2;
    background: $forge-chrome-background;
    color: $forge-chrome-text;
    border: tall $forge-border;
}

#login-method-title,
#login-provider-title,
#theme-picker-title,
#model-picker-title {
    height: 1;
    color: $forge-chrome-text;
    text-style: bold;
    margin-bottom: 1;
}

#model-picker-tabs {
    height: 1;
    color: $forge-muted-text;
    margin-bottom: 1;
}

#login-method-list,
#login-provider-list,
#theme-picker-list,
#model-picker-list {
    height: auto;
    max-height: 12;
    background: $forge-transcript-background;
    color: $forge-screen-text;
    border: tall $forge-border;
}

#login-method-list ListItem Label,
#login-provider-list ListItem Label,
#theme-picker-list ListItem Label,
#model-picker-list ListItem Label {
    color: $forge-screen-text;
}

#login-method-intro {
    height: 1;
    color: $forge-muted-text;
    margin-bottom: 1;
}

#login-method-list {
    height: auto;
    max-height: 10;
}

#model-picker-search {
    height: 3;
    margin-bottom: 1;
    background: $forge-prompt-background;
    color: $forge-prompt-text;
    border: tall $forge-prompt-border;
}

#login-method-help,
#login-provider-help,
#theme-picker-help,
#model-picker-help {
    height: 1;
    margin-top: 1;
    color: $forge-muted-text;
}

CustomProviderLoginScreen,
LoginScreen,
OAuthLoginScreen {
    align: center middle;
}

#login-screen {
    width: 72;
    max-width: 92%;
    height: auto;
    padding: 1 2;
    background: $forge-chrome-background;
    border: tall $forge-border;
}

#login-title {
    height: 1;
    color: $forge-chrome-text;
    text-style: bold;
    margin-bottom: 1;
}

#login-help,
#custom-provider-help {
    height: 1;
    color: $forge-muted-text;
    margin-bottom: 1;
}

#login-api-key,
#login-oauth-code,
#custom-provider-name,
#custom-provider-display-name,
#custom-provider-base-url,
#custom-provider-api-key-env,
#custom-provider-models,
#custom-provider-default-model,
#custom-provider-api-key {
    background: $forge-prompt-background;
    color: $forge-prompt-text;
    border: tall $forge-prompt-border;
    margin-bottom: 1;
}

#login-oauth-url {
    min-height: 1;
    max-height: 4;
    color: $forge-chrome-text;
    margin-bottom: 1;
}

#login-footer {
    height: 1;
    color: $forge-muted-text;
}
"""
