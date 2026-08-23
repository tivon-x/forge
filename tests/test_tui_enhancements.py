"""Pi-inspired TUI enhancements: kill ring, queue restore, transcript search,
session picker management, user themes, hot reload, and git branch caching."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from textual.widgets import Input, Label, ListView

from forge_cli.tui import app as tui_app
from forge_cli.tui import prompt as tui_prompt
from forge_cli.tui.app import (
    CommandOutputScreen,
    ForgeTuiApp,
    PromptInput,
    SessionImportTrustScreen,
    SessionPickerScreen,
    TranscriptSearchScreen,
    _activity_prompt_border_color,
)
from forge_cli.tui.config import FORGE_DARK_THEME, TuiKeybindings, TuiSettings
from forge_cli.tui.widgets import _GIT_BRANCH_CACHE, _git_branch
from forge_coding.sessions.importing import SessionImportTrustRequired
from forge_coding.sessions.manager import CodingSessionRecord
from forge_coding.sessions.tree import SessionTreeBranchResult, SessionTreeChoice
from test_tui_app import FakeSession, _screen_is

# --------------------------------------------------------------------------- #
# Kill ring and prompt editing
# --------------------------------------------------------------------------- #


def test_prompt_input_kill_ring_deletes_and_yanks_words() -> None:
    prompt = PromptInput()
    prompt.text = "one two three"
    prompt.move_cursor((0, len("one two")))

    prompt._kill_word_backward()

    assert prompt.text == "one  three"
    assert prompt._kill_ring == ["two"]

    prompt._yank()

    assert prompt.text == "one two three"

    prompt.move_cursor((0, 4))
    prompt._kill_word_forward()
    assert prompt.text == "one three"


def test_prompt_input_kill_ring_deletes_to_line_edges() -> None:
    prompt = PromptInput()
    prompt.text = "prefix\nkeep rest"
    prompt.move_cursor((1, 4))

    prompt._kill_to_line_start()
    assert prompt.text == "prefix\n rest"

    prompt.text = "alpha beta\ngamma"
    prompt.move_cursor((1, 0))

    prompt._kill_to_line_end()
    assert prompt.text == "alpha beta\n"


def test_prompt_input_kill_to_line_end_joins_next_line() -> None:
    prompt = PromptInput()
    prompt.text = "first\nsecond"
    prompt.move_cursor((0, 5))

    prompt._kill_to_line_end()

    assert prompt.text == "firstsecond"


def test_prompt_input_yank_pop_cycles_previous_kills() -> None:
    prompt = PromptInput()
    prompt.text = "a b c"
    prompt.move_cursor((0, 2))
    prompt._kill_word_backward()  # kills "a ", text = "b c"
    prompt.move_cursor((0, 2))
    prompt._kill_word_backward()  # kills "b ", text = "c"
    prompt._yank()  # inserts "b "

    assert prompt.text == "b c"

    prompt._yank_pop()  # replaces "b " with "a "

    assert prompt.text == "a c"


def test_prompt_input_yank_pop_ignores_edited_yank() -> None:
    prompt = PromptInput()
    prompt.text = "a b"
    prompt.move_cursor((0, 2))
    prompt._kill_word_backward()
    prompt._yank()
    prompt.text = "a bX"  # simulate the user editing the yanked text

    prompt._yank_pop()

    assert prompt.text == "a bX"


def test_external_editor_command_resolves_env_then_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui_prompt.os, "environ", {"EDITOR": "my-editor"}, raising=False)
    monkeypatch.setattr(tui_prompt.sys, "platform", "linux")

    assert tui_prompt._external_editor_command() == ["my-editor"]

    monkeypatch.setattr(tui_prompt.os, "environ", {}, raising=False)
    monkeypatch.setattr(tui_prompt.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert tui_prompt._external_editor_command() == ["/usr/bin/nano"]

    monkeypatch.setattr(tui_prompt.shutil, "which", lambda name: None)
    assert tui_prompt._external_editor_command() is None


class _FakeEditorProcess:
    async def wait(self) -> int:
        return 0


@pytest.mark.anyio
async def test_prompt_input_external_editor_replaces_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_subprocess(*args: object, **kwargs: object) -> _FakeEditorProcess:
        del kwargs
        path = Path(str(args[-1]))
        path.write_text("edited prompt", encoding="utf-8")
        return _FakeEditorProcess()

    monkeypatch.setattr(tui_prompt, "_external_editor_command", lambda: ["fake-editor"])
    monkeypatch.setattr(tui_prompt.asyncio, "create_subprocess_exec", fake_subprocess)

    app = ForgeTuiApp(FakeSession())
    async with app.run_test(size=(120, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.text = "original prompt"
        await pilot.pause()

        await pilot.press("ctrl+g")
        await pilot.pause()

        assert prompt.text == "edited prompt"


# --------------------------------------------------------------------------- #
# Thinking-level border colors
# --------------------------------------------------------------------------- #


def test_activity_prompt_border_uses_thinking_level_color() -> None:
    theme = FORGE_DARK_THEME

    assert (
        _activity_prompt_border_color(
            theme, frame=0, running=True, shell_mode=False, thinking_level="high"
        )
        == theme.thinking_borders["high"]
    )
    assert theme.thinking_borders["high"] != theme.accent
    assert theme.thinking_border("unknown-level") == theme.accent


# --------------------------------------------------------------------------- #
# /hotkeys reflects configured keys
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_tui_hotkeys_modal_reflects_configured_keys() -> None:
    app = ForgeTuiApp(
        FakeSession(),
        tui_settings=TuiSettings(keybindings=TuiKeybindings(command_palette="ctrl+j")),
    )

    async with app.run_test(size=(120, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/hotkeys"
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        assert await _screen_is(pilot, app, CommandOutputScreen)
        body = app.screen.query_one("#command-output-body")
        rendered = str(body.render())
        assert "Open slash-command completions: Ctrl+J" in rendered
        assert "Restore queued messages to editor: Alt+Up" in rendered


# --------------------------------------------------------------------------- #
# Escape restores queued messages / Alt+Up dequeues
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_tui_escape_restores_queued_messages_to_editor() -> None:
    session = FakeSession()
    session.queued_follow_up_messages = ("follow up",)
    session.queued_steering_messages = ("steer me",)
    app = ForgeTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        app.state.running = True
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert session.queued_follow_up_messages == ()
        assert session.queued_steering_messages == ()
        assert "steer me" in prompt.text
        assert "follow up" in prompt.text


@pytest.mark.anyio
async def test_tui_alt_up_dequeues_queued_message() -> None:
    session = FakeSession()
    session.queued_follow_up_messages = ("queued text",)
    app = ForgeTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        app.state.running = True
        await pilot.pause()

        await pilot.press("alt+up")
        await pilot.pause()

        assert session.queued_follow_up_messages == ()
        assert prompt.text == "queued text"


# --------------------------------------------------------------------------- #
# Transcript search
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_tui_transcript_search_finds_and_navigates_matches() -> None:
    session = FakeSession(messages=[HumanMessage(content="Find this needle")])
    app = ForgeTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+shift+f")
        assert await _screen_is(pilot, app, TranscriptSearchScreen)

        await pilot.press("n", "e", "e", "d", "l", "e")
        await pilot.pause()

        assert app._search_matches
        status = app.screen.query_one("#transcript-search-status")
        assert "1/1 matches" in str(status.render())

        await pilot.press("escape")
        await pilot.pause()

        assert not app._search_matches


@pytest.mark.anyio
async def test_tui_transcript_search_next_and_previous_wrap() -> None:
    session = FakeSession(
        messages=[HumanMessage(content="one needle"), HumanMessage(content="two needle")]
    )
    app = ForgeTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+shift+f")
        assert await _screen_is(pilot, app, TranscriptSearchScreen)

        await pilot.press("n", "e", "e", "d", "l", "e")
        await pilot.pause()

        assert len(app._search_matches) == 2
        assert app._search_index == 0

        await pilot.press("enter")
        await pilot.pause()
        assert app._search_index == 1

        await pilot.press("enter")
        await pilot.pause()
        assert app._search_index == 0


# --------------------------------------------------------------------------- #
# Session picker: search, sort, rename, delete
# --------------------------------------------------------------------------- #


class _ManagedFakeSessionManager:
    def __init__(self, records: list[CodingSessionRecord]) -> None:
        self.records = records
        self.renames: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    def list_sessions(self, cwd: Path | None = None) -> list[CodingSessionRecord]:
        del cwd
        return self.records

    def rename_session(self, session_id: str, title: str) -> CodingSessionRecord | None:
        self.renames.append((session_id, title))
        for index, record in enumerate(self.records):
            if record.id == session_id:
                updated = CodingSessionRecord(
                    id=record.id,
                    path=record.path,
                    cwd=record.cwd,
                    model=record.model,
                    title=title or None,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                )
                self.records[index] = updated
                return updated
        return None

    def delete_session(self, session_id: str) -> bool:
        self.deleted.append(session_id)
        before = len(self.records)
        self.records = [record for record in self.records if record.id != session_id]
        return len(self.records) < before


def _picker_records() -> list[CodingSessionRecord]:
    return [
        CodingSessionRecord(
            id="session-a",
            path=Path("/tmp/a.jsonl"),
            cwd=Path("/workspace/project"),
            model="model-a",
            title="Alpha",
            created_at=1.0,
            updated_at=3.0,
        ),
        CodingSessionRecord(
            id="session-b",
            path=Path("/tmp/b.jsonl"),
            cwd=Path("/workspace/project"),
            model="model-b",
            title="Beta",
            created_at=1.0,
            updated_at=2.0,
        ),
    ]


def _picker_labels(app: ForgeTuiApp) -> list[str]:
    return [
        str(item.query_one(Label).content)
        for item in app.screen.query_one("#session-picker-list", ListView).children
    ]


@pytest.mark.anyio
async def test_tui_session_picker_search_filters_records() -> None:
    session = FakeSession()
    session.session_manager = _ManagedFakeSessionManager(_picker_records())
    app = ForgeTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+r")
        assert await _screen_is(pilot, app, SessionPickerScreen)
        search_input = app.screen.query_one("#session-picker-search", Input)
        search_input.value = "beta"
        await pilot.pause()

        labels = _picker_labels(app)
        assert len(labels) == 1
        assert "Beta" in labels[0]


@pytest.mark.anyio
async def test_tui_session_picker_renames_selected_session() -> None:
    session = FakeSession()
    manager = _ManagedFakeSessionManager(_picker_records())
    session.session_manager = manager
    app = ForgeTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+r")
        assert await _screen_is(pilot, app, SessionPickerScreen)

        await pilot.press("ctrl+r")
        await pilot.pause()
        rename_input = app.screen.query_one("#session-rename-input", Input)
        rename_input.value = "Renamed"
        await pilot.press("enter")
        await pilot.pause()

        assert manager.renames == [("session-a", "Renamed")]
        assert any("Renamed" in label for label in _picker_labels(app))


@pytest.mark.anyio
async def test_tui_session_picker_delete_confirms_and_removes() -> None:
    session = FakeSession()
    manager = _ManagedFakeSessionManager(_picker_records())
    session.session_manager = manager
    app = ForgeTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+r")
        assert await _screen_is(pilot, app, SessionPickerScreen)

        await pilot.press("ctrl+d")
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()

        assert manager.deleted == ["session-a"]
        assert len(manager.records) == 1


@pytest.mark.anyio
async def test_tui_session_picker_sort_toggle_reorders_records() -> None:
    session = FakeSession()
    session.session_manager = _ManagedFakeSessionManager(_picker_records())
    app = ForgeTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+r")
        assert await _screen_is(pilot, app, SessionPickerScreen)

        await pilot.press("ctrl+s")
        await pilot.pause()

        labels = _picker_labels(app)
        assert "Alpha" in labels[0]
        assert "Beta" in labels[1]


# --------------------------------------------------------------------------- #
# Hot reload
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_tui_settings_hot_reload_applies_theme_and_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = ForgeTuiApp(FakeSession())
    signatures = iter(((1.0,), (2.0,)))
    monkeypatch.setattr(tui_app, "tui_settings_signature", lambda: next(signatures))
    monkeypatch.setattr(
        tui_app,
        "load_tui_settings",
        lambda: TuiSettings(
            keybindings=TuiKeybindings(command_palette="f10"),
            theme="forge-light",
        ),
    )

    async with app.run_test(size=(120, 30)) as pilot:
        app._settings_signature = (0.0,)
        app._maybe_reload_settings()
        await pilot.pause()

        assert app.tui_settings.theme == "forge-light"
        prompt = app.query_one("#prompt", PromptInput)
        assert "f10" in prompt.tui_keybindings.keys_for("command_palette")


@pytest.mark.anyio
async def test_tui_hot_reload_skips_unchanged_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = ForgeTuiApp(FakeSession())
    monkeypatch.setattr(tui_app, "tui_settings_signature", lambda: (1.0,))
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())

    async with app.run_test(size=(120, 30)) as pilot:
        app._settings_signature = (1.0,)
        app._maybe_reload_settings()
        await pilot.pause()

        assert app.tui_settings == TuiSettings()


# --------------------------------------------------------------------------- #
# Git branch cache
# --------------------------------------------------------------------------- #


def test_git_branch_cache_queries_once_per_head_mtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    calls: list[Path] = []

    def fake_branch(cwd: Path) -> str:
        calls.append(cwd)
        return "main" if len(calls) == 1 else "other"

    monkeypatch.setattr("forge_cli.tui.widgets._run_git_branch", fake_branch)
    _GIT_BRANCH_CACHE.clear()

    assert _git_branch(tmp_path) == "main"
    assert _git_branch(tmp_path) == "main"
    assert len(calls) == 1

    import os
    import time

    (git_dir / "HEAD").write_text("ref: refs/heads/other\n", encoding="utf-8")
    future = time.time() + 5
    os.utime(git_dir / "HEAD", (future, future))

    assert _git_branch(tmp_path) == "other"
    assert len(calls) == 2
    _GIT_BRANCH_CACHE.clear()


def test_git_branch_cache_handles_missing_git_dir(tmp_path: Path) -> None:
    _GIT_BRANCH_CACHE.clear()

    assert _git_branch(tmp_path) == "--"
    _GIT_BRANCH_CACHE.clear()


# --------------------------------------------------------------------------- #
# Session clone, fork, import, and copy commands
# --------------------------------------------------------------------------- #


class _R5Command:
    handled = True

    def __init__(self, **values: object) -> None:
        self._values = values

    def __getattr__(self, name: str) -> object:
        if name in {
            "compact_summary",
            "export_destination",
            "export_format",
            "goal_action",
            "import_path",
            "login_provider",
            "message",
            "logout_provider",
            "resume_session_id",
            "theme",
            "thinking_level",
        }:
            return self._values.get(name)
        return self._values.get(name, False)


class _R5Session(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.next_command: _R5Command | None = None
        self.forked_entry: str | None = None
        self.clone_count = 0
        self.import_decisions: list[str | None] = []
        self.import_requires_retrust = False
        self.copy_count = 0

    def handle_command(self, text: str) -> _R5Command:
        del text
        command = self.next_command
        self.next_command = None
        return command or _R5Command()

    async def clone_current_session(self) -> str:
        self.clone_count += 1
        self.messages = (HumanMessage(content="cloned transcript"),)
        return "Cloned session."

    async def fork_choices(self) -> tuple[SessionTreeChoice, ...]:
        return (
            SessionTreeChoice(entry_id="root", label="user: Root"),
            SessionTreeChoice(entry_id="tool", label="tool call: read", is_tool_call=True),
            SessionTreeChoice(entry_id="last", label="user: Last", active=True),
        )

    async def fork_from_entry(self, entry_id: str) -> SessionTreeBranchResult:
        self.forked_entry = entry_id
        self.messages = ()
        return SessionTreeBranchResult(message="Forked session.", input_prefill="fork prompt")

    async def prepare_session_import(self, path: Path) -> SimpleNamespace:
        return SimpleNamespace(trust_required=True, cwd=path.parent)

    async def commit_session_import(
        self,
        plan: object,
        *,
        trust_decision: str | None = None,
    ) -> str:
        del plan
        if self.import_requires_retrust:
            self.import_requires_retrust = False
            raise SessionImportTrustRequired("Import requires a project trust decision")
        self.import_decisions.append(trust_decision)
        self.messages = (HumanMessage(content="imported transcript"),)
        return "Imported session."

    async def copy_last_assistant(self) -> SimpleNamespace:
        self.copy_count += 1
        return SimpleNamespace(copied=True)


@pytest.mark.anyio
async def test_tui_fork_prefills_without_sending() -> None:
    session = _R5Session()
    session.next_command = _R5Command(fork_picker_requested=True)
    app = ForgeTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/fork"
        await pilot.press("enter")
        await pilot.pause()

        assert await _screen_is(pilot, app, tui_app.TreePickerScreen)
        labels = [
            str(item.query_one(Label).content)
            for item in app.screen.query_one("#tree-picker-list", ListView).children
        ]
        assert labels == ["  user: Root", "* user: Last"]
        await pilot.press("up", "enter")
        await pilot.pause()

        assert session.forked_entry == "root"
        assert session.prompt_texts == []
        assert prompt.value == "fork prompt"
        assert prompt.has_focus


@pytest.mark.anyio
async def test_tui_clone_runs_in_worker_and_refreshes_transcript() -> None:
    session = _R5Session()
    session.next_command = _R5Command(clone_requested=True)
    app = ForgeTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/clone"
        await pilot.press("enter")
        await pilot.pause()

        assert session.clone_count == 1
        assert [(item.role, item.text) for item in app.state.items] == [
            ("user", "cloned transcript"),
        ]


@pytest.mark.anyio
async def test_tui_import_trust_cancel_does_not_commit_then_allows_once(tmp_path: Path) -> None:
    session = _R5Session()
    app = ForgeTuiApp(session)
    notices: list[str] = []
    app._notify = lambda message, **kwargs: notices.append(message)  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        session.next_command = _R5Command(import_path=tmp_path / "session.jsonl")
        prompt.value = "/import session.jsonl"
        await pilot.press("enter")
        await pilot.pause()
        assert await _screen_is(pilot, app, SessionImportTrustScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert session.import_decisions == []

        session.next_command = _R5Command(import_path=tmp_path / "session.jsonl")
        prompt.value = "/import session.jsonl"
        await pilot.press("enter")
        await pilot.pause()
        assert await _screen_is(pilot, app, SessionImportTrustScreen)
        await pilot.press("enter")
        await pilot.pause()

        assert session.import_decisions == ["once"]
        assert [(item.role, item.text) for item in app.state.items] == [
            ("user", "imported transcript"),
        ]


@pytest.mark.anyio
async def test_tui_import_reopens_trust_after_commit_revalidation(tmp_path: Path) -> None:
    session = _R5Session()
    session.import_requires_retrust = True
    app = ForgeTuiApp(session)
    plan = SimpleNamespace(trust_required=False, cwd=tmp_path)

    async with app.run_test() as pilot:
        await app._commit_session_import(plan, tmp_path / "session.jsonl")
        await pilot.pause()

        assert await _screen_is(pilot, app, SessionImportTrustScreen)
        assert session.import_decisions == []


@pytest.mark.anyio
async def test_tui_copy_reports_success() -> None:
    session = _R5Session()
    session.next_command = _R5Command(copy_requested=True)
    app = ForgeTuiApp(session)
    notices: list[str] = []
    app._notify = lambda message, **kwargs: notices.append(message)  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/copy"
        await pilot.press("enter")
        await pilot.pause()

    assert session.copy_count == 1
    assert notices == ["Copied last assistant response."]
