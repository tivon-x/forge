from pathlib import Path

import pytest

from forge_cli.tui.config import (
    BUILTIN_TUI_THEME_NAMES,
    FORGE_DARK_THEME,
    HIGH_CONTRAST_THEME,
    TuiConfigError,
    TuiKeybindings,
    TuiSettings,
    available_theme_names,
    get_tui_theme,
    load_tui_settings,
    load_user_themes,
    save_tui_settings,
    tui_settings_from_json,
    tui_settings_path,
    tui_settings_signature,
)
from forge_coding.paths import ForgePaths


def test_tui_settings_path_uses_forge_home(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")

    assert tui_settings_path(paths) == tmp_path / ".forge" / "tui.json"


def test_load_tui_settings_returns_defaults_when_file_is_missing(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")

    assert load_tui_settings(paths) == TuiSettings()
    assert load_tui_settings(paths).keybindings.quit == "ctrl+d"


def test_load_tui_settings_reads_keybindings(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    path = tui_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text(
        """
        {
          "keybindings": {
            "command_palette": "ctrl+j",
            "session_picker": "ctrl+y",
            "queue_follow_up": "f5",
            "accept_completion": "f2",
            "thinking_cycle": "f3",
            "model_cycle": "f6",
            "toggle_thinking": "f4",
            "copy_message": "ctrl+b"
          },
          "theme": "high-contrast"
        }
        """,
        encoding="utf-8",
    )

    settings = load_tui_settings(paths)

    assert settings.keybindings.command_palette == "ctrl+j"
    assert settings.keybindings.session_picker == "ctrl+y"
    assert settings.keybindings.queue_follow_up == "f5"
    assert settings.keybindings.toggle_tool_results == "ctrl+o"
    assert settings.keybindings.toggle_thinking == "f4"
    assert settings.keybindings.accept_completion == "f2"
    assert settings.keybindings.thinking_cycle == "f3"
    assert settings.keybindings.model_cycle == "f6"
    assert settings.keybindings.copy_message == "ctrl+b"
    assert settings.keybindings.cancel == "escape"
    assert settings.theme == "high-contrast"
    assert settings.resolved_theme == HIGH_CONTRAST_THEME


def test_save_tui_settings_writes_json(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")

    path = save_tui_settings(TuiSettings(theme="forge-light"), paths)

    assert path == tmp_path / ".forge" / "tui.json"
    assert load_tui_settings(paths).theme == "forge-light"


def test_tui_settings_ignores_removed_message_selection_keybindings() -> None:
    settings = tui_settings_from_json(
        {
            "keybindings": {
                "message_previous": "alt+up",
                "message_next": "alt+down",
            }
        }
    )

    assert settings == TuiSettings()


def test_tui_settings_reject_unknown_fields() -> None:
    with pytest.raises(TuiConfigError, match="Unknown TUI settings field"):
        tui_settings_from_json({"palette": {}})


def test_tui_keybindings_reject_duplicate_keys() -> None:
    with pytest.raises(TuiConfigError, match="assigned to both"):
        tui_settings_from_json(
            {
                "keybindings": {
                    "cancel": "escape",
                    "command_palette": "escape",
                }
            }
        )


def test_tui_settings_reject_unknown_theme() -> None:
    with pytest.raises(TuiConfigError, match="Unknown TUI theme"):
        tui_settings_from_json({"theme": "solarized"})


def test_tui_settings_accept_light_theme() -> None:
    settings = tui_settings_from_json({"theme": "forge-light"})

    assert settings.theme == "forge-light"
    assert settings.resolved_theme.screen_background == "#ffffff"
    assert settings.resolved_theme.syntax_theme == "ansi_light"
    assert settings.resolved_theme.markdown_heading == settings.resolved_theme.accent
    assert settings.resolved_theme.markdown_bullet == settings.resolved_theme.accent


def test_tui_settings_load_auto_copy_selection() -> None:
    settings = tui_settings_from_json({"auto_copy_selection": True})

    assert settings.auto_copy_selection is True
    assert settings.to_json()["auto_copy_selection"] is True


def test_tui_settings_reject_invalid_auto_copy_selection() -> None:
    with pytest.raises(TuiConfigError, match="auto_copy_selection"):
        tui_settings_from_json({"auto_copy_selection": "yes"})


def test_tui_keybindings_serialize_to_json() -> None:
    settings = TuiSettings(
        keybindings=TuiKeybindings(
            command_palette="ctrl+j",
            session_picker="ctrl+y",
            queue_follow_up="f5",
            accept_completion="f2",
            thinking_cycle="f3",
            model_cycle="f6",
            toggle_thinking="f4",
            copy_message="ctrl+b",
        ),
        theme="high-contrast",
    )

    assert settings.to_json()["keybindings"]["command_palette"] == "ctrl+j"
    assert settings.to_json()["keybindings"]["session_picker"] == "ctrl+y"
    assert settings.to_json()["keybindings"]["queue_follow_up"] == "f5"
    assert settings.to_json()["keybindings"]["toggle_tool_results"] == "ctrl+o"
    assert settings.to_json()["keybindings"]["toggle_thinking"] == "f4"
    assert settings.to_json()["keybindings"]["accept_completion"] == "f2"
    assert settings.to_json()["keybindings"]["thinking_cycle"] == "f3"
    assert settings.to_json()["keybindings"]["model_cycle"] == "f6"
    assert settings.to_json()["keybindings"]["copy_message"] == "ctrl+b"
    assert settings.to_json()["theme"] == "high-contrast"
    assert settings.to_json()["auto_copy_selection"] is False


def test_get_tui_theme_returns_builtin_theme() -> None:
    assert get_tui_theme("high-contrast").prompt_border == "#00ff66"
    assert get_tui_theme("forge-light").prompt_border == "#2563eb"
    assert get_tui_theme("forge-dark").screen_background == "#000000"
    assert get_tui_theme("forge-dark").accent == "#6ea8fe"


def test_tui_theme_exposes_semantic_tool_status_colors() -> None:
    themes = (
        TuiSettings().resolved_theme,
        tui_settings_from_json({"theme": "forge-light"}).resolved_theme,
        HIGH_CONTRAST_THEME,
    )
    for theme in themes:
        assert theme.success.startswith("#")
        assert theme.error.startswith("#")


def test_tui_theme_exposes_subagent_semantic_styles() -> None:
    themes = (
        TuiSettings().resolved_theme,
        tui_settings_from_json({"theme": "forge-light"}).resolved_theme,
        HIGH_CONTRAST_THEME,
    )
    for theme in themes:
        for role in ("subagent", "subagent-running", "subagent-success", "subagent-error"):
            style = theme.role_styles[role]
            assert style.border
            assert style.body


def test_tui_keybindings_accept_multi_key_arrays() -> None:
    settings = tui_settings_from_json(
        {
            "keybindings": {
                "cancel": ["escape", "ctrl+c"],
                "yank": ["ctrl+y", "f9"],
            }
        }
    )

    assert settings.keybindings.keys_for("cancel") == ("escape", "ctrl+c")
    assert settings.keybindings.keys_for("yank") == ("ctrl+y", "f9")
    assert settings.keybindings.to_json()["cancel"] == ["escape", "ctrl+c"]


def test_tui_keybindings_explicit_key_overrides_other_default() -> None:
    settings = tui_settings_from_json({"keybindings": {"session_picker": "ctrl+y"}})

    assert settings.keybindings.keys_for("session_picker") == ("ctrl+y",)
    assert settings.keybindings.keys_for("yank") == ()


def test_tui_keybindings_reject_duplicate_explicit_keys() -> None:
    with pytest.raises(TuiConfigError, match="assigned to both"):
        tui_settings_from_json(
            {
                "keybindings": {
                    "cancel": ["escape", "f9"],
                    "command_palette": "f9",
                }
            }
        )


def test_tui_keybindings_round_trip_stable() -> None:
    settings = TuiSettings()
    reloaded = tui_settings_from_json(settings.to_json())

    assert reloaded.keybindings == settings.keybindings
    assert reloaded.to_json() == settings.to_json()


def test_tui_keybindings_key_display_joins_keys() -> None:
    settings = tui_settings_from_json({"keybindings": {"cancel": ["escape", "ctrl+c"]}})

    assert settings.keybindings.key_display("cancel") == "Escape / Ctrl+C"


def test_load_user_themes_reads_home_theme_files(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    themes_dir = paths.home / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "my.json").write_text(
        """
        {
          "name": "my-theme",
          "accent": "#ff0000",
          "shell_border": "#00ff00",
          "thinking_borders": {"high": "#ffff00"}
        }
        """,
        encoding="utf-8",
    )

    themes = load_user_themes(paths)

    assert set(themes) == {"my-theme"}
    theme = get_tui_theme("my-theme", paths)
    assert theme.accent == "#ff0000"
    assert theme.shell_border == "#00ff00"
    assert theme.thinking_border("high") == "#ffff00"
    assert theme.thinking_border("unknown") == theme.accent
    # Fields that were not overridden fall back to the dark theme.
    assert theme.screen_background == FORGE_DARK_THEME.screen_background
    assert theme.role_styles["user"].border == FORGE_DARK_THEME.role_styles["user"].border


def test_user_theme_json_rejects_unknown_fields(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    themes_dir = paths.home / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "bad.json").write_text(
        '{"name": "bad", "accent_color": "#ff0000"}',
        encoding="utf-8",
    )

    assert load_user_themes(paths) == {}


def test_available_theme_names_lists_user_themes_after_builtins(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    themes_dir = paths.home / "themes"
    themes_dir.mkdir(parents=True)
    (themes_dir / "z.json").write_text('{"name": "z-theme"}', encoding="utf-8")
    (themes_dir / "a.json").write_text('{"name": "a-theme"}', encoding="utf-8")

    names = available_theme_names(paths)

    assert names[:3] == BUILTIN_TUI_THEME_NAMES
    assert names[3:] == ("a-theme", "z-theme")


def test_theme_name_accepts_user_themes(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge_cli.tui import config as tui_config

    monkeypatch.setattr(tui_config, "load_user_themes", lambda paths=None: {"my-theme": object()})  # type: ignore[arg-type,return-value]

    settings = tui_settings_from_json({"theme": "my-theme"})

    assert settings.theme == "my-theme"


def test_tui_settings_signature_tracks_theme_files(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    empty_signature = tui_settings_signature(paths)

    themes_dir = paths.home / "themes"
    themes_dir.mkdir(parents=True)
    theme_path = themes_dir / "a.json"
    theme_path.write_text('{"name": "a-theme"}', encoding="utf-8")

    assert tui_settings_signature(paths) != empty_signature

    before = tui_settings_signature(paths)
    import time

    time.sleep(0.01)
    theme_path.write_text('{"name": "a-theme", "accent": "#123456"}', encoding="utf-8")

    assert tui_settings_signature(paths) != before
