from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

import forge_coding.sessions.clipboard as clipboard


def test_linux_falls_back_to_xclip_and_writes_unicode_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    available = {"xclip": "C:/bin/xclip"}
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: available.get(name))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(clipboard.subprocess, "run", run)

    result = clipboard.copy_to_clipboard("中文\nemoji 🦄")

    assert result.status is clipboard.ClipboardStatus.COPIED
    assert result.command == "xclip"
    assert calls == [
        (
            ["C:/bin/xclip", "-selection", "clipboard"],
            {
                "input": "中文\nemoji 🦄",
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "shell": False,
                "timeout": 5.0,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.PIPE,
                "check": False,
            },
        )
    ]


def test_windows_uses_clip_exe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "win32")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: name)
    seen: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen["command"] = command
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(clipboard.subprocess, "run", run)

    result = clipboard.copy_to_clipboard("你好")

    assert result.copied
    assert seen["command"] == ["clip.exe"]
    assert seen["shell"] is False
    assert seen["input"] == "你好"


def test_no_command_is_distinct_from_failed_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: None)

    result = clipboard.copy_to_clipboard("text")

    assert result.status is clipboard.ClipboardStatus.NO_COMMAND
    assert result.error == "No supported clipboard command found."


def test_timeout_is_distinct_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: name)

    def run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise subprocess.TimeoutExpired("pbcopy", clipboard.CLIPBOARD_TIMEOUT_SECONDS)

    monkeypatch.setattr(clipboard.subprocess, "run", run)

    result = clipboard.copy_to_clipboard("text")

    assert result.status is clipboard.ClipboardStatus.TIMEOUT
    assert result.command == "pbcopy"
    assert len(result.error or "") <= clipboard.MAX_CLIPBOARD_ERROR_LENGTH


def test_failed_commands_fall_back_then_return_bounded_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: name)
    calls = 0

    def run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stderr="x" * 500)

    monkeypatch.setattr(clipboard.subprocess, "run", run)

    result = clipboard.copy_to_clipboard("text")

    assert calls == 3
    assert result.status is clipboard.ClipboardStatus.FAILED
    assert result.command == "xsel"
    assert len(result.error or "") == clipboard.MAX_CLIPBOARD_ERROR_LENGTH


def test_unknown_platform_has_no_supported_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "plan9")

    result = clipboard.copy_to_clipboard("text")

    assert result.status is clipboard.ClipboardStatus.NO_COMMAND
