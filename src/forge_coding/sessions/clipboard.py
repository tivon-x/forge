"""Small, platform-specific clipboard helpers for coding sessions."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

CLIPBOARD_TIMEOUT_SECONDS = 5.0
MAX_CLIPBOARD_ERROR_LENGTH = 256


class ClipboardStatus(StrEnum):
    """Outcome of a clipboard write."""

    COPIED = "copied"
    NO_COMMAND = "no_command"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ClipboardResult:
    """Bounded result for a clipboard write attempt."""

    status: ClipboardStatus
    command: str | None = None
    error: str | None = None

    @property
    def copied(self) -> bool:
        return self.status is ClipboardStatus.COPIED


def _commands_for_platform(platform_name: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if platform_name.startswith("win"):
        return (("clip.exe", ()),)
    if platform_name == "darwin":
        return (("pbcopy", ()),)
    if platform_name.startswith("linux"):
        return (
            ("wl-copy", ()),
            ("xclip", ("-selection", "clipboard")),
            ("xsel", ("--clipboard", "--input")),
        )
    return ()


def _bounded_error(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    message = str(value).strip()
    if not message:
        return None
    if len(message) <= MAX_CLIPBOARD_ERROR_LENGTH:
        return message
    return message[: MAX_CLIPBOARD_ERROR_LENGTH - 1] + "…"


def copy_to_clipboard(text: str) -> ClipboardResult:
    """Copy ``text`` through the first available native clipboard command."""

    candidates = _commands_for_platform(sys.platform)
    last_failure: ClipboardResult | None = None
    found_command = False

    for command, args in candidates:
        executable = shutil.which(command)
        if executable is None:
            continue
        found_command = True
        try:
            completed = subprocess.run(
                [executable, *args],
                input=text,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                timeout=CLIPBOARD_TIMEOUT_SECONDS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ClipboardResult(
                ClipboardStatus.TIMEOUT,
                command=command,
                error=f"Clipboard command timed out after {CLIPBOARD_TIMEOUT_SECONDS:g} seconds.",
            )
        except OSError as exc:
            last_failure = ClipboardResult(
                ClipboardStatus.FAILED,
                command=command,
                error=_bounded_error(exc) or "Unable to start clipboard command.",
            )
            continue

        if completed.returncode == 0:
            return ClipboardResult(ClipboardStatus.COPIED, command=command)

        detail = _bounded_error(completed.stderr)
        last_failure = ClipboardResult(
            ClipboardStatus.FAILED,
            command=command,
            error=detail or f"Clipboard command exited with status {completed.returncode}.",
        )

    if last_failure is not None:
        return last_failure
    if not found_command:
        return ClipboardResult(
            ClipboardStatus.NO_COMMAND,
            error="No supported clipboard command found.",
        )
    return ClipboardResult(ClipboardStatus.FAILED, error="Clipboard write failed.")
