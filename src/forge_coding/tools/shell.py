"""Shell process, cancellation, and output helpers."""

from __future__ import annotations

import asyncio
import contextlib
import locale
import ntpath
import os
import shutil
import signal
import subprocess
import tempfile
from typing import Any

from forge_agent.tools import ToolCancellationToken


def _prefixed_shell_command(command: str, prefix: str | None) -> str:
    """Return a shell command with an opt-in setup prefix applied."""
    if prefix is None:
        return command
    return f"{prefix}\n{command}"


def _windows_bash_path() -> str | None:
    """Return a real bash executable on Windows, or None when unavailable.

    ``asyncio.create_subprocess_shell`` runs ``cmd.exe`` on Windows, which
    cannot execute bash commands such as ``ls`` unless Git's tools happen to
    be on PATH. Git for Windows ships ``bash.exe``; prefer it when installed.
    """
    candidates: list[str] = []
    found = shutil.which("bash")
    if found and not _is_windows_wsl_bridge(found):
        candidates.append(found)
    roots = [
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    ]
    for root in roots:
        if not root:
            continue
        candidates.append(os.path.join(root, "Git", "bin", "bash.exe"))
        candidates.append(os.path.join(root, "Git", "usr", "bin", "bash.exe"))
        candidates.append(os.path.join(root, "Programs", "Git", "bin", "bash.exe"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _is_windows_wsl_bridge(path: str) -> bool:
    """Return whether ``path`` is the Windows WSL compatibility bridge.

    ``C:\\Windows\\System32\\bash.exe`` launches WSL rather than a native
    bash process. It emits UTF-16 diagnostics and is not suitable for the
    subprocess contract used by Forge, so Git Bash candidates should win.
    """

    candidate = ntpath.normpath(path)
    return (
        ntpath.basename(candidate).casefold() == "bash.exe"
        and ntpath.basename(ntpath.dirname(candidate)).casefold() == "system32"
        and ntpath.basename(ntpath.dirname(ntpath.dirname(candidate))).casefold() == "windows"
    )


def _shell_output_encodings() -> list[str]:
    """Return codepages that may encode shell output on this machine.

    cmd.exe emits the OEM codepage (e.g. cp936 on Chinese Windows), which is
    not valid UTF-8; bash (POSIX or Git for Windows) emits UTF-8. Prefer the
    OEM codepage on Windows, then the locale encoding.
    """
    encodings: list[str] = []
    if os.name == "nt":
        try:
            import ctypes

            encodings.append(f"cp{ctypes.windll.kernel32.GetOEMCP()}")
        except Exception:
            pass
    with contextlib.suppress(Exception):
        encodings.append(locale.getpreferredencoding(False))
    return encodings


def _decode_shell_output(data: bytes, *, encodings: list[str] | None = None) -> str:
    """Decode shell output as UTF-8, falling back to local codepages.

    Decoding with ``errors="replace"`` from the start turns cmd.exe error
    text on non-UTF-8 Windows locales into mojibake. Prefer strict UTF-8;
    only when that fails, try the OEM and locale codepages before giving up.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    fallback_encodings = _shell_output_encodings() if encodings is None else encodings
    for encoding in dict.fromkeys(fallback_encodings):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


async def _communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    *,
    timeout: float | None,
    signal: ToolCancellationToken | None,
) -> tuple[bytes, bytes | None, bool, bool]:
    communicate = asyncio.create_task(process.communicate())
    cancel_watch: asyncio.Task[None] | None = None
    try:
        wait_for: set[asyncio.Task[Any]] = {communicate}
        if signal is not None:
            cancel_watch = asyncio.create_task(_wait_for_cancel(signal))
            wait_for.add(cancel_watch)

        done, _pending = await asyncio.wait(
            wait_for,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if communicate in done:
            output_bytes, stderr = communicate.result()
            return output_bytes, stderr, False, False

        cancelled = cancel_watch is not None and cancel_watch in done
        _kill_process_tree(process)
        try:
            output_bytes, stderr = await communicate
        except asyncio.CancelledError:
            output_bytes = b""
            stderr_result: bytes | None = None
        else:
            stderr_result = stderr
        return output_bytes, stderr_result, not cancelled, cancelled
    except asyncio.CancelledError:
        _kill_process_tree(process)
        if not communicate.done():
            communicate.cancel()
        raise
    finally:
        if cancel_watch is not None:
            cancel_watch.cancel()


async def _wait_for_cancel(signal: ToolCancellationToken) -> None:
    while not signal.is_cancelled():
        await asyncio.sleep(0.05)


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except ProcessLookupError:
            return
    else:
        # The shell may be cmd.exe (Windows fallback) or a direct bash process
        # (Git for Windows); either way, killing only that direct process
        # leaves grandchildren holding the captured output pipe open, so
        # ``communicate()`` cannot return promptly on cancellation.
        # taskkill receives a validated integer PID as a distinct argument.
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return
        try:
            process.kill()
        except ProcessLookupError:
            return


def _write_temp_output(output: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="forge-bash-",
        suffix=".log",
        delete=False,
    ) as handle:
        handle.write(output)
        return handle.name
