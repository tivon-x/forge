"""Shared execution and display helpers for native search tools."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Mapping
from pathlib import Path

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken
from forge_agent.types import JSONValue
from forge_coding.tools.base import ToolInputError
from forge_coding.tools.common import _path_arg, _workspace_root
from forge_coding.tools.shell import OUTPUT_TRUNCATION_MARKER, _run_executable
from forge_coding.tools.tool_manager import ToolManager, ToolManagerError
from forge_coding.tools.truncation import truncate_head

DEFAULT_SEARCH_TIMEOUT_SECONDS = 30.0


def search_path(
    arguments: Mapping[str, JSONValue],
    *,
    root: Path,
    context: ForgeRuntimeContext | None,
) -> tuple[Path, Path]:
    workspace = _workspace_root(context, root).resolve()
    value = arguments.get("path", ".")
    if value is None:
        value = "."
    if not isinstance(value, str):
        raise ToolInputError("path must be a string")
    try:
        path = _path_arg({"path": value}, "path", cwd=workspace)
    except ToolInputError as exc:
        message = str(exc).casefold()
        if "outside" in message:
            raise ToolInputError("Path is outside the project workspace") from None
        raise ToolInputError("Invalid search path") from None
    try:
        relative = path.relative_to(workspace)
    except ValueError as exc:
        raise ToolInputError("Path is outside the project workspace") from exc
    if not path.exists():
        raise ToolInputError("Path does not exist inside the project workspace")
    return workspace, relative


async def run_search(
    *,
    manager: ToolManager,
    tool_name: str,
    arguments: list[str],
    workspace: Path,
    timeout: float | None,
    signal: ToolCancellationToken | None,
) -> tuple[AgentToolResult | None, bytes, int | None, bool, bool]:
    """Ensure and run a managed search executable."""

    if signal is not None and signal.is_cancelled():
        return _cancelled_search(tool_name)
    try:
        executable = await _ensure_tool(manager, tool_name, signal)
    except ToolManagerError as exc:
        if signal is not None and signal.is_cancelled():
            return _cancelled_search(tool_name)
        offline = manager.environ.get("FORGE_OFFLINE", "").strip().casefold()
        offline_state = "enabled" if offline in {"1", "true", "yes"} else "disabled"
        detail = str(exc).splitlines()[0][:160]
        message = (
            f"{tool_name} acquisition failed: {detail}. "
            f"FORGE_OFFLINE is {offline_state}. "
            f"Install {tool_name} manually on PATH or in ~/.forge/bin."
        )
        return (
            AgentToolResult(
                tool_call_id="",
                name=tool_name,
                ok=False,
                content=message,
                error=message,
            ),
            b"",
            None,
            False,
            False,
        )
    try:
        output, returncode, timed_out, cancelled = await _run_executable(
            str(executable),
            arguments,
            cwd=workspace,
            timeout=timeout,
            signal=signal,
        )
    except (OSError, ValueError) as exc:
        message = type(exc).__name__
        return (
            AgentToolResult(
                tool_call_id="",
                name=tool_name,
                ok=False,
                content=f"Unable to run {tool_name}: {message}",
                error=message,
            ),
            b"",
            None,
            False,
            False,
        )
    return None, output, returncode, timed_out, cancelled


def _cancelled_search(
    tool_name: str,
) -> tuple[AgentToolResult, bytes, None, bool, bool]:
    message = f"{tool_name} cancelled"
    return (
        AgentToolResult(
            tool_call_id="",
            name=tool_name,
            ok=False,
            content=message,
            error=message,
            data={"cancelled": True},
        ),
        b"",
        None,
        False,
        True,
    )


async def _ensure_tool(
    manager: ToolManager,
    name: str,
    signal: ToolCancellationToken | None,
) -> Path:
    # The manager's network path is blocking by design; keep it outside the
    # event loop and bridge task cancellation into its synchronous checks.
    thread_signal = _AcquisitionSignal(signal)
    pending = asyncio.create_task(
        asyncio.to_thread(manager.ensure_tool, name, signal=thread_signal)
    )
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        thread_signal.cancel()
        with contextlib.suppress(Exception):
            await asyncio.shield(pending)
        raise


class _AcquisitionSignal:
    def __init__(self, upstream: ToolCancellationToken | None) -> None:
        self._upstream = upstream
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set() or (
            self._upstream is not None and self._upstream.is_cancelled()
        )


def timeout_arg(arguments: Mapping[str, JSONValue]) -> float:
    value = arguments.get("timeout")
    if value is None:
        return DEFAULT_SEARCH_TIMEOUT_SECONDS
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError("timeout must be greater than 0")
    return float(value)


def strip_output_marker(data: bytes) -> tuple[bytes, bool]:
    truncated = OUTPUT_TRUNCATION_MARKER in data
    return data.replace(OUTPUT_TRUNCATION_MARKER, b""), truncated


def display_path(value: str, *, workspace: Path) -> str | None:
    normalized = value.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        try:
            normalized = candidate.resolve().relative_to(workspace).as_posix()
        except ValueError:
            return None
    else:
        try:
            normalized = (workspace / candidate).resolve().relative_to(workspace).as_posix()
        except ValueError:
            return None
    return normalized or "."


def bounded_result(
    *,
    name: str,
    content: str,
    data: dict[str, JSONValue],
    returncode: int | None,
    timed_out: bool,
    cancelled: bool,
) -> AgentToolResult:
    truncation = truncate_head(content)
    details = dict(data)
    details["return_code"] = returncode
    details["timed_out"] = timed_out
    details["cancelled"] = cancelled
    details["truncation"] = truncation.to_json()
    if timed_out:
        message = f"{name} timed out"
        return AgentToolResult(
            tool_call_id="",
            name=name,
            ok=False,
            content=message,
            data=details,
            error=message,
        )
    if cancelled:
        message = f"{name} cancelled"
        return AgentToolResult(
            tool_call_id="",
            name=name,
            ok=False,
            content=message,
            data=details,
            error=message,
        )
    return AgentToolResult(
        tool_call_id="",
        name=name,
        ok=returncode in (0, 1),
        content=truncation.content or "(no matches)",
        data=details,
        error=None if returncode in (0, 1) else f"{name} exited with code {returncode}",
    )


__all__ = [
    "DEFAULT_SEARCH_TIMEOUT_SECONDS",
    "bounded_result",
    "display_path",
    "run_search",
    "search_path",
    "strip_output_marker",
    "timeout_arg",
]
