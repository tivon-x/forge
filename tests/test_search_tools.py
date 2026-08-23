from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from time import monotonic, sleep

import pytest

from forge_agent.tools import ToolCancellationToken
from forge_coding.tools import ToolInputError, search_common
from forge_coding.tools.find import create_find_tool
from forge_coding.tools.grep import create_grep_tool
from forge_coding.tools.ls import create_ls_tool
from forge_coding.tools.shell import OUTPUT_TRUNCATION_MARKER, _run_executable
from forge_coding.tools.tool_manager import ToolManager


class StubManager(ToolManager):
    def __init__(self, executable: Path) -> None:
        super().__init__(home=executable.parent)
        self.executable = executable

    def ensure_tool(self, name: str, *, signal: ToolCancellationToken | None = None) -> Path:
        del name, signal
        return self.executable


class CancelledSignal:
    def is_cancelled(self) -> bool:
        return True


class EnsureMustNotRun(StubManager):
    def ensure_tool(self, name: str, *, signal: ToolCancellationToken | None = None) -> Path:
        del name, signal
        raise AssertionError("ensure_tool must not run after cancellation")


@pytest.mark.anyio
async def test_fixed_executable_timeout_kills_the_child(tmp_path: Path) -> None:
    started = monotonic()
    output, returncode, timed_out, cancelled = await _run_executable(
        sys.executable,
        ["-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        timeout=0.1,
        signal=None,
    )

    assert output == b""
    assert returncode is not None
    assert timed_out is True
    assert cancelled is False
    assert monotonic() - started < 5


def _match_event(path: str, line_number: int, text: str, *, event_type: str = "match") -> bytes:
    return (
        json.dumps(
            {
                "type": event_type,
                "data": {
                    "path": {"text": path},
                    "line_number": line_number,
                    "lines": {"text": text + "\n"},
                },
            }
        ).encode()
        + b"\n"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("FORGE_RUN_INTEGRATION", "").casefold() not in {"1", "true", "yes"},
    reason="set FORGE_RUN_INTEGRATION=1 to run preinstalled rg/fd integration",
)
@pytest.mark.skipif(
    shutil.which("rg") is None or (shutil.which("fd") or shutil.which("fdfind")) is None,
    reason="rg and fd/fdfind are not installed",
)
@pytest.mark.anyio
async def test_real_search_tools_include_hidden_but_respect_gitignore(tmp_path: Path) -> None:
    rg = shutil.which("rg")
    fd = shutil.which("fd") or shutil.which("fdfind")
    assert rg is not None and fd is not None
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "visible.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("needle\n", encoding="utf-8")
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "skip.py").write_text("needle\n", encoding="utf-8")

    grep_result = await create_grep_tool(
        cwd=tmp_path, tool_manager=StubManager(Path(rg))
    ).execute({"pattern": "needle"})
    find_result = await create_find_tool(
        cwd=tmp_path, tool_manager=StubManager(Path(fd))
    ).execute({"pattern": "*.py"})
    assert grep_result.ok is True
    assert ".hidden.py:1: needle" in grep_result.content
    assert "ignored/skip.py" not in grep_result.content
    assert find_result.ok is True
    assert ".hidden.py" in find_result.content
    assert "visible.py" in find_result.content
    assert "ignored/skip.py" not in find_result.content


@pytest.mark.anyio
async def test_grep_schema_argv_json_parsing_and_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float | None,
        signal: ToolCancellationToken | None,
    ) -> tuple[bytes, int, bool, bool]:
        captured.update(executable=executable, arguments=arguments, cwd=cwd, timeout=timeout)
        return _match_event("src\\main.py", 4, "Needle"), 0, False, False

    monkeypatch.setattr(search_common, "_run_executable", fake_run)
    tool = create_grep_tool(cwd=tmp_path, tool_manager=StubManager(tmp_path / "rg"))
    result = await tool.execute(
        {
            "pattern": "Needle",
            "path": ".",
            "glob": "*.py",
            "ignore_case": True,
            "literal": True,
            "context": 2,
            "limit": 1,
            "timeout": 0.5,
        }
    )

    assert result.ok is True
    assert result.content == "src/main.py:4: Needle"
    assert result.data is not None
    assert result.data["matches"] == 1
    arguments = captured["arguments"]
    assert isinstance(arguments, list)
    assert arguments[:4] == ["--json", "--line-number", "--color=never", "--hidden"]
    assert "--ignore-case" in arguments
    assert "--fixed-strings" in arguments
    assert "--context" in arguments
    assert "--glob" in arguments
    assert captured["timeout"] == 0.5


@pytest.mark.anyio
async def test_grep_rejects_workspace_escape(tmp_path: Path) -> None:
    tool = create_grep_tool(cwd=tmp_path, tool_manager=StubManager(tmp_path / "rg"))
    with pytest.raises(ToolInputError, match="outside the project workspace"):
        await tool.execute({"pattern": "secret", "path": "../outside"})


@pytest.mark.anyio
async def test_managed_tool_failure_is_bounded_and_actionable(tmp_path: Path) -> None:
    manager = ToolManager(
        home=tmp_path / "home",
        system="linux",
        machine="x86_64",
        environ={"PATH": "", "FORGE_OFFLINE": "yes"},
    )
    result = await create_grep_tool(cwd=tmp_path, tool_manager=manager).execute(
        {"pattern": "needle"}
    )

    assert result.ok is False
    assert "acquisition failed" in result.content
    assert "FORGE_OFFLINE is enabled" in result.content
    assert "Install rg manually" in result.content
    assert len(result.content) < 500


@pytest.mark.anyio
async def test_find_sorts_posix_relative_results_and_uses_default_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float | None,
        signal: ToolCancellationToken | None,
    ) -> tuple[bytes, int, bool, bool]:
        captured.update(arguments=arguments, cwd=cwd)
        return b"z\\nested\\z.py\0A.py\0a.py\0", 0, False, False

    monkeypatch.setattr(search_common, "_run_executable", fake_run)
    tool = create_find_tool(cwd=tmp_path, tool_manager=StubManager(tmp_path / "fd"))
    result = await tool.execute({"pattern": "*.py"})

    assert result.ok is True
    assert result.content.splitlines() == ["A.py", "a.py", "z/nested/z.py"]
    arguments = captured["arguments"]
    assert isinstance(arguments, list)
    assert arguments[:3] == ["--glob", "--color=never", "--hidden"]
    assert "--print0" in arguments
    assert "1000" in arguments


@pytest.mark.anyio
async def test_find_drops_external_output_and_reports_bounded_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float | None,
        signal: ToolCancellationToken | None,
    ) -> tuple[bytes, int, bool, bool]:
        del executable, arguments, cwd, timeout, signal
        return b"../../secret\0inside.py\0" + OUTPUT_TRUNCATION_MARKER, 0, False, False

    monkeypatch.setattr(search_common, "_run_executable", fake_run)
    tool = create_find_tool(cwd=tmp_path, tool_manager=StubManager(tmp_path / "fd"))
    result = await tool.execute({"pattern": "*"})

    assert result.ok is True
    assert result.content == "inside.py"
    assert result.data is not None
    assert result.data["output_truncated"] is True


@pytest.mark.anyio
async def test_find_preserves_legal_whitespace_in_nul_delimited_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float | None,
        signal: ToolCancellationToken | None,
    ) -> tuple[bytes, int, bool, bool]:
        del executable, arguments, cwd, timeout, signal
        return b" leading name.py\0line\nbreak.py\0", 0, False, False

    monkeypatch.setattr(search_common, "_run_executable", fake_run)
    result = await create_find_tool(
        cwd=tmp_path, tool_manager=StubManager(tmp_path / "fd")
    ).execute({"pattern": "*.py"})

    assert result.ok is True
    assert result.content == " leading name.py\nline\nbreak.py"


@pytest.mark.anyio
async def test_search_process_error_does_not_expose_executable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "private" / "fd"

    async def fail_run(*_args: object, **_kwargs: object) -> tuple[bytes, int, bool, bool]:
        raise OSError(f"cannot execute {secret}")

    monkeypatch.setattr(search_common, "_run_executable", fail_run)
    result = await create_find_tool(
        cwd=tmp_path, tool_manager=StubManager(secret)
    ).execute({"pattern": "*"})

    assert result.ok is False
    assert result.content == "Unable to run fd: OSError"
    assert str(secret) not in result.content


@pytest.mark.anyio
async def test_grep_stops_parsing_after_match_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _match_event("inside.py", 0, "before", event_type="context")
    output += _match_event("inside.py", 1, "first")
    output += _match_event("inside.py", 2, "after", event_type="context")
    output += _match_event("inside.py", 3, "second")

    async def fake_run(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float | None,
        signal: ToolCancellationToken | None,
    ) -> tuple[bytes, int, bool, bool]:
        del executable, arguments, cwd, timeout, signal
        return output, 0, False, False

    monkeypatch.setattr(search_common, "_run_executable", fake_run)
    tool = create_grep_tool(cwd=tmp_path, tool_manager=StubManager(tmp_path / "rg"))
    result = await tool.execute({"pattern": "first", "limit": 1})

    assert result.ok is True
    assert result.content.splitlines() == [
        "inside.py:0: before",
        "inside.py:1: first",
        "inside.py:2: after",
    ]


@pytest.mark.anyio
async def test_search_reports_timeout_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float | None,
        signal: ToolCancellationToken | None,
    ) -> tuple[bytes, int, bool, bool]:
        del executable, arguments, cwd, timeout, signal
        return b"", 137, True, False

    monkeypatch.setattr(search_common, "_run_executable", fake_run)
    tool = create_find_tool(cwd=tmp_path, tool_manager=StubManager(tmp_path / "fd"))
    result = await tool.execute({"pattern": "*.py"})
    assert result.ok is False
    assert "timed out" in result.content


@pytest.mark.anyio
async def test_search_reports_nonzero_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float | None,
        signal: ToolCancellationToken | None,
    ) -> tuple[bytes, int, bool, bool]:
        del executable, arguments, cwd, timeout, signal
        return b"rg failed", 2, False, False

    monkeypatch.setattr(search_common, "_run_executable", fake_run)
    tool = create_grep_tool(cwd=tmp_path, tool_manager=StubManager(tmp_path / "rg"))
    result = await tool.execute({"pattern": "needle"})
    assert result.ok is False
    assert result.error == "grep exited with code 2"
    assert result.data is not None
    assert result.data["return_code"] == 2


@pytest.mark.anyio
async def test_search_reports_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(
        executable: str,
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float | None,
        signal: ToolCancellationToken | None,
    ) -> tuple[bytes, int | None, bool, bool]:
        del executable, arguments, cwd, timeout, signal
        return b"", None, False, True

    monkeypatch.setattr(search_common, "_run_executable", fake_run)
    tool = create_find_tool(cwd=tmp_path, tool_manager=StubManager(tmp_path / "fd"))
    result = await tool.execute({"pattern": "*"})
    assert result.ok is False
    assert result.content == "find cancelled"
    assert result.data is not None
    assert result.data["cancelled"] is True


@pytest.mark.anyio
async def test_search_does_not_ensure_or_spawn_when_already_cancelled(tmp_path: Path) -> None:
    manager = EnsureMustNotRun(tmp_path / "fd")
    result = await create_find_tool(cwd=tmp_path, tool_manager=manager).execute(
        {"pattern": "*"}, signal=CancelledSignal()
    )

    assert result.ok is False
    assert result.content == "fd cancelled"
    assert result.data == {"cancelled": True}


@pytest.mark.anyio
async def test_search_task_cancellation_stops_and_joins_tool_acquisition(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    stopped = threading.Event()

    class BlockingManager(StubManager):
        def ensure_tool(
            self,
            name: str,
            *,
            signal: ToolCancellationToken | None = None,
        ) -> Path:
            del name
            started.set()
            while signal is None or not signal.is_cancelled():
                sleep(0.01)
            stopped.set()
            raise RuntimeError("cancelled")

    task = asyncio.create_task(
        create_find_tool(
            cwd=tmp_path,
            tool_manager=BlockingManager(tmp_path / "fd"),
        ).execute({"pattern": "*"})
    )
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.anyio
async def test_ls_stable_order_type_size_and_symlink(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("123", encoding="utf-8")
    (tmp_path / "A.txt").write_text("1", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(tmp_path / "z.txt")
    except OSError:
        link = None  # type: ignore[assignment]

    result = await create_ls_tool(cwd=tmp_path).execute({})
    assert result.ok is True
    lines = result.content.splitlines()
    assert lines[0].startswith("A.txt")
    assert any(line.startswith("folder/") for line in lines)
    assert any(line.startswith("z.txt") and "3B" in line for line in lines)
    if link is not None:
        assert any(line.startswith("link") and "symlink" in line for line in lines)


@pytest.mark.anyio
async def test_ls_empty_directory_and_workspace_escape(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = await create_ls_tool(cwd=tmp_path).execute({"path": "empty"})
    assert result.ok is True
    assert result.content == "(empty directory)"
    assert result.data is not None
    assert result.data["path"] == "empty"
    assert result.data["entry_count"] == 0

    with pytest.raises(ToolInputError, match="outside the project workspace"):
        await create_ls_tool(cwd=tmp_path).execute({"path": "../outside"})

    cancelled = await create_ls_tool(cwd=tmp_path).execute({}, signal=CancelledSignal())
    assert cancelled.ok is False
    assert cancelled.data == {"cancelled": True}


@pytest.mark.anyio
async def test_ls_reports_head_truncation(tmp_path: Path) -> None:
    for index in range(2_001):
        (tmp_path / f"file-{index:04d}.txt").write_text("x", encoding="utf-8")
    result = await create_ls_tool(cwd=tmp_path).execute({})
    assert result.ok is True
    assert result.data is not None
    truncation = result.data["truncation"]
    assert isinstance(truncation, dict)
    assert truncation["truncated"] is True
    assert len(result.content.splitlines()) == 2_000
