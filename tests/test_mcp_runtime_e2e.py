"""Offline MCP acceptance written before the runtime implementation.

Failure modes: implicit connection, stale tools after revocation/config edit,
credential echo in structured content, unpaired failures and orphan processes.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent import AgentHarness, AgentHarnessConfig
from forge_agent.events import ToolExecutionEndEvent, ToolExecutionStartEvent
from forge_coding.mcp.runtime import MCPError, MCPRuntime
from forge_coding.paths import ForgePaths


def assert_processes_closed(workspace: Path) -> None:
    pids = workspace / "pids"
    if not pids.exists():
        return
    for pid in map(int, pids.read_text().splitlines()):
        if sys.platform == "win32":
            import ctypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                try:
                    code = ctypes.c_ulong()
                    assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                    assert code.value != 259, f"Owned MCP process is still running: {pid}"
                finally:
                    kernel.CloseHandle(handle)
        else:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise AssertionError(f"Owned MCP process is still running: {pid}")


@pytest.mark.anyio
async def test_mcp_authorization_execution_and_close(tmp_path: Path, monkeypatch) -> None:
    paths = ForgePaths(home=tmp_path / "user")
    paths.home.mkdir()
    shutil.copy(Path(__file__).parent / "fixtures/tool_ecosystem/mcp_server.py", tmp_path)
    config = {
        "mcpServers": {
            "fixture": {
                "command": sys.executable,
                "args": ["mcp_server.py"],
                "env": {"FIXTURE_SECRET": {"env": "FORGE_FIXTURE_SECRET"}},
                "exposure": "direct",
                "toolExposure": {"hidden_large": "hidden"},
            }
        }
    }
    paths.mcp_config_path.write_text(json.dumps(config))
    monkeypatch.setenv("FORGE_FIXTURE_SECRET", "fixture-secret-value")
    runtime = MCPRuntime(tmp_path, paths=paths)
    assert not runtime.tools and not (tmp_path / "pids").exists()
    evidence = tmp_path / "mcp-evidence"
    evidence.mkdir()
    passed = False
    try:
        await runtime.enable("fixture")
        assert all("fixture-secret-value" not in t.description for t in runtime.tools)
        large = next(t for t in runtime.tools if t.name == "fixture_hidden_large")
        assert large.metadata["forge.exposure"] == "hidden"
        echo = next(t for t in runtime.tools if t.name == "fixture_echo")
        model = ScriptedChatModel(
            [
                tool_call_ai("echo", "fixture_echo", {"value": "fixture-secret-value"}),
                tool_call_ai("fail", "fixture_fail", {}),
                tool_call_ai("mixed", "fixture_mixed", {}),
                tool_call_ai("input", "fixture_request_input", {}),
                AIMessage(content="done"),
            ]
        )
        harness = AgentHarness(AgentHarnessConfig(provider=model, tools=runtime.tools))
        async with asyncio.timeout(30):
            events = [event async for event in harness.prompt("exercise MCP")]
        roots = [
            e for e in events if isinstance(e, (ToolExecutionStartEvent, ToolExecutionEndEvent))
        ]
        assert len(roots) == 8
        results = [m for m in harness.messages if isinstance(m, ToolMessage)]
        assert [m.status for m in results] == ["success", "error", "success", "error"]
        assert any(block.get("type") == "image" for block in results[2].content)
        assert "Saved binary content" in str(results[2].content)
        assert "fixture-secret-value" not in json.dumps(
            [m.model_dump(mode="json") for m in results]
        )
        assert results[0].artifact["forge.mcp.v1"]["structuredContent"] == {"value": "[redacted]"}
        await runtime.disable("fixture")
        with pytest.raises(MCPError):
            await echo.ainvoke({"value": "stale"})
        await runtime.enable("fixture")
        config["mcpServers"]["fixture"]["args"] = ["mcp_server.py", "changed"]
        paths.mcp_config_path.write_text(json.dumps(config))
        with pytest.raises(MCPError):
            await next(t for t in runtime.tools if t.name == "fixture_echo").ainvoke(
                {"value": "stale"}
            )
        assert not runtime.tools
        passed = True
        (evidence / "events.json").write_text(
            json.dumps([e.model_dump(mode="json") for e in events])
        )
    finally:
        await runtime.aclose()
        assert_processes_closed(tmp_path)
        (evidence / "summary.json").write_text(
            json.dumps({"passed": passed, "active": runtime.active_count})
        )
        print(f"MCP evidence: {evidence}")
    assert runtime.active_count == 0


@pytest.mark.anyio
async def test_mcp_cancel_drains_owned_connections(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / "user")
    paths.home.mkdir()
    shutil.copy(Path(__file__).parent / "fixtures/tool_ecosystem/mcp_server.py", tmp_path)
    paths.mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fixture": {
                        "command": sys.executable,
                        "args": ["mcp_server.py"],
                        "exposure": "direct",
                    }
                }
            }
        )
    )
    runtime = MCPRuntime(tmp_path, paths=paths)
    await runtime.enable("fixture")
    task = asyncio.create_task(
        next(t for t in runtime.tools if t.name == "fixture_wait").ainvoke({})
    )
    try:
        async with asyncio.timeout(20):
            while not (tmp_path / "waiting").exists():
                await asyncio.sleep(0.05)
        await runtime.aclose()
        assert task.done() and runtime.active_count == 0
        assert_processes_closed(tmp_path)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.aclose()


@pytest.mark.anyio
async def test_print_mcp_codemode_and_mounted_commands(tmp_path: Path, monkeypatch, capsys) -> None:
    from conftest import isolate_home
    from forge_agent.session import JsonlSessionStorage
    from forge_cli.cli import PrintOutputMode, run_print_mode
    from forge_cli.tui.app import ForgeTuiApp
    from forge_cli.tui.prompt import PromptInput
    from forge_coding.sessions.session import CodingSession, CodingSessionConfig

    isolate_home(monkeypatch, tmp_path)
    paths = ForgePaths()
    paths.home.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).parent / "fixtures/tool_ecosystem/mcp_server.py", tmp_path)
    paths.mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fixture": {
                        "command": sys.executable,
                        "args": ["mcp_server.py"],
                    }
                }
            }
        )
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    model = ScriptedChatModel(
        [
            tool_call_ai(
                "script",
                "codemode",
                {
                    "code": "text(await tools.fixture_echo({value:'combined'})); "
                    "store('combined',true);"
                },
            ),
            AIMessage(content="done"),
        ]
    )
    assert await run_print_mode(
        prompt="combine",
        model="fake",
        cwd=tmp_path,
        provider=model,
        output=PrintOutputMode.json,
        storage=storage,
        mcp_servers=("fixture",),
        codemode="only",
    )
    output = capsys.readouterr().out
    assert "combined" in output
    assert "fixture_echo" not in {t.name for t in model.calls[0]["tools"]}
    initial_processes = (tmp_path / "pids").read_text()
    restored = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel([]),
            model="fake",
            storage=storage,
            cwd=tmp_path,
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    try:
        assert restored.codemode_store == {"combined": True}
        assert not any(t.name == "fixture_echo" for t in restored.tools)
        assert (tmp_path / "pids").read_text() == initial_processes
        app = ForgeTuiApp(restored)
        async with app.run_test(size=(120, 32)) as pilot:
            app.query_one("#prompt", PromptInput).text = "/mcp enable fixture"
            await app.action_submit_prompt()
            async with asyncio.timeout(20):
                while not any(t.name == "fixture_echo" for t in restored.tools):
                    await pilot.pause(0.05)
            app.query_one("#prompt", PromptInput).text = "/mcp disable fixture"
            await app.action_submit_prompt()
            async with asyncio.timeout(10):
                while any(t.name == "fixture_echo" for t in restored.tools):
                    await pilot.pause(0.05)
            (tmp_path / "mounted-mcp.svg").write_text(app.export_screenshot(), encoding="utf-8")
        (tmp_path / "combined-evidence.json").write_text(
            json.dumps(
                {
                    "print_json": True,
                    "standard_mcp_result": True,
                    "restore_did_not_authorize": True,
                    "mounted_enable_disable": True,
                }
            )
        )
    finally:
        await restored.aclose()
        assert_processes_closed(tmp_path)
