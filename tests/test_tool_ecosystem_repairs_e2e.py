"""Review regressions, written before repairs.

Failure modes: stale child permissions, shared parent controls, false context
pressure, silent name collisions, partial refresh publication, revoked old
tools after discovery failure, and transport buffering before size checks.
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import httpx2
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from conftest import isolate_home
from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent.session import JsonlSessionStorage
from forge_coding.mcp import runtime as mcp
from forge_coding.paths import ForgePaths
from forge_coding.resources.base import ForgeResourcePaths
from forge_coding.resources.trust import TrustStore
from forge_coding.sessions.session import CodingSession, CodingSessionConfig
from forge_coding.tools import ToolDefinition, ToolSet


def config(root, model, **kwargs):
    paths = ForgePaths(home=root / "home")
    paths.home.mkdir(exist_ok=True)
    return CodingSessionConfig(
        provider=model,
        model="fake",
        cwd=root,
        storage=JsonlSessionStorage(root / "session.jsonl"),
        resource_paths=ForgeResourcePaths(root=paths.home, cwd=root, agents_root=None, paths=paths),
        trust_store=TrustStore(root / "trust.json"),
        auto_compact_enabled=False,
        **kwargs,
    )


def evidence(root, **values):
    (root / "repair-evidence.json").write_text(json.dumps(values, indent=2), encoding="utf-8")


@pytest.mark.anyio
async def test_child_catalog_follows_mcp_and_only_mode(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    shutil.copy(Path(__file__).parent / "fixtures/tool_ecosystem/mcp_server.py", tmp_path)
    monkeypatch.setenv("FORGE_FIXTURE_SECRET", "offline-fixture")
    opts = config(
        tmp_path,
        ScriptedChatModel(
            [
                tool_call_ai("read", "read", {"path": "note.txt"}),
                tool_call_ai("echo", "fixture_echo", {"value": "child"}),
                AIMessage(content="done"),
            ]
        ),
        codemode="only",
        mcp_servers=("fixture",),
    )
    opts.resource_paths.paths.mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fixture": {
                        "command": sys.executable,
                        "args": ["mcp_server.py"],
                        "env": {"FIXTURE_SECRET": {"env": "FORGE_FIXTURE_SECRET"}},
                    }
                }
            }
        )
    )
    (tmp_path / "note.txt").write_text("readable")
    session = await CodingSession.load(opts)
    try:
        worker = next(s for s in session._subagent_runner.specs if s.name == "worker")
        assert "fixture_echo" in {t.name for t in worker.tools}
        await session._subagent_runner.run("worker", "Read and echo")
        results = [
            m for c in opts.provider.calls for m in c["messages"] if isinstance(m, ToolMessage)
        ]
        assert results and all(m.status == "success" for m in results)
        assert any(m.name == "fixture_echo" for m in results)
        await session.apply_mcp_action("disable", "fixture")
        assert all(
            t.name != "fixture_echo" for s in session._subagent_runner.specs for t in s.tools
        )
        await session.apply_mcp_action("enable", "fixture")
        assert any(
            t.name == "fixture_echo" for s in session._subagent_runner.specs for t in s.tools
        )
        evidence(tmp_path, child_read_and_mcp=True, enable_disable_synced=True)
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_reload_does_not_share_parent_controls(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    tool = StructuredTool.from_function(
        func=lambda value: value, name="remote_echo", description="Echo"
    )
    opts = config(
        tmp_path,
        ScriptedChatModel(
            [
                tool_call_ai("search", "tool_search", {"names": ["remote_echo"]}),
                tool_call_ai("script", "codemode", {"code": "store('child',1);"}),
                AIMessage(content="done"),
            ]
        ),
        tools=ToolSet([ToolDefinition(tool, label="echo", exposure="deferred")]),
        codemode="on",
    )
    session = await CodingSession.load(opts)
    try:
        session.reload()
        assert {"tool_search", "codemode"} <= {t.name for t in session.tools}
        worker = next(s for s in session._subagent_runner.specs if s.name == "worker")
        assert {t.name for t in worker.tools} == {"remote_echo"}
        await session._subagent_runner.run("worker", "Try parent controls")
        assert session._tool_discovery.loaded == {} and session.codemode_store == {}
        evidence(
            tmp_path, child_catalog=[t.name for t in worker.tools], parent_state_unchanged=True
        )
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_context_counts_only_declared_tools_and_invalidates_cache(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    tool = StructuredTool.from_function(
        func=lambda value: value, name="remote_echo", description="x" * 100000
    )
    opts = config(
        tmp_path,
        ScriptedChatModel(
            [
                tool_call_ai("search", "tool_search", {"names": ["remote_echo"]}),
                AIMessage(content="done"),
            ]
        ),
        tools=ToolSet([ToolDefinition(tool, label="echo", exposure="deferred")]),
        enable_subagents=False,
    )
    session = await CodingSession.load(opts)
    try:
        before = session.context_usage.total_tokens
        assert before < 2000 and session.context_token_estimate < 2000
        _ = [e async for e in session.prompt("load")]
        after = session.context_usage.total_tokens
        assert after > before + 10000 and session.context_token_estimate > 10000
        evidence(tmp_path, before=before, after=after)
    finally:
        await session.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["tool_search", "codemode"])
async def test_reserved_feature_names_fail_explicitly(tmp_path, monkeypatch, name):
    isolate_home(monkeypatch, tmp_path)
    tool = StructuredTool.from_function(func=lambda: "custom", name=name, description="Caller tool")
    with pytest.raises(ValueError, match="reserved"):
        await CodingSession.load(
            config(tmp_path, ScriptedChatModel([]), tools=[tool], enable_subagents=False)
        )
    evidence(tmp_path, reserved_name=name, explicit_rejection=True)


@pytest.mark.anyio
async def test_mcp_publication_does_not_silently_remove_caller_tools(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    custom = StructuredTool.from_function(
        func=lambda: "caller", name="read_mcp_resource", description="Custom"
    )
    opts = config(tmp_path, ScriptedChatModel([]), tools=[custom], enable_subagents=False)
    session = await CodingSession.load(opts)
    try:
        await session.apply_mcp_action("list")
        assert "read_mcp_resource" in {t.name for t in session.tools}
        assert await session.tools[0].ainvoke({}) == "caller"
        runtime = session._mcp_runtime
        runtime._tools = {
            "tool": (
                StructuredTool.from_function(
                    func=lambda: "remote",
                    name="tool_search",
                    description="Remote collision",
                ),
            )
        }
        with pytest.raises(ValueError, match="collides"):
            await session.apply_mcp_action("list")
        assert "read_mcp_resource" in {t.name for t in session.tools}
        evidence(tmp_path, caller_preserved=True, remote_control_collision_rejected=True)
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_cancelled_refresh_publishes_runtime_catalog(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    opts = config(tmp_path, ScriptedChatModel([]), tools=[], enable_subagents=False)
    paths = opts.resource_paths.paths
    paths.mcp_config_path.write_text(
        json.dumps({"mcpServers": {n: {"command": sys.executable} for n in ("a", "b")}})
    )
    runtime = mcp.MCPRuntime(tmp_path, paths=paths)

    def tool(name):
        return StructuredTool.from_function(func=lambda value: value, name=name, description="Echo")

    runtime._authorized = {n: s.fingerprint for n, s in runtime.servers.items()}
    runtime._tools = {n: (tool(n + "_old"),) for n in ("a", "b")}
    entered = asyncio.Event()

    async def enable(name):
        if name == "a":
            runtime._revoke(name)
            runtime._authorized[name] = runtime.servers[name].fingerprint
            runtime._tools[name] = (tool("a_new"),)
        else:
            current = asyncio.current_task()
            runtime._tasks[current] = name
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                runtime._tasks.pop(current, None)

    monkeypatch.setattr(runtime, "enable", enable)
    session = await CodingSession.load(opts)
    session._mcp_runtime = runtime
    await session.apply_mcp_action("list")
    try:
        task = asyncio.create_task(session.apply_mcp_action("reload"))
        await entered.wait()
        session.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        actual = {t.name for t in session.tools if t.name.startswith(("a_", "b_"))}
        assert actual == {t.name for t in runtime.tools} == {"a_new", "b_old"}
        assert runtime.active_count == 0
        evidence(tmp_path, published=sorted(actual), active=0)
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_failed_discovery_keeps_old_tools_callable(tmp_path, monkeypatch):
    paths = ForgePaths(home=tmp_path / "home")
    paths.home.mkdir()
    paths.mcp_config_path.write_text(json.dumps({"mcpServers": {"a": {"command": sys.executable}}}))
    runtime = mcp.MCPRuntime(tmp_path, paths=paths)
    old = StructuredTool.from_function(func=lambda: "retained", name="a_old", description="Echo")
    runtime._tools = {"a": (old,)}
    runtime._authorized = {"a": runtime.servers["a"].fingerprint}

    class Client:
        server_info = None

        async def __aenter__(self):
            raise ConnectionError("offline fixture")

        async def __aexit__(self, *args):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(runtime, "_client", lambda *a, **k: (Client(), None))
    try:
        await runtime.reload(None)
        assert (old.metadata or {}).get("forge.available") is not False
        assert await old.ainvoke({}) == "retained"
        evidence(tmp_path, old_available=True, active=runtime.active_count)
    finally:
        await runtime.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["json", "sse", "compressed"])
async def test_http_read_budget_precedes_full_buffering(tmp_path, kind):
    consumed = 0
    closed = False

    class Stream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            nonlocal consumed
            for _ in range(1024):
                consumed += 65536
                yield b"x" * 65536

        async def aclose(self):
            nonlocal closed
            closed = True

    def handler(request):
        if request.url.path == "/small":
            return httpx2.Response(200, json={"ok": True})
        return httpx2.Response(
            200,
            headers={
                "content-type": "text/event-stream" if kind == "sse" else "application/json",
                **({"content-encoding": "gzip"} if kind == "compressed" else {}),
            },
            stream=Stream(),
        )

    async with mcp._http_client(transport=httpx2.MockTransport(handler)) as client:
        assert (await client.get("http://fixture/small")).json() == {"ok": True}
        with pytest.raises(mcp.MCPError, match="budget|encoding"):
            await client.get("http://fixture/large")
    assert closed and consumed <= 8 * 1024 * 1024 + 65536
    evidence(tmp_path, kind=kind, consumed=consumed, closed=closed)


@pytest.mark.anyio
async def test_discovery_budget_spans_pages(tmp_path, monkeypatch):
    from mcp.types import ListToolsResult, Tool

    paths = ForgePaths(home=tmp_path / "home")
    paths.home.mkdir()
    paths.mcp_config_path.write_text(json.dumps({"mcpServers": {"a": {"command": sys.executable}}}))
    runtime = mcp.MCPRuntime(tmp_path, paths=paths)
    pages = 0

    class Client:
        server_info = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def close(self):
            pass

        def redact(self, value):
            return value

        async def list_tools_mcp(self, **kwargs):
            nonlocal pages
            pages += 1
            return ListToolsResult(
                tools=[
                    Tool(
                        name=f"echo{pages}",
                        description="x" * (5 * 1024 * 1024),
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
                next_cursor="next" if pages == 1 else None,
            )

    monkeypatch.setattr(runtime, "_client", lambda *a, **k: (Client(), None))
    try:
        with pytest.raises(mcp.MCPError, match="catalog byte budget"):
            await runtime.enable("a")
        assert not runtime.tools and runtime.active_count == 0
        evidence(tmp_path, pages=pages, catalog_rejected=True)
    finally:
        await runtime.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
@pytest.mark.parametrize(
    "media", ["text/event-stream", "Text/Event-Stream", "Text/Event-Stream ; charset=utf-8"]
)
async def test_sse_budget_resets_at_event_boundaries(tmp_path, ending, media):
    class Stream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(10):
                # Split CRLF and empty delimiters across transport chunks.
                yield b"data:" + b"x" * (1024 * 1024)
                for value in ending + ending:
                    yield bytes([value])

    def handler(request):
        return httpx2.Response(200, headers={"content-type": media}, stream=Stream())

    async with (
        mcp._http_client(transport=httpx2.MockTransport(handler)) as client,
        client.stream("GET", "http://fixture/events") as response,
    ):
        count = sum([len(chunk) async for chunk in response.aiter_bytes()])
    assert count > 8 * 1024 * 1024
    evidence(tmp_path, events=10, bytes=count, independent_budgets=True)


@pytest.mark.anyio
async def test_stdio_rejects_large_line_before_json_parsing(tmp_path):
    from test_mcp_runtime_e2e import assert_processes_closed

    script = tmp_path / "large_server.py"
    script.write_text("""import os
from pathlib import Path
from fastmcp import FastMCP
Path('pids').write_text(str(os.getpid()))
server = FastMCP('large-fixture')
@server.tool(description='x' * (12 * 1024 * 1024))
def echo(value: str) -> str:
    return value
server.run(transport='stdio', show_banner=False)
""")
    paths = ForgePaths(home=tmp_path / "home")
    paths.home.mkdir()
    paths.mcp_config_path.write_text(
        json.dumps({"mcpServers": {"large": {"command": sys.executable, "args": [str(script)]}}})
    )
    runtime = mcp.MCPRuntime(tmp_path, paths=paths)
    try:
        with pytest.raises(mcp.MCPError):
            await runtime.enable("large")
        assert "stdio message byte budget exceeded" in runtime.describe("logs", "large")
        assert runtime.active_count == 0 and not runtime.tools
    finally:
        await runtime.aclose()
    assert_processes_closed(tmp_path)
    evidence(tmp_path, before_json_parsing=True, process_closed=True)


@pytest.mark.anyio
async def test_catalog_budget_covers_servers_and_resources(tmp_path, monkeypatch):
    from mcp.types import ListResourcesResult, ListToolsResult, Resource, Tool

    monkeypatch.setattr(mcp, "MAX_MCP_CATALOG_BYTES", 65536)
    paths = ForgePaths(home=tmp_path / "home")
    paths.home.mkdir()
    paths.mcp_config_path.write_text(
        json.dumps({"mcpServers": {n: {"command": sys.executable} for n in ("a", "b")}})
    )
    runtime = mcp.MCPRuntime(tmp_path, paths=paths)

    class Client:
        server_info = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def close(self):
            pass

        def redact(self, value):
            return value

        async def list_tools_mcp(self, **kwargs):
            return ListToolsResult(
                tools=[
                    Tool(
                        name="echo",
                        description="x" * 40000,
                        input_schema={"type": "object", "properties": {}},
                    )
                ]
            )

        async def list_resources_mcp(self, cursor=None, **kwargs):
            return ListResourcesResult(
                resources=[Resource(uri="fixture://note", name="note", description="x" * 40000)],
                next_cursor="next" if cursor is None else None,
            )

    monkeypatch.setattr(runtime, "_client", lambda *a, **k: (Client(), None))
    try:
        await runtime.enable("a")
        with pytest.raises(mcp.MCPError, match="catalog byte budget"):
            await runtime.enable("b")
        assert {t.name for t in runtime.tools} == {"a_echo"}
        with pytest.raises(mcp.MCPError, match="catalog byte budget"):
            await runtime._resource_operation("a", "resources")
        evidence(tmp_path, servers_bounded=True, resources_bounded=True)
    finally:
        await runtime.aclose()


@pytest.mark.anyio
async def test_child_copies_still_obey_runtime_unavailability(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    opts = config(
        tmp_path,
        ScriptedChatModel(
            [
                tool_call_ai("first", "a_echo", {}),
                tool_call_ai("again", "a_echo", {}),
                AIMessage(content="done"),
            ]
        ),
        tools=[],
    )
    paths = opts.resource_paths.paths
    paths.mcp_config_path.write_text(json.dumps({"mcpServers": {"a": {"command": sys.executable}}}))
    runtime = mcp.MCPRuntime(tmp_path, paths=paths)
    server = runtime.servers["a"]
    connections = 0

    class Client:
        async def close(self):
            pass

    def client(*args, **kwargs):
        nonlocal connections
        connections += 1
        return Client(), None

    monkeypatch.setattr(runtime, "_client", client)

    async def fail(client):
        raise ConnectionError("fixture execution connection failed")

    async def invoke():
        return await runtime._operate(server, fail)

    runtime._tools = {
        "a": (
            StructuredTool.from_function(
                coroutine=invoke,
                name="a_echo",
                description="Offline MCP failure",
            ),
        )
    }
    runtime._authorized = {"a": server.fingerprint}
    session = await CodingSession.load(opts)
    session._mcp_runtime = runtime
    try:
        await session.apply_mcp_action("list")
        await session._subagent_runner.run("worker", "Try the unavailable tool twice")
        assert connections == 1
        assert (runtime.tools[0].metadata or {}).get("forge.available") is False
        evidence(tmp_path, connections=connections, child_retry_rejected=True)
    finally:
        await session.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_conflicting_refresh_revokes_and_publishes_consistently(
    tmp_path, monkeypatch, cancel
):
    from mcp.types import ListToolsResult, Tool

    isolate_home(monkeypatch, tmp_path)
    opts = config(tmp_path, ScriptedChatModel([]), tools=[], codemode="on")
    paths = opts.resource_paths.paths
    paths.mcp_config_path.write_text(
        json.dumps({"mcpServers": {name: {"command": sys.executable} for name in ("a", "b")}})
    )
    runtime = mcp.MCPRuntime(tmp_path, paths=paths)
    names = ["old"]

    class Client:
        server_info = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def close(self):
            pass

        def redact(self, value):
            return value

        async def list_tools_mcp(self, **kwargs):
            return ListToolsResult(
                tools=[
                    Tool(name=n, input_schema={"type": "object", "properties": {}}) for n in names
                ]
            )

    monkeypatch.setattr(runtime, "_client", lambda *args, **kwargs: (Client(), None))
    session = await CodingSession.load(opts)
    session._mcp_runtime = runtime
    try:
        await session.apply_mcp_action("enable", "a")
        await session.apply_mcp_action("enable", "b")
        old = tuple(runtime.tools)
        names[:] = ["foo-bar", "foo_bar"]
        if cancel:
            enable = runtime.enable
            entered = asyncio.Event()

            async def partial(name):
                if name == "a":
                    await enable(name)
                else:
                    task = asyncio.current_task()
                    runtime._tasks[task] = name
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        runtime._tasks.pop(task, None)

            monkeypatch.setattr(runtime, "enable", partial)
            action = asyncio.create_task(session.apply_mcp_action("reload"))
            await entered.wait()
            session.cancel()
            with pytest.raises(asyncio.CancelledError):
                await action
        else:
            with pytest.raises(ValueError, match="collides"):
                await session.apply_mcp_action("reload")
        assert not runtime.tools and not runtime._authorized
        assert not {t.name for t in session.tools if t.name.startswith(("a_", "b_"))}
        assert all((t.metadata or {}).get("forge.available") is False for t in old)
        await session.apply_mcp_action("list")
        assert not runtime.tools and runtime.active_count == 0
        assert "codemode" in {t.name for t in session.tools}
        assert all(
            not t.name.startswith(("a_", "b_"))
            for s in session._subagent_runner.specs
            for t in s.tools
        )
        evidence(tmp_path, cancelled=cancel, rejected_catalog_revoked=True, directories_equal=True)
    finally:
        await session.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "action", ["resources", "templates", "read_resources", "read_templates", "read_content"]
)
async def test_resource_read_failures_preserve_native_tools(tmp_path, monkeypatch, action):
    from mcp.types import ListResourcesResult, Resource

    paths = ForgePaths(home=tmp_path / "home")
    paths.home.mkdir()
    paths.mcp_config_path.write_text(json.dumps({"mcpServers": {"a": {"command": sys.executable}}}))
    runtime = mcp.MCPRuntime(tmp_path, paths=paths)
    healthy = StructuredTool.from_function(
        func=lambda: "healthy", name="a_echo", description="Echo"
    )
    runtime._tools = {"a": (healthy,)}
    runtime._authorized = {"a": runtime.servers["a"].fingerprint}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def close(self):
            pass

        async def list_resources_mcp(self, **kwargs):
            if action == "read_templates":
                return ListResourcesResult(resources=[Resource(uri="fixture://note", name="note")])
            raise ConnectionError("offline discovery")

        async def list_resource_templates_mcp(self, **kwargs):
            raise ConnectionError("offline template discovery")

        async def read_resource(self, uri):
            raise ConnectionError("offline resource read")

    monkeypatch.setattr(runtime, "_client", lambda *a, **k: (Client(), None))
    if action == "read_content":
        runtime._resource_uris["a"] = {"fixture://note"}
        runtime._resource_templates["a"] = ()
    try:
        with pytest.raises(mcp.MCPError):
            await runtime._resource_operation(
                "a", "read" if action.startswith("read") else action, "fixture://note"
            )
        assert (healthy.metadata or {}).get("forge.available") is not False

        async def normal(client):
            return "healthy"

        assert await runtime._operate(runtime.servers["a"], normal) == "healthy"
        evidence(tmp_path, action=action, native_tools_available=True)
    finally:
        await runtime.aclose()


@pytest.mark.anyio
async def test_split_crlf_counts_all_wire_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "MAX_MCP_MESSAGE_BYTES", 508)

    class Stream(httpx2.AsyncByteStream):
        def __init__(self, count):
            self.count = count

        async def __aiter__(self):
            yield b"data:" + b"x" * self.count
            for c in b"\r\n\r\n":
                yield bytes([c])

    def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Stream(499 if request.url.path == "/small" else 500),
        )

    async with mcp._http_client(transport=httpx2.MockTransport(handler)) as client:
        assert len((await client.get("http://fixture/small")).content) == 508
        with pytest.raises(mcp.MCPError, match="budget"):
            await client.get("http://fixture/large")
    evidence(tmp_path, exact_budget_accepted=True, one_byte_over_rejected=True)
