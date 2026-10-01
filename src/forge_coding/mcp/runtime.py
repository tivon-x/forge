"""Explicitly authorized SDK clients, with owned operations and bounded results."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from forge_coding.mcp.config import MCPConfigError, MCPServer, load_mcp_servers
from forge_coding.paths import ForgePaths
from forge_coding.resources.trust import TrustResult, project_path_is_safe
from forge_coding.tools.output import bounded_text, save_output

T = TypeVar("T")
MAX_MCP_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_MCP_CATALOG_BYTES = 8 * 1024 * 1024


class MCPError(RuntimeError):
    """A redacted MCP operation failure."""


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
            if secret.lower().startswith("bearer "):
                value = value.replace(secret[7:], "[redacted]")
        return value
    if isinstance(value, dict):
        return {_redact(k, secrets): _redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secrets) for v in value]
    return value


def _http_client(
    headers: dict[str, str] | None = None,
    timeout: Any = None,
    auth: Any = None,
    **kwargs: Any,
) -> Any:
    import httpx2

    class BoundedStream(httpx2.AsyncByteStream):
        def __init__(self, source: Any, *, sse: bool) -> None:
            self.source = source
            self.sse = sse

        async def __aiter__(self) -> Any:
            size = 0
            line_size = 0
            pending_cr = False
            async for raw_chunk in self.source:
                chunk = raw_chunk
                if not self.sse:
                    size += len(chunk)
                    if size > MAX_MCP_MESSAGE_BYTES:
                        raise MCPError("MCP response byte budget exceeded")
                else:
                    if pending_cr:
                        chunk = b"\r" + chunk
                    pending_cr = chunk.endswith(b"\r")
                    if pending_cr:
                        chunk = chunk[:-1]
                    start = 0
                    # Track SSE event/line lengths without retaining their bytes.
                    for match in re.finditer(rb"\r\n|\r|\n", chunk):
                        part_size = match.start() - start
                        size += part_size + len(match.group())
                        line_size += part_size
                        if size > MAX_MCP_MESSAGE_BYTES:
                            raise MCPError("MCP SSE message byte budget exceeded")
                        if line_size == 0:
                            size = 0
                        line_size = 0
                        start = match.end()
                    tail_size = len(chunk) - start
                    size += tail_size
                    line_size += tail_size
                    if size + int(pending_cr) > MAX_MCP_MESSAGE_BYTES:
                        raise MCPError("MCP SSE message byte budget exceeded")
                yield raw_chunk

        async def aclose(self) -> None:
            await self.source.aclose()

    async def bound_response(response: Any) -> None:
        # Reject unsolicited compression before the decoder can inflate its body.
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            await response.aclose()
            raise MCPError("Unsupported MCP response encoding")
        media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        sse = media_type == "text/event-stream"
        length = response.headers.get("content-length")
        if (
            not sse
            and length is not None
            and length.isdigit()
            and int(length) > MAX_MCP_MESSAGE_BYTES
        ):
            await response.aclose()
            raise MCPError("MCP response byte budget exceeded")
        response.stream = BoundedStream(response.stream, sse=sse)

    # SDK defaults to redirects; custom authentication headers must never follow them.
    kwargs["follow_redirects"] = False
    kwargs["trust_env"] = False
    hooks = dict(kwargs.pop("event_hooks", {}))
    hooks["response"] = [bound_response, *hooks.get("response", ())]
    return httpx2.AsyncClient(
        headers={**(headers or {}), "Accept-Encoding": "identity"},
        timeout=timeout,
        auth=auth,
        event_hooks=hooks,
        **kwargs,
    )


async def _reject_elicitation(*args: Any, **kwargs: Any) -> Any:
    from mcp.types import ElicitResult

    return ElicitResult(action="decline")


def _private_directory(path: Path, root: Path) -> None:
    if not project_path_is_safe(path, project_root=root):
        raise MCPError("Unsafe MCP authentication directory")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if sys.platform == "win32":
        import ntsecuritycon  # type: ignore[import-untyped]
        import win32api  # type: ignore[import-untyped]
        import win32con  # type: ignore[import-untyped]
        import win32security  # type: ignore[import-untyped]

        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        try:
            sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
            acl = win32security.ACL()
            acl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION,
                win32con.OBJECT_INHERIT_ACE | win32con.CONTAINER_INHERIT_ACE,
                ntsecuritycon.FILE_ALL_ACCESS,
                sid,
            )
            win32security.SetNamedSecurityInfo(
                str(path),
                win32security.SE_FILE_OBJECT,
                win32security.DACL_SECURITY_INFORMATION
                | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
                None,
                None,
                acl,
                None,
            )
        finally:
            token.Close()
    else:
        path.chmod(0o700)


class MCPRuntime:
    def __init__(
        self, cwd: Path, *, paths: ForgePaths | None = None, trust: TrustResult | None = None
    ) -> None:
        self.cwd = cwd
        self.paths = paths or ForgePaths()
        self.trust = trust
        self.servers = load_mcp_servers(cwd, paths=self.paths, trust=trust)
        self._authorized: dict[str, str] = {}
        self._tools: dict[str, tuple[BaseTool, ...]] = {}
        self._tasks: dict[asyncio.Task[Any], str] = {}
        self._clients: set[Any] = set()
        self._catalog_bytes: dict[str, int] = {}
        self._logs: dict[str, deque[str]] = {}
        self._closed = False
        self._resource_tools: tuple[BaseTool, ...] | None = None
        self._resource_uris: dict[str, set[str]] = {}
        self._resource_templates: dict[str, tuple[str, ...]] = {}
        self._cleanup_failed = False

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        return tuple(tool for group in self._tools.values() for tool in group)

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    def cancel(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()

    def _log(self, name: str, message: str) -> None:
        self._logs.setdefault(name, deque(maxlen=200)).append(message)

    def _revoke(self, name: str) -> None:
        self._authorized.pop(name, None)
        self._catalog_bytes.pop(name, None)
        self._resource_uris.pop(name, None)
        self._resource_templates.pop(name, None)
        for tool in self._tools.pop(name, ()):
            tool.metadata = {**(tool.metadata or {}), "forge.available": False}

    def reject_catalog(self) -> None:
        """Fail closed when the owning session cannot accept the composed catalog."""
        for name in tuple(self._authorized.keys() | self._tools.keys()):
            self._revoke(name)
            self._log(
                name, "Catalog rejected; authorization revoked; enable again after correction"
            )

    async def _read_once_retry(self, callback: Callable[[], Awaitable[T]]) -> T:
        import httpx2

        try:
            return await callback()
        except (httpx2.TransportError, ConnectionError):
            return await callback()

    def _check(self, server: MCPServer, *, authorized: bool = True) -> None:
        if self._closed:
            raise MCPError("MCP runtime is closed")
        try:
            current = load_mcp_servers(self.cwd, paths=self.paths, trust=self.trust).get(
                server.name
            )
        except MCPConfigError:
            self._revoke(server.name)
            raise MCPError("MCP configuration invalid; authorization revoked") from None
        if current is None or current.fingerprint != server.fingerprint:
            self._revoke(server.name)
            self._log(server.name, "Configuration changed; authorization revoked")
            raise MCPError("MCP configuration changed; enable the server again")
        if not server.config.enabled:
            raise MCPError("MCP server disabled by configuration")
        if authorized and self._authorized.get(server.name) != server.fingerprint:
            raise MCPError("MCP server is not authorized in this session")
        if authorized and any(
            (tool.metadata or {}).get("forge.available") is False
            for tool in self._tools.get(server.name, ())
        ):
            raise MCPError("MCP server tools are unavailable; enable the server again")

    def _client(
        self, server: MCPServer, *, auth: Any = None, auth_secrets: tuple[str, ...] = ()
    ) -> Any:
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        refs = (*server.config.env.values(), *server.config.headers.values())
        values = {reference.env: os.environ.get(reference.env) for reference in refs}
        if any(value is None or value == "" for value in values.values()):
            raise MCPError("MCP environment reference is missing")
        secrets = tuple(
            sorted({v for v in values.values() if v} | set(auth_secrets), key=len, reverse=True)
        )
        transport: Any
        if server.config.command is not None:
            environment = {
                key: os.environ[key]
                for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE")
                if key in os.environ
            }
            environment.update({k: cast(str, values[v.env]) for k, v in server.config.env.items()})
            from forge_coding.mcp.stdio import BoundedStdioTransport

            sink = None
            transport = BoundedStdioTransport(
                server.config.command,
                server.config.args,
                env=environment,
                cwd=str(server.workspace / (server.config.cwd or ".")),
                max_bytes=MAX_MCP_MESSAGE_BYTES,
                on_limit=lambda: self._log(server.name, "MCP stdio message byte budget exceeded"),
            )
        else:
            sink = None
            transport = StreamableHttpTransport(
                cast(str, server.config.url),
                headers={k: cast(str, values[v.env]) for k, v in server.config.headers.items()},
                auth=auth,
                httpx_client_factory=_http_client,
            )

        workspace = self.cwd

        class SafeClient(Client[Any]):
            raw_result: dict[str, Any] | None = None

            def redact(self, value: Any) -> Any:
                current = getattr(getattr(auth, "context", None), "current_tokens", None)
                refreshed = tuple(
                    candidate
                    for key in ("access_token", "refresh_token")
                    if isinstance(candidate := getattr(current, key, None), str) and candidate
                )
                return _redact(value, secrets + refreshed)

            async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
                result = await super().call_tool(*args, **kwargs)
                result.content = [
                    type(block).model_validate(self.redact(block.model_dump(mode="json")))
                    for block in result.content
                ]
                result.structured_content = self.redact(result.structured_content)
                result.meta = self.redact(result.meta)
                self.raw_result = {
                    "content": [
                        block.model_dump(mode="json", by_alias=True) for block in result.content
                    ],
                    "structuredContent": result.structured_content,
                    "isError": result.is_error,
                }
                from mcp.types import TextContent

                converted: list[Any] = []
                for block in result.content:
                    resource = getattr(block, "resource", None)
                    binary = block.type == "audio" or (
                        resource is not None
                        and hasattr(resource, "blob")
                        and not str(getattr(resource, "mime_type", "")).startswith("image/")
                    )
                    if binary:
                        path = await save_output(
                            workspace,
                            json.dumps(block.model_dump(mode="json", by_alias=True)),
                            prefix="mcp-binary",
                        )
                        converted.append(
                            TextContent(type="text", text=f"Saved binary content: {path}")
                        )
                    else:
                        converted.append(block)
                result.content = converted
                return result

        client = SafeClient(
            transport,
            elicitation_handler=_reject_elicitation,
            timeout=server.config.timeoutSeconds,
            init_timeout=10,
        )
        return client, sink

    async def _operate(
        self,
        server: MCPServer,
        callback: Callable[[Any], Awaitable[T]],
        *,
        authorized: bool = True,
        timeout: float | None = None,
        auth: Any = None,
        discovery: bool = False,
    ) -> T:
        self._check(server, authorized=authorized)

        async def run() -> T:
            active_auth = auth
            auth_secrets: tuple[str, ...] = ()
            if server.config.url is not None and active_auth is None:
                try:
                    active_auth, auth_secrets = await asyncio.wait_for(
                        self._stored_auth(server), 10
                    )
                except Exception:
                    raise MCPError("MCP authentication cache could not be loaded") from None
            client, sink = self._client(server, auth=active_auth, auth_secrets=auth_secrets)
            self._clients.add(client)
            try:
                async with asyncio.timeout(timeout or server.config.timeoutSeconds):
                    return await callback(client)
            except (asyncio.CancelledError, MCPError):
                raise
            except Exception as exc:
                if not discovery:
                    for tool in self._tools.get(server.name, ()):
                        tool.metadata = {**(tool.metadata or {}), "forge.available": False}
                message = (
                    "MCP discovery failed or timed out"
                    if discovery
                    else "MCP operation failed or timed out; remote effects may be unknown"
                )
                self._log(server.name, f"{message} ({type(exc).__name__})")
                raise MCPError(message) from None
            finally:

                async def close_client() -> None:
                    try:
                        await client.close()
                    except Exception:
                        # Failed SDK connect is re-raised by close; also close the public transport.
                        await client.transport.close()

                closing = asyncio.create_task(close_client())
                try:
                    await asyncio.wait_for(asyncio.shield(closing), 5)
                except asyncio.CancelledError:
                    await asyncio.wait_for(closing, 5)
                    raise
                except TimeoutError:
                    self._closed = True
                    self._cleanup_failed = True
                    closing.cancel()
                    raise MCPError("MCP cleanup failed; runtime closed") from None
                finally:
                    self._clients.discard(client)
                    if sink is not None:
                        sink.close()

        task = asyncio.create_task(run())
        self._tasks[task] = server.name
        try:
            return await task
        finally:
            self._tasks.pop(task, None)

    async def enable(self, name: str) -> None:
        latest = load_mcp_servers(self.cwd, paths=self.paths, trust=self.trust)
        server = latest.get(name)
        if server is None:
            raise MCPError("Unknown MCP server")

        async def discover(client: Any) -> tuple[tuple[BaseTool, ...], int]:
            from langchain.mcp import as_langchain_tool

            remote_tools: list[Any] = []
            catalog_bytes = 0
            cursors: set[str] = set()
            cursor = None
            async with client:
                for _ in range(250):

                    async def fetch_page(cursor: str | None = cursor) -> Any:
                        return await client.list_tools_mcp(cursor=cursor, cache_mode="refresh")

                    page = await self._read_once_retry(fetch_page)
                    catalog_bytes += len(page.model_dump_json().encode())
                    if catalog_bytes > MAX_MCP_CATALOG_BYTES:
                        raise MCPError("MCP catalog byte budget exceeded")
                    remote_tools.extend(page.tools)
                    if len(remote_tools) > 2000:
                        raise MCPError("MCP tool count exceeded")
                    cursor = page.next_cursor
                    if cursor is None:
                        break
                    if cursor in cursors:
                        raise MCPError("Repeated MCP discovery cursor")
                    cursors.add(cursor)
                else:
                    raise MCPError("MCP discovery page limit exceeded")
            tools: list[BaseTool] = []
            for remote in remote_tools:
                remote = type(remote).model_validate(client.redact(remote.model_dump(mode="json")))
                if len(json.dumps(remote.input_schema).encode()) > 1024 * 1024:
                    raise MCPError("MCP tool schema exceeded")
                native = await as_langchain_tool(remote, client)
                native.name = f"{name}_{remote.name}"
                if len(native.name) > 128 or any(t.name == native.name for t in tools):
                    raise MCPError("Invalid or duplicate MCP tool name")

                def make_invoke(selected_remote: Any) -> Any:
                    async def invoke(**arguments: Any) -> tuple[Any, Any]:
                        async def call(owned_client: Any) -> tuple[Any, Any]:
                            adapted = await as_langchain_tool(selected_remote, owned_client)
                            result = await adapted.ainvoke(
                                {
                                    "type": "tool_call",
                                    "id": "mcp",
                                    "name": adapted.name,
                                    "args": arguments,
                                },
                                config={"callbacks": []},
                            )
                            if not isinstance(result, ToolMessage):
                                raise MCPError("Invalid native MCP result")
                            raw = owned_client.raw_result or {}
                            original = result.artifact
                            if hasattr(original, "model_dump"):
                                original = original.model_dump(mode="json")
                            artifact = {
                                "forge.mcp.v1": raw,
                                "forge.mcp.artifact": original,
                                "forge.status": result.status,
                            }
                            serialized = json.dumps(artifact, ensure_ascii=False)
                            if len(serialized.encode()) > 1024 * 1024:
                                path = await save_output(self.cwd, serialized, prefix="mcp")
                                artifact = {
                                    "forge.mcp.v1": {
                                        "isError": raw.get("isError", False),
                                        "truncated": True,
                                        "full_output_path": path,
                                    },
                                    "forge.status": result.status,
                                }
                            content = result.content
                            if len(json.dumps(content, ensure_ascii=False).encode()) > 20480:
                                path = await save_output(
                                    self.cwd, json.dumps(content, ensure_ascii=False), prefix="mcp"
                                )
                                blocks = (
                                    content
                                    if isinstance(content, list)
                                    else [{"type": "text", "text": content}]
                                )
                                bounded: list[Any] = []
                                remaining = 19000
                                # Keep images and binary paths before spending the text budget.
                                for block in sorted(
                                    blocks,
                                    key=lambda b: (
                                        isinstance(b, dict)
                                        and b.get("type") == "text"
                                        and not str(b.get("text", "")).startswith("Saved binary")
                                    ),
                                ):
                                    size = len(json.dumps(block, ensure_ascii=False).encode())
                                    if size <= remaining:
                                        bounded.append(block)
                                        remaining -= size
                                    elif (
                                        isinstance(block, dict)
                                        and block.get("type") == "text"
                                        and remaining > 256
                                    ):
                                        text = bounded_text(
                                            str(block.get("text", "")), max(1, remaining // 2 - 128)
                                        )
                                        bounded.append({"type": "text", "text": text})
                                        remaining = 0
                                bounded.append({"type": "text", "text": f"Full output: {path}"})
                                content = bounded
                            return content, artifact

                        return await self._operate(server, call)

                    return invoke

                # Retain the official native schema; only execution ownership changes.
                if not isinstance(native, StructuredTool):
                    raise MCPError("Unsupported native MCP tool")
                native.coroutine = make_invoke(remote)
                native.handle_tool_error = True
                exposure = server.config.toolExposure.get(remote.name, server.config.exposure)
                if len(json.dumps(remote.input_schema).encode()) > 16384 and exposure in {
                    "direct",
                    "deferred",
                }:
                    exposure = "codemode"

                native.metadata = {
                    **client.redact(native.metadata or {}),
                    "forge.namespace": name,
                    "forge.mcp.server": name,
                    "forge.mcp.remote_name": remote.name,
                    "forge.exposure": exposure,
                }
                tools.append(native)
            return tuple(tools), catalog_bytes

        tools, catalog_bytes = await self._operate(
            server, discover, authorized=False, timeout=10, discovery=True
        )
        self._check(server, authorized=False)
        if (
            catalog_bytes + sum(v for n, v in self._catalog_bytes.items() if n != name)
            > MAX_MCP_CATALOG_BYTES
        ):
            raise MCPError("MCP catalog byte budget exceeded")
        other_names = {tool.name for n, group in self._tools.items() if n != name for tool in group}
        if len(other_names) + len(tools) > 2000:
            raise MCPError("Session tool count exceeded")
        if other_names.intersection(tool.name for tool in tools):
            raise MCPError("MCP tool name collision")
        self.servers = latest
        self._revoke(name)
        self._authorized[name] = server.fingerprint
        self._tools[name] = tools
        self._catalog_bytes[name] = catalog_bytes
        self._log(name, f"Authorized; discovered {len(tools)} tools")

    async def disable(self, name: str) -> None:
        self._revoke(name)
        tasks = [task for task, owner in self._tasks.items() if owner == name]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        self._log(name, "Authorization revoked")

    async def reload(self, trust: TrustResult | None) -> None:
        if self._tasks:
            raise MCPError("Cannot reload MCP while operations are active")
        latest = load_mcp_servers(self.cwd, paths=self.paths, trust=trust)
        self.trust = trust
        for name, fingerprint in tuple(self._authorized.items()):
            server = latest.get(name)
            if server is None or server.fingerprint != fingerprint:
                await self.disable(name)
        self.servers = latest
        for name in tuple(self._authorized):
            try:
                await self.enable(name)
            except MCPError:
                self._log(name, "Discovery refresh failed; previous directory retained")

    def _auth_directory(self, server: MCPServer) -> Path:
        return self.paths.home / "mcp-auth" / server.fingerprint[:20]

    def _oauth(self, server: MCPServer, *, interactive: bool) -> Any:
        from fastmcp.client.auth import OAuth
        from key_value.aio.stores.filetree import (
            FileTreeStore,
            FileTreeV1CollectionSanitizationStrategy,
            FileTreeV1KeySanitizationStrategy,
        )

        if server.config.url is None:
            raise MCPError("OAuth requires an HTTP MCP server")
        directory = self._auth_directory(server)
        _private_directory(directory, self.paths.home)

        class ExplicitOAuth(OAuth):
            async def redirect_handler(self, authorization_url: str) -> None:
                if not interactive:
                    raise MCPError("MCP login required; use /mcp login explicitly")
                await super().redirect_handler(authorization_url)

        return ExplicitOAuth(
            server.config.url,
            client_name="Forge MCP",
            callback_host="127.0.0.1",
            callback_timeout=300,
            token_storage=FileTreeStore(
                data_directory=directory,
                key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(directory),
                collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(
                    directory
                ),
            ),
            httpx_client_factory=_http_client,
        )

    async def _stored_auth(self, server: MCPServer) -> tuple[Any, tuple[str, ...]]:
        if not self._auth_directory(server).exists():
            return None, ()
        auth = self._oauth(server, interactive=False)
        tokens = await auth.token_storage_adapter.get_tokens()
        if tokens is None:
            return None, ()
        values = [tokens.access_token, tokens.refresh_token]
        info = await auth.token_storage_adapter.get_client_info()
        if info is not None:
            values.append(info.client_secret)
        return auth, tuple(value for value in values if value)

    async def login(self, name: str) -> None:
        server = load_mcp_servers(self.cwd, paths=self.paths, trust=self.trust).get(name)
        if server is None:
            raise MCPError("Unknown MCP server")
        auth = self._oauth(server, interactive=True)

        async def authenticate(client: Any) -> None:
            async with client:
                pass

        await self._operate(server, authenticate, authorized=False, timeout=300, auth=auth)
        await self.disable(name)
        self._log(name, "Login completed; enable separately to authorize tools")

    async def logout(self, name: str) -> None:
        await self.disable(name)
        server = self.servers.get(name)
        if server is None:
            raise MCPError("Unknown MCP server")
        if self._auth_directory(server).exists():
            auth = self._oauth(server, interactive=False)
            await auth.token_storage_adapter.clear()
        self._log(name, "Logged out")

    @property
    def resource_tools(self) -> tuple[BaseTool, ...]:
        if self._resource_tools is not None:
            return self._resource_tools

        async def resources(server: str) -> tuple[str, dict[str, Any]]:
            return await self._resource_operation(server, "resources")

        async def templates(server: str) -> tuple[str, dict[str, Any]]:
            return await self._resource_operation(server, "templates")

        async def read(server: str, uri: str) -> tuple[str, dict[str, Any]]:
            return await self._resource_operation(server, "read", uri)

        self._resource_tools = tuple(
            StructuredTool.from_function(
                coroutine=callback,
                name=name,
                description=description,
                response_format="content_and_artifact",
                metadata={"forge.namespace": "mcp"},
            )
            for name, callback, description in (
                (
                    "list_mcp_resources",
                    resources,
                    "List resources of an explicitly authorized MCP server.",
                ),
                (
                    "list_mcp_resource_templates",
                    templates,
                    "List resource templates of an authorized MCP server.",
                ),
                (
                    "read_mcp_resource",
                    read,
                    "Read a resource URI through an authorized MCP server.",
                ),
            )
        )
        return self._resource_tools

    async def _resource_operation(
        self, name: str, action: str, uri: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        server = self.servers.get(name)
        if server is None:
            raise MCPError("Unknown MCP server")
        if uri is not None and len(uri) > 4096:
            raise MCPError("Resource URI exceeded limit")

        async def operation(client: Any) -> tuple[str, dict[str, Any]]:
            async def listing(kind: str) -> list[dict[str, Any]]:
                cursor = None
                seen: set[str] = set()
                items: list[dict[str, Any]] = []
                catalog_bytes = 0
                for _ in range(250):

                    async def fetch_page(cursor: str | None = cursor) -> Any:
                        return await (
                            client.list_resources_mcp(cursor=cursor, cache_mode="refresh")
                            if kind == "resources"
                            else client.list_resource_templates_mcp(
                                cursor=cursor, cache_mode="refresh"
                            )
                        )

                    page = await self._read_once_retry(fetch_page)
                    catalog_bytes += len(page.model_dump_json().encode())
                    if catalog_bytes > MAX_MCP_CATALOG_BYTES:
                        raise MCPError("MCP resource catalog byte budget exceeded")
                    values = page.resources if kind == "resources" else page.resource_templates
                    items.extend(
                        r.model_dump(mode="json", by_alias=True)
                        for r in values
                        if getattr(r, "mime_type", None) != "text/html"
                    )
                    if len(items) > 2000:
                        raise MCPError("Resource count exceeded limit")
                    cursor = page.next_cursor
                    if cursor is None:
                        break
                    if cursor in seen:
                        raise MCPError("Repeated resource cursor")
                    seen.add(cursor)
                else:
                    raise MCPError("Resource page limit exceeded")
                if kind == "resources":
                    self._resource_uris[name] = {str(item["uri"]) for item in items}
                else:
                    self._resource_templates[name] = tuple(
                        str(item["uriTemplate"]) for item in items
                    )
                return items

            async with client:
                if action == "read":
                    from fastmcp.resources.template import match_uri_template

                    if name not in self._resource_uris:
                        await listing("resources")
                    if name not in self._resource_templates:
                        await listing("templates")
                    if uri not in self._resource_uris[name] and not any(
                        match_uri_template(cast(str, uri), template) is not None
                        for template in self._resource_templates[name]
                    ):
                        raise MCPError("Resource URI is not listed or templated")
                    raw = [
                        r.model_dump(mode="json", by_alias=True)
                        for r in await self._read_once_retry(lambda: client.read_resource(uri))
                        if getattr(r, "mime_type", None) != "text/html"
                    ]
                else:
                    raw = await listing(action)
            raw = client.redact(raw)
            serialized = json.dumps(raw, ensure_ascii=False)
            artifact: dict[str, Any] = {"forge.mcp.resource.v1": raw}
            if len(serialized.encode()) > 1024 * 1024:
                artifact = {
                    "forge.mcp.resource.v1": {
                        "truncated": True,
                        "full_output_path": await save_output(
                            self.cwd, serialized, prefix="mcp-resource"
                        ),
                    }
                }
            view = bounded_text(serialized, 20480)
            if len(serialized.encode()) > 20480:
                view += "\nFull output: " + await save_output(
                    self.cwd, serialized, prefix="mcp-resource"
                )
            return view, artifact

        # Lists and reads are side-effect-free, including read's implicit discovery.
        return await self._operate(server, operation, discovery=True)

    def describe(self, action: str = "list", name: str | None = None, *, limit: int = 50) -> str:
        if action == "logs":
            return (
                "\n".join(list(self._logs.get(name or "", ()))[-max(1, min(200, limit)) :])
                or "No MCP logs."
            )
        if action == "tools":
            return (
                "\n".join(t.name for t in self._tools.get(name or "", ()))
                or "No authorized MCP tools."
            )
        return (
            "\n".join(
                f"{n}: {'authorized' if n in self._authorized else 'not authorized'}; "
                f"{s.config.exposure}; active={sum(owner == n for owner in self._tasks.values())}; "
                + (
                    "project"
                    if s.source == self.paths.project_mcp_config_path(self.cwd)
                    else "user"
                )
                for n, s in self.servers.items()
            )
            or "No MCP servers configured."
        )

    async def aclose(self) -> None:
        self._closed = True
        self._authorized.clear()
        self._tools.clear()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        if self._cleanup_failed:
            raise MCPError("MCP cleanup could not be confirmed; replacement refused")
