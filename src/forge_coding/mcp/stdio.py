"""Bounded stdio framing; protocol/session handling remains owned by the SDK."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastmcp.client.transports import ClientTransport
from fastmcp.client.transports.base import TransportOptions
from mcp.shared.message import SessionMessage
from mcp.types import jsonrpc_message_adapter

from forge_coding.tools.shell import _kill_process_tree


class BoundedStdioTransport(ClientTransport):
    """Use SDK streams with a byte limit before parsing each JSON-RPC line."""

    legacy_only = True

    def __init__(
        self,
        command: str,
        args: list[str],
        *,
        env: dict[str, str],
        cwd: str,
        max_bytes: int,
        on_limit: Callable[[], None],
    ) -> None:
        self.command, self.args, self.env, self.cwd = command, args, env, cwd
        self.max_bytes, self.on_limit = max_bytes, on_limit

    @asynccontextmanager
    async def connect_session(
        self,
        *,
        transport_options: TransportOptions | None = None,
        **session_kwargs: Any,
    ) -> AsyncIterator[Any]:
        from forge_coding.mcp.runtime import MCPError

        process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            env=self.env,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=self.max_bytes,
            start_new_session=sys.platform != "win32",
        )
        assert process.stdout is not None and process.stdin is not None
        reader, writer = process.stdout, process.stdin
        incoming, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](0)
        write_stream, outgoing = anyio.create_memory_object_stream[SessionMessage](0)

        async def receive() -> None:
            async with incoming:
                while True:
                    try:
                        line = await reader.readline()
                        if not line:
                            return
                        if len(line) > self.max_bytes:
                            raise ValueError("Line limit")
                    except ValueError:
                        self.on_limit()
                        await incoming.send(MCPError("MCP stdio message byte budget exceeded"))
                        return
                    try:
                        message = SessionMessage(
                            jsonrpc_message_adapter.validate_json(line, by_name=False)
                        )
                    except Exception:
                        await incoming.send(MCPError("Invalid MCP stdio message"))
                        return
                    await incoming.send(message)

        async def send() -> None:
            async with outgoing:
                async for message in outgoing:
                    data = message.message.model_dump_json(
                        by_alias=True, exclude_unset=True
                    ).encode()
                    if len(data) > self.max_bytes:
                        self.on_limit()
                        raise MCPError("MCP stdio request byte budget exceeded")
                    writer.write(data + b"\n")
                    await writer.drain()

        try:
            async with read_stream, write_stream, anyio.create_task_group() as group:
                group.start_soon(receive)
                group.start_soon(send)
                try:
                    options = transport_options or TransportOptions()
                    async with options.session_class(
                        read_stream, write_stream, **session_kwargs
                    ) as session:
                        yield session
                finally:
                    group.cancel_scope.cancel()
        finally:

            async def cleanup() -> None:
                if process.returncode is None:
                    await asyncio.to_thread(_kill_process_tree, process)
                await process.wait()

            closing = asyncio.create_task(cleanup())
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                await closing
                raise
