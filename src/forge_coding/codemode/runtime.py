"""Async parent host for one-shot QuickJS workers and native nested tools."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from forge_agent.session import CustomEntry
from forge_agent.tool_execution import get_nested_tool_executor, tool_exposure
from forge_agent.types import JSONValue
from forge_coding.features.tool_discovery import (
    PersistFeature,
    SearchInput,
    ToolDiscovery,
    js_identifier,
)
from forge_coding.tools.output import bounded_text, save_output
from forge_coding.tools.shell import _kill_process_tree

STORE_NAMESPACE = "forge.codemode_store.v1"
MAX_LINE = 2 * 1024 * 1024


def _valid_store(values: Any) -> bool:
    if not isinstance(values, dict):
        return False
    try:
        return len(json.dumps(values, allow_nan=False).encode()) <= 1024 * 1024 and all(
            isinstance(key, str)
            and len(key) <= 256
            and len(json.dumps(value, allow_nan=False).encode()) <= 65536
            for key, value in values.items()
        )
    except (ValueError, TypeError):
        return False


class CodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str = Field(min_length=1, max_length=256 * 1024)


class CodemodeRuntime:
    def __init__(self, cwd: Path, discovery: ToolDiscovery, persist: PersistFeature) -> None:
        self.cwd = cwd
        self.discovery = discovery
        self.persist = persist
        self.store: dict[str, JSONValue] = {}
        self.diagnostics: list[str] = []
        self.workers: set[asyncio.subprocess.Process] = set()

    def restore(self, entries: Sequence[CustomEntry]) -> None:
        self.store = {}
        self.diagnostics = []
        for entry in entries:
            if entry.namespace != STORE_NAMESPACE:
                continue
            if entry.data.get("version") != 1:
                if "Unknown Codemode store version ignored" not in self.diagnostics:
                    self.diagnostics.append("Unknown Codemode store version ignored")
                continue
            if entry.data.get("version") == 1:
                values = entry.data.get("values")
                if _valid_store(values):
                    self.store = dict(cast(dict[str, JSONValue], values))
                elif "Invalid Codemode store snapshot ignored" not in self.diagnostics:
                    self.diagnostics.append("Invalid Codemode store snapshot ignored")

    def _catalog(self) -> list[dict[str, Any]]:
        catalog = []
        identifiers: set[str] = set()
        for tool in self.discovery.reader():
            if (
                tool_exposure(tool) in {"hidden", "model-only"}
                or (tool.metadata or {}).get("forge.available") is False
            ):
                continue
            ident = js_identifier(tool.name)
            if ident in identifiers:
                raise ValueError("Codemode tool identifier collision")
            identifiers.add(ident)
            catalog.append(
                {"name": tool.name, "jsName": ident, "description": tool.description[:128]}
            )
        return catalog

    def api_prompt(self) -> str:
        self._catalog()
        lines = [
            "Codemode: use top-level await/return with tools.<jsName>(args).",
            "text/image/console/exit/store/load/searchTools/describeTool are available.",
            "No process, filesystem, fetch, require or module imports exist in JavaScript.",
            "Tool failures reject promises; use Promise.allSettled for independent failures.",
        ]
        namespaces: dict[str, list[BaseTool]] = {}
        for tool in self.discovery.reader():
            if (
                tool_exposure(tool) not in {"hidden", "model-only"}
                and (tool.metadata or {}).get("forge.available") is not False
            ):
                namespaces.setdefault(
                    str((tool.metadata or {}).get("forge.namespace", "builtin")), []
                ).append(tool)
        budget = 12000
        for name, group in namespaces.items():
            lines.append(f"Namespace {name}: {len(group)} tools")
            for tool in group:
                signature = f"tools.{js_identifier(tool.name)}(args: {json.dumps(tool.args)})"
                if budget >= len(signature):
                    lines.append(signature)
                    budget -= len(signature)
        lines.append(
            "Use searchTools/describeTool for omitted signatures. Total timeout defaults to 300s."
        )
        return "\n".join(lines)

    async def execute(self, code: str) -> tuple[Any, dict[str, Any]]:
        started = monotonic()
        timeout_ms = 300000
        output_tokens = 10000
        if code.startswith("// @options:"):
            line, _, code = code.partition("\n")
            options = json.loads(line.removeprefix("// @options:").strip())
            if not isinstance(options, dict) or set(options) - {"timeout_ms", "max_output_tokens"}:
                raise ValueError("Invalid Codemode options")
            timeout_ms = options.get("timeout_ms", timeout_ms)
            output_tokens = options.get("max_output_tokens", output_tokens)
        if (
            type(timeout_ms) is not int
            or not 1 <= timeout_ms <= 300000
            or type(output_tokens) is not int
            or not 1 <= output_tokens <= 10000
        ):
            raise ValueError("Invalid Codemode limits")
        nested = get_nested_tool_executor()
        catalog = self._catalog()
        tools = {
            t.name: t
            for t in self.discovery.reader()
            if tool_exposure(t) not in {"hidden", "model-only"}
            and (t.metadata or {}).get("forge.available") is not False
        }
        run_id = uuid4().hex
        process: asyncio.subprocess.Process | None = None
        pending: dict[int, asyncio.Task[None]] = {}
        seen: set[int] = set()
        output: list[dict[str, Any]] = []
        output_bytes = 0
        image_bytes = 0
        lost = False
        write_lock = asyncio.Lock()
        write_capacity = asyncio.Condition()
        queued_bytes = 0
        slots = asyncio.Semaphore(32)
        result: dict[str, Any] = {"ok": False, "error": "Script failed"}
        stderr_task: asyncio.Task[None] | None = None
        stage = "worker startup"

        async def send(kind: str, **payload: Any) -> None:
            nonlocal queued_bytes
            if process is None or process.stdin is None:
                raise ValueError("Worker input unavailable")
            data = (
                json.dumps(
                    {"v": 1, "run": run_id, "type": kind, **payload},
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
            if len(data) > MAX_LINE:
                raise ValueError("IPC payload exceeded limit")
            async with write_capacity:
                await write_capacity.wait_for(lambda: queued_bytes + len(data) <= 8 * 1024 * 1024)
                queued_bytes += len(data)
            try:
                async with write_lock:
                    process.stdin.write(data)
                    await process.stdin.drain()
            finally:
                async with write_capacity:
                    queued_bytes -= len(data)
                    write_capacity.notify_all()

        async def dispatch(message: dict[str, Any]) -> None:
            ident = message["id"]
            try:
                async with slots:
                    name, arguments = message["name"], message["args"]
                    if message.get("target") == "metadata":
                        if name == "search":
                            options = SearchInput.model_validate(arguments)
                            value: Any = self.discovery.search(
                                options.query, limit=options.limit, namespace=options.namespace
                            )
                        elif name == "describe" and arguments.get("name") in tools:
                            tool = tools[arguments["name"]]
                            schema = tool.tool_call_schema
                            schema = (
                                (
                                    schema.model_json_schema()
                                    if hasattr(schema, "model_json_schema")
                                    else schema.schema()
                                )
                                if isinstance(schema, type)
                                else schema
                            )
                            encoded = json.dumps(schema, ensure_ascii=False)
                            offset = arguments.get("offset", 0)
                            limit = arguments.get("limit", 4096)
                            if (
                                type(offset) is not int
                                or offset < 0
                                or type(limit) is not int
                                or not 1 <= limit <= 8192
                            ):
                                raise ValueError("Invalid schema page")
                            value = {
                                "name": tool.name,
                                "description": tool.description[:512],
                                "schema": encoded[offset : offset + limit],
                                "next_offset": offset + limit
                                if offset + limit < len(encoded)
                                else None,
                            }
                        else:
                            raise ValueError("Unknown metadata request")
                    elif message.get("target") == "tool" and name in tools:
                        child = await nested.call(name, arguments)
                        if child.status != "success":
                            await send(
                                "tool_result", id=ident, ok=False, error="Tool execution failed"
                            )
                            return
                        artifact = child.artifact
                        if isinstance(artifact, dict) and "forge.mcp.v1" in artifact:
                            value = artifact["forge.mcp.v1"]
                        elif isinstance(artifact, dict) and artifact.get("data") is not None:
                            value = (
                                {**artifact["data"], "content": child.content}
                                if isinstance(artifact["data"], dict)
                                else artifact["data"]
                            )
                        else:
                            value = child.content
                        if len(json.dumps(value, ensure_ascii=False).encode()) > 1024 * 1024:
                            path = await save_output(
                                self.cwd,
                                json.dumps(value, ensure_ascii=False),
                                prefix="codemode-tool",
                            )
                            value = {"truncated": True, "full_output_path": path}
                    else:
                        raise ValueError("Unknown tool request")
                await send("tool_result", id=ident, ok=True, value=value)
            except asyncio.CancelledError:
                raise
            except Exception:
                await send("tool_result", id=ident, ok=False, error="Tool execution failed")

        async def drain_stderr() -> None:
            if process is not None and process.stderr is not None:
                while await process.stderr.read(65536):
                    pass

        try:
            async with asyncio.timeout(timeout_ms / 1000):
                environment = {
                    k: os.environ[k]
                    for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
                    if k in os.environ
                }
                # Package initialization needs a home path; use the workspace as its home.
                environment.update({"HOME": str(self.cwd), "USERPROFILE": str(self.cwd)})
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "forge_coding.codemode.worker",
                    cwd=str(self.cwd),
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=MAX_LINE,
                    start_new_session=sys.platform != "win32",
                )
                self.workers.add(process)
                stderr_task = asyncio.create_task(drain_stderr())
                await send(
                    "init", code=code, tools=catalog, store=self.store, timeout=timeout_ms / 1000
                )
                assert process.stdout is not None
                while True:
                    stage = "worker output"
                    packet = await process.stdout.readline()
                    if not packet or len(packet) > MAX_LINE:
                        raise ValueError("Worker exited or exceeded IPC limit")
                    stage = "IPC decoding"
                    message = json.loads(packet)
                    stage = "IPC envelope"
                    if (
                        not isinstance(message, dict)
                        or message.get("v") != 1
                        or message.get("run") != run_id
                    ):
                        raise ValueError("Invalid IPC envelope")
                    kind = message.get("type")
                    if kind == "tool_call":
                        stage = "IPC tool request"
                        ident = message.get("id")
                        args = message.get("args")
                        if (
                            type(ident) is not int
                            or ident in seen
                            or len(seen) >= 256
                            or not isinstance(message.get("name"), str)
                            or not isinstance(args, dict)
                            or len(json.dumps(args).encode()) > 65536
                            or len(pending) >= 64
                        ):
                            raise ValueError("Invalid IPC tool request")
                        seen.add(ident)
                        pending[ident] = asyncio.create_task(dispatch(message))

                        def remove(task: asyncio.Task[None], ident: int = ident) -> None:
                            pending.pop(ident, None)
                            if (
                                not task.cancelled()
                                and task.exception() is not None
                                and process is not None
                                and process.returncode is None
                            ):
                                process.kill()

                        pending[ident].add_done_callback(remove)
                    elif kind == "output":
                        stage = "IPC output item"
                        item = message.get("item")
                        if not isinstance(item, dict) or item.get("type") not in {"text", "image"}:
                            raise ValueError("Invalid output item")
                        if item["type"] == "text" and not isinstance(item.get("text"), str):
                            raise ValueError("Invalid text output")
                        if item["type"] == "image":
                            value = item.get("image")
                            if isinstance(value, dict):
                                if value.get("type") == "image" and "mimeType" in value:
                                    value = (
                                        f"data:{value['mimeType']};base64,{value.get('data', '')}"
                                    )
                                else:
                                    value = value.get("image_url", value.get("data"))
                            if not isinstance(value, str) or not re.fullmatch(
                                r"data:image/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/=]+", value
                            ):
                                raise ValueError("Invalid inline image")
                            decoded = base64.b64decode(value.split(",", 1)[1], validate=True)
                            if (
                                len(decoded) > 512 * 1024
                                or image_bytes + len(decoded) > 4 * 1024 * 1024
                            ):
                                raise ValueError("Image budget exceeded")
                            image_bytes += len(decoded)
                            item = {"type": "image", "image_url": value}
                        size = len(json.dumps(item, ensure_ascii=False).encode())
                        if output_bytes + size <= 8 * 1024 * 1024:
                            output.append(item)
                            output_bytes += size
                        else:
                            lost = True
                    elif kind == "done":
                        stage = "IPC completion"
                        if type(message.get("ok")) is not bool:
                            raise ValueError("Invalid done result")
                        result = message
                        break
                    else:
                        raise ValueError("Unknown IPC message")
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            result = {
                "ok": False,
                "error": "Script timed out; completed tool effects are not rolled back",
            }
        except Exception as exc:
            result = {
                "ok": False,
                "error": (
                    f"Script failed or violated execution limits ({type(exc).__name__}; {stage})"
                ),
            }
        finally:
            children = list(pending.values())
            for child in children:
                child.cancel()
            if children:
                await asyncio.gather(*children, return_exceptions=True)
            if process is not None:
                if process.returncode is None:
                    await asyncio.to_thread(_kill_process_tree, process)
                await process.wait()
                self.workers.discard(process)
            if stderr_task is not None:
                await stderr_task
        if result["ok"]:
            snapshot = result.get("store", {})
            values = snapshot.get("values") if isinstance(snapshot, dict) else None
            if not _valid_store(values):
                result = {"ok": False, "error": "Invalid store result"}
            else:
                lost = lost or snapshot.get("lost") is True
                if values != self.store:
                    await self.persist(
                        STORE_NAMESPACE,
                        {"version": 1, "values": cast(dict[str, JSONValue], values)},
                    )
                    self.store = cast(dict[str, JSONValue], values)
        payload = {
            "ok": result["ok"],
            "output": output,
            "value": result.get("value"),
            "error": result.get("error"),
            "lost_output": lost,
            "duration_ms": int((monotonic() - started) * 1000),
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        view = bounded_text(serialized, output_tokens * 4)
        artifact = {
            "forge.codemode.v1": payload,
            "forge.status": "success" if result["ok"] else "error",
        }
        if len(serialized.encode()) > output_tokens * 4:
            path = await save_output(self.cwd, serialized, prefix="codemode")
            view += f"\nCollected output: {path}"
            artifact = {
                "forge.codemode.v1": {
                    "ok": result["ok"],
                    "truncated": True,
                    "full_output_path": path,
                    "lost_output": lost,
                },
                "forge.status": artifact["forge.status"],
            }
        return ("Script completed\n" if result["ok"] else "Script failed\n") + view, artifact

    def create_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            coroutine=self.execute,
            name="codemode",
            description=self.api_prompt(),
            args_schema=CodeInput,
            response_format="content_and_artifact",
            metadata={"forge.exposure": "model-only", "forge.execution_mode": "sequential"},
        )

    async def aclose(self) -> None:
        for worker in tuple(self.workers):
            if worker.returncode is None:
                await asyncio.to_thread(_kill_process_tree, worker)
            await worker.wait()
            self.workers.discard(worker)
