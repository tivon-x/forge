"""BM25 search and durable branch loadout over the native tool catalog."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from forge_agent.session import CustomEntry
from forge_agent.tool_execution import tool_exposure
from forge_agent.types import JSONValue

LOADOUT_NAMESPACE = "forge.tool_loadout.v1"
type PersistFeature = Callable[[str, dict[str, JSONValue]], Awaitable[None]]


def tool_fingerprint(tool: BaseTool) -> str:
    schema = tool.tool_call_schema
    if isinstance(schema, type):
        schema = (
            schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()
        )
    native = {
        "name": tool.name,
        "description": tool.description,
        "schema": schema,
        "metadata": {
            k: v
            for k, v in (tool.metadata or {}).items()
            if k not in {"forge.loaded", "forge.declared", "forge.codemode_only", "forge.available"}
        },
    }
    return hashlib.sha256(json.dumps(native, sort_keys=True, default=str).encode()).hexdigest()


def js_identifier(name: str) -> str:
    return (
        "".join(
            c if re.fullmatch(r"[A-Za-z_$]" if i == 0 else r"[A-Za-z0-9_$]", c) else "_"
            for i, c in enumerate(name)
        )
        or "_"
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(default="", max_length=2048)
    names: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=5, ge=1, le=20)
    namespace: str | None = Field(default=None, max_length=128)


class ToolDiscovery:
    def __init__(self, reader: Callable[[], Sequence[BaseTool]], persist: PersistFeature) -> None:
        self.reader = reader
        self.persist = persist
        self.loaded: dict[str, str] = {}
        self.diagnostics: list[str] = []
        self.mode: Literal["on", "only"] | None = None

    def restore(self, entries: Sequence[CustomEntry], mode: Literal["on", "only"] | None) -> None:
        self.loaded = {}
        self.diagnostics = []
        self.mode = mode
        for entry in entries:
            if entry.namespace != LOADOUT_NAMESPACE:
                continue
            if entry.data.get("version") != 1:
                if "Unknown tool loadout version ignored" not in self.diagnostics:
                    self.diagnostics.append("Unknown tool loadout version ignored")
                continue
            loaded = entry.data.get("loaded")
            if isinstance(loaded, dict) and len(loaded) <= 2000:
                self.loaded = {
                    k: v for k, v in loaded.items() if isinstance(v, str) and len(v) == 64
                }
            restored_mode = entry.data.get("codemode")
            if mode is None and restored_mode in {"on", "only"}:
                self.mode = restored_mode  # type: ignore[assignment]
        self.sync()

    def sync(self) -> None:
        tools = self.reader()
        current = {tool.name: tool_fingerprint(tool) for tool in tools}
        if any(current.get(name) != fingerprint for name, fingerprint in self.loaded.items()):
            message = (
                "Some loaded tools are unavailable or changed; history does not authorize them"
            )
            if message not in self.diagnostics:
                self.diagnostics.append(message)
        for tool in tools:
            loaded = self.loaded.get(tool.name) == tool_fingerprint(tool)
            metadata = dict(tool.metadata or {})
            metadata["forge.loaded"] = loaded and tool_exposure(tool) == "deferred"
            metadata["forge.codemode_only"] = (
                self.mode == "only" and tool_exposure(tool) == "direct"
            )
            tool.metadata = metadata

    def search(
        self, query: str, *, limit: int = 5, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        tools = [
            t
            for t in self.reader()
            if tool_exposure(t) not in {"hidden", "model-only"}
            and (t.metadata or {}).get("forge.available") is not False
            and (
                namespace is None
                or (t.metadata or {}).get("forge.namespace", "builtin") == namespace
            )
        ]
        documents = [
            Counter(
                _tokens(
                    f"{t.name} {t.name} "
                    f"{(t.metadata or {}).get('forge.namespace', 'builtin')} {t.description}"
                )
            )
            for t in tools
        ]
        query_tokens = set(_tokens(query))
        average = sum(sum(d.values()) for d in documents) / max(1, len(documents)) or 1
        frequencies = {word: sum(word in d for d in documents) for word in query_tokens}
        scored = []
        for order, (tool, document) in enumerate(zip(tools, documents, strict=True)):
            length = sum(document.values())
            score = sum(
                math.log(1 + (len(tools) - frequencies[w] + 0.5) / (frequencies[w] + 0.5))
                * document[w]
                * 2.2
                / (document[w] + 1.2 * (0.25 + 0.75 * length / average))
                for w in query_tokens
                if document[w]
            )
            if score > 0 or not query_tokens:
                scored.append((score, order, tool))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            {
                "name": t.name,
                "jsName": js_identifier(t.name),
                "description": t.description[:512],
                "loaded": bool((t.metadata or {}).get("forge.loaded")),
            }
            for _, _, t in scored[: max(1, min(20, limit))]
        ]

    async def snapshot(self) -> None:
        await self.persist(
            LOADOUT_NAMESPACE,
            {
                "version": 1,
                "loaded": dict(self.loaded),
                "codemode": self.mode,
            },
        )

    def create_tool(self) -> BaseTool:
        async def search(
            query: str = "",
            names: list[str] | None = None,
            limit: int = 5,
            namespace: str | None = None,
        ) -> str:
            catalog = {
                t.name: t
                for t in self.reader()
                if tool_exposure(t) not in {"hidden", "model-only"}
                and (t.metadata or {}).get("forge.available") is not False
                and (
                    namespace is None
                    or (t.metadata or {}).get("forge.namespace", "builtin") == namespace
                )
            }
            results = self.search(query, limit=limit, namespace=namespace)
            selected = names or [r["name"] for r in results]
            if len(set(selected)) != len(selected) or any(n not in catalog for n in selected):
                raise ToolException("Unknown or duplicate tool name")
            if any(len(json.dumps(catalog[name].args).encode()) > 16384 for name in selected):
                raise ToolException("Tool schema exceeds direct declaration limit; use Codemode")
            added = {
                name: tool_fingerprint(catalog[name])
                for name in selected
                if tool_exposure(catalog[name]) == "deferred"
            }
            previous = self.loaded
            self.loaded = {**previous, **added}
            try:
                await self.snapshot()
            except BaseException:
                self.loaded = previous
                raise
            self.sync()
            if names:
                results = [
                    {"name": n, "description": catalog[n].description[:512]} for n in selected
                ]
            return json.dumps(
                {
                    "tools": results,
                    "loaded": list(added),
                    "next_turn": "Loaded schemas are available on the next model request.",
                }
            )

        return StructuredTool.from_function(
            coroutine=search,
            name="tool_search",
            description="Find tools by intent or names; load deferred schemas for the next turn.",
            args_schema=SearchInput,
            handle_tool_error=True,
            metadata={"forge.exposure": "model-only", "forge.execution_mode": "sequential"},
        )
