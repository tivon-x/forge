"""Bounded text views and safe workspace artifacts for composed tools."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_core.messages import ToolMessage

from forge_coding.tools.write import create_write_tool_definition


def bounded_text(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    marker = b"\n[output truncated]\n"
    if limit <= len(marker):
        return marker[: max(0, limit)].decode("utf-8", errors="ignore")
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return (
        encoded[:head].decode("utf-8", errors="ignore")
        + marker.decode()
        + encoded[-tail:].decode("utf-8", errors="ignore")
    )


async def save_output(cwd: Path, value: str, *, prefix: str) -> str:
    relative = f".forge/artifacts/{prefix}-{uuid4().hex}.json"
    writer = create_write_tool_definition(cwd=cwd).tool
    result = await writer.ainvoke(
        {
            "type": "tool_call",
            "id": uuid4().hex,
            "name": writer.name,
            "args": {"path": relative, "content": value},
        },
        config={"callbacks": []},
    )
    if not isinstance(result, ToolMessage) or result.status == "error":
        raise ValueError("Unable to save collected output inside the workspace")
    return str(cwd / relative)
