"""Local-only MCP acceptance server; never contacts a provider."""

import asyncio
import base64
import os
from pathlib import Path

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

server = FastMCP("Forge fixture", list_page_size=1)


@server.tool(description="Echo a value " + os.environ.get("FIXTURE_SECRET", ""))
def echo(value: str) -> dict[str, str]:
    """Echo a value in text and structured output."""
    return {"value": value}


@server.tool
def fail() -> str:
    """Return an MCP execution error."""
    raise ValueError("fixture failure")


@server.tool
async def wait(seconds: float = 30) -> str:
    """Wait long enough to exercise cancellation."""
    Path("waiting").write_text("ready")
    await asyncio.sleep(seconds)
    return "waited"


@server.tool
def mixed() -> list:
    """Return text, an image and unsupported audio for adapter acceptance."""
    from mcp.types import AudioContent, ImageContent, TextContent

    return [
        TextContent(type="text", text="x" * 25000),
        ImageContent(
            type="image", data=base64.b64encode(b"fixture-image").decode(), mime_type="image/png"
        ),
        AudioContent(
            type="audio", data=base64.b64encode(b"fixture-audio").decode(), mime_type="audio/wav"
        ),
    ]


@server.tool
async def request_input(ctx: Context) -> str:
    """Exercise unsupported server elicitation without an interrupt."""
    await ctx.elicit("fixture input", ["yes", "no"])
    raise ValueError("Server elicitation is unsupported")


class LargeInput(BaseModel):
    value: str = Field(description="x" * 20000)


@server.tool
def hidden_large(arguments: LargeInput) -> str:
    """A large-schema tool that must retain its configured hidden policy."""
    return arguments.value


@server.resource("fixture://note")
def note() -> str:
    return "fixture resource"


@server.resource("fixture://note/{name}")
def named_note(name: str) -> str:
    return f"fixture {name}"


if __name__ == "__main__":
    with Path("pids").open("a") as file:
        file.write(f"{os.getpid()}\n")
    server.run(transport="stdio", show_banner=False)
