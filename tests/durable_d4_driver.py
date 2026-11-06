"""Offline D4 execution recovery with real process exits."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command
from pydantic import AfterValidator

from fake_models import StreamingScriptedChatModel, tool_call_ai
from forge_agent.session import CustomEntry
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.storage import JsonlSessionStorage


class RecoveryModel(StreamingScriptedChatModel):
    root: Path = Path(".")
    action: str = ""
    fault: str = ""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if self.fault == "declarations":
            with (self.root / "declarations.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps([convert_to_openai_tool(t) for t in tools]) + "\n")
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        with (self.root / "calls.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps([m.model_dump(mode="json") for m in messages]) + "\n")
        if self.action == "crash" and self.fault in {"request", "declarations"}:
            os._exit(17)
        if self.action == "crash" and self.fault in {"partial", "reasoning-partial"}:
            content = (
                "partial"
                if self.fault == "partial"
                else [{"type": "reasoning", "reasoning": "partial"}]
            )
            yield ChatGenerationChunk(message=AIMessageChunk(content=content, id="partial"))
            await asyncio.sleep(30)
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            yield chunk


async def main(action: str, root: Path, fault: str, replay: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    home = root / "home"
    home.mkdir(exist_ok=True)
    os.environ.update(HOME=str(home), USERPROFILE=str(home), LANGSMITH_TRACING="false")
    (root / f"{action}-pid.json").write_text(json.dumps({"pid": os.getpid()}))

    async def effect(
        value: Annotated[str, AfterValidator(lambda text: text + "!")], runtime: ToolRuntime
    ) -> str | Command:
        with (root / "effects.jsonl").open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "call": runtime.tool_call_id,
                        "value": value,
                        "request_index": runtime.state.get("forge_request_index"),
                        "cwd": runtime.context.workspace_root,
                    }
                )
                + "\n"
            )
        if action == "crash" and fault in {"effect", "permission", "environment", "new-call"}:
            os._exit(17)
        if fault == "command-outcome":
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="effect complete",
                            tool_call_id=runtime.tool_call_id,
                            name="effect",
                        )
                    ]
                }
            )
        return "effect complete"

    tool = StructuredTool.from_function(
        coroutine=effect,
        name="effect",
        description="Controlled offline effect.",
        metadata={"forge.replay": replay},
    )
    storage = JsonlSessionStorage(root / "session")
    original_commit = storage.commit_records

    async def commit(writes, entries=()):
        sequence = await original_commit(writes, entries)
        with (root / "commit-log.jsonl").open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "sequence": sequence,
                        "writes": writes,
                        "entries": [e.model_dump(mode="json") for e in entries],
                    }
                )
                + "\n"
            )
        if action == "crash":
            serialized = json.dumps(writes)
            if fault == "response" and '"phase": "tools"' in serialized:
                os._exit(17)
            if (
                fault == "intent"
                and '"phase": "execute"' in serialized
                and '"replay"' in serialized
            ):
                os._exit(17)
            if (
                fault in {"outcome", "command-outcome"}
                and '"phase": "done"' in serialized
                and '"call": "old-call"' in serialized
            ):
                os._exit(17)
        return sequence

    storage.commit_records = commit
    responses = (
        [AIMessage(content="answer")]
        if fault in {"request", "partial", "reasoning-partial", "declarations"}
        else [tool_call_ai("old-call", "effect", {"value": "one"}), AIMessage(content="answer")]
    )
    if action == "resume":
        responses = [AIMessage(content="answer")]
        if fault == "new-call":
            responses = [
                tool_call_ai("old-call", "effect", {"value": "two"}),
                AIMessage(content="answer"),
            ]
        if fault == "permission":
            tool.metadata["forge.available"] = False
        if fault == "declarations":
            tool.description = "Changed only for the next prepare."
            tool.args_schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    cwd = root / "environment" if action == "resume" and fault == "environment" else root
    cwd.mkdir(exist_ok=True)
    model = RecoveryModel(responses)
    model.root, model.action, model.fault = root, action, fault
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            storage=storage,
            cwd=cwd,
            tools=[tool],
            enable_subagents=False,
        )
    )
    if action == "crash":
        receipt = await session.submit("controlled request", request_id="d4-request")
        (root / "submission.json").write_text(json.dumps({"id": receipt.id}))
        if fault in {"partial", "reasoning-partial"}:
            async with asyncio.timeout(10):
                field = "assistant" if fault == "partial" else "thinking"
                while session.live_activity[field] != "partial":
                    await asyncio.sleep(0.01)
            os._exit(17)
        await receipt.wait(timeout=10)
        raise AssertionError("fault was not reached")
    receipt = await session.submission(json.loads((root / "submission.json").read_text())["id"])
    assert receipt is not None
    if action == "abort":
        await session.abort()
    result = await receipt.wait(timeout=10)
    messages = session.state.messages
    calls = [call["id"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]
    results = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    (root / "result.json").write_text(
        json.dumps(
            {
                "settlement": result,
                "calls": calls,
                "results": results,
                "messages": [m.model_dump(mode="json") for m in messages],
                "snapshot": session.snapshot(),
                "aborted_attempts": [
                    e.data
                    for e in session.state.entries
                    if isinstance(e, CustomEntry) and e.namespace == "forge.aborted_attempt.v1"
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    await session.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]))
