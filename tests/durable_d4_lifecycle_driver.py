"""Controlled batch, nested, retry, questionnaire and host-lifecycle faults."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from fake_models import StreamingScriptedChatModel, tool_call_ai
from forge_agent.retry import RetryPolicy
from forge_agent.tool_execution import get_nested_tool_executor
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.storage import JsonlSessionStorage


async def main(action: str, root: Path, fault: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    home = root / "home"
    home.mkdir(exist_ok=True)
    os.environ.update(HOME=str(home), USERPROFILE=str(home), LANGSMITH_TRACING="false")
    (root / f"{action}-pid.json").write_text(json.dumps({"pid": os.getpid()}))
    first_outcome = asyncio.Event()

    def log(kind: str, data: object) -> None:
        with (root / f"{kind}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(data) + "\n")

    async def effect(value: str, runtime: ToolRuntime) -> str:
        log("entered", {"value": value, "call": runtime.tool_call_id})
        if action == "crash" and value == "second":
            await first_outcome.wait()
        log("effects", {"value": value, "call": runtime.tool_call_id})
        if action == "crash" and value == "second":
            os._exit(17)
        if action == "crash" and fault in {"close", "abort"}:
            await asyncio.sleep(30)
        return value

    async def composite() -> str:
        result = await get_nested_tool_executor().call("effect", {"value": "nested"})
        if action == "crash":
            os._exit(17)
        return str(result.content)

    tool = StructuredTool.from_function(
        coroutine=effect,
        name="effect",
        description="Offline effect.",
        metadata={
            "forge.replay": "safe" if fault in {"nested", "safe-close"} else "unsafe",
            "forge.execution_mode": "sequential" if fault == "sequential" else "parallel",
        },
    )
    parent = StructuredTool.from_function(
        coroutine=composite,
        name="composite",
        description="Idempotent owner.",
        metadata={"forge.replay": "safe"},
    )
    storage = JsonlSessionStorage(root / "session")
    original = storage.commit_records

    async def commit(writes, entries=()):
        sequence = await original(writes, entries)
        log("commit-log", {"sequence": sequence, "writes": writes})
        text = json.dumps(writes)
        if '"call": "first-call"' in text and '"phase": "done"' in text:
            first_outcome.set()
        if action == "crash" and fault == "retry" and '"phase": "retry"' in text:
            os._exit(17)
        return sequence

    storage.commit_records = commit
    if fault in {"parallel", "sequential"}:
        response = AIMessage(
            content="",
            tool_calls=[
                {"name": "effect", "id": "first-call", "args": {"value": "first"}},
                {"name": "effect", "id": "second-call", "args": {"value": "second"}},
            ],
        )
    elif fault == "nested":
        response = tool_call_ai("parent-call", "composite", {})
    elif fault.startswith("questionnaire"):
        response = tool_call_ai(
            "question-call",
            "ask_user_question",
            {
                "questions": [
                    {
                        "question": "Choose?",
                        "header": "choice",
                        "options": [
                            {"label": "A", "description": "First choice"},
                            {"label": "B", "description": "Second choice"},
                        ],
                        "multi_select": False,
                    }
                ]
            },
        )
    else:
        response = tool_call_ai("old-call", "effect", {"value": "one"})

    class Model(StreamingScriptedChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            log("requests", [m.model_dump(mode="json") for m in messages])
            if fault == "retry":
                raise ConnectionError("controlled timeout")
            async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
                yield chunk

    model = Model(
        [response, AIMessage(content="answer")]
        if action == "crash"
        else [AIMessage(content="answer")]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            cwd=root,
            storage=storage,
            tools=[tool, parent],
            enable_subagents=False,
            interactive=fault.startswith("questionnaire"),
            retry=RetryPolicy(initial_delay=0.05, max_delay=0.2),
        )
    )
    if action == "crash":
        receipt = await session.submit("controlled", request_id="lifecycle")
        (root / "submission.json").write_text(json.dumps({"id": receipt.id}))
        if fault in {"close", "abort"}:
            async with asyncio.timeout(10):
                while not (root / "effects.jsonl").exists():
                    await asyncio.sleep(0.01)
            if fault == "close":
                await session.aclose()
            else:
                await session.abort()
                await receipt.wait(timeout=10)
                await session.aclose()
            os._exit(17)
        if fault.startswith("questionnaire"):
            async with asyncio.timeout(10):
                while not session.is_waiting_for_input or session.is_running:
                    await asyncio.sleep(0.01)
            if fault == "questionnaire-close":
                await session.aclose()
            if fault == "questionnaire-abort":
                await session.abort()
                await receipt.wait(timeout=10)
                await session.aclose()
            os._exit(17)
        await receipt.wait(timeout=10)
        raise AssertionError("fault was not reached")
    receipt = await session.submission(json.loads((root / "submission.json").read_text())["id"])
    assert receipt is not None
    if fault == "questionnaire-close":
        _ = [e async for e in session.respond_to_human_input("A")]
    waiting = asyncio.create_task(receipt.wait(timeout=10))
    if fault.startswith("questionnaire") and fault not in {
        "questionnaire-abort",
        "questionnaire-close",
    }:
        async with asyncio.timeout(10):
            while not session.is_waiting_for_input or session.is_running:
                await asyncio.sleep(0.01)
        _ = [e async for e in session.respond_to_human_input("A")]
    result = await waiting
    messages = session.state.messages
    (root / "result.json").write_text(
        json.dumps(
            {
                "settlement": result,
                "calls": [
                    c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls
                ],
                "results": [m.tool_call_id for m in messages if isinstance(m, ToolMessage)],
                "snapshot": session.snapshot(),
            }
        ),
        encoding="utf-8",
    )
    await session.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2]), sys.argv[3]))
