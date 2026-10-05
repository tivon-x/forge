"""Offline D3 partial-answer process exit/reopen driver."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from fake_models import ScriptedChatModel, StreamingScriptedChatModel, tool_call_ai
from forge_agent import MessageDeltaEvent, SubagentRuntime
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.storage import JsonlSessionStorage


class PausedModel(StreamingScriptedChatModel):
    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        self._record_call(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content="已确认的部分回答", id="partial"))
        await asyncio.sleep(30)
        yield ChatGenerationChunk(message=AIMessageChunk(content="不会显示", id="partial"))


async def main(action: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    home = root / "home"
    home.mkdir(exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    model = PausedModel([AIMessage(content="unused")])
    storage = JsonlSessionStorage(root / "session")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            storage=storage,
            cwd=root,
            enable_subagents=action == "background",
            interactive=action == "background",
        )
    )
    if action == "background":
        session._harness.config.provider = ScriptedChatModel(
            [
                tool_call_ai(
                    "bg-exit",
                    "task",
                    {
                        "agent": "scout",
                        "instruction": "partial",
                        "mode": "background",
                        "context": "fresh",
                    },
                ),
                AIMessage(content="root ready"),
            ]
        )
        child = PausedModel()

        async def runtime(spec):
            del spec
            return SubagentRuntime(provider=child, model="fake")

        session._subagent_model_runtime = runtime
        _ = [event async for event in session.prompt("launch background")]
        async with asyncio.timeout(5):
            while not session.background_subagents or not session.background_subagents[0].get(
                "output"
            ):
                await asyncio.sleep(0.02)
        (root / "confirmed.json").write_text(
            json.dumps(session.snapshot(), ensure_ascii=False), encoding="utf-8"
        )
        (root / "effects.json").write_text(
            json.dumps(
                {
                    "model_calls": len(child.calls) + len(session._harness.config.provider.calls),
                    "child_model_calls": len(child.calls),
                    "root_model_calls": len(session._harness.config.provider.calls),
                    "tool_calls": 1,
                    "pid": os.getpid(),
                }
            ),
            encoding="utf-8",
        )
        os._exit(17)
    if action == "partial":
        async for event in session.prompt("controlled process exit"):
            if isinstance(event, MessageDeltaEvent):
                (root / "confirmed.json").write_text(
                    json.dumps(session.snapshot(), ensure_ascii=False), encoding="utf-8"
                )
                (root / "effects.json").write_text(
                    json.dumps(
                        {"model_calls": len(model.calls), "tool_calls": 0, "pid": os.getpid()}
                    ),
                    encoding="utf-8",
                )
                os._exit(17)
        raise AssertionError("partial progress was not published")
    snapshot = session.snapshot()
    live = session.live_activity
    (root / "reopened.json").write_text(
        json.dumps(
            {"snapshot": snapshot, "live": live, "model_calls": len(model.calls)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    await session.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
