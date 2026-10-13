"""Public saver, historical branch and history-only fork acceptance."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.channels import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel
from pydantic.v1 import BaseModel as LegacyModel

from fake_models import ScriptedChatModel, tool_call_ai
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.features.execution import GenerationMiddleware
from forge_coding.sessions.checkpoints import JournalCheckpointSaver
from forge_coding.sessions.copying import copy_active_branch
from forge_coding.sessions.manager import CodingSessionRecord
from forge_coding.sessions.storage import JsonlSessionStorage


def add_values(value, writes):
    return (value or 0) + sum(writes)


class DeltaState(TypedDict):
    total: Annotated[int, DeltaChannel(add_values, snapshot_frequency=2)]


class UnsupportedClient(BaseModel):
    label: str


class LegacyInput(LegacyModel):
    value: str


async def input_contract(root):
    effects = []

    async def effect(value: str) -> str:
        effects.append(value)
        return value

    async def unsupported() -> tuple[str, object]:
        effects.append("unsupported")
        return "effect completed", UnsupportedClient(label="must not persist")

    async def command_tool(runtime: ToolRuntime) -> Command:
        effects.append("command")
        return Command(
            update={
                "messages": [
                    ToolMessage(content="command completed", tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    json_tool = StructuredTool.from_function(
        coroutine=effect,
        name="json_tool",
        description="Controlled JSON schema.",
        args_schema=schema,
    )
    legacy = StructuredTool.from_function(
        coroutine=effect,
        name="legacy",
        description="Controlled legacy input.",
        args_schema=LegacyInput,
    )
    remote = StructuredTool.from_function(
        coroutine=effect,
        name="remote",
        description="Remote references must not be fetched.",
        args_schema={
            "type": "object",
            "properties": {"value": {"$ref": "https://invalid.test/no-schema"}},
        },
    )
    unsupported_tool = StructuredTool.from_function(
        coroutine=unsupported,
        name="unsupported",
        description="Unsupported artifact.",
        response_format="content_and_artifact",
    )
    command = StructuredTool.from_function(
        coroutine=command_tool, name="command_tool", description="Native command."
    )
    model = ScriptedChatModel(
        [
            tool_call_ai("invalid-json", "json_tool", {"value": 42}),
            tool_call_ai("valid-json", "json_tool", {"value": "valid"}),
            tool_call_ai("legacy-call", "legacy", {"value": "legacy"}),
            tool_call_ai("remote-ref", "remote", {"value": "forbidden"}),
            tool_call_ai("unsupported-result", "unsupported", {}),
            tool_call_ai("command-result", "command_tool", {}),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            cwd=root,
            storage=JsonlSessionStorage(root / "input-contract"),
            tools=[json_tool, legacy, remote, unsupported_tool, command],
            enable_subagents=False,
        )
    )
    try:
        _ = [e async for e in session.prompt("validate native inputs")]
        assert effects == ["valid", "legacy", "unsupported", "command"]
        results = [m for m in session.messages if isinstance(m, ToolMessage)]
        assert [m.status for m in results] == [
            "error",
            "success",
            "success",
            "error",
            "error",
            "success",
        ]
        assert session.live_activity["tools"] == {}
        records = await session._config.storage.read_records()
        tool_outcomes = [
            d["value"]
            for d in records["documents"]
            if d["record"]["kind"] == "forge.execution.v1" and "call" in d["value"]
        ]
        assert len(tool_outcomes) == 6
        assert all(v["phase"] == "done" and "outcome" in v for v in tool_outcomes)
        return {
            "effects": effects,
            "json_schema": True,
            "legacy_schema": True,
            "remote_reference_rejected": True,
        }
    finally:
        await session.aclose()


def delta_graph(saver):
    graph = StateGraph(DeltaState)
    graph.add_node("increment", lambda state: {"total": 10})
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile(checkpointer=saver)


async def resume_delta(root):
    home = root / "home"
    os.environ.update(HOME=str(home), USERPROFILE=str(home), LANGSMITH_TRACING="false")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel([]),
            model="fake",
            system="offline",
            cwd=root,
            storage=JsonlSessionStorage(root / "session"),
            enable_subagents=False,
        )
    )
    graph = delta_graph(JournalCheckpointSaver(session.documents))
    configs = json.loads((root / "delta-configs.json").read_text())
    assert (await graph.aget_state(configs["original"])).values["total"] == 11
    assert (await graph.aget_state(configs["latest"])).values["total"] == 23
    assert (await graph.aget_state(configs["branch"])).values["total"] == 28
    await session.aclose()


async def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    home = root / "home"
    home.mkdir(exist_ok=True)
    os.environ.update(HOME=str(home), USERPROFILE=str(home), LANGSMITH_TRACING="false")
    provider = ScriptedChatModel([AIMessage(content="answer")])
    storage = JsonlSessionStorage(root / "session")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="offline",
            storage=storage,
            cwd=root,
            enable_subagents=False,
        )
    )
    _ = [e async for e in session.prompt("initial")]
    runtime = session._harness._runtime_state
    assert runtime is not None and runtime.durable
    saver = runtime.checkpointer
    assert isinstance(saver, JournalCheckpointSaver)
    config = {"configurable": {"thread_id": runtime.thread_id}}
    latest = await saver.aget_tuple(config)
    assert latest is not None
    history = [item async for item in saver.alist(config)]
    assert len(history) > 3
    assert history[0].checkpoint["id"] == latest.checkpoint["id"]
    assert [item async for item in saver.alist(config, limit=0)] == []
    assert all(
        item.metadata["source"] == "loop"
        for item in [item async for item in saver.alist(config, filter={"source": "loop"})]
    )
    assert [item async for item in saver.alist(config, before=latest.config)]
    assert (await asyncio.to_thread(saver.get_tuple, latest.config)).checkpoint == latest.checkpoint
    assert len(await asyncio.to_thread(lambda: list(saver.list(config)))) == len(history)
    graph = create_agent(
        provider,
        tools=list(session._harness.config.tools),
        system_prompt="offline",
        checkpointer=saver,
        middleware=[GenerationMiddleware()],
    )
    snapshot = await graph.aget_state(config)
    assert [m.content for m in snapshot.values["messages"]] == ["initial", "answer"]
    branch = await graph.aupdate_state(
        latest.config, {"messages": [HumanMessage(content="history branch")]}
    )
    assert len(provider.calls) == 1
    assert len((await graph.aget_state(branch)).values["messages"]) == 3
    assert len((await graph.aget_state(latest.config)).values["messages"]) == 2
    delta = await saver.aget_delta_channel_history(config=branch, channels=["messages"])
    assert "messages" in delta
    await saver.aput_writes(branch, [("contract-channel", 1)], "contract-task")
    await saver.aput_writes(branch, [("contract-channel", 2)], "contract-task")
    pending = (await saver.aget_tuple(branch)).pending_writes
    assert [v for t, c, v in pending if t == "contract-task" and c == "contract-channel"] == [1]
    await saver.aput_writes(branch, [("__error__", "one")], "error-task")
    await saver.aput_writes(branch, [("__error__", "two")], "error-task")
    pending = (await saver.aget_tuple(branch)).pending_writes
    assert [v for t, c, v in pending if t == "error-task" and c == "__error__"] == ["two"]
    before = (await storage.read_records())["sequence"]
    for value in [
        {"closure": lambda: None},
        {"client": object()},
        ToolMessage(
            content="controlled",
            tool_call_id="client-check",
            artifact={"client": UnsupportedClient(label="not-runtime-state")},
        ),
    ]:
        try:
            saver.encode(value)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Runtime objects must not be serializable")
    try:
        saver.decode(["pickle", ""])
    except ValueError:
        pass
    else:
        raise AssertionError("Pickle must be rejected")
    assert (await storage.read_records())["sequence"] == before
    isolated = {"configurable": {"thread_id": "contract-native", "checkpoint_ns": "isolated"}}
    copied = await saver.aput(
        isolated, latest.checkpoint, latest.metadata, latest.checkpoint["channel_versions"]
    )
    assert (await saver.aget_tuple(copied)).checkpoint == latest.checkpoint
    await saver.aput_writes(copied, [("probe", "kept")], "isolated-task")
    await saver.aput(
        isolated, latest.checkpoint, latest.metadata, latest.checkpoint["channel_versions"]
    )
    assert (await saver.aget_tuple(copied)).pending_writes == [("isolated-task", "probe", "kept")]
    await saver.aput(isolated, latest.checkpoint, latest.metadata, {})
    assert (await saver.aget_tuple(copied)).checkpoint == latest.checkpoint
    await asyncio.to_thread(saver.delete_thread, "contract-native")
    assert await saver.aget_tuple(isolated) is None
    assert await saver.aget_tuple(latest.config) is not None
    namespace_config = {
        "configurable": {"thread_id": runtime.thread_id, "checkpoint_ns": "foreign-namespace"}
    }
    await saver.aput(
        namespace_config,
        {**latest.checkpoint, "id": "zzzz-namespace"},
        latest.metadata,
        latest.checkpoint["channel_versions"],
    )
    assert (await saver.aget_tuple(config)).checkpoint["id"] == branch["configurable"][
        "checkpoint_id"
    ]
    delta_native = delta_graph(saver)
    delta_config = {"configurable": {"thread_id": "delta-contract"}}
    assert (await delta_native.ainvoke({"total": 1}, delta_config, durability="sync"))[
        "total"
    ] == 11
    original_delta = (await delta_native.aget_state(delta_config)).config
    assert (await delta_native.ainvoke({"total": 2}, delta_config, durability="sync"))[
        "total"
    ] == 23
    latest_delta = (await delta_native.aget_state(delta_config)).config
    branch_delta = await delta_native.aupdate_state(latest_delta, {"total": 5})
    (root / "delta-configs.json").write_text(
        json.dumps({"original": original_delta, "latest": latest_delta, "branch": branch_delta})
    )
    source = CodingSessionRecord(
        id="source",
        path=storage.path,
        cwd=root,
        model="fake",
        title=None,
        provider_name="fake",
        created_at=0,
        updated_at=0,
    )
    destination = CodingSessionRecord(
        id="fork",
        path=root / "fork.jsonl",
        cwd=root,
        model="fake",
        title=None,
        provider_name="fake",
        created_at=0,
        updated_at=0,
    )
    await copy_active_branch(source, session.state.active_leaf_id, destination)
    fork_storage = JsonlSessionStorage(destination.path)
    fork_model = ScriptedChatModel([AIMessage(content="fork answer")])
    fork = await CodingSession.load(
        CodingSessionConfig(
            provider=fork_model,
            model="fake",
            system="offline",
            storage=fork_storage,
            cwd=root,
            enable_subagents=False,
        )
    )
    records = await fork_storage.read_records()
    assert not records["tasks"]
    assert not [d for d in records["documents"] if d["record"]["kind"] == "forge.graph.v1"]
    assert len(fork_model.calls) == 0
    _ = [e async for e in fork.prompt("new work")]
    assert len(fork_model.calls) == 1
    input_report = await input_contract(root)
    (root / "report.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "pid": os.getpid(),
                "checkpoints": len(history),
                "channel_versions": latest.checkpoint["channel_versions"],
                "pending_write_dedup": True,
                "pickle_rejected": True,
                "runtime_objects_rejected": True,
                "branch_provider_calls": len(provider.calls),
                "fork_provider_calls_before_submit": 0,
                "fork_provider_calls_after_submit": len(fork_model.calls),
                "native_inputs": input_report,
            }
        ),
        encoding="utf-8",
    )
    await fork.aclose()
    await session.aclose()
    restart = subprocess.run(
        [sys.executable, __file__, str(root), "delta-resume"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert restart.returncode == 0, restart.stdout + restart.stderr


if __name__ == "__main__":
    asyncio.run(resume_delta(Path(sys.argv[1])) if len(sys.argv) > 2 else main(Path(sys.argv[1])))
