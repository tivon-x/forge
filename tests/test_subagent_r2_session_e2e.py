"""R2 offline session acceptance with real parent/child LangChain graphs.

Failure contract: role models must not switch the parent; fork must exclude the
whole pending delegation batch and hidden metadata; fresh must omit parent
facts; unavailable models and failed summaries must not launch a child; summary
usage must survive failure. Artifacts contain synthetic messages only.
"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from conftest import isolate_home
from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent.events import ToolExecutionUpdateEvent
from forge_agent.session import CustomEntry, JsonlSessionStorage
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.providers.config import OpenAICompatibleProviderConfig, ProviderSettings
from forge_coding.sessions.usage import USAGE_NAMESPACE


async def consume(events):
    return [event async for event in events]


def profile(tmp_path, name, context, model="test/child"):
    path = tmp_path / ".forge" / "agents" / name
    path.mkdir(parents=True)
    (path / "AGENT.md").write_text(
        f"---\ndescription: Offline role\nmodel: {model}\ncontext: {context}\n"
        "---\nReport the requested fact.",
        encoding="utf-8",
    )


@pytest.mark.parametrize("context", ["fresh", "fork", "summary"])
def test_role_context_and_usage(tmp_path, monkeypatch, context):
    isolate_home(monkeypatch, tmp_path)
    profile(tmp_path, "oracle", context)
    profile(tmp_path, "second", "fresh", "test/second")
    monkeypatch.setenv("FORGE_OFFLINE_TEST_KEY", "synthetic")
    child = ScriptedChatModel(
        [
            AIMessage(
                content="CHILD_FACT",
                usage_metadata={
                    "input_tokens": 11,
                    "output_tokens": 3,
                    "total_tokens": 14,
                },
            )
        ]
    )
    second = ScriptedChatModel([AIMessage(content="SECOND_FACT")])
    parent = ScriptedChatModel(
        [
            tool_call_ai("prior", "task", {"agent": "second", "instruction": "earlier task"}),
            AIMessage(content="remembered", additional_kwargs={"reasoning_content": "HIDDEN"}),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "delegate",
                        "name": "task",
                        "args": {"agent": "oracle", "instruction": "child task"},
                        "type": "tool_call",
                    },
                    {
                        "id": "second",
                        "name": "task",
                        "args": {"agent": "second", "instruction": "second task"},
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(
                content="TASK_BRIEF",
                usage_metadata={
                    "input_tokens": 7,
                    "output_tokens": 2,
                    "total_tokens": 9,
                },
            )
            if context == "summary"
            else AIMessage(content="PARENT_DONE"),
            AIMessage(content="PARENT_DONE"),
        ]
    )
    built = []

    def build(config, **kwargs):
        built.append((config.max_retries, kwargs["model"]))
        return child if kwargs["model"] == "child" else second

    monkeypatch.setattr("forge_coding.sessions.model_selection.create_model_provider", build)
    provider = OpenAICompatibleProviderConfig(
        name="test",
        models=("parent", "child", "second"),
        default_model="parent",
        api_key_env="FORGE_OFFLINE_TEST_KEY",
        context_windows={"child": 128000},
    )

    async def run():
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=parent,
                provider_name="test",
                model="parent",
                provider_settings=ProviderSettings(providers=(provider,), default_provider="test"),
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
                tools=[],
                system="Offline",
                trust_override="yes",
                auto_compact_enabled=False,
            )
        )
        try:
            await consume(session.prompt("PARENT_SECRET_FACT"))
            events = await consume(session.prompt("delegate now"))
            assert sorted(built) == [(0, "child"), (0, "second"), (0, "second")]
            assert session.model == "parent"
            assert child.calls
            assert second.calls
            paired = [m for m in session.messages if isinstance(m, ToolMessage)]
            assert {m.tool_call_id for m in paired} == {"prior", "delegate", "second"}
            assert all(m.artifact["data"]["status"] == "completed" for m in paired)
            inherited = child.calls[0]["messages"]
            content = repr(inherited)
            assert ("PARENT_SECRET_FACT" in content) == (context == "fork")
            assert "HIDDEN" not in content
            inherited_calls = {
                call["id"] for m in inherited if isinstance(m, AIMessage) for call in m.tool_calls
            }
            inherited_results = {m.tool_call_id for m in inherited if isinstance(m, ToolMessage)}
            assert (
                inherited_calls == inherited_results == ({"prior"} if context == "fork" else set())
            )
            traces = [
                event.data
                for event in events
                if isinstance(event, ToolExecutionUpdateEvent)
                and event.tool_call_id == "delegate"
                and event.data
                and event.data.get("kind") == "subagent_trace"
            ]
            assert traces and not any(
                item["kind"] in {"tool_call", "tool_result"} for item in traces[-1]["items"]
            )
            assert "TASK_BRIEF" in content if context == "summary" else True
            entries = await session._config.storage.read_all()
            usages = [
                e.data
                for e in entries
                if isinstance(e, CustomEntry) and e.namespace == USAGE_NAMESPACE
            ]
            actual = [u for u in usages if u["purpose"] == "subagent"]
            child_usage = [u for u in actual if u["requested_model"] == "child"]
            assert len(child_usage) == 1
            assert child_usage[0]["provider"] == "test"
            assert child_usage[0]["total_tokens"] == 14
            assert {u["requested_model"] for u in actual} == {"child", "second"}
            if context == "summary":
                assert any(
                    u["purpose"] == "subagent_summary" and u["total_tokens"] == 9 for u in usages
                )
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "case": context,
                        "pass": True,
                        "status": "completed",
                        "paired_calls": len(paired),
                        "child_calls": len(child.calls),
                        "residual_tasks": 0,
                        "artifacts": ["parent.jsonl"],
                    }
                ),
                encoding="utf-8",
            )
        finally:
            await session.aclose()
        assert child.closed
        assert second.closed

    asyncio.run(run())


def test_unconfigured_role_does_not_construct_or_launch(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    profile(tmp_path, "oracle", "fresh", "unconfigured/child")
    parent = ScriptedChatModel(
        [
            tool_call_ai("bad", "task", {"agent": "oracle", "instruction": "child task"}),
            AIMessage(content="PARENT_DONE"),
        ]
    )

    async def run():
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=parent,
                model="parent",
                tools=[],
                system="Offline",
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
                trust_override="yes",
                auto_compact_enabled=False,
            )
        )
        try:
            await consume(session.prompt("launch"))
            results = [m for m in session.messages if isinstance(m, ToolMessage)]
            assert results and "configured" in str(results[0].content).lower()
            assert len(parent.calls) == 2
        finally:
            await session.aclose()

    asyncio.run(run())


def test_failed_summary_keeps_usage_and_does_not_launch_child(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    profile(tmp_path, "oracle", "summary", "inherit")
    parent = ScriptedChatModel(
        [
            tool_call_ai("bad-summary", "task", {"agent": "oracle", "instruction": "child task"}),
            AIMessage(
                content="",
                usage_metadata={
                    "input_tokens": 7,
                    "output_tokens": 0,
                    "total_tokens": 7,
                },
            ),
            AIMessage(content="PARENT_DONE"),
        ]
    )

    async def run():
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=parent,
                model="parent",
                tools=[],
                system="Offline",
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
                trust_override="yes",
                auto_compact_enabled=False,
            )
        )
        try:
            await consume(session.prompt("FACT"))
            results = [m for m in session.messages if isinstance(m, ToolMessage)]
            assert results and "summary" in str(results[0].content).lower()
            assert len(parent.calls) == 3
            entries = await session._config.storage.read_all()
            assert any(
                isinstance(e, CustomEntry)
                and e.namespace == USAGE_NAMESPACE
                and e.data["purpose"] == "subagent_summary"
                and e.data["total_tokens"] == 7
                for e in entries
            )
        finally:
            await session.aclose()

    asyncio.run(run())


def test_interrupted_summary_stream_keeps_reported_usage(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    profile(tmp_path, "oracle", "summary", "inherit")

    class InterruptedSummary(ScriptedChatModel):
        async def _astream(self, messages, **kwargs):
            self._record_call(messages)
            if "context summarization assistant" in str(messages[0].content):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="partial",
                        usage_metadata={
                            "input_tokens": 7,
                            "output_tokens": 1,
                            "total_tokens": 8,
                        },
                    )
                )
                raise RuntimeError("503 synthetic summary stream failure")
            response = self._next_response()
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

    parent = InterruptedSummary(
        [
            tool_call_ai(
                "summary-stream", "task", {"agent": "oracle", "instruction": "child task"}
            ),
            AIMessage(content="PARENT_DONE"),
        ]
    )

    async def run():
        session = await CodingSession.load(
            CodingSessionConfig(
                cwd=tmp_path,
                provider=parent,
                model="parent",
                tools=[],
                system="Offline",
                storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
                trust_override="yes",
                auto_compact_enabled=False,
            )
        )
        try:
            await consume(session.prompt("FACT"))
            assert len(parent.calls) == 3
            entries = await session._config.storage.read_all()
            assert any(
                isinstance(e, CustomEntry)
                and e.namespace == USAGE_NAMESPACE
                and e.data["purpose"] == "subagent_summary"
                and e.data["total_tokens"] == 8
                for e in entries
            )
        finally:
            await session.aclose()

    asyncio.run(run())
