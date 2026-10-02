"""Offline review regressions through real session and child graphs.

Failure contract: canonical model names retain colons; owned clients close on
completion, cancellation, failed summary/preparation/construction, while borrowed
parent clients stay alive; signatures survive only their originating candidate.
Capacity must count retained native content and signatures before the next call;
cancellation during client close must drain every owned client before propagating,
including repeated cancellation, without closing the borrowed parent.
"""

import asyncio
import base64
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from conftest import isolate_home
from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent import SubagentModelCandidate, SubagentRunner, SubagentRuntime, SubagentSpec
from forge_agent.session import JsonlSessionStorage
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.providers.config import OpenAICompatibleProviderConfig, ProviderSettings


class ClosingModel(ScriptedChatModel):
    close_count: int = 0
    fail_calls: tuple[int, ...] = ()

    async def aclose(self):
        self.close_count += 1
        await super().aclose()

    async def _agenerate(self, messages, **kwargs):
        self._record_call(messages)
        response = self._next_response()
        if len(self.calls) in self.fail_calls:
            raise RuntimeError("temporary 503")
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=response)])


async def consume(events):
    return [event async for event in events]


def role(tmp_path, *, model="gateway/child", fallback="", context="fresh"):
    path = tmp_path / ".forge/agents/oracle"
    path.mkdir(parents=True)
    (path / "AGENT.md").write_text(
        f"---\ndescription: Offline review\nmodel: {model}\ncontext: {context}\n"
        + (f"fallback-models: {fallback}\n" if fallback else "")
        + "---\nReport evidence.",
        encoding="utf-8",
    )


async def load(tmp_path, parent, models, *, catalog=True):
    config = OpenAICompatibleProviderConfig(
        name="gateway",
        models=tuple(models),
        default_model="parent",
        api_key_env="FORGE_OFFLINE_REVIEW_KEY",
    )
    return await CodingSession.load(
        CodingSessionConfig(
            cwd=tmp_path,
            provider=parent,
            provider_name="gateway",
            model="parent",
            tools=[],
            system="Offline",
            trust_override="yes",
            auto_compact_enabled=False,
            storage=JsonlSessionStorage(tmp_path / "parent.jsonl"),
            provider_settings=ProviderSettings(providers=(config,), default_provider="gateway")
            if catalog
            else None,
        )
    )


@pytest.mark.parametrize(
    "primary,fallback",
    [
        ("vendor/model:free", "backup"),
        ("child", "vendor/backup:free"),
    ],
)
def test_canonical_colon_models_launch_primary_and_fallback(
    tmp_path, monkeypatch, primary, fallback
):
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv("FORGE_OFFLINE_REVIEW_KEY", "synthetic")
    role(tmp_path, model=f"gateway/{primary}", fallback=f"gateway/{fallback}")
    parent = ClosingModel(
        [
            tool_call_ai("task", "task", {"agent": "oracle", "instruction": "inspect"}),
            AIMessage(content="parent done"),
        ]
    )
    built = []

    def build(config, *, model, **kwargs):
        child = ClosingModel([AIMessage(content="answer")])
        if model == primary:
            child.fail_calls = (1,)
        built.append((model, child))
        return child

    monkeypatch.setattr("forge_coding.sessions.model_selection.create_model_provider", build)

    async def run():
        session = await load(tmp_path, parent, ("parent", primary, fallback))
        try:
            await consume(session.prompt("delegate"))
            results = [m for m in session.messages if isinstance(m, ToolMessage)]
            assert results[0].artifact["data"]["status"] == "completed"
            assert [name for name, _ in built] == [primary, fallback]
            assert all(len(child.calls) == 1 for _, child in built)
            assert session.model == "parent" and parent.close_count == 0
        finally:
            await session.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["none", "summary", "prepare", "construct"])
def test_invocation_clients_release_before_session_close(tmp_path, monkeypatch, failure):
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv("FORGE_OFFLINE_REVIEW_KEY", "synthetic")
    role(
        tmp_path,
        fallback="gateway/backup" if failure == "construct" else "",
        context="summary" if failure == "summary" else "fresh",
    )
    count = 3 if failure == "none" else 1
    responses = []
    for index in range(count):
        responses.append(
            tool_call_ai(f"task-{index}", "task", {"agent": "oracle", "instruction": "inspect"})
        )
        if failure == "summary":
            responses.append(AIMessage(content=""))
        responses.append(AIMessage(content="parent done"))
    parent = ClosingModel(responses)
    children = []

    def build(config, *, model, **kwargs):
        if model == "backup":
            raise RuntimeError("synthetic construction failure")
        child = ClosingModel([AIMessage(content="answer")])
        children.append(child)
        return child

    monkeypatch.setattr("forge_coding.sessions.model_selection.create_model_provider", build)
    if failure == "prepare":
        monkeypatch.setattr(
            SubagentRunner,
            "prepare",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("synthetic preparation failure")),
        )

    async def run():
        session = await load(tmp_path, parent, ("parent", "child", "backup"))
        baseline = len(session._owned_providers)
        try:
            for _ in range(count):
                await consume(session.prompt("delegate"))
                assert children and all(child.close_count == 1 for child in children)
                assert len(session._owned_providers) == baseline
                assert parent.close_count == 0
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "case": failure,
                        "pass": True,
                        "owned_clients": len(children),
                        "closed_clients": sum(child.close_count for child in children),
                        "parent_closed_before_session_close": parent.close_count,
                    }
                ),
                encoding="utf-8",
            )
        finally:
            await session.aclose()
        assert all(child.close_count == 1 for child in children)

    asyncio.run(run())


def test_borrowed_parent_is_not_closed_by_child(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    role(tmp_path, model="inherit")
    parent = ClosingModel(
        [
            tool_call_ai("task", "task", {"agent": "oracle", "instruction": "inspect"}),
            AIMessage(content="child"),
            AIMessage(content="parent done"),
        ]
    )

    async def run():
        session = await load(tmp_path, parent, ("parent",), catalog=False)
        try:
            await consume(session.prompt("delegate"))
            assert len(parent.calls) == 3 and parent.close_count == 0
        finally:
            await session.aclose()

    asyncio.run(run())


def test_foreground_cancel_releases_owned_client(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv("FORGE_OFFLINE_REVIEW_KEY", "synthetic")
    role(tmp_path)

    class BlockingModel(ClosingModel):
        async def _agenerate(self, messages, **kwargs):
            self._record_call(messages)
            self.started.set()
            await asyncio.Event().wait()

    async def run():
        child = BlockingModel()
        object.__setattr__(child, "started", asyncio.Event())
        parent = ClosingModel(
            [tool_call_ai("cancel-task", "task", {"agent": "oracle", "instruction": "inspect"})]
        )
        monkeypatch.setattr(
            "forge_coding.sessions.model_selection.create_model_provider", lambda *a, **k: child
        )
        session = await load(tmp_path, parent, ("parent", "child"))
        try:
            prompt = asyncio.create_task(consume(session.prompt("delegate")))
            await asyncio.wait_for(child.started.wait(), 5)
            session.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(prompt, 5)
            assert child.close_count == 1 and parent.close_count == 0
            results = [m for m in session.messages if isinstance(m, ToolMessage)]
            assert [m.tool_call_id for m in results] == ["cancel-task"]
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "case": "foreground_cancel",
                        "pass": True,
                        "closed_clients": 1,
                        "paired_calls": 1,
                        "parent_closed_before_session_close": 0,
                    }
                ),
                encoding="utf-8",
            )
        finally:
            await session.aclose()

    asyncio.run(run())


@pytest.mark.anyio
@pytest.mark.parametrize("hop", [False, True])
async def test_native_signatures_stay_with_originating_candidate(tmp_path, hop):
    signature_key = "__gemini_function_call_thought_signatures__"

    def signed(name):
        message = tool_call_ai(name, "write_marker", {"value": name})
        message.id = f"message-{name}"
        message.additional_kwargs[signature_key] = {
            name: base64.b64encode(f"signature-{name}".encode()).decode()
        }
        return message

    @tool
    def write_marker(value: str) -> str:
        """Append one synthetic marker."""
        with (tmp_path / "writes.txt").open("a", encoding="utf-8") as stream:
            stream.write(value + "\n")
        return value

    primary = ClosingModel([signed("a"), AIMessage(content="done")])
    fallback = ClosingModel([signed("b"), AIMessage(content="done")])
    if hop:
        primary.fail_calls = (2, 3)
    candidates = (
        SubagentModelCandidate(primary, "same", "provider-a"),
        SubagentModelCandidate(fallback, "same", "provider-b"),
    )
    runtime = SubagentRuntime(primary, model="same", candidates=candidates)
    runner = SubagentRunner(
        lambda: runtime,
        [
            SubagentSpec(
                "oracle",
                "inspect",
                "Offline",
                tools=(write_marker,),
                model="provider-a/same",
                fallback_models=("provider-b/same",) if hop else (),
            )
        ],
    )
    history = [
        HumanMessage(content="parent"),
        signed("parent"),
        ToolMessage(content="parent result", tool_call_id="parent"),
    ]
    result = await runner.run("oracle", "inspect", context="fork", parent_messages=history)
    assert result.status == "completed"
    from langchain_google_genai.chat_models import _parse_chat_history

    own = next(
        m
        for m in primary.calls[1]["messages"]
        if isinstance(m, AIMessage) and m.tool_calls and m.tool_calls[0]["id"] == "a"
    )
    assert own.additional_kwargs[signature_key] == signed("a").additional_kwargs[signature_key]
    _, encoded = _parse_chat_history(primary.calls[1]["messages"], model="gemini-3-flash")
    assert any(
        part.thought_signature == b"signature-a" for content in encoded for part in content.parts
    )
    inherited = next(
        m
        for m in primary.calls[0]["messages"]
        if isinstance(m, AIMessage) and m.tool_calls and m.tool_calls[0]["id"] == "parent"
    )
    assert signed("parent").additional_kwargs[signature_key]
    assert inherited.additional_kwargs == {}
    if hop:
        for call in fallback.calls:
            assert all(
                signature_key not in m.additional_kwargs
                for m in call["messages"]
                if isinstance(m, AIMessage) and m.tool_calls and m.tool_calls[0]["id"] == "a"
            )
        own_b = next(
            m
            for m in fallback.calls[1]["messages"]
            if isinstance(m, AIMessage) and m.tool_calls and m.tool_calls[0]["id"] == "b"
        )
        assert (
            own_b.additional_kwargs[signature_key] == signed("b").additional_kwargs[signature_key]
        )
        assert (tmp_path / "writes.txt").read_text().splitlines() == ["a", "b"]
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "case": "signature_hopping" if hop else "signature_same_candidate",
                "pass": True,
                "original_signature_encoded": True,
                "parent_metadata_removed": True,
                "foreign_metadata_removed": hop,
                "attempts": len(result.attempts),
                "tool_calls": result.tool_calls,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("payload", ["inline", "reasoning_block", "signature"])
@pytest.mark.parametrize("window", [2000, 16000])
async def test_native_request_capacity_counts_sent_content(tmp_path, payload, window):
    marker = tmp_path / "writes.txt"

    @tool
    def write_marker() -> str:
        """Append one offline marker."""
        with marker.open("a", encoding="utf-8") as output:
            output.write("once\n")
        return "written"

    response = tool_call_ai("write-1", "write_marker", {})
    if payload == "inline":
        response.content = "<think>" + "x" * 4000 + "</think>"
    elif payload == "reasoning_block":
        response.content = [{"type": "reasoning", "reasoning": "x" * 4000}]
    else:
        response.additional_kwargs["__gemini_function_call_thought_signatures__"] = {
            "write-1": base64.b64encode(b"x" * 4000).decode()
        }
    model = ClosingModel([response, AIMessage(content="done")])
    runtime = SubagentRuntime(
        model,
        "child",
        candidates=(SubagentModelCandidate(model, "child", "offline", window, 128),),
    )
    runner = SubagentRunner(
        lambda: runtime,
        [SubagentSpec("worker", "inspect", "Offline", tools=(write_marker,))],
    )
    result = await runner.run("worker", "write once")
    overflow = window == 2000
    assert result.status == ("failed" if overflow else "completed")
    assert len(model.calls) == (1 if overflow else 2)
    assert marker.read_text(encoding="utf-8").splitlines() == ["once"]
    assert result.tool_calls == 1
    if overflow:
        assert "window" in result.error
    else:
        sent = next(m for m in model.calls[1]["messages"] if isinstance(m, AIMessage))
        assert sent.content == response.content
        assert sent.additional_kwargs == response.additional_kwargs
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "case": "native_capacity",
                "pass": True,
                "payload": payload,
                "window": window,
                "status": result.status,
                "attempts": len(result.attempts),
                "tool_calls": result.tool_calls,
                "artifacts": ["writes.txt"],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("repeat_cancel", [False, True])
def test_foreground_cancel_during_close_drains_all_clients(tmp_path, monkeypatch, repeat_cancel):
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv("FORGE_OFFLINE_REVIEW_KEY", "synthetic")
    role(tmp_path, fallback="gateway/backup")

    class BlockingCloseModel(ClosingModel):
        async def aclose(self):
            self.closing.set()
            await self.release.wait()
            await super().aclose()

    async def run():
        child = ClosingModel([AIMessage(content="child done")])
        backup = BlockingCloseModel()
        object.__setattr__(backup, "closing", asyncio.Event())
        object.__setattr__(backup, "release", asyncio.Event())
        parent = ClosingModel(
            [
                tool_call_ai("close-task", "task", {"agent": "oracle", "instruction": "inspect"}),
                AIMessage(content="parent done"),
            ]
        )
        monkeypatch.setattr(
            "forge_coding.sessions.model_selection.create_model_provider",
            lambda config, model, **kwargs: child if model == "child" else backup,
        )
        session = await load(tmp_path, parent, ("parent", "child", "backup"))
        task_tool = next(tool for tool in session._harness.config.tools if tool.name == "task")
        prompt = asyncio.create_task(
            task_tool.ainvoke(
                {
                    "type": "tool_call",
                    "id": "close-task",
                    "name": "task",
                    "args": {"agent": "oracle", "instruction": "inspect"},
                }
            )
        )
        try:
            await asyncio.wait_for(backup.closing.wait(), 5)
            prompt.cancel()
            if repeat_cancel:
                await asyncio.sleep(0)
                prompt.cancel()
            done, _ = await asyncio.wait({prompt}, timeout=0.1)
            assert not done, "Cancellation must wait until client cleanup settles"
            backup.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(prompt, 5)
            assert child.close_count == backup.close_count == 1
            assert parent.close_count == 0
            (tmp_path / "summary.json").write_text(
                json.dumps(
                    {
                        "case": "cancel_during_close",
                        "pass": True,
                        "repeat_cancel": repeat_cancel,
                        "closed_clients": child.close_count + backup.close_count,
                        "parent_closed_before_session_close": parent.close_count,
                        "child_calls": len(child.calls),
                    }
                ),
                encoding="utf-8",
            )
        finally:
            backup.release.set()
            await asyncio.gather(prompt, return_exceptions=True)
            await session.aclose()
        assert child.close_count == backup.close_count == 1

    asyncio.run(run())
