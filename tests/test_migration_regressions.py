"""Regression tests for the LangChain-native migration review findings.

Each test pins one reported bug so a future refactor cannot silently regress
the native-message production path:

- P1-1a cancellation persists a durable synthetic tool result (no dangling call)
- P1-1b load-time repair covers native AIMessage/ToolMessage transcripts
- P1-2  auto-compaction does not assume ``.role`` on native messages
- P1-3  ``max_turns`` maps to exactly that many assistant replies (no
        ``GRAPH_RECURSION_LIMIT`` approximation)
- P2-4  raised tool failures produce a ``ToolMessage`` with ``status="error"``
- P2-5  TUI restore tolerates a foreign native tool artifact
- P2-6  native user/assistant messages are branchable in the session tree
- G1    direct ``BaseChatModel`` callers get native messages by default
- G4    the injected ``ToolRuntime.context`` reaches the execution function
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from forge_agent import AgentHarness, AgentHarnessConfig
from forge_agent.events import ErrorEvent
from forge_agent.langchain_runtime import run_langchain_agent
from forge_agent.session import JsonlSessionStorage, LeafEntry, MessageEntry
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.session import (
    _first_recent_context_index,
    _interrupted_tool_repair_plan,
    _is_branchable_tree_entry,
    _is_tool_call_tree_entry,
)
from forge_coding.tools import ToolDefinition
from forge_coding.tui.state import TuiState


@tool
def _echo_tool(value: str) -> str:
    """Echo a value."""
    return f"echo:{value}"


class _ToolAgentChatModel(BaseChatModel):
    """Preset-response model that can play tool calls (supports ``bind_tools``)."""

    responses: list[AIMessage]
    _calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-forge-tool-agent"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        response = self.responses[min(self._calls, len(self.responses) - 1)]
        self._calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def _tool_call_ai(
    tool_call_id: str, name: str = "echo", args: dict[str, object] | None = None
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": name,
                "args": args or {"value": "x"},
                "type": "tool_call",
            }
        ],
    )


def _session_config(
    tmp_path: Path,
    provider: object,
    storage: JsonlSessionStorage,
    tools: object | None = None,
) -> CodingSessionConfig:
    return CodingSessionConfig(
        provider=provider,  # type: ignore[arg-type]
        model="fake",
        system="You are Forge.",
        storage=storage,
        cwd=tmp_path,
        tools=tools,
    )


# --------------------------------------------------------------------------- #
# P1-3: max_turns maps to exactly one assistant reply (no recursion-limit gap)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_max_turns_produces_exactly_one_assistant_reply_and_error() -> None:
    model = _ToolAgentChatModel(responses=[_tool_call_ai("call-1")])
    transcript: list[object] = []

    events = [
        event
        async for event in run_langchain_agent(
            provider=model,
            model="fake",
            system="You are Forge.",
            messages=transcript,  # type: ignore[arg-type]
            tools=[_echo_tool],
            max_turns=1,
        )
    ]

    assistant_replies = [m for m in transcript if isinstance(m, AIMessage) and m.tool_calls]
    errors = [event for event in events if isinstance(event, ErrorEvent)]

    assert model._calls == 1
    assert len(assistant_replies) == 1
    assert errors == [
        ErrorEvent(message="Agent loop stopped after reaching max_turns=1", recoverable=True)
    ]
    assert not any("RECURSION_LIMIT" in str(e.message) for e in errors)


# --------------------------------------------------------------------------- #
# P1-1b: load-time repair covers native ToolMessage/AIMessage transcripts
# --------------------------------------------------------------------------- #
def test_interrupted_tool_repair_plan_recognizes_native_messages() -> None:
    messages = (
        HumanMessage(content="Go"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call-1", "name": "read", "args": {"path": "x"}, "type": "tool_call"}
            ],
        ),
    )
    plan = _interrupted_tool_repair_plan(messages, context_entry_ids=("a", "b"))
    assert plan is not None
    parent_id, suffix = plan
    assert parent_id == "b"
    assert isinstance(suffix[0], ToolMessage)
    assert suffix[0].tool_call_id == "call-1"
    assert suffix[0].status == "error"
    assert "interrupted" in suffix[0].content


def test_repair_plan_leaves_a_balanced_native_transcript_untouched() -> None:
    messages = (
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "read", "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content="ok", tool_call_id="call-1", name="read", status="success"),
    )
    assert _interrupted_tool_repair_plan(messages, context_entry_ids=("a", "b")) is None


@pytest.mark.anyio
async def test_load_persists_repair_for_native_interrupted_tool_call(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    await storage.append(MessageEntry(message=HumanMessage(content="Read README.md")))
    await storage.append(
        MessageEntry(
            message=AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "read",
                        "args": {"path": "README.md"},
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )
    messages = [e for e in await storage.read_all() if e.type == "message"]
    await storage.append(LeafEntry(parent_id=messages[-1].id, entry_id=messages[-1].id))

    session = await CodingSession.load(
        _session_config(tmp_path, FakeListChatModel(responses=["Recovered."]), storage)
    )
    entries = await storage.read_all()
    message_entries = [entry for entry in entries if entry.type == "message"]
    repairs = [e.message for e in message_entries if isinstance(e.message, ToolMessage)]
    assert len(repairs) == 1
    assert repairs[0].tool_call_id == "call-1"
    assert repairs[0].status == "error"
    assert session.messages[-1] == repairs[0]


# --------------------------------------------------------------------------- #
# P1-1a: cancelling mid-tool persists the synthetic ToolMessage (no dangling call)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_cancel_persists_synthetic_tool_result(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    started = asyncio.Event()

    async def blocking_executor(
        arguments: dict[str, object], signal: object | None = None
    ) -> object:
        del arguments, signal
        started.set()
        await asyncio.Event().wait()  # never completes until the run is cancelled
        raise AssertionError("blocking executor must not return after cancel")

    blocking_tool = ToolDefinition(
        name="block",
        description="Blocks until cancelled.",
        prompt_snippet="Block until cancelled.",
        prompt_guidelines=(),
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        executor=blocking_executor,
    ).to_langchain_tool()

    model = _ToolAgentChatModel(
        responses=[
            _tool_call_ai("call-1", name="block", args={"value": "x"}),
            AIMessage(content="recovered"),
        ]
    )
    session = await CodingSession.load(
        _session_config(tmp_path, model, storage, tools=[blocking_tool])
    )

    async def run_prompt() -> None:
        async for _event in session.prompt("Go"):
            pass

    task = asyncio.create_task(run_prompt())
    await started.wait()
    session.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    message_entries = [e for e in await storage.read_all() if e.type == "message"]
    tool_results = [e.message for e in message_entries if isinstance(e.message, ToolMessage)]
    assert any(
        tool_result.tool_call_id == "call-1" and tool_result.status == "error"
        for tool_result in tool_results
    )


# --------------------------------------------------------------------------- #
# P1-2: compaction helpers must not assume a `.role` attribute on native msgs
# --------------------------------------------------------------------------- #
def test_first_recent_context_index_handles_native_messages() -> None:
    from forge_coding.context_window import estimate_message_tokens

    rows = (
        ("e1", HumanMessage(content="hello")),
        ("e2", AIMessage(content="world")),
        ("e3", ToolMessage(content="result", tool_call_id="call-1")),
    )
    keep = sum(estimate_message_tokens(message) * 5 for _, message in rows)
    index = _first_recent_context_index(rows, keep_recent_tokens=keep)
    # Must reach the native user message without raising AttributeError.
    assert index == 0


# --------------------------------------------------------------------------- #
# P2-4: raised tool exceptions surface as ToolMessage(status="error")
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_tool_execution_error_yields_error_status_message() -> None:
    async def boom(arguments: dict[str, object], signal: object | None = None) -> object:
        del arguments, signal
        raise RuntimeError("kaboom")

    boom_tool = ToolDefinition(
        name="boom",
        description="Always fails.",
        prompt_snippet="Always fails.",
        prompt_guidelines=(),
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        executor=boom,
    ).to_langchain_tool()

    # Direct ``ainvoke`` of the coroutine does not inject ToolRuntime; the
    # native agent graph is the production path that does.  So observe the
    # raised failure as a ``status="error"`` ToolMessage in the harness
    # transcript instead.
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=_ToolAgentChatModel(
                responses=[
                    _tool_call_ai("call-1", name="boom", args={"value": "x"}),
                    AIMessage(content="done"),
                ]
            ),
            model="fake",
            system="You are Forge.",
            tools=[boom_tool],
        )
    )
    [event async for event in harness.prompt("Go")]
    failure_messages = [
        message
        for message in harness.messages
        if isinstance(message, ToolMessage) and message.tool_call_id == "call-1"
    ]
    assert failure_messages
    assert failure_messages[0].status == "error"
    assert "kaboom" in str(failure_messages[0].content)


# --------------------------------------------------------------------------- #
# P2-5: TUI restore tolerates a third-party native tool artifact
# --------------------------------------------------------------------------- #
def test_tui_load_messages_ignores_foreign_tool_artifact() -> None:
    state = TuiState()
    foreign = ToolMessage(
        content="business data",
        tool_call_id="call-1",
        name="third_party_tool",
        artifact={"totally": ["unrelated", "business", "payload"]},
    )
    state.load_messages([foreign])  # must not raise a ValidationError
    assert any(item.role == "tool" and item.tool_call_id == "call-1" for item in state.items)


# ---------------------------------------------------------------------------
# P2-6: native HumanMessage/AIMessage entries are branchable in the tree
# ---------------------------------------------------------------------------
def test_tree_branchable_accepts_native_messages() -> None:
    answer_entry = MessageEntry(message=AIMessage(content="answer"))
    assert _is_branchable_tree_entry(answer_entry)
    user_entry = MessageEntry(message=HumanMessage(content="question"))
    assert _is_branchable_tree_entry(user_entry)
    tool_entry = MessageEntry(message=_tool_call_ai("call-1"))
    assert _is_branchable_tree_entry(tool_entry)
    assert _is_tool_call_tree_entry(tool_entry)


# ---------------------------------------------------------------------------
# P1-1c: the interrupt flag belongs to the most recent turn only (no leak into
# the next normally-completed run, which would re-persist JSONL rows)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_interrupt_flag_does_not_leak_into_next_run() -> None:
    started = asyncio.Event()

    async def blocking_executor(
        arguments: dict[str, object], signal: object | None = None
    ) -> object:
        del arguments, signal
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("blocking executor must not return after cancel")

    blocking_tool = ToolDefinition(
        name="block",
        description="Blocks until cancelled.",
        prompt_snippet="Block until cancelled.",
        prompt_guidelines=(),
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        executor=blocking_executor,
    ).to_langchain_tool()

    harness = AgentHarness(
        AgentHarnessConfig(
            provider=_ToolAgentChatModel(
                responses=[
                    _tool_call_ai("call-1", name="block", args={"value": "x"}),
                    AIMessage(content="recovered"),
                ]
            ),
            model="fake",
            system="You are Forge.",
            tools=[blocking_tool],
        )
    )

    async def run_prompt() -> None:
        async for _event in harness.prompt("Go"):
            pass

    task = asyncio.create_task(run_prompt())
    await started.wait()
    harness.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert harness.was_last_run_interrupted

    # A subsequent normally-completed run must clear the flag; the session's
    # ``finally`` flush keys off it and would otherwise re-persist messages.
    [event async for event in harness.prompt("Again")]
    assert not harness.was_last_run_interrupted


# ---------------------------------------------------------------------------
# P2-6b: model-assisted branch summaries must serialize native messages
# ---------------------------------------------------------------------------
def test_branch_summary_source_handles_native_messages() -> None:
    from forge_coding.branch_summary import (
        _branch_file_operations,
        _format_summary_source_message,
    )

    assert _format_summary_source_message(HumanMessage(content="hello")) == "[User]: hello"
    assistant = _format_summary_source_message(
        AIMessage(
            content="hi",
            tool_calls=[{"id": "c1", "name": "read", "args": {"path": "x"}, "type": "tool_call"}],
        )
    )
    assert "[Assistant]: hi" in assistant
    assert 'read(path="x")' in assistant
    tool = _format_summary_source_message(
        ToolMessage(content="data", tool_call_id="c1", name="read", status="success")
    )
    assert "[Tool result: read (ok)]: data" in tool
    failed = _format_summary_source_message(
        ToolMessage(content="boom", tool_call_id="c1", name="bash", status="error")
    )
    assert "(failed)" in failed

    read_files, modified = _branch_file_operations(
        (
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "read", "args": {"path": "a.py"}, "type": "tool_call"},
                    {"id": "c2", "name": "write", "args": {"path": "b.py"}, "type": "tool_call"},
                ],
            ),
        )
    )
    assert read_files == ["a.py"]
    assert modified == ["b.py"]


@pytest.mark.anyio
async def test_branch_summary_with_model_handles_native_messages() -> None:
    from forge_coding.branch_summary import summarize_branch_messages_with_model

    model = FakeListChatModel(responses=["A structured summary of the branch."])
    summary = await summarize_branch_messages_with_model(
        provider=model,
        model="fake",
        messages=(
            HumanMessage(content="hello"),
            AIMessage(
                content="hi",
                tool_calls=[
                    {"id": "c1", "name": "read", "args": {"path": "x"}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="data", tool_call_id="c1", name="read", status="success"),
        ),
    )
    assert summary is not None
    assert "A structured summary" in summary
    assert "read-files" in summary  # file operations extracted from native calls


# ---------------------------------------------------------------------------
# G2: the production runtime/session no longer embed legacy protocol code
# ---------------------------------------------------------------------------
def test_production_runtime_is_langchain_native() -> None:
    import inspect

    from forge_agent import harness, langchain_runtime
    from forge_coding import session

    for module in (harness, langchain_runtime, session):
        source = inspect.getsource(module)
        assert "forge_ai" not in source, f"{module.__name__} still imports forge_ai"

    runtime_source = inspect.getsource(langchain_runtime)
    for legacy_name in (
        "ForgeProviderChatModel",
        "ForgeProviderRuntimeError",
        "_langchain_tool",
        "_from_langchain_messages",
        "_to_langchain_message",
        "native_transcript",
        "ModelProvider",
    ):
        assert legacy_name not in runtime_source, f"runtime still defines {legacy_name}"

    from forge_agent import compat

    assert "ForgeProviderChatModel" in inspect.getsource(compat)
    assert "run_compat_agent" in inspect.getsource(compat)


# ---------------------------------------------------------------------------
# G4: executors consume the injected ForgeRuntimeContext (workspace + prefix)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tool_executor_uses_injected_context_workspace(tmp_path: Path) -> None:
    from forge_coding.session import CodingSession
    from forge_coding.session import CodingSessionConfig as SessionConfig
    from forge_coding.tools import create_read_tool

    session_dir = tmp_path / "session-workspace"
    session_dir.mkdir()
    (session_dir / "target.txt").write_text("from session workspace", encoding="utf-8")
    other_dir = tmp_path / "other-workspace"
    other_dir.mkdir()

    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        SessionConfig(
            provider=_ToolAgentChatModel(
                responses=[
                    _tool_call_ai("call-1", name="read", args={"path": "target.txt"}),
                    AIMessage(content="done"),
                ]
            ),
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=session_dir,
            # Tool created with a different cwd than the session: the injected
            # runtime context must win, otherwise the read would fail.
            tools=[create_read_tool(cwd=other_dir)],
        )
    )

    [event async for event in session.prompt("Go")]
    tool_messages = [
        m for m in session.messages if isinstance(m, ToolMessage) and m.tool_call_id == "call-1"
    ]
    assert tool_messages
    assert tool_messages[0].status == "success"
    assert "from session workspace" in tool_messages[0].content
    artifact = tool_messages[0].artifact or {}
    assert (artifact.get("details") or {}).get("workspace_root") == str(session_dir)


# ---------------------------------------------------------------------------
# F3: TUI user-message recognition covers native HumanMessage rows
# ---------------------------------------------------------------------------
def test_tui_user_message_helpers_accept_native_messages() -> None:
    from forge_agent.events import MessageEndEvent
    from forge_coding.tui.app import (
        _is_user_message_end_event,
    )

    assert _is_user_message_end_event(MessageEndEvent(message=HumanMessage(content="hi")))
    assert not _is_user_message_end_event(MessageEndEvent(message=AIMessage(content="yo")))


# ---------------------------------------------------------------------------
# G1: a harness built on a chat model keeps native messages by default
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_harness_defaults_to_native_messages_for_chat_model() -> None:
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=FakeListChatModel(responses=["hello"]),
            model="fake",
            system="You are Forge.",
        )
    )
    [event async for event in harness.prompt("hi")]
    assert isinstance(harness.messages[-1], AIMessage)


# ---------------------------------------------------------------------------
# G4: the injected ToolRuntime.context reaches the tool execution boundary
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tool_runtime_context_reaches_result_details(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")

    async def capture(arguments: dict[str, object], signal: object | None = None) -> object:
        del arguments, signal
        from forge_agent.tools import AgentToolResult

        return AgentToolResult(
            tool_call_id="call-1",
            name="capture",
            ok=True,
            content="captured",
        )

    capture_tool = ToolDefinition(
        name="capture",
        description="Captures runtime context into the result.",
        prompt_snippet="Capture runtime context.",
        prompt_guidelines=(),
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        executor=capture,
    ).to_langchain_tool()

    model = _ToolAgentChatModel(
        responses=[
            _tool_call_ai("call-1", name="capture", args={"value": "x"}),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        _session_config(tmp_path, model, storage, tools=[capture_tool])
    )
    # The harness forwards ForgeRuntimeContext into the graph's ToolRuntime.
    assert session._harness.config.runtime_context is not None
    events = [event async for event in session.prompt("Go")]
    assert any(event.type == "tool_execution_end" for event in events)

    tool_messages = [
        m for m in session.messages if isinstance(m, ToolMessage) and m.artifact is not None
    ]
    assert tool_messages
    details = tool_messages[0].artifact.get("details") or {}
    assert details.get("workspace_root") == str(tmp_path)
    assert details.get("session_id") is None  # session_id unset in this config
