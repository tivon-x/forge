import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)

from conftest import isolate_home
from fake_models import (
    ScriptedChatModel,
    ScriptedErrorChatModel,
    ThrowingChatModel,
    message_signatures,
    message_texts,
    tool_call_ai,
)
from forge_agent import QueueUpdateEvent
from forge_agent.session import (
    CompactionEntry,
    JsonlSessionStorage,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    ThinkingLevelChangeEntry,
)
from forge_coding import (
    CodingSession,
    CodingSessionConfig,
    FileCredentialStore,
    ForgePaths,
    ForgeResourcePaths,
    ModelChoice,
    OpenAICodexProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderConfigError,
    ProviderSettings,
    ScopedModelConfig,
    SessionManager,
    SessionTreeBranchResult,
    load_provider_settings,
    save_provider_settings,
)
from forge_coding import session as coding_session_module
from forge_coding.session import (
    _first_recent_context_index,
    _interrupted_tool_repair_plan,
    _is_branchable_tree_entry,
    _is_tool_call_tree_entry,
    _ordered_tree_entries,
    parse_terminal_command,
)
from forge_coding.tools import ToolDefinition


async def _collect_session_events(session_stream: object) -> list[object]:
    return [event async for event in session_stream]  # type: ignore[attr-defined]


def _config(
    tmp_path: Path,
    provider: BaseChatModel,
    storage: JsonlSessionStorage,
    tools: Any = None,
) -> CodingSessionConfig:
    return CodingSessionConfig(
        provider=provider,
        model="fake",
        system="You are Forge.",
        storage=storage,
        cwd=tmp_path,
        tools=tools,
    )


class SwitchableChatModel(BaseChatModel):
    """Replaceable provider used to swap session models under test."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.swapped = False

    @property
    def _llm_type(self) -> str:
        return "forge-switchable-chat"

    async def aclose(self) -> None:
        self.closed = True

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        content = "Generated" if self.switch else "Pre-change"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class RaisingChatModel(BaseChatModel):
    def __init__(self, fail_on_call: int = 1, success_content: str = "Generated title") -> None:
        super().__init__()
        object.__setattr__(self, "fail_on_call", fail_on_call)
        object.__setattr__(self, "success_content", success_content)
        object.__setattr__(self, "call_count", 0)

    @property
    def _llm_type(self) -> str:
        return "forge-raising-chat"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        count = int(getattr(self, "call_count", 0)) + 1
        object.__setattr__(self, "call_count", count)
        if count == getattr(self, "fail_on_call", 1):
            raise RuntimeError("provider exploded")
        message = AIMessage(content=getattr(self, "success_content", "Generated title"))
        return ChatResult(generations=[ChatGeneration(message=message)])


class WaitingChatModel(BaseChatModel):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "started", asyncio.Event())
        object.__setattr__(self, "release", asyncio.Event())
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "call_count", 0)

    @property
    def _llm_type(self) -> str:
        return "forge-waiting-chat"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del stop, run_manager, kwargs
        call_index = int(getattr(self, "call_count", 0))
        object.__setattr__(self, "call_count", call_index + 1)
        self.calls.append(list(messages))
        if call_index == 0:
            self.started.set()
            await self.release.wait()
            text = "First"
        else:
            text = "Second"
        yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del stop, run_manager, kwargs
        call_index = int(getattr(self, "call_count", 0))
        object.__setattr__(self, "call_count", call_index + 1)
        self.calls.append(list(messages))
        text = "First" if call_index == 0 else "Second"
        if call_index == 0:
            self.started.set()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class CancellableWaitingChatModel(BaseChatModel):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "started", asyncio.Event())
        object.__setattr__(self, "release", asyncio.Event())
        object.__setattr__(self, "calls", [])

    @property
    def _llm_type(self) -> str:
        return "forge-cancellable-waiting-chat"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del stop, run_manager, kwargs
        self.calls.append(list(messages))
        self.started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.005)
        yield ChatGenerationChunk(message=AIMessageChunk(content="Finished"))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del stop, run_manager, kwargs
        self.calls.append(list(messages))
        self.started.set()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Finished"))])


@pytest.mark.anyio
async def test_load_empty_session_defers_transcript_file(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")

    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    entries = await storage.read_all()
    assert entries == []
    assert not storage.path.exists()
    assert session.messages == ()
    assert session.state.model == "fake"
    assert session.thinking_level == "medium"
    assert session.available_thinking_levels == ("off", "minimal", "low", "medium", "high", "xhigh")
    assert session.cwd == tmp_path
    assert session.model == "fake"
    assert [tool.name for tool in session.tools] == ["read", "write", "edit", "bash"]


@pytest.mark.anyio
async def test_session_export_defaults_to_cwd(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / ".forge" / "sessions" / "session-1.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    await storage.append(MessageEntry(id="root", message=HumanMessage(content="Export me")))

    output_path = await session.export()

    assert output_path == tmp_path / "session-1.html"
    html = output_path.read_text(encoding="utf-8")
    assert "Export me" in html
    assert str(storage.path) in html


@pytest.mark.anyio
async def test_session_export_writes_jsonl_to_destination_directory(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / ".forge" / "sessions" / "session-1.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    await storage.append(MessageEntry(id="root", message=HumanMessage(content="Export me")))

    output_path = await session.export(Path("exports"), format="jsonl")

    assert output_path == tmp_path / "exports" / "session-1.jsonl"
    assert "Export me" in output_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_prompt_logs_unexpected_agent_call_exception(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    forge_paths = ForgePaths(home=tmp_path / "forge-home", agents_home=tmp_path / "agents-home")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ThrowingChatModel(error="provider exploded"),
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            provider_name="fake-provider",
            session_id="session-1",
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    await _collect_session_events(session.prompt("Hello"))

    log_path = forge_paths.agent_calls_log_path
    assert session.last_diagnostic_log_path == log_path
    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["kind"] == "error_event"
    assert entry["phase"] == "agent_loop"
    assert entry["provider_name"] == "fake-provider"
    assert entry["model"] == "fake"
    assert entry["session_id"] == "session-1"
    assert entry["cwd"] == str(tmp_path)
    assert entry["error"] == {"message": "provider exploded", "recoverable": False}
    assert "Hello" not in log_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_prompt_logs_error_event_diagnostic_data(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    forge_paths = ForgePaths(home=tmp_path / "forge-home", agents_home=tmp_path / "agents-home")
    provider = ThrowingChatModel(error="provider failed")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            provider_name="fake-provider",
            session_id="session-1",
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    await _collect_session_events(session.prompt("Hello"))

    log_path = forge_paths.agent_calls_log_path
    assert session.last_diagnostic_log_path == log_path
    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["kind"] == "error_event"
    assert entry["error"] == {
        "message": "provider failed",
        "recoverable": False,
    }
    assert "Hello" not in log_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_load_persists_repair_for_session_with_interrupted_tail_tool_call(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    user_entry = MessageEntry(message=HumanMessage(content="Read README.md"))
    await storage.append(user_entry)
    assistant_entry = MessageEntry(
        parent_id=user_entry.id,
        message=AIMessage(
            content="I'll read it.",
            tool_calls=[
                {"id": "call-1", "name": "read", "args": {"path": "README.md"}, "type": "tool_call"}
            ],
        ),
    )
    await storage.append(assistant_entry)
    await storage.append(LeafEntry(parent_id=assistant_entry.id, entry_id=assistant_entry.id))

    provider = ScriptedChatModel([AIMessage(content="Recovered.")])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
        )
    )

    assert provider.calls == []
    assert message_signatures(session.messages) == [
        ("human", "Read README.md", (), None),
        ("ai", "I'll read it.", ("call-1",), None),
        ("tool", "Tool call interrupted by user", (), "call-1"),
    ]

    entries = await storage.read_all()
    message_entries = [entry for entry in entries if entry.type == "message"]
    assert message_signatures([entry.message for entry in message_entries]) == [
        ("human", "Read README.md", (), None),
        ("ai", "I'll read it.", ("call-1",), None),
        ("tool", "Tool call interrupted by user", (), "call-1"),
    ]


@pytest.mark.anyio
async def test_load_persists_repair_for_historical_interrupted_tool_call(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    user_entry = MessageEntry(message=HumanMessage(content="Read README.md"))
    await storage.append(user_entry)
    assistant_entry = MessageEntry(
        parent_id=user_entry.id,
        message=AIMessage(
            content="I'll read it.",
            tool_calls=[
                {"id": "call-1", "name": "read", "args": {"path": "README.md"}, "type": "tool_call"}
            ],
        ),
    )
    await storage.append(assistant_entry)
    continued_entry = MessageEntry(
        parent_id=assistant_entry.id,
        message=HumanMessage(content="continue"),
    )
    await storage.append(continued_entry)
    await storage.append(LeafEntry(parent_id=continued_entry.id, entry_id=continued_entry.id))

    provider = ScriptedChatModel()
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
        )
    )

    assert provider.calls == []
    assert message_signatures(session.messages) == [
        ("human", "Read README.md", (), None),
        ("ai", "I'll read it.", ("call-1",), None),
        ("tool", "Tool call interrupted by user", (), "call-1"),
        ("human", "continue", (), None),
    ]

    entries = await storage.read_all()
    message_entries = [entry for entry in entries if entry.type == "message"]
    assert message_signatures([entry.message for entry in message_entries[-2:]]) == [
        ("tool", "Tool call interrupted by user", (), "call-1"),
        ("human", "continue", (), None),
    ]

    restored = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
        )
    )
    assert message_signatures(restored.messages) == message_signatures(session.messages)


@pytest.mark.anyio
async def test_prompt_persists_user_assistant_and_leaf_entries(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="Hi")])
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    _events = await _collect_session_events(session.prompt("Hello"))

    entries = await storage.read_all()
    assert storage.path.exists()
    assert isinstance(entries[0], SessionInfoEntry)
    assert entries[0].cwd == str(tmp_path)
    assert entries[1] == ModelChangeEntry(
        id=entries[1].id, parent_id=entries[0].id, model="fake", timestamp=entries[1].timestamp
    )
    assert entries[2] == ThinkingLevelChangeEntry(
        id=entries[2].id,
        parent_id=entries[1].id,
        thinking_level="medium",
        timestamp=entries[2].timestamp,
    )
    message_entries = [entry for entry in entries if entry.type == "message"]
    leaf_entries = [entry for entry in entries if entry.type == "leaf"]
    assert message_signatures([entry.message for entry in message_entries]) == [
        ("human", "Hello", (), None),
        ("ai", "Hi", (), None),
    ]
    assert [entry.entry_id for entry in leaf_entries] == [entry.id for entry in message_entries]
    assert entries[-1].type == "leaf"
    assert entries[-1].entry_id == message_entries[-1].id
    assert message_signatures(session.messages) == [
        ("human", "Hello", (), None),
        ("ai", "Hi", (), None),
    ]


@pytest.mark.anyio
async def test_terminal_command_can_persist_output_to_context(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    result = await session.run_terminal_command("echo hello", add_to_context=True)

    assert result.ok is True
    assert result.output.strip() == "hello"
    assert result.added_to_context is True
    entries = await storage.read_all()
    messages = [entry.message for entry in entries if isinstance(entry, MessageEntry)]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert "Terminal command executed by the user." in messages[0].content
    assert "echo hello" in messages[0].content
    assert "hello" in messages[0].content


@pytest.mark.anyio
async def test_terminal_command_can_run_without_context(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    result = await session.run_terminal_command("echo hidden", add_to_context=False)

    assert result.ok is True
    assert result.output.strip() == "hidden"
    assert result.added_to_context is False
    entries = await storage.read_all()
    assert not any(isinstance(entry, MessageEntry) for entry in entries)


# The shell_command_prefix feature routes commands through bash only on POSIX
# (see create_bash_tool); on Windows they run under the default shell.
requires_posix_shell = pytest.mark.skipif(
    sys.platform == "win32", reason="shell_command_prefix uses bash only on POSIX"
)


@requires_posix_shell
@pytest.mark.anyio
async def test_terminal_command_uses_configured_shell_command_prefix(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            shell_command_prefix="shopt -s expand_aliases\nalias greet='printf terminal-alias'",
        )
    )

    result = await session.run_terminal_command("greet", add_to_context=False)

    assert result.ok is True
    assert result.output == "terminal-alias"
    assert result.added_to_context is False


@requires_posix_shell
@pytest.mark.anyio
async def test_agent_bash_tool_uses_configured_shell_command_prefix(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            shell_command_prefix="shopt -s expand_aliases\nalias greet='printf agent-alias'",
        )
    )
    bash_tool = next(tool for tool in session.tools if tool.name == "bash")

    result = await bash_tool.execute({"command": "greet"})

    assert result.ok is True
    assert result.content == "agent-alias"
    assert result.data is not None
    assert result.data["shell_command_prefix_applied"] is True


@pytest.mark.anyio
async def test_session_persists_no_shell_prefix_sentinel(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    prefix = "export FORGE_SENTINEL=not-a-real-secret"
    provider = ScriptedChatModel(
        [
            tool_call_ai("call-1", "bash", {"command": "echo hello"}),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            shell_command_prefix=prefix,
        )
    )

    events = await _collect_session_events(session.prompt("Run it"))

    assert any(event.type == "tool_execution_end" for event in events)
    raw = storage.path.read_text(encoding="utf-8")
    assert "FORGE_SENTINEL" not in raw
    assert "not-a-real-secret" not in raw
    # The bash tool still reports whether the prefix was applied, without
    # persisting the prefix value itself.
    tool_messages = [
        message
        for message in session.messages
        if isinstance(message, ToolMessage) and message.artifact is not None
    ]
    assert any(
        (message.artifact or {}).get("data", {}).get("shell_command_prefix_applied") is True
        for message in tool_messages
    )
    jsonl_path = await session.export(destination=tmp_path / "export.jsonl", format="jsonl")
    html_path = await session.export(destination=tmp_path / "export.html", format="html")
    assert "FORGE_SENTINEL" not in jsonl_path.read_text(encoding="utf-8")
    assert "FORGE_SENTINEL" not in html_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_steering_during_blocking_tool_is_persisted_and_seen(
    tmp_path: Path,
) -> None:
    from forge_agent.tools import AgentToolResult
    from forge_coding.tools import ToolDefinition

    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_executor(
        arguments: dict[str, object],
        signal: object | None = None,
        context: object | None = None,
    ) -> object:
        del arguments, signal, context
        started.set()
        await release.wait()
        return AgentToolResult(
            tool_call_id="",
            name="block",
            ok=True,
            content="unblocked",
        )

    block_tool = ToolDefinition(
        name="block",
        description="Blocks until released.",
        prompt_snippet="Block until released.",
        prompt_guidelines=(),
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        executor=blocking_executor,
    ).to_langchain_tool()
    provider = ScriptedChatModel(
        [
            tool_call_ai("call-1", "block", {"value": "x"}),
            AIMessage(content="Second"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            tools=[block_tool],
        )
    )
    run_events: list[object] = []

    async def run_prompt() -> None:
        async for event in session.prompt("Hello"):
            run_events.append(event)

    task = asyncio.create_task(run_prompt())
    await started.wait()

    queue_events = await _collect_session_events(
        session.prompt("Queued steering", streaming_behavior="steer")
    )
    assert queue_events == [QueueUpdateEvent(steering=("Queued steering",))]

    release.set()
    await task

    # The second model call already contains the steering message.
    assert message_texts(provider.calls[1]["messages"][1:]) == [
        "Hello",
        "",
        "unblocked",
        "Queued steering",
    ]
    assert message_signatures(session.messages) == [
        ("human", "Hello", (), None),
        ("ai", "", ("call-1",), None),
        ("tool", "unblocked", (), "call-1"),
        ("human", "Queued steering", (), None),
        ("ai", "Second", (), None),
    ]
    # The queue update fires mid-run (before agent_end) so the TUI clears the
    # pending steering display without waiting for the run to finish.
    assert any(isinstance(event, QueueUpdateEvent) for event in run_events)
    entries = await storage.read_all()
    message_entries = [entry for entry in entries if entry.type == "message"]
    assert message_signatures([entry.message for entry in message_entries]) == (
        message_signatures(list(session.messages))
    )


def test_parse_terminal_command_prefixes() -> None:
    assert parse_terminal_command("! pwd") is not None
    add_request = parse_terminal_command("! pwd")
    assert add_request is not None
    assert add_request.command == "pwd"
    assert add_request.add_to_context is True
    hidden_request = parse_terminal_command("!! pwd")
    assert hidden_request is not None
    assert hidden_request.command == "pwd"
    assert hidden_request.add_to_context is False
    assert parse_terminal_command("hello") is None


@pytest.mark.anyio
async def test_prompt_queues_steering_while_session_is_running(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = WaitingChatModel()
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    run_events: list[object] = []

    async def run_prompt() -> None:
        async for event in session.prompt("Hello"):
            run_events.append(event)

    task = asyncio.create_task(run_prompt())
    await provider.started.wait()

    with pytest.raises(RuntimeError, match="already running"):
        await _collect_session_events(session.prompt("Dropped overlap"))

    queue_events = await _collect_session_events(
        session.prompt("Queued steering", streaming_behavior="steer")
    )
    entries_before_release = await storage.read_all()

    provider.release.set()
    await task

    assert queue_events == [QueueUpdateEvent(steering=("Queued steering",))]
    before_release_messages = [
        entry.message for entry in entries_before_release if entry.type == "message"
    ]
    assert before_release_messages == [HumanMessage(content="Hello")]
    assert entries_before_release[-1].type == "leaf"
    assert entries_before_release[-1].entry_id == next(
        entry.id for entry in entries_before_release if entry.type == "message"
    )
    assert message_signatures(session.messages) == [
        ("human", "Hello", (), None),
        ("ai", "First", (), None),
        ("human", "Queued steering", (), None),
        ("ai", "Second", (), None),
    ]
    assert message_texts(provider.calls[1][1:]) == message_texts(session.messages[:3])
    entries = await storage.read_all()
    message_entries = [entry for entry in entries if entry.type == "message"]
    leaf_entries = [entry for entry in entries if entry.type == "leaf"]
    assert message_signatures([entry.message for entry in message_entries]) == message_signatures(
        list(session.messages)
    )
    assert [entry.entry_id for entry in leaf_entries] == [entry.id for entry in message_entries]
    assert any(isinstance(event, QueueUpdateEvent) for event in run_events)


@pytest.mark.anyio
async def test_tree_can_branch_from_first_user_message_before_assistant_response(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = CancellableWaitingChatModel()
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    async def run_prompt() -> None:
        try:
            async for _event in session.prompt("Start here"):
                pass
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(run_prompt())
    await provider.started.wait()

    choices = await session.tree_choices()
    with pytest.raises(RuntimeError, match="Forge is still working"):
        await session.branch_to_entry(choices[0].entry_id)

    session.cancel()
    await task
    result = await session.branch_to_entry(choices[0].entry_id)
    entries = await storage.read_all()
    message_entries = [entry for entry in entries if entry.type == "message"]

    assert [choice.label for choice in choices] == ["user: Start here"]
    assert result == SessionTreeBranchResult(
        message=f"Branched session before {choices[0].entry_id}.",
        input_prefill="Start here",
    )
    assert session.messages == ()
    assert message_signatures([entry.message for entry in message_entries]) == [
        ("human", "Start here", (), None),
    ]
    assert isinstance(entries[-1], LeafEntry)
    assert entries[-1].entry_id == message_entries[0].parent_id


@pytest.mark.anyio
async def test_tree_choices_handles_deep_session_without_recursion_error(
    tmp_path: Path,
) -> None:
    # A long conversation is a deep root-to-leaf chain of entries. Building the
    # tree picker must not exceed Python's recursion limit. Regression for #277:
    # "/tree" on a long session raised "maximum recursion depth exceeded".
    depth = sys.getrecursionlimit() + 500
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    parent_id: str | None = None
    for index in range(depth):
        entry = MessageEntry(
            id=f"m{index}",
            parent_id=parent_id,
            message=HumanMessage(content=f"message {index}"),
        )
        await storage.append(entry)
        parent_id = entry.id
    await storage.append(LeafEntry(parent_id=parent_id, entry_id=parent_id))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    choices = await session.tree_choices()

    assert len(choices) == depth
    assert choices[0].entry_id == "m0"
    assert choices[-1].entry_id == f"m{depth - 1}"


def test_ordered_tree_entries_preserves_branch_order() -> None:
    # Locks the traversal contract the iterative walk must preserve: emit a
    # node's direct children before descending, then depth-first into each child.
    entries = [
        MessageEntry(id="A", parent_id=None, message=HumanMessage(content="A")),
        MessageEntry(id="B", parent_id=None, message=HumanMessage(content="B")),
        MessageEntry(id="C", parent_id="A", message=HumanMessage(content="C")),
        MessageEntry(id="D", parent_id="A", message=HumanMessage(content="D")),
        MessageEntry(id="E", parent_id="B", message=HumanMessage(content="E")),
        MessageEntry(id="F", parent_id="C", message=HumanMessage(content="F")),
    ]

    ordered = _ordered_tree_entries(entries)

    assert [entry.id for entry in ordered] == ["A", "B", "C", "D", "F", "E"]


def test_ordered_tree_entries_terminates_on_parent_cycle() -> None:
    # A malformed parent cycle must terminate (not hang or overflow) and still
    # emit each entry exactly once. Guards the iterative walk's cycle safety.
    entries = [
        MessageEntry(id="a", parent_id="b", message=HumanMessage(content="a")),
        MessageEntry(id="b", parent_id="a", message=HumanMessage(content="b")),
    ]

    ordered = _ordered_tree_entries(entries)

    assert sorted(entry.id for entry in ordered) == ["a", "b"]
    assert len(ordered) == 2


@pytest.mark.anyio
async def test_tree_branching_preserves_active_model(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    await storage.append(MessageEntry(id="first", message=HumanMessage(content="Earlier")))
    await storage.append(ModelChangeEntry(id="historical-model", parent_id="first", model="old"))
    await storage.append(
        MessageEntry(
            id="assistant",
            parent_id="historical-model",
            message=AIMessage(content="Old answer"),
        )
    )
    await storage.append(ModelChangeEntry(id="current-model", parent_id="assistant", model="new"))
    await storage.append(
        MessageEntry(
            id="latest",
            parent_id="current-model",
            message=HumanMessage(content="Latest"),
        )
    )
    await storage.append(LeafEntry(entry_id="latest"))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    result = await session.branch_to_entry("assistant")

    assert result == SessionTreeBranchResult(message="Branched session at assistant.")
    assert session.model == "new"
    assert session.state.model == "old"
    assert message_signatures(session.messages) == [
        ("human", "Earlier", (), None),
        ("ai", "Old answer", (), None),
    ]


@pytest.mark.anyio
async def test_context_usage_is_cached_until_session_context_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    calls = 0
    original_estimate = coding_session_module.estimate_context_usage

    def wrapped_estimate(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original_estimate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(coding_session_module, "estimate_context_usage", wrapped_estimate)

    initial_usage = session.context_usage
    cached_usage = session.context_usage

    assert cached_usage is initial_usage
    assert calls == 1


@pytest.mark.anyio
async def test_context_usage_recalculates_after_prompt_and_compaction(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel(
        [
            AIMessage(content="Long answer " * 80),
        ]
    )
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    initial_usage = session.context_usage

    _events = await _collect_session_events(session.prompt("Explain context accounting."))
    after_prompt_usage = session.context_usage

    assert after_prompt_usage.message_count == 2
    assert after_prompt_usage.total_tokens > initial_usage.total_tokens
    assert session.context_token_estimate == after_prompt_usage.total_tokens

    _message = await session.compact("Context accounting was discussed.")
    after_compaction_usage = session.context_usage

    assert after_compaction_usage.message_count == 1
    assert after_compaction_usage.total_tokens < after_prompt_usage.total_tokens
    assert session.context_token_estimate == after_compaction_usage.total_tokens


@pytest.mark.anyio
async def test_session_persists_and_replays_thinking_level_changes(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    message = await session.set_thinking_level("high")
    entries = await storage.read_all()
    thinking_entries = [entry for entry in entries if entry.type == "thinking_level_change"]
    leaves = [entry for entry in entries if entry.type == "leaf"]

    restored = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    assert message == "Thinking mode: high"
    assert session.thinking_level == "high"
    assert len(thinking_entries) == 2
    assert thinking_entries[-1].thinking_level == "high"
    assert leaves[-1].entry_id == thinking_entries[-1].id
    assert restored.thinking_level == "high"
    assert restored.state.thinking_level == "high"


@pytest.mark.anyio
async def test_session_cycles_thinking_level(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    message = await session.cycle_thinking_level()

    assert message == "Thinking mode: high"
    assert session.thinking_level == "high"


@pytest.mark.anyio
async def test_session_uses_active_model_thinking_capabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    provider_config = OpenAICompatibleProviderConfig(
        name="openai",
        models=("reasoner", "plain"),
        default_model="reasoner",
        thinking_levels=("off", "low", "high"),
        thinking_models=("reasoner",),
        thinking_default="low",
        thinking_parameter="reasoning_effort",
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="reasoner",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=ProviderSettings(providers=(provider_config,)),
        )
    )

    assert session.available_thinking_levels == ("off", "low", "high")
    assert session.thinking_level == "low"
    assert session.thinking_unavailable_reason is None
    assert await session.set_thinking_level("high") == "Thinking mode: high"

    with pytest.raises(ValueError, match="not available"):
        await session.set_thinking_level("medium")

    session.set_model("plain")

    assert session.available_thinking_levels == ()
    assert session.thinking_unavailable_reason == "openai:plain is not declared in thinking_models"
    with pytest.raises(ValueError, match="openai:plain is not declared in thinking_models"):
        await session.cycle_thinking_level()

    session.set_model("reasoner")

    assert session.available_thinking_levels == ("off", "low", "high")
    assert session.thinking_level == "high"
    assert session.thinking_unavailable_reason is None


@pytest.mark.anyio
async def test_session_persists_thinking_preference_for_new_sessions(tmp_path: Path) -> None:
    forge_paths = ForgePaths(home=tmp_path / ".forge")
    provider_config = OpenAICodexProviderConfig(
        thinking_levels=("off", "minimal", "low", "medium", "high", "xhigh"),
        thinking_models=("gpt-5.5",),
        thinking_default="medium",
        thinking_parameter="reasoning.effort",
    )
    settings = ProviderSettings(
        default_provider="openai-codex",
        providers=(provider_config,),
    )
    storage = JsonlSessionStorage(tmp_path / "codex-session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5.5",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            provider_name="openai-codex",
            provider_settings=settings,
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    assert session.thinking_level == "medium"
    assert await session.set_thinking_level("low") == "Thinking mode: low"

    saved = load_provider_settings(forge_paths)
    assert saved.get_provider("openai-codex").thinking_defaults == {"gpt-5.5": "low"}

    new_session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5.5",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "new-codex-session.jsonl"),
            cwd=tmp_path,
            provider_name="openai-codex",
            provider_settings=saved,
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    assert new_session.thinking_level == "low"


@pytest.mark.anyio
async def test_resumed_session_history_overrides_saved_thinking_preference(
    tmp_path: Path,
) -> None:
    provider_config = OpenAICompatibleProviderConfig(
        name="openai",
        models=("reasoner",),
        default_model="reasoner",
        thinking_levels=("low", "high"),
        thinking_default="low",
        thinking_parameter="reasoning_effort",
        thinking_defaults={"reasoner": "low"},
    )
    storage = JsonlSessionStorage(tmp_path / "resume-thinking-session.jsonl")
    info = SessionInfoEntry(id="info", cwd=str(tmp_path))
    model = ModelChangeEntry(id="model", parent_id="info", model="reasoner")
    thinking = ThinkingLevelChangeEntry(
        id="thinking",
        parent_id="model",
        thinking_level="high",
    )
    leaf = LeafEntry(id="leaf", parent_id="thinking", entry_id="thinking")
    for entry in (info, model, thinking, leaf):
        await storage.append(entry)

    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="reasoner",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=ProviderSettings(providers=(provider_config,)),
        )
    )

    assert session.thinking_level == "high"


@pytest.mark.anyio
async def test_session_uses_codex_subscription_thinking_capabilities(
    tmp_path: Path,
) -> None:
    provider_config = OpenAICodexProviderConfig(
        thinking_levels=("off", "minimal", "low", "medium", "high", "xhigh"),
        thinking_models=("gpt-5.5",),
        thinking_default="medium",
        thinking_parameter="reasoning.effort",
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5.5",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "codex-session.jsonl"),
            cwd=tmp_path,
            provider_name="openai-codex",
            provider_settings=ProviderSettings(providers=(provider_config,)),
        )
    )

    assert session.available_thinking_levels == (
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    )
    assert session.thinking_unavailable_reason is None
    assert await session.set_thinking_level("high") == "Thinking mode: high"


@pytest.mark.anyio
async def test_session_refreshes_runtime_provider_for_thinking_level(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[tuple[str | None, str | None]] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del provider_config, credential_store
        created.append((model, thinking_level))
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    provider_config = OpenAICompatibleProviderConfig(
        name="openai",
        models=("reasoner",),
        default_model="reasoner",
        thinking_levels=("low", "high"),
        thinking_default="low",
        thinking_parameter="reasoning_effort",
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="reasoner",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "runtime-session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=ProviderSettings(providers=(provider_config,)),
            runtime_provider_config=provider_config,
            thinking_level="high",
        )
    )

    assert created == [("reasoner", "high")]

    await session.set_thinking_level("low")

    assert created[-1] == ("reasoner", "low")

    await session.aclose()


@pytest.mark.anyio
async def test_load_restores_existing_transcript(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    user_entry = MessageEntry(id="user", message=HumanMessage(content="Earlier"))
    assistant_entry = MessageEntry(
        id="assistant",
        parent_id="user",
        message=AIMessage(content="Restored"),
    )
    await storage.append(user_entry)
    await storage.append(assistant_entry)

    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    assert message_signatures(session.messages) == [
        ("human", "Earlier", (), None),
        ("ai", "Restored", (), None),
    ]


@pytest.mark.anyio
async def test_load_detaches_missing_root_parent_from_imported_branch(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(
        id="root",
        parent_id="missing-external-parent",
        message=HumanMessage(content="Root"),
    )
    assistant = MessageEntry(
        id="assistant",
        parent_id="root",
        message=AIMessage(content="Restored"),
    )
    await storage.append(root)
    await storage.append(assistant)
    await storage.append(LeafEntry(entry_id="assistant"))

    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    assert message_signatures(session.messages) == [
        ("human", "Root", (), None),
        ("ai", "Restored", (), None),
    ]
    assert session.state.active_leaf_id == "assistant"


@pytest.mark.anyio
async def test_tree_branching_detaches_missing_root_parent_from_imported_branch(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(
        id="root",
        parent_id="missing-external-parent",
        message=HumanMessage(content="Root"),
    )
    assistant = MessageEntry(
        id="assistant",
        parent_id="root",
        message=AIMessage(content="Restored"),
    )
    await storage.append(root)
    await storage.append(assistant)
    await storage.append(LeafEntry(parent_id="assistant", entry_id="assistant"))

    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))
    choices = await session.tree_choices()
    result = await session.branch_to_entry("root")

    assert [choice.entry_id for choice in choices] == ["root", "assistant"]
    assert result == SessionTreeBranchResult(
        message="Branched session before root.",
        input_prefill="Root",
    )
    assert session.messages == ()


@pytest.mark.anyio
async def test_load_restores_explicit_empty_leaf_branch(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    await storage.append(root)
    await storage.append(LeafEntry(entry_id="root"))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    result = await session.branch_to_entry("root")
    reloaded = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    assert result == SessionTreeBranchResult(
        message="Branched session before root.",
        input_prefill="Root",
    )
    assert session.messages == ()
    assert reloaded.messages == ()
    assert reloaded.state.active_leaf_id is None


@pytest.mark.anyio
async def test_load_restores_active_leaf_branch(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    left = MessageEntry(
        id="left",
        parent_id="root",
        message=AIMessage(content="Inactive branch"),
    )
    right = MessageEntry(
        id="right",
        parent_id="root",
        message=AIMessage(content="Active branch"),
    )
    await storage.append(root)
    await storage.append(left)
    await storage.append(right)
    await storage.append(LeafEntry(entry_id="right"))

    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    assert message_signatures(session.messages) == [
        ("human", "Root", (), None),
        ("ai", "Active branch", (), None),
    ]
    assert session.state.active_leaf_id == "right"


@pytest.mark.anyio
async def test_session_tree_choices_indent_only_diverged_branches(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    main = MessageEntry(id="main", parent_id="root", message=AIMessage(content="Main"))
    first_branch = MessageEntry(
        id="first-branch",
        parent_id="root",
        message=AIMessage(content="First branch"),
    )
    first_branch_child = MessageEntry(
        id="first-branch-child",
        parent_id="first-branch",
        message=HumanMessage(content="Follow-up"),
    )
    main_child = MessageEntry(
        id="main-child",
        parent_id="main",
        message=HumanMessage(content="Main follow-up"),
    )
    second_branch = MessageEntry(
        id="second-branch",
        parent_id="root",
        message=AIMessage(content="Second branch"),
    )
    await storage.append(root)
    await storage.append(main)
    await storage.append(first_branch)
    await storage.append(first_branch_child)
    await storage.append(main_child)
    await storage.append(second_branch)
    await storage.append(LeafEntry(entry_id="second-branch"))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    choices = await session.tree_choices()

    assert [choice.label for choice in choices] == [
        "user: Root",
        "assistant: Main",
        "  assistant: First branch",
        "  assistant: Second branch",
        "user: Main follow-up",
        "  user: Follow-up",
    ]


@pytest.mark.anyio
async def test_session_branches_to_previous_entry_without_destroying_history(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    left = MessageEntry(id="left", parent_id="root", message=AIMessage(content="Left"))
    right = MessageEntry(id="right", parent_id="root", message=AIMessage(content="Right"))
    await storage.append(root)
    await storage.append(left)
    await storage.append(right)
    await storage.append(LeafEntry(entry_id="right"))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    result = await session.branch_to_entry("left")

    entries = await storage.read_all()
    assert result == SessionTreeBranchResult(message="Branched session at left.")
    assert message_signatures(session.messages) == [
        ("human", "Root", (), None),
        ("ai", "Left", (), None),
    ]
    assert [entry.id for entry in entries if entry.type == "message"] == ["root", "left", "right"]
    assert isinstance(entries[-1], LeafEntry)
    assert entries[-1].entry_id == "left"


@pytest.mark.anyio
async def test_persist_after_branch_keeps_state_on_active_branch(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="New answer")])
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    answer = MessageEntry(id="answer", parent_id="root", message=AIMessage(content="Answer"))
    abandoned = MessageEntry(
        id="abandoned",
        parent_id="answer",
        message=HumanMessage(content="Abandoned follow-up"),
    )
    abandoned_answer = MessageEntry(
        id="abandoned-answer",
        parent_id="abandoned",
        message=AIMessage(content="Abandoned answer"),
    )
    await storage.append(root)
    await storage.append(answer)
    await storage.append(abandoned)
    await storage.append(abandoned_answer)
    await storage.append(LeafEntry(entry_id="abandoned-answer"))
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    await session.branch_to_entry("answer")
    _events = await _collect_session_events(session.prompt("New follow-up"))

    assert message_signatures(session.state.messages) == [
        ("human", "Root", (), None),
        ("ai", "Answer", (), None),
        ("human", "New follow-up", (), None),
        ("ai", "New answer", (), None),
    ]
    assert "abandoned" not in session.state.context_entry_ids
    assert "abandoned-answer" not in session.state.context_entry_ids

    await session.compact()
    compactions = [entry for entry in await storage.read_all() if entry.type == "compaction"]
    assert len(compactions) == 1
    assert "abandoned" not in compactions[0].replaces_entry_ids
    assert "abandoned-answer" not in compactions[0].replaces_entry_ids
    assert "Abandoned" not in provider.calls[1]["messages"][1].content


@pytest.mark.anyio
async def test_session_branches_to_before_selected_user_message_with_prefill(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    assistant = MessageEntry(
        id="assistant",
        parent_id="root",
        message=AIMessage(content="Answer"),
    )
    followup = MessageEntry(
        id="followup",
        parent_id="assistant",
        message=HumanMessage(content="Try this again"),
    )
    await storage.append(root)
    await storage.append(assistant)
    await storage.append(followup)
    await storage.append(LeafEntry(entry_id="followup"))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    result = await session.branch_to_entry("followup")

    entries = await storage.read_all()
    assert result == SessionTreeBranchResult(
        message="Branched session before followup.",
        input_prefill="Try this again",
    )
    assert message_signatures(session.messages) == [
        ("human", "Root", (), None),
        ("ai", "Answer", (), None),
    ]
    assert [entry.id for entry in entries if entry.type == "message"] == [
        "root",
        "assistant",
        "followup",
    ]
    assert isinstance(entries[-1], LeafEntry)
    assert entries[-1].entry_id == "assistant"


@pytest.mark.anyio
async def test_session_branch_preserves_active_model(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first_model = ModelChangeEntry(id="model-a", model="first-model")
    left = MessageEntry(
        id="left",
        parent_id="model-a",
        message=HumanMessage(content="Before switch"),
    )
    second_model = ModelChangeEntry(
        id="model-b",
        parent_id="left",
        model="second-model",
    )
    right = MessageEntry(
        id="right",
        parent_id="model-b",
        message=AIMessage(content="After switch"),
    )
    await storage.append(first_model)
    await storage.append(left)
    await storage.append(second_model)
    await storage.append(right)
    await storage.append(LeafEntry(entry_id="right"))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    assert session.model == "second-model"

    await session.branch_to_entry("left")

    assert session.state.model == "first-model"
    assert session.model == "second-model"


@pytest.mark.anyio
async def test_session_branch_with_summary_keeps_pre_branch_model_and_messages(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first_model = ModelChangeEntry(id="model-a", model="first-model")
    left = MessageEntry(
        id="left",
        parent_id="model-a",
        message=HumanMessage(content="Before switch"),
    )
    second_model = ModelChangeEntry(
        id="model-b",
        parent_id="left",
        model="second-model",
    )
    right = MessageEntry(
        id="right",
        parent_id="model-b",
        message=AIMessage(content="After switch"),
    )
    await storage.append(first_model)
    await storage.append(left)
    await storage.append(second_model)
    await storage.append(right)
    await storage.append(LeafEntry(entry_id="right"))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    assert session.model == "second-model"

    await session.branch_to_entry("left", summarize=True)

    assert session.state.model == "first-model"
    assert session.model == "second-model"
    assert len(session.messages) == 2
    assert session.messages[0] == HumanMessage(content="Before switch")
    assert session.messages[1].content.startswith(
        "The following is a summary of a branch that this conversation came back from:"
    )


@pytest.mark.anyio
async def test_session_branch_with_summary_rebuilds_context(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="The abandoned branch went left.")])
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    left = MessageEntry(id="left", parent_id="root", message=AIMessage(content="Left"))
    right = MessageEntry(
        id="right",
        parent_id="left",
        message=HumanMessage(content="Abandoned follow-up"),
    )
    await storage.append(root)
    await storage.append(left)
    await storage.append(right)
    await storage.append(LeafEntry(entry_id="right"))
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    result = await session.branch_to_entry("root", summarize=True)
    entries = await storage.read_all()
    summary = entries[-2]

    assert "with branch summary" in result.message
    assert summary.type == "branch_summary"
    assert summary.parent_id == "root"
    assert summary.branch_root_id == "root"
    assert summary.summary.startswith(
        "The user explored a different conversation branch before returning here."
    )
    assert "The abandoned branch went left." in summary.summary
    assert provider.calls[0]["tools"] == []
    assert "<conversation>" in provider.calls[0]["messages"][1].content
    assert "Use this EXACT format:" in provider.calls[0]["messages"][1].content
    assert "Abandoned follow-up" in provider.calls[0]["messages"][1].content
    assert len(session.messages) == 2
    assert session.messages[0] == HumanMessage(content="Root")
    assert session.messages[1].type == "human"
    assert isinstance(session.messages[1].content, str)
    assert session.messages[1].content.startswith(
        "The following is a summary of a branch that this conversation came back from:"
    )
    assert "The abandoned branch went left." in session.messages[1].content


@pytest.mark.anyio
async def test_session_branch_with_summary_accepts_custom_instructions(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="Custom branch summary.")])
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    left = MessageEntry(id="left", parent_id="root", message=AIMessage(content="Left"))
    right = MessageEntry(
        id="right",
        parent_id="left",
        message=HumanMessage(content="Abandoned follow-up"),
    )
    await storage.append(root)
    await storage.append(left)
    await storage.append(right)
    await storage.append(LeafEntry(entry_id="right"))
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    await session.branch_to_entry(
        "root",
        summarize=True,
        custom_instructions="Focus on failing commands.",
    )

    prompt = provider.calls[0]["messages"][1].content
    assert "Use this EXACT format:" in prompt
    assert "Additional focus: Focus on failing commands." in prompt


@pytest.mark.anyio
async def test_session_branch_with_summary_tracks_file_operations(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="File work summary.")])
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    assistant = MessageEntry(
        id="assistant",
        parent_id="root",
        message=AIMessage(
            content="Using tools",
            tool_calls=[
                {
                    "id": "read-1",
                    "name": "read",
                    "args": {"path": "src/read_only.py"},
                    "type": "tool_call",
                },
                {
                    "id": "edit-1",
                    "name": "edit",
                    "args": {"path": "src/changed.py"},
                    "type": "tool_call",
                },
            ],
        ),
    )
    await storage.append(root)
    await storage.append(assistant)
    await storage.append(LeafEntry(entry_id="assistant"))
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    await session.branch_to_entry("root", summarize=True)
    entries = await storage.read_all()
    summary = entries[-2]

    assert summary.type == "branch_summary"
    assert "<read-files>\nsrc/read_only.py\n</read-files>" in summary.summary
    assert "<modified-files>\nsrc/changed.py\n</modified-files>" in summary.summary


@pytest.mark.anyio
async def test_session_branch_with_summary_falls_back_when_model_summary_is_unavailable(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    left = MessageEntry(id="left", parent_id="root", message=AIMessage(content="Left"))
    right = MessageEntry(
        id="right",
        parent_id="left",
        message=HumanMessage(content="Abandoned follow-up"),
    )
    await storage.append(root)
    await storage.append(left)
    await storage.append(right)
    await storage.append(LeafEntry(entry_id="right"))
    session = await CodingSession.load(_config(tmp_path, ScriptedChatModel(), storage))

    result = await session.branch_to_entry("root", summarize=True)
    entries = await storage.read_all()
    summary = entries[-2]

    assert "with branch summary" in result.message
    assert summary.type == "branch_summary"
    assert "Automatically compacted 2 prior message(s)." in summary.summary
    assert "Abandoned follow-up" in summary.summary
    assert len(session.messages) == 2
    assert session.messages[0] == HumanMessage(content="Root")
    assert "Abandoned follow-up" in session.messages[1].content


@pytest.mark.anyio
async def test_continue_persists_only_new_messages(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    await storage.append(MessageEntry(id="user", message=HumanMessage(content="Continue me")))
    provider = ScriptedChatModel([AIMessage(content="Continued")])
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    _events = await _collect_session_events(session.continue_())

    entries = await storage.read_all()
    message_entries = [entry for entry in entries if entry.type == "message"]
    assert message_signatures([entry.message for entry in message_entries]) == [
        ("human", "Continue me", (), None),
        ("ai", "Continued", (), None),
    ]


@pytest.mark.anyio
async def test_tool_results_are_persisted(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel(
        [
            AIMessage(
                content="Using tool",
                tool_calls=[{"id": "call-1", "name": "missing", "args": {}, "type": "tool_call"}],
            ),
            AIMessage(content="Done"),
        ]
    )
    session = await CodingSession.load(_config(tmp_path, provider, storage))

    _events = await _collect_session_events(session.prompt("Use a tool"))

    messages = [entry.message for entry in await storage.read_all() if entry.type == "message"]
    assert any(isinstance(message, ToolMessage) for message in messages)
    assert any(
        isinstance(message, ToolMessage) and message.status == "error" for message in messages
    )


@pytest.mark.anyio
async def test_session_preserves_explicit_empty_system_prompt(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="Done")])
    config = CodingSessionConfig(
        provider=provider,
        model="fake",
        system="",
        storage=storage,
        cwd=tmp_path,
    )
    session = await CodingSession.load(config)

    _events = await _collect_session_events(session.prompt("Hello"))

    assert provider.calls[0]["messages"][0].content == ""


@pytest.mark.anyio
async def test_session_builds_system_prompt_when_system_is_omitted(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    skills_dir = resource_root / "skills"
    skills_dir.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("Follow project rules.", encoding="utf-8")
    (skills_dir / "testing").mkdir()
    (skills_dir / "testing" / "SKILL.md").write_text(
        "---\ndescription: Test code\n---\n# Testing",
        encoding="utf-8",
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="Done")])
    config = CodingSessionConfig(
        provider=provider,
        model="fake",
        storage=storage,
        cwd=tmp_path,
        resource_paths=ForgeResourcePaths(root=resource_root, agents_root=None),
    )
    session = await CodingSession.load(config)

    _events = await _collect_session_events(session.prompt("Hello"))

    assert (
        "Available tools:\n- read: Read file contents" in provider.calls[0]["messages"][0].content
    )
    assert '<project_instructions path="' in provider.calls[0]["messages"][0].content
    assert "Follow project rules." in provider.calls[0]["messages"][0].content
    assert "<available_skills>" in provider.calls[0]["messages"][0].content
    assert "<name>testing</name>" in provider.calls[0]["messages"][0].content
    assert [Path(context_file.path).name for context_file in session.context_files] == ["AGENTS.md"]


@pytest.mark.anyio
async def test_session_touches_session_manager_after_persisting_messages(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake")
    provider = ScriptedChatModel([AIMessage(content="Greeting")])
    config = CodingSessionConfig(
        provider=provider,
        model="fake",
        system="You are Forge.",
        storage=storage,
        cwd=tmp_path,
        session_id=record.id,
        session_manager=manager,
        resource_paths=ForgeResourcePaths(root=tmp_path / "resources", agents_root=None),
    )
    session = await CodingSession.load(config)

    _events = await _collect_session_events(session.prompt("Hello"))

    updated = manager.get_session(record.id)
    assert updated is not None
    assert updated.updated_at >= record.updated_at


@pytest.mark.anyio
async def test_session_auto_names_first_unnamed_managed_session(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake")
    provider = ScriptedChatModel([AIMessage(content='"Fix broken CLI output now"')])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            session_id=record.id,
            session_manager=manager,
        )
    )

    await _collect_session_events(session.prompt("Please fix the broken CLI output."))

    renamed = manager.get_session(record.id)
    assert renamed is not None
    assert renamed.title == "Fix broken CLI output"
    assert provider.calls[0]["tools"] == []
    assert "Please fix the broken CLI output." in provider.calls[0]["messages"][1].content
    assert message_texts(provider.calls[1]["messages"][1:]) == ["Please fix the broken CLI output."]


@pytest.mark.anyio
async def test_session_auto_name_falls_back_when_provider_fails(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake")
    record = manager.create_session(cwd=tmp_path, model="fake")
    provider = RaisingChatModel(fail_on_call=1, success_content="Done")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            session_id=record.id,
            session_manager=manager,
        )
    )

    await _collect_session_events(session.prompt("Investigate flaky session restore tests"))

    renamed = manager.get_session(record.id)
    assert renamed is not None
    assert renamed.title == "Investigate flaky session restore"
    assert message_signatures(session.messages) == [
        ("human", "Investigate flaky session restore tests", (), None),
        ("ai", "Done", (), None),
    ]


@pytest.mark.anyio
async def test_session_auto_name_falls_back_when_provider_returns_unusable_title(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake")
    provider = ScriptedChatModel([AIMessage(content="!!!")])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            session_id=record.id,
            session_manager=manager,
        )
    )

    await _collect_session_events(session.prompt("Debug failing model picker"))

    renamed = manager.get_session(record.id)
    assert renamed is not None
    assert renamed.title == "Debug failing model picker"


@pytest.mark.anyio
async def test_session_auto_name_does_not_overwrite_manual_name(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake", title="Manual name")
    provider = ScriptedChatModel([AIMessage(content="Done")])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            session_id=record.id,
            session_manager=manager,
        )
    )

    await _collect_session_events(session.prompt("Rename this automatically"))

    unchanged = manager.get_session(record.id)
    assert unchanged is not None
    assert unchanged.title == "Manual name"
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_session_auto_name_does_not_index_new_session_before_first_persist(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.prepare_session(cwd=tmp_path, model="fake")
    provider = ScriptedChatModel([AIMessage(content="Generated title")])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=record.cwd,
            session_id=record.id,
            session_manager=manager,
            index_on_first_persist=True,
        )
    )

    stream = session.prompt("Stop before the first persisted message")
    _first_event = await anext(stream)
    await stream.aclose()

    assert manager.get_session(record.id) is None
    assert await storage.read_all() == []


@pytest.mark.anyio
async def test_session_loads_and_expands_skills(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    skills_dir = resource_root / "skills" / "testing"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Testing\nRun pytest.", encoding="utf-8")
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="Done")])
    config = CodingSessionConfig(
        provider=provider,
        model="fake",
        system="You are Forge.",
        storage=storage,
        cwd=tmp_path,
        resource_paths=ForgeResourcePaths(root=resource_root, agents_root=None),
    )
    session = await CodingSession.load(config)

    _events = await _collect_session_events(session.prompt("/skill:testing add tests"))

    assert [skill.name for skill in session.skills] == ["testing"]
    assert '<skill name="testing" location="' in provider.calls[0]["messages"][1].content
    assert "References are relative to" in provider.calls[0]["messages"][1].content
    assert provider.calls[0]["messages"][1].content.endswith("</skill>\n\nadd tests")
    assert session.handle_command("/skill:testing").handled is False


@pytest.mark.anyio
async def test_system_command_shows_prompt_without_persisting_or_adding_context(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel()
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
        )
    )

    before_messages = session.messages
    before_entries = await storage.read_all()

    result = session.handle_command("/system")

    assert result.handled is True
    assert result.message == "You are Forge."
    assert session.messages == before_messages
    assert await storage.read_all() == before_entries
    assert provider.calls == []


@pytest.mark.anyio
async def test_session_expands_prompt_templates_as_slash_commands(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    prompts_dir = resource_root / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "example.md").write_text(
        "Custom prompt for {{ arguments }}.",
        encoding="utf-8",
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="Done")])
    config = CodingSessionConfig(
        provider=provider,
        model="fake",
        system="You are Forge.",
        storage=storage,
        cwd=tmp_path,
        resource_paths=ForgeResourcePaths(root=resource_root, agents_root=None),
    )
    session = await CodingSession.load(config)

    assert [template.name for template in session.prompt_templates] == ["example"]
    assert session.handle_command("/example src/app.py").handled is False

    _events = await _collect_session_events(session.prompt("/example src/app.py"))

    assert provider.calls[0]["messages"][1].content == "Custom prompt for src/app.py."


@pytest.mark.anyio
async def test_session_skill_index_lets_agent_read_relevant_skill_file(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    skills_dir = resource_root / "skills" / "testing"
    skills_dir.mkdir(parents=True)
    skill_path = skills_dir / "SKILL.md"
    skill_path.write_text(
        "---\ndescription: Use when writing tests\n---\n# Testing\nRun pytest.",
        encoding="utf-8",
    )
    provider = ScriptedChatModel(
        [
            AIMessage(
                content="Reading skill.",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "read",
                        "args": {"path": str(skill_path)},
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            storage=storage,
            cwd=tmp_path,
            resource_paths=ForgeResourcePaths(root=resource_root, agents_root=None),
        )
    )

    _events = await _collect_session_events(session.prompt("Add tests."))

    assert "<available_skills>" in provider.calls[0]["messages"][0].content
    assert f"<location>{skill_path}</location>" in provider.calls[0]["messages"][0].content
    assert len(provider.calls) == 2
    tool_result = provider.calls[1]["messages"][-1]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == "call-1"
    assert tool_result.name == "read"
    assert "# Testing\nRun pytest." in tool_result.content
    assert tool_result.artifact is not None
    assert tool_result.artifact["data"]["path"] == str(skill_path)


@pytest.mark.anyio
async def test_session_loads_with_resource_diagnostics_instead_of_failing(
    tmp_path: Path,
) -> None:
    resource_root = tmp_path / "resources"
    skills_dir = resource_root / "skills"
    (skills_dir / "good").mkdir(parents=True)
    (skills_dir / "good" / "SKILL.md").write_text("# Directory skill", encoding="utf-8")
    (skills_dir / "legacy.md").write_text("# Legacy bare-md skill", encoding="utf-8")
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    config = CodingSessionConfig(
        provider=ScriptedChatModel(),
        model="fake",
        system="You are Forge.",
        storage=storage,
        cwd=tmp_path,
        resource_paths=ForgeResourcePaths(root=resource_root, agents_root=None),
    )

    session = await CodingSession.load(config)

    assert [skill.name for skill in session.skills] == ["good"]
    assert len(session.resource_diagnostics) == 1
    assert (
        "bare .md files are no longer treated as skills" in session.resource_diagnostics[0].message
    )
    assert "Resource diagnostics: 1" in (session.handle_command("/session").message or "")


@pytest.mark.anyio
async def test_session_reload_refreshes_resources_and_system_prompt(tmp_path: Path) -> None:
    resource_root = tmp_path / "resources"
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel([AIMessage(content="Done")])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            storage=storage,
            cwd=tmp_path,
            resource_paths=ForgeResourcePaths(root=resource_root, agents_root=None),
        )
    )
    assert session.skills == ()
    assert session.context_files == ()

    skills_dir = resource_root / "skills" / "testing"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\ndescription: Test code\n---\n# Testing\nRun pytest.",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text("Reloaded project rules.", encoding="utf-8")

    entries_before = await storage.read_all()
    result = session.handle_command("/reload")
    entries_after = await storage.read_all()
    _events = await _collect_session_events(session.prompt("Hello"))

    assert result.message is not None
    assert "Reloaded local coding resources and project context." in result.message
    assert "Skills: 1 total (changed, +1)" in result.message
    assert "Project context files: 1 total (changed, +1)" in result.message
    assert "Next-turn system prompt: rebuilt" in result.message
    assert "Not refreshed by /reload" in result.message
    assert entries_after == entries_before
    assert [skill.name for skill in session.skills] == ["testing"]
    assert [Path(context_file.path).name for context_file in session.context_files] == ["AGENTS.md"]
    assert "Reloaded project rules." in provider.calls[0]["messages"][0].content
    assert "<name>testing</name>" in provider.calls[0]["messages"][0].content


@pytest.mark.anyio
async def test_session_reload_skips_provider_settings_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_load_provider_settings(paths: ForgePaths | None = None) -> ProviderSettings:
        del paths
        raise AssertionError("/reload should not refresh provider settings")

    monkeypatch.setattr(
        coding_session_module,
        "load_provider_settings",
        fail_load_provider_settings,
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_settings=ProviderSettings(
                providers=(OpenAICompatibleProviderConfig(name="openai"),)
            ),
        )
    )

    result = session.handle_command("/reload")

    assert result.message is not None
    assert "Provider config:" in result.message
    assert "Not refreshed by /reload" in result.message


@pytest.mark.anyio
async def test_session_reload_leaves_system_prompt_when_inputs_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            storage=storage,
            cwd=tmp_path,
        )
    )

    def fail_build_system_prompt(options: object) -> str:
        del options
        raise AssertionError("system prompt should not be rebuilt")

    monkeypatch.setattr(
        coding_session_module,
        "build_system_prompt",
        fail_build_system_prompt,
    )

    result = session.handle_command("/reload")

    assert result.message is not None
    assert "Next-turn system prompt: unchanged" in result.message


@pytest.mark.anyio
async def test_session_provider_settings_reload_uses_session_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    forge_paths = ForgePaths(home=tmp_path / "forge-home", agents_home=tmp_path / "agents-home")
    seen_paths: list[ForgePaths | None] = []

    def load_provider_settings(paths: ForgePaths | None = None) -> ProviderSettings:
        seen_paths.append(paths)
        return ProviderSettings(providers=(OpenAICompatibleProviderConfig(name="openai"),))

    monkeypatch.setattr(coding_session_module, "load_provider_settings", load_provider_settings)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "provider-reload-session.jsonl"),
            cwd=tmp_path,
            provider_settings=ProviderSettings(
                providers=(OpenAICompatibleProviderConfig(name="openai"),)
            ),
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    session.reload_provider_settings()

    assert seen_paths == [forge_paths]


@pytest.mark.anyio
async def test_session_compact_persists_summary_and_rebuilds_context(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = ScriptedChatModel(
        [
            AIMessage(content="Session answer"),
            AIMessage(content="Generated session summary"),
            AIMessage(content="Next answer"),
        ]
    )
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    _events = await _collect_session_events(session.prompt("Explain sessions."))

    message_count_before = len(session.messages)
    message_entries_before = [
        entry.id for entry in await storage.read_all() if entry.type == "message"
    ]

    result = await session.compact("Focus on session persistence.")
    entries_after_compact = await storage.read_all()
    compactions = [entry for entry in entries_after_compact if entry.type == "compaction"]
    leaves = [entry for entry in entries_after_compact if entry.type == "leaf"]

    _next_events = await _collect_session_events(session.prompt("Continue."))

    assert result == f"Compacted {message_count_before} context entries."
    assert len(compactions) == 1
    assert isinstance(compactions[0], CompactionEntry)
    assert compactions[0].summary == "Generated session summary"
    assert compactions[0].replaces_entry_ids == message_entries_before
    assert leaves[-1].entry_id == compactions[0].id
    assert provider.calls[1]["messages"][0].content.startswith(
        "You are a context summarization assistant."
    )
    assert (
        "Additional focus: Focus on session persistence."
        in provider.calls[1]["messages"][1].content
    )
    assert message_texts(provider.calls[2]["messages"][1:]) == [
        "Previous conversation summary:\nGenerated session summary",
        "Continue.",
    ]


@pytest.mark.anyio
async def test_session_auto_compacts_after_response_when_threshold_is_exceeded(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    large_prompt = "Explain sessions.\n" + ("old context " * 12_000)
    provider = ScriptedChatModel(
        [
            AIMessage(content="First answer"),
            AIMessage(content="Second answer"),
            AIMessage(content="Generated automatic summary"),
            AIMessage(content="Third answer"),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            auto_compact_token_threshold=1,
        )
    )
    _first_events = await _collect_session_events(session.prompt(large_prompt))

    _second_events = await _collect_session_events(session.prompt("Continue."))
    _third_events = await _collect_session_events(session.prompt("Next."))

    entries = await storage.read_all()
    compactions = [entry for entry in entries if entry.type == "compaction"]

    assert len(compactions) == 1
    assert compactions[0].summary == "Generated automatic summary"
    assert "Explain sessions." in provider.calls[2]["messages"][1].content
    assert message_texts(provider.calls[3]["messages"][1:]) == [
        f"Previous conversation summary:\n{compactions[0].summary}",
        "Continue.",
        "Second answer",
        "Next.",
    ]


@pytest.mark.anyio
async def test_session_auto_compacts_with_pi_style_default_threshold(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    large_prompt = "Explain sessions.\n" + ("old context " * 12_000)
    provider = ScriptedChatModel(
        [
            AIMessage(content="First answer"),
            AIMessage(content="Second answer"),
            AIMessage(content="Default threshold summary"),
        ]
    )
    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                models=("fake",),
                default_model="fake",
                context_windows={"fake": 20_000},
            ),
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            provider_name="local",
            provider_settings=settings,
        )
    )

    assert session.context_window_tokens == 20_000
    assert session.auto_compact_token_threshold == 3_616

    _first_events = await _collect_session_events(session.prompt(large_prompt))
    _second_events = await _collect_session_events(session.prompt("Continue."))

    compactions = [entry for entry in await storage.read_all() if entry.type == "compaction"]

    assert len(compactions) == 1
    assert compactions[0].summary == "Default threshold summary"


@pytest.mark.anyio
async def test_session_compacts_and_retries_once_after_context_overflow(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    large_prompt = "Collect context.\n" + ("old context " * 12_000)
    provider = ScriptedErrorChatModel(
        [
            AIMessage(content="First answer"),
            AIMessage(content="Second answer"),
            AIMessage(content="Overflow recovery summary"),
            AIMessage(content="Recovered answer"),
        ],
        error_on_call=3,
        error_message="This model's maximum context length was exceeded.",
    )
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    _first_events = await _collect_session_events(session.prompt(large_prompt))
    _second_events = await _collect_session_events(session.prompt("Keep this recent turn."))

    retry_events = await _collect_session_events(session.prompt("Trigger overflow."))
    entries = await storage.read_all()
    compactions = [entry for entry in entries if entry.type == "compaction"]

    assert len(compactions) == 1
    assert compactions[0].summary == "Overflow recovery summary"
    assert any(
        getattr(event, "type", None) == "message_end"
        and getattr(event, "message", None).content == "Recovered answer"
        for event in retry_events
    )
    assert message_texts(provider.calls[4]["messages"][1:]) == [
        "Previous conversation summary:\nOverflow recovery summary",
        "Keep this recent turn.",
        "Second answer",
        "Trigger overflow.",
    ]


@pytest.mark.anyio
async def test_session_switches_configured_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created_providers: list[BaseChatModel] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del credential_store, model, thinking_level
        provider = ScriptedChatModel([])
        created_providers.append(provider)
        return provider

    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCAL_API_KEY", "test-key")
    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(name="openai"),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("qwen", "llama"),
                default_model="qwen",
            ),
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=settings,
        )
    )

    session.set_provider("local")

    assert session.provider_name == "local"
    assert session.model == "qwen"
    assert session.available_models == ("qwen", "llama")
    assert [(choice.provider_name, choice.model) for choice in session.available_model_choices] == [
        ("local", "qwen"),
        ("local", "llama"),
    ]
    assert len(created_providers) == 1

    session.set_provider("local")

    assert len(created_providers) == 2

    await session.aclose()

    assert [provider.closed for provider in created_providers] == [True, True]


@pytest.mark.anyio
async def test_session_switch_uses_session_credential_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    forge_paths = ForgePaths(home=tmp_path / "forge-home", agents_home=tmp_path / "agents-home")
    FileCredentialStore(forge_paths.home / "credentials.json").set("openai", "stored-key")
    credential_store_paths: list[Path] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del provider_config, model, thinking_level
        assert credential_store is not None
        credential_store_paths.append(credential_store.path)
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                api_key_env="LOCAL_API_KEY",
                credential_name=None,
                models=("qwen",),
                default_model="qwen",
            ),
            OpenAICompatibleProviderConfig(name="openai", credential_name="openai"),
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "switch-store-session.jsonl"),
            cwd=tmp_path,
            provider_name="local",
            provider_settings=settings,
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    session.set_provider("openai")

    assert credential_store_paths == [forge_paths.home / "credentials.json"]


@pytest.mark.anyio
async def test_available_model_choices_hide_unusable_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_API_KEY", "local-key")
    forge_paths = ForgePaths(home=tmp_path / "forge-home", agents_home=tmp_path / "agents-home")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(name="openai"),
            OpenAICompatibleProviderConfig(
                name="local",
                api_key_env="LOCAL_API_KEY",
                credential_name=None,
                models=("qwen", "llama"),
                default_model="qwen",
            ),
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=settings,
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    assert session.available_models == ()
    assert session.available_providers == ("local",)
    assert [(choice.provider_name, choice.model) for choice in session.available_model_choices] == [
        ("local", "qwen"),
        ("local", "llama"),
    ]


@pytest.mark.anyio
async def test_available_model_choices_include_stored_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    forge_paths = ForgePaths(home=tmp_path / "forge-home", agents_home=tmp_path / "agents-home")
    FileCredentialStore(forge_paths.home / "credentials.json").set("openai", "stored-key")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(OpenAICompatibleProviderConfig(name="openai", credential_name="openai"),),
    )

    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "stored-session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=settings,
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    assert session.available_providers == ("openai",)
    assert ("openai", "gpt-5.4") in [
        (choice.provider_name, choice.model) for choice in session.available_model_choices
    ]


@pytest.mark.anyio
async def test_session_toggles_and_cycles_scoped_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_API_KEY", "local-key")
    forge_paths = ForgePaths(home=tmp_path / "forge-home", agents_home=tmp_path / "agents-home")
    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                api_key_env="LOCAL_API_KEY",
                credential_name=None,
                models=("qwen", "llama"),
                default_model="qwen",
            ),
        ),
        scoped_models=(ScopedModelConfig(provider="local", model="qwen"),),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="qwen",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "scoped-session.jsonl"),
            cwd=tmp_path,
            provider_name="local",
            provider_settings=settings,
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    llama = ModelChoice(provider_name="local", model="llama")
    scoped = session.toggle_scoped_model(llama)
    choice = session.cycle_scoped_model()
    saved = json.loads((forge_paths.home / "providers.json").read_text(encoding="utf-8"))

    assert [(item.provider_name, item.model) for item in scoped] == [
        ("local", "qwen"),
        ("local", "llama"),
    ]
    assert choice == llama
    assert session.model == "llama"
    assert saved["scoped_models"] == [
        {"provider": "local", "model": "qwen"},
        {"provider": "local", "model": "llama"},
    ]


@requires_posix_shell
@pytest.mark.anyio
async def test_session_resume_preserves_shell_command_prefix(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    first_record = manager.create_session(cwd=first_cwd, model="fake", title="First")
    second_record = manager.create_session(cwd=second_cwd, model="fake", title="Second")
    second_storage = JsonlSessionStorage(second_record.path)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(first_record.path),
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
            shell_command_prefix="shopt -s expand_aliases\nalias greet='printf resumed-alias'",
        )
    )
    await second_storage.append(SessionInfoEntry(cwd=str(second_record.cwd)))
    await second_storage.append(ModelChangeEntry(model="fake"))

    await session.resume(second_record.id)
    result = await session.run_terminal_command("greet", add_to_context=False)

    assert result.ok is True
    assert result.output == "resumed-alias"


@pytest.mark.anyio
async def test_session_resumes_indexed_session(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_record = manager.create_session(cwd=tmp_path / "first", model="fake", title="First")
    second_cwd = tmp_path / "second"
    second_cwd.mkdir(parents=True)
    second_record = manager.create_session(cwd=second_cwd, model="fake", title="Second")
    first_storage = JsonlSessionStorage(first_record.path)
    second_storage = JsonlSessionStorage(second_record.path)
    provider = ScriptedChatModel([AIMessage(content="Second answer")])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=first_storage,
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
        )
    )
    await second_storage.append(SessionInfoEntry(cwd=str(second_record.cwd)))
    await second_storage.append(ModelChangeEntry(model="fake"))
    await second_storage.append(MessageEntry(message=HumanMessage(content="Earlier")))
    await second_storage.append(MessageEntry(message=AIMessage(content="Restored")))

    message = await session.resume(second_record.id)
    _events = await _collect_session_events(session.prompt("Continue."))

    assert message == f"Resumed session: {second_record.id}"
    assert session.session_id == second_record.id
    assert session.cwd == second_record.cwd
    assert [item.content for item in session.messages[:2]] == ["Earlier", "Restored"]
    assert message_texts(provider.calls[0]["messages"][1:]) == [
        "Earlier",
        "Restored",
        "Continue.",
    ]


@pytest.mark.anyio
async def test_session_toggle_scoped_model_preserves_newer_provider_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_API_KEY", "local-key")
    monkeypatch.setenv("REMOTE_API_KEY", "remote-key")
    forge_paths = ForgePaths(home=tmp_path / "forge-home", agents_home=tmp_path / "agents-home")
    loaded_settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                api_key_env="LOCAL_API_KEY",
                models=("qwen", "llama"),
                default_model="qwen",
            ),
        ),
        scoped_models=(ScopedModelConfig(provider="local", model="qwen"),),
    )
    newer_settings = ProviderSettings(
        default_provider="local",
        providers=(
            loaded_settings.get_provider("local"),
            OpenAICompatibleProviderConfig(
                name="remote",
                api_key_env="REMOTE_API_KEY",
                models=("sonnet",),
                default_model="sonnet",
            ),
        ),
        scoped_models=(
            ScopedModelConfig(provider="local", model="qwen"),
            ScopedModelConfig(provider="remote", model="sonnet"),
        ),
    )
    save_provider_settings(newer_settings, forge_paths)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="qwen",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "scoped-session.jsonl"),
            cwd=tmp_path,
            provider_name="local",
            provider_settings=loaded_settings,
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    session.toggle_scoped_model(ModelChoice(provider_name="local", model="llama"))

    saved = coding_session_module.load_provider_settings(forge_paths)
    assert saved.get_provider("remote").default_model == "sonnet"
    assert saved.scoped_models == (
        ScopedModelConfig(provider="local", model="qwen"),
        ScopedModelConfig(provider="remote", model="sonnet"),
        ScopedModelConfig(provider="local", model="llama"),
    )


@pytest.mark.anyio
async def test_session_set_model_rejects_model_not_declared_for_provider(tmp_path: Path) -> None:
    provider_config = OpenAICompatibleProviderConfig(
        name="openai",
        models=("gpt-5",),
        default_model="gpt-5",
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=ProviderSettings(providers=(provider_config,)),
        )
    )

    with pytest.raises(
        coding_session_module.ProviderConfigError,
        match="Model is not configured for provider openai: gpt-5.5",
    ):
        session.set_model("gpt-5.5")

    assert session.model == "gpt-5"


@pytest.mark.anyio
async def test_session_load_falls_back_when_persisted_model_does_not_match_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[tuple[str, str | None]] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del credential_store, thinking_level
        created.append((provider_config.name, model))  # type: ignore[attr-defined]
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    await storage.append(SessionInfoEntry(cwd=str(tmp_path)))
    await storage.append(ModelChangeEntry(model="gpt-5"))
    provider_config = OpenAICodexProviderConfig(
        models=("gpt-5.5",),
        default_model="gpt-5.5",
    )

    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5.5",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            provider_name="openai-codex",
            provider_settings=ProviderSettings(
                default_provider="openai-codex",
                providers=(provider_config,),
            ),
            runtime_provider_config=provider_config,
        )
    )

    assert session.state.model == "gpt-5"
    assert session.model == "gpt-5.5"
    assert created == [("openai-codex", "gpt-5.5")]


@pytest.mark.anyio
async def test_session_set_model_persists_default_provider_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    forge_paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    provider_config = OpenAICompatibleProviderConfig(
        name="openai",
        models=("gpt-5", "gpt-5-mini"),
        default_model="gpt-5",
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=ProviderSettings(providers=(provider_config,)),
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    session.set_model("gpt-5-mini")

    saved = coding_session_module.load_provider_settings(forge_paths)
    assert saved.default_provider == "openai"
    assert saved.get_provider("openai").default_model == "gpt-5-mini"


@pytest.mark.anyio
async def test_session_set_model_choice_persists_default_provider_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    forge_paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("qwen", "llama"),
                default_model="qwen",
            ),
        ),
    )
    created: list[tuple[str, str | None]] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del credential_store, thinking_level
        created.append((provider_config.name, model))  # type: ignore[attr-defined]
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("openai"),
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )
    created.clear()

    session.set_model_choice(ModelChoice(provider_name="local", model="llama"))

    saved = coding_session_module.load_provider_settings(forge_paths)
    assert saved.default_provider == "local"
    assert saved.get_provider("local").default_model == "llama"
    assert created == [("local", "llama")]


@pytest.mark.anyio
async def test_session_set_model_choice_switches_provider_model_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCAL_API_KEY", "local-key")
    forge_paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("qwen", "llama"),
                default_model="qwen",
            ),
        ),
        scoped_models=(
            ScopedModelConfig(provider="openai", model="gpt-5"),
            ScopedModelConfig(provider="local", model="llama"),
        ),
    )
    created: list[tuple[str, str | None]] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del credential_store, thinking_level
        created.append((provider_config.name, model))  # type: ignore[attr-defined]
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("openai"),
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )
    created.clear()

    choice = session.cycle_scoped_model()

    assert choice == ModelChoice(provider_name="local", model="llama")
    assert session.provider_name == "local"
    assert session.model == "llama"
    assert created == [("local", "llama")]


@pytest.mark.anyio
async def test_session_set_model_preserves_newer_provider_file_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    forge_paths = ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents")
    loaded_provider = OpenAICompatibleProviderConfig(
        name="openai",
        models=("gpt-5", "gpt-5-mini"),
        default_model="gpt-5",
    )
    newer_settings = ProviderSettings(
        default_provider="openai",
        providers=(
            loaded_provider,
            OpenAICompatibleProviderConfig(
                name="openrouter",
                api_key_env="OPENROUTER_API_KEY",
                credential_name="openrouter",
                models=("openai/gpt-5.5",),
                default_model="openai/gpt-5.5",
                headers={"X-Title": "Forge"},
            ),
        ),
        scoped_models=(ScopedModelConfig(provider="openrouter", model="openai/gpt-5.5"),),
    )
    save_provider_settings(newer_settings, forge_paths)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="openai",
            provider_settings=ProviderSettings(providers=(loaded_provider,)),
            resource_paths=ForgeResourcePaths(root=forge_paths.home, paths=forge_paths),
        )
    )

    session.set_model("gpt-5-mini")

    saved = coding_session_module.load_provider_settings(forge_paths)
    assert saved.get_provider("openai").default_model == "gpt-5-mini"
    assert saved.get_provider("openrouter").headers == {"X-Title": "Forge"}
    assert saved.scoped_models == (
        ScopedModelConfig(provider="openrouter", model="openai/gpt-5.5"),
    )


@pytest.mark.anyio
async def test_session_new_session_uses_default_provider_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    current_record = manager.create_session(
        cwd=tmp_path,
        model="openai/gpt-5.5",
        provider_name="openrouter",
    )
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
            OpenAICompatibleProviderConfig(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
                models=("openai/gpt-5.5",),
                default_model="openai/gpt-5.5",
            ),
        ),
    )
    created: list[tuple[str, str | None]] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del credential_store, thinking_level
        created.append((provider_config.name, model))  # type: ignore[attr-defined]
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="openai/gpt-5.5",
            system="You are Forge.",
            storage=JsonlSessionStorage(current_record.path),
            cwd=current_record.cwd,
            session_id=current_record.id,
            session_manager=manager,
            provider_name="openrouter",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("openrouter"),
        )
    )
    created.clear()

    message = await session.new_session()

    assert message.startswith("Started new session: ")
    assert session.provider_name == "openai"
    assert session.model == "gpt-5"
    assert manager.get_session(session.session_id) is None
    assert created == [("openai", "gpt-5")]


@pytest.mark.anyio
async def test_session_new_session_is_indexed_after_first_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    current_record = manager.create_session(cwd=tmp_path, model="fake", provider_name="fake")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
        ),
    )

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ScriptedChatModel:
        del provider_config, credential_store, model, thinking_level
        return ScriptedChatModel([AIMessage(content="Greeting")])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(current_record.path),
            cwd=current_record.cwd,
            session_id=current_record.id,
            session_manager=manager,
            provider_name="fake",
            provider_settings=settings,
        )
    )

    _message = await session.new_session()
    pending_id = session.session_id

    assert pending_id is not None
    assert manager.get_session(pending_id) is None
    assert all(record.id != pending_id for record in manager.list_sessions(tmp_path))

    _events = await _collect_session_events(session.prompt("Hello"))

    indexed = manager.get_session(pending_id)
    assert indexed is not None
    assert indexed.provider_name == "openai"
    assert indexed.model == "gpt-5"
    assert indexed.title == "Greeting"
    assert indexed.path.exists()


@pytest.mark.anyio
async def test_session_name_indexes_pending_session_without_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    current_record = manager.create_session(cwd=tmp_path, model="fake", provider_name="fake")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
        ),
    )

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ScriptedChatModel:
        del provider_config, credential_store, model, thinking_level
        return ScriptedChatModel()

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(current_record.path),
            cwd=current_record.cwd,
            session_id=current_record.id,
            session_manager=manager,
            provider_name="fake",
            provider_settings=settings,
        )
    )

    _message = await session.new_session()
    pending_id = session.session_id

    assert pending_id is not None
    assert manager.get_session(pending_id) is None

    result = session.handle_command("/name Customer bugfix")

    indexed = manager.get_session(pending_id)
    assert result.message == "Session renamed: Customer bugfix"
    assert indexed is not None
    assert indexed.title == "Customer bugfix"
    assert indexed.provider_name == "openai"
    assert indexed.model == "gpt-5"
    assert indexed.path.exists()
    assert await JsonlSessionStorage(indexed.path).read_all()


@pytest.mark.anyio
async def test_session_resume_uses_target_session_provider_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_record = manager.create_session(
        cwd=tmp_path / "first",
        model="gpt-5",
        provider_name="openai",
        title="First",
    )
    second_cwd = tmp_path / "second"
    second_cwd.mkdir(parents=True)
    second_record = manager.create_session(
        cwd=second_cwd,
        model="qwen",
        provider_name="local",
        title="Second",
    )
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("qwen",),
                default_model="qwen",
            ),
        ),
    )
    created: list[tuple[str, str | None]] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del credential_store, thinking_level
        created.append((provider_config.name, model))  # type: ignore[attr-defined]
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    second_storage = JsonlSessionStorage(second_record.path)
    await second_storage.append(SessionInfoEntry(cwd=str(second_record.cwd)))
    await second_storage.append(ModelChangeEntry(model="qwen"))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5",
            system="You are Forge.",
            storage=JsonlSessionStorage(first_record.path),
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
            provider_name="openai",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("openai"),
        )
    )
    created.clear()

    await session.resume(second_record.id)

    assert session.provider_name == "local"
    assert session.model == "qwen"
    assert created == [("local", "qwen")]


@pytest.mark.anyio
async def test_session_resume_missing_provider_preserves_active_provider_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_record = manager.create_session(
        cwd=tmp_path / "first",
        model="gpt-5",
        provider_name="openai",
        title="First",
    )
    second_cwd = tmp_path / "second"
    second_cwd.mkdir(parents=True)
    second_record = manager.create_session(
        cwd=second_cwd,
        model="qwen",
        provider_name=None,
        title="Legacy second",
    )
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("qwen",),
                default_model="qwen",
            ),
        ),
    )
    created: list[tuple[str, str | None]] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del credential_store, thinking_level
        created.append((provider_config.name, model))  # type: ignore[attr-defined]
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    second_storage = JsonlSessionStorage(second_record.path)
    await second_storage.append(SessionInfoEntry(cwd=str(second_record.cwd)))
    await second_storage.append(ModelChangeEntry(model="qwen"))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5",
            system="You are Forge.",
            storage=JsonlSessionStorage(first_record.path),
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
            provider_name="openai",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("openai"),
        )
    )
    created.clear()

    await session.resume(second_record.id)

    assert session.provider_name == "openai"
    assert session.model == "gpt-5"
    assert created == [("openai", "gpt-5"), ("openai", "gpt-5")]


@pytest.mark.anyio
async def test_session_resume_rejects_incompatible_provider_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_record = manager.create_session(
        cwd=tmp_path / "first",
        model="gpt-5",
        provider_name="openai",
        title="First",
    )
    second_cwd = tmp_path / "second"
    second_cwd.mkdir(parents=True)
    second_record = manager.create_session(
        cwd=second_cwd,
        model="gpt-5.5",
        provider_name="local",
        title="Bad second",
    )
    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("qwen",),
                default_model="qwen",
            ),
        ),
    )

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> BaseChatModel:
        del credential_store, model, thinking_level
        return ScriptedChatModel([])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="gpt-5",
            system="You are Forge.",
            storage=JsonlSessionStorage(first_record.path),
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
            provider_name="openai",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("openai"),
        )
    )

    with pytest.raises(
        ProviderConfigError,
        match="Model is not configured for provider local: gpt-5.5",
    ):
        await session.resume(second_record.id)

    assert session.provider_name == "openai"
    assert session.model == "gpt-5"


@pytest.mark.anyio
async def test_session_context_usage_recalculates_after_resume(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_record = manager.create_session(cwd=tmp_path / "first", model="fake", title="First")
    second_cwd = tmp_path / "second"
    second_cwd.mkdir(parents=True)
    second_record = manager.create_session(cwd=second_cwd, model="fake", title="Second")
    first_storage = JsonlSessionStorage(first_record.path)
    second_storage = JsonlSessionStorage(second_record.path)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=first_storage,
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
        )
    )
    before_resume_usage = session.context_usage
    await second_storage.append(SessionInfoEntry(cwd=str(second_record.cwd)))
    await second_storage.append(ModelChangeEntry(model="fake"))
    await second_storage.append(MessageEntry(message=HumanMessage(content="Earlier " * 20)))
    await second_storage.append(MessageEntry(message=AIMessage(content="Restored " * 20)))

    _message = await session.resume(second_record.id)
    after_resume_usage = session.context_usage

    assert before_resume_usage.message_count == 0
    assert after_resume_usage.message_count == 2
    assert after_resume_usage.total_tokens > before_resume_usage.total_tokens
    assert session.context_token_estimate == after_resume_usage.total_tokens


def test_minimal_commands_are_handled(tmp_path: Path) -> None:
    session = CodingSession(
        _config(tmp_path, ScriptedChatModel(), JsonlSessionStorage(tmp_path / "session.jsonl")),
        state=object(),  # type: ignore[arg-type]
        harness=object(),  # type: ignore[arg-type]
        last_parent_id=None,
    )

    assert session.handle_command("hello").handled is False
    assert session.handle_command("/new").new_session_requested is True
    assert session.handle_command("/clear").message == "Unknown command: /clear"
    assert session.handle_command("/quit").exit_requested is True
    assert session.handle_command("/exit").exit_requested is True
    assert session.handle_command("/unknown").message == "Unknown command: /unknown"


# --------------------------------------------------------------------------- #
# 5.1: resume()/new_session() adopt a fully-loaded replacement session
# atomically.  All durable state moves over, retired providers close, and a
# failed replacement leaves the original session and its provider usable.
# --------------------------------------------------------------------------- #
def _entry_parent_chain(entries: list[object]) -> list[str | None]:
    by_id = {entry.id: entry for entry in entries}  # type: ignore[attr-defined]
    broken: list[str | None] = []
    for entry in entries:  # type: ignore[attr-defined]
        parent = getattr(entry, "parent_id", None)
        if parent is not None and parent not in by_id:
            broken.append(parent)
    return broken


@pytest.mark.anyio
async def test_new_session_first_persist_writes_metadata_before_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    current_record = manager.create_session(cwd=tmp_path, model="fake", provider_name="fake")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(OpenAICompatibleProviderConfig(name="openai"),),
    )

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ScriptedChatModel:
        del provider_config, credential_store, model, thinking_level
        return ScriptedChatModel([AIMessage(content="Greeting"), AIMessage(content="Second")])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(current_record.path),
            cwd=current_record.cwd,
            session_id=current_record.id,
            session_manager=manager,
            provider_name="fake",
            provider_settings=settings,
        )
    )

    await session.new_session()
    pending_id = session.session_id
    assert pending_id is not None

    _events = await _collect_session_events(session.prompt("Hello"))

    entries = await session.storage.read_all()
    types = [entry.type for entry in entries]
    assert types[:3] == ["session_info", "model_change", "thinking_level_change"]
    assert types[3:] == ["message", "leaf", "message", "leaf"]
    assert _entry_parent_chain(entries) == []
    # The session is recoverable from the manager after the first prompt.
    indexed = manager.get_session(pending_id)
    assert indexed is not None
    reloaded = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel([AIMessage(content="Recovered")]),
            model=indexed.model,
            system="You are Forge.",
            storage=JsonlSessionStorage(indexed.path),
            cwd=indexed.cwd,
            session_id=indexed.id,
            session_manager=manager,
            provider_name=indexed.provider_name or "openai",
            provider_settings=settings,
        )
    )
    # The auto-name helper consumed the first scripted response; the durable
    # transcript holds the main answer.
    assert [item.content for item in reloaded.messages] == ["Hello", "Second"]


@pytest.mark.anyio
async def test_new_session_from_pending_metadata_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    current_record = manager.create_session(cwd=tmp_path, model="fake", provider_name="fake")
    settings = ProviderSettings(
        default_provider="openai",
        providers=(OpenAICompatibleProviderConfig(name="openai"),),
    )

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ScriptedChatModel:
        del provider_config, credential_store, model, thinking_level
        return ScriptedChatModel([AIMessage(content="Answer")])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "fresh.jsonl"),
            cwd=tmp_path,
            session_id=current_record.id,
            session_manager=manager,
            provider_name="fake",
            provider_settings=settings,
        )
    )
    assert session._pending_initial_entries  # not yet persisted

    await session.new_session()

    # The adopted pending entries still initialize the new storage on first
    # persist, and the new session stays recoverable.
    _events = await _collect_session_events(session.prompt("Hello"))
    entries = await session.storage.read_all()
    assert [entry.type for entry in entries[:3]] == [
        "session_info",
        "model_change",
        "thinking_level_change",
    ]
    assert [item.content for item in session.messages] == ["Hello", "Answer"]


@pytest.mark.anyio
async def test_resume_transfers_owned_providers_and_closes_retired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_record = manager.create_session(cwd=tmp_path, model="fake", provider_name="fake")
    second_cwd = tmp_path / "second"
    second_cwd.mkdir()
    second_record = manager.create_session(cwd=second_cwd, model="fake", provider_name="fake")
    second_storage = JsonlSessionStorage(second_record.path)
    await second_storage.append(SessionInfoEntry(cwd=str(second_cwd)))
    await second_storage.append(ModelChangeEntry(model="fake"))
    await second_storage.append(MessageEntry(message=HumanMessage(content="Earlier")))
    await second_storage.append(MessageEntry(message=AIMessage(content="Restored")))

    created: list[ScriptedChatModel] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ScriptedChatModel:
        del provider_config, credential_store, model, thinking_level
        provider = ScriptedChatModel([AIMessage(content="Resumed answer")])
        created.append(provider)
        return provider

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    settings = ProviderSettings(
        default_provider="fake",
        providers=(
            OpenAICompatibleProviderConfig(name="fake", models=("fake",), default_model="fake"),
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(first_record.path),
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
            provider_name="fake",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("fake"),
        )
    )
    assert len(created) == 1
    first_provider = created[0]

    message = await session.resume(second_record.id)

    assert message == f"Resumed session: {second_record.id}"
    assert len(created) == 2
    second_provider = created[1]
    # The retired provider from the previous session is closed immediately.
    assert first_provider.closed is True
    assert second_provider.closed is False
    assert session._owned_providers == [second_provider]
    assert [item.content for item in session.messages[:2]] == ["Earlier", "Restored"]

    await session.aclose()
    assert second_provider.closed is True


@pytest.mark.anyio
async def test_resume_to_different_provider_switches_runtime_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_record = manager.create_session(cwd=tmp_path, model="fake", provider_name="fake")
    second_cwd = tmp_path / "second"
    second_cwd.mkdir()
    second_record = manager.create_session(
        cwd=second_cwd, model="other-model", provider_name="other"
    )
    second_storage = JsonlSessionStorage(second_record.path)
    await second_storage.append(SessionInfoEntry(cwd=str(second_cwd)))
    await second_storage.append(ModelChangeEntry(model="other-model"))
    await second_storage.append(MessageEntry(message=HumanMessage(content="From other")))

    created: list[tuple[str, str]] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ScriptedChatModel:
        del credential_store, thinking_level
        created.append((provider_config.name, model or ""))  # type: ignore[attr-defined]
        return ScriptedChatModel([AIMessage(content="Other answer")])

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    settings = ProviderSettings(
        default_provider="fake",
        providers=(
            OpenAICompatibleProviderConfig(name="fake", models=("fake",), default_model="fake"),
            OpenAICompatibleProviderConfig(
                name="other", models=("other-model",), default_model="other-model"
            ),
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(first_record.path),
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
            provider_name="fake",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("fake"),
        )
    )

    message = await session.resume(second_record.id)

    assert message == f"Resumed session: {second_record.id}"
    assert session.provider_name == "other"
    assert session.model == "other-model"
    assert created[-1] == ("other", "other-model")
    assert [item.content for item in session.messages] == ["From other"]
    # The adopted session answers with the new provider.
    _events = await _collect_session_events(session.prompt("Continue"))
    assert session.messages[-1].content == "Other answer"


@pytest.mark.anyio
async def test_resume_failure_leaves_original_session_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    first_record = manager.create_session(cwd=tmp_path, model="fake", provider_name="fake")
    second_cwd = tmp_path / "second"
    second_cwd.mkdir()
    second_record = manager.create_session(cwd=second_cwd, model="fake", provider_name="other")
    second_storage = JsonlSessionStorage(second_record.path)
    await second_storage.append(SessionInfoEntry(cwd=str(second_cwd)))
    await second_storage.append(ModelChangeEntry(model="fake"))

    created: list[ScriptedChatModel] = []

    def create_provider(
        provider_config: object,
        *,
        credential_store: FileCredentialStore | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ScriptedChatModel:
        del credential_store, model, thinking_level
        if provider_config.name == "other":  # type: ignore[attr-defined]
            raise coding_session_module.ProviderConfigError("other provider exploded")
        provider = ScriptedChatModel([AIMessage(content="Original answer")])
        created.append(provider)
        return provider

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    settings = ProviderSettings(
        default_provider="fake",
        providers=(
            OpenAICompatibleProviderConfig(name="fake", models=("fake",), default_model="fake"),
            OpenAICompatibleProviderConfig(name="other", models=("fake",), default_model="fake"),
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(first_record.path),
            cwd=first_record.cwd,
            session_id=first_record.id,
            session_manager=manager,
            provider_name="fake",
            provider_settings=settings,
            runtime_provider_config=settings.get_provider("fake"),
        )
    )
    original_provider = created[0]

    with pytest.raises(coding_session_module.ProviderConfigError, match="exploded"):
        await session.resume(second_record.id)

    # The original session and its provider remain fully usable.
    assert session.provider_name == "fake"
    assert session.session_id == first_record.id
    assert original_provider.closed is False
    _events = await _collect_session_events(session.prompt("Still alive"))
    assert [item.content for item in session.messages] == ["Still alive", "Original answer"]


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
        _config(tmp_path, FakeListChatModel(responses=["Recovered."]), storage)
    )
    entries = await storage.read_all()
    message_entries = [entry for entry in entries if entry.type == "message"]
    repairs = [e.message for e in message_entries if isinstance(e.message, ToolMessage)]
    assert len(repairs) == 1
    assert repairs[0].tool_call_id == "call-1"
    assert repairs[0].status == "error"
    assert session.messages[-1] == repairs[0]


# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_cancel_persists_synthetic_tool_result(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    started = asyncio.Event()

    async def blocking_executor(
        arguments: dict[str, object],
        signal: object | None = None,
        context: object | None = None,
    ) -> object:
        del arguments, signal, context
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

    model = ScriptedChatModel(
        responses=[
            tool_call_ai("call-1", name="block", args={"value": "x"}),
            AIMessage(content="recovered"),
        ]
    )
    session = await CodingSession.load(_config(tmp_path, model, storage, tools=[blocking_tool]))

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


# ---------------------------------------------------------------------------
def test_tree_branchable_accepts_native_messages() -> None:
    answer_entry = MessageEntry(message=AIMessage(content="answer"))
    assert _is_branchable_tree_entry(answer_entry)
    user_entry = MessageEntry(message=HumanMessage(content="question"))
    assert _is_branchable_tree_entry(user_entry)
    tool_entry = MessageEntry(message=tool_call_ai("call-1", "echo", {"value": "x"}))
    assert _is_branchable_tree_entry(tool_entry)
    assert _is_tool_call_tree_entry(tool_entry)


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
            provider=ScriptedChatModel(
                responses=[
                    tool_call_ai("call-1", name="read", args={"path": "target.txt"}),
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
    # The injected context steers execution but must never be dumped into the
    # persisted artifact (workspace root / session id / shell prefix leak).
    assert artifact.get("details") is None or not (artifact.get("details") or {}).get(
        "workspace_root"
    )


# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tool_runtime_context_reaches_result_details(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")

    async def capture(
        arguments: dict[str, object],
        signal: object | None = None,
        context: object | None = None,
    ) -> object:
        del arguments, signal, context
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

    model = ScriptedChatModel(
        responses=[
            tool_call_ai("call-1", name="capture", args={"value": "x"}),
            AIMessage(content="done"),
        ]
    )
    session = await CodingSession.load(_config(tmp_path, model, storage, tools=[capture_tool]))
    # The harness forwards ForgeRuntimeContext into the graph's ToolRuntime.
    assert session._harness.config.runtime_context is not None
    events = [event async for event in session.prompt("Go")]
    assert any(event.type == "tool_execution_end" for event in events)

    tool_messages = [
        m for m in session.messages if isinstance(m, ToolMessage) and m.artifact is not None
    ]
    assert tool_messages
    details = tool_messages[0].artifact.get("details") or {}
    # The runtime context reaches the executor (the tool ran), but none of its
    # fields may be copied into the persisted artifact.
    assert "workspace_root" not in details
    assert "session_id" not in details
    assert "shell_command_prefix" not in details
