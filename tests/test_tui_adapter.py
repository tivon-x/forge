from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from forge_agent import (
    AgentEndEvent,
    AgentStartEvent,
    AgentToolResult,
    ErrorEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    QueueUpdateEvent,
    RetryEvent,
    ThinkingDeltaEvent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from forge_cli.formatting import format_tool_call_block, format_tool_result_block
from forge_cli.tui import TuiEventAdapter, TuiState
from forge_coding.skills import Skill, format_skill_invocation


def test_tui_adapter_tracks_running_state() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(AgentStartEvent())
    assert state.running is True

    adapter.apply(AgentEndEvent())
    assert state.running is False


def test_tui_adapter_builds_assistant_items_from_streamed_messages() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(MessageStartEvent())
    adapter.apply(MessageDeltaEvent(delta="Hel"))
    adapter.apply(MessageDeltaEvent(delta="lo"))
    assert state.assistant_buffer == "Hello"
    assert state.items == []

    adapter.apply(MessageEndEvent(message=AIMessage(content="Hello")))

    assert state.assistant_buffer == ""
    assert [(item.role, item.text) for item in state.items] == [("assistant", "Hello")]


def test_tui_adapter_builds_user_items_from_streamed_messages() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(MessageStartEvent(message_role="user"))
    adapter.apply(MessageEndEvent(message=HumanMessage(content="Hello Forge")))

    assert state.assistant_buffer == ""
    assert [(item.role, item.text) for item in state.items] == [("user", "Hello Forge")]


def test_tui_adapter_compacts_streamed_skill_invocations() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    skill = Skill(
        name="review",
        path=Path("/workspace/.forge/skills/review.md"),
        content="# Review\nFull noisy instructions.",
        description="Review code",
    )

    adapter.apply(
        MessageEndEvent(
            message=HumanMessage(content=format_skill_invocation(skill, "check the auth flow"))
        )
    )

    assert [(item.role, item.text) for item in state.items] == [
        ("skill", "Using skill: review"),
        ("user", "check the auth flow"),
    ]


def test_tui_adapter_groups_thinking_deltas_separately() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(ThinkingDeltaEvent(delta="hidden "))
    adapter.apply(ThinkingDeltaEvent(delta="reasoning"))

    assert [(item.role, item.text) for item in state.items] == [("thinking", "hidden reasoning")]
    assert state.show_thinking is False


def test_tui_adapter_flushes_assistant_buffer_before_tool_events() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(MessageDeltaEvent(delta="Before tool"))
    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
        )
    )

    assert state.assistant_buffer == ""
    assert state.items[0].role == "assistant"
    assert state.items[0].text == "Before tool"
    assert state.items[1].role == "tool"
    assert "→ read" in state.items[1].text


def test_tui_adapter_renders_skill_file_reads_with_skill_style() -> None:
    skill = Skill(
        name="review",
        path=Path("/workspace/.forge/skills/review.md"),
        content="# Review",
        description="Review code",
    )
    state = TuiState(skills=(skill,))
    adapter = TuiEventAdapter(state)

    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="call-1",
                name="read",
                arguments={"path": "/workspace/.forge/skills/review.md"},
            )
        )
    )
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="call-1",
                name="read",
                ok=True,
                content="# Review\nFull instructions.",
            )
        )
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        ("skill", "Loading skill: review", "✓ read\n# Review\nFull instructions.")
    ]


def test_tui_adapter_leaves_ordinary_reads_as_tool_items() -> None:
    skill = Skill(
        name="review",
        path=Path("/workspace/.forge/skills/review.md"),
        content="# Review",
        description="Review code",
    )
    state = TuiState(skills=(skill,))
    adapter = TuiEventAdapter(state)

    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="call-1",
                name="read",
                arguments={"path": "/workspace/README.md"},
            )
        )
    )

    assert [(item.role, item.text) for item in state.items] == [
        ("tool", "→ read /workspace/README.md")
    ]


def test_tool_call_blocks_use_human_readable_invocations() -> None:
    assert (
        format_tool_call_block(
            ToolCall(
                id="call-1",
                name="read",
                arguments={"path": "tests/test_tui_app.py", "offset": 1, "limit": 80},
            )
        )
        == "→ read tests/test_tui_app.py:1-80"
    )
    assert (
        format_tool_call_block(
            ToolCall(id="call-2", name="edit", arguments={"path": "src/forge_coding/tui/app.py"})
        )
        == "→ edit src/forge_coding/tui/app.py"
    )
    assert (
        format_tool_call_block(
            ToolCall(
                id="call-3",
                name="bash",
                arguments={
                    "command": "git log --oneline --decorate --graph --max-count=8",
                    "timeout": 30,
                },
            )
        )
        == "$ git log --oneline --decorate --graph --max-count=8 (timeout 30s)"
    )


def test_tui_adapter_records_tool_updates_and_results() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(ToolExecutionUpdateEvent(tool_call_id="call-1", message="reading"))
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(tool_call_id="call-1", name="read", ok=True, content="done")
        )
    )
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="call-2",
                name="bash",
                ok=False,
                content="failed",
            )
        )
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        ("tool", "… reading", None),
        ("tool", "✓ read", "✓ read\ndone"),
        ("tool", "✗ bash", "✗ bash\nfailed"),
    ]


def test_tui_adapter_records_retry_status() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(
        RetryEvent(
            attempt=2,
            max_attempts=3,
            delay_seconds=0,
            message="Retrying provider request 2/3 after HTTP 503.",
        )
    )

    assert [(item.role, item.text) for item in state.items] == [
        ("status", "… Retrying provider request 2/3 after HTTP 503.")
    ]


def test_tui_adapter_records_queue_updates() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(QueueUpdateEvent(steering=("adjust",), follow_up=("after",)))

    assert state.queued_steering == ("adjust",)
    assert state.queued_follow_up == ("after",)
    assert state.queued_message_count == 2


def test_tool_result_blocks_preview_long_content() -> None:
    content = "\n".join(f"line {index}" for index in range(1, 12))

    block = format_tool_result_block(name="read", ok=True, content=content)

    assert "line 1" in block
    assert "line 8" in block
    assert "line 9" not in block
    assert "3 more lines" in block


def test_tui_adapter_renders_live_edit_patch() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="call-1",
                name="edit",
                ok=True,
                content="Successfully replaced 1 block.",
                data={"patch": "--- a.py\n+++ a.py\n@@\n-old\n+new"},
            )
        )
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        (
            "tool",
            "✓ edit",
            "✓ edit\nSuccessfully replaced 1 block.\n\nPatch:\n--- a.py\n+++ a.py\n@@\n-old\n+new",
        )
    ]


def test_tui_adapter_records_errors_and_stops_on_non_recoverable_error() -> None:
    state = TuiState(running=True, assistant_buffer="partial")
    adapter = TuiEventAdapter(state)

    adapter.apply(ErrorEvent(message="provider failed", recoverable=False))

    assert state.running is False
    assert state.error == "provider failed"
    assert [(item.role, item.text) for item in state.items] == [
        ("assistant", "partial"),
        ("error", "Error: provider failed"),
    ]


def test_tui_adapter_renders_cancellation_as_status() -> None:
    state = TuiState(running=True, assistant_buffer="partial")
    adapter = TuiEventAdapter(state)

    adapter.apply(ErrorEvent(message="Agent run cancelled", recoverable=True))

    assert state.running is True
    assert state.error is None
    assert [(item.role, item.text) for item in state.items] == [
        ("assistant", "partial"),
        ("status", "Agent run cancelled."),
    ]


def test_tui_adapter_maps_task_lifecycle_to_one_subagent_display() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="task-1",
                name="task",
                arguments={"agent": "scout", "instruction": "Inspect auth flow"},
            )
        )
    )
    adapter.apply(
        ToolExecutionUpdateEvent(
            tool_call_id="task-1",
            message="Reading session.py",
            data={
                "kind": "subagent_activity",
                "agent": "scout",
                "status": "running",
                "activity": {
                    "phase": "started",
                    "tool": "read",
                    "summary": "Reading session.py",
                },
            },
        )
    )
    adapter.apply(
        ToolExecutionUpdateEvent(
            tool_call_id="task-1",
            message="Writing noisy child output that must not become a row",
            data={"kind": "unknown_nested_event"},
        )
    )
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="task-1",
                name="task",
                ok=True,
                content="done",
                data={
                    "kind": "subagent_run",
                    "version": 1,
                    "agent": "scout",
                    "status": "completed",
                    "instruction": "Inspect auth flow",
                    "final_output": "Auth flow is healthy.",
                    "tool_calls": 1,
                    "queued_ms": 4,
                    "duration_ms": 1200,
                    "truncated": False,
                    "error": None,
                },
            )
        )
    )

    assert len(state.items) == 1
    item = state.items[0]
    assert item.role == "subagent"
    assert item.subagent is not None
    assert item.subagent.status == "completed"
    assert item.subagent.activity == "completed"
    assert item.subagent.tool_calls == 1
    assert item.subagent.final_output == "Auth flow is healthy."


def test_tui_state_bad_task_artifact_falls_back_to_ordinary_tool_item() -> None:
    state = TuiState()
    state.add_subagent_task(
        ToolCall(
            id="task-bad",
            name="task",
            arguments={"agent": "reviewer", "instruction": "Review persistence"},
        )
    )
    state.finish_subagent_task(
        AgentToolResult(
            tool_call_id="task-bad",
            name="task",
            ok=True,
            content="legacy result",
            data={"kind": "subagent_run", "version": 99},
        )
    )

    assert len(state.items) == 1
    assert state.items[0].role == "tool"
    assert state.items[0].subagent is None
    assert state.items[0].tool_result_text == "✓ task\nlegacy result"


def test_tui_state_extreme_artifact_usage_falls_back_without_formatting_it() -> None:
    state = TuiState()
    state.add_subagent_task(
        ToolCall(
            id="task-huge-usage",
            name="task",
            arguments={"agent": "reviewer", "instruction": "Review persistence"},
        )
    )

    accepted = state.finish_subagent_task(
        AgentToolResult(
            tool_call_id="task-huge-usage",
            name="task",
            ok=True,
            content="legacy result",
            data={
                "kind": "subagent_run",
                "version": 2,
                "status": "completed",
                "total_tokens": 1 << 20_000,
            },
        )
    )

    assert accepted is False
    assert state.items[0].role == "tool"
    assert state.items[0].subagent is None


def test_tui_state_restores_interrupted_task_as_cancelled() -> None:
    from langchain_core.messages import ToolMessage

    state = TuiState()
    state.load_messages(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "task-cancelled",
                        "name": "task",
                        "args": {"agent": "worker", "instruction": "Implement change"},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                tool_call_id="task-cancelled",
                name="task",
                content="Tool call interrupted by user",
                status="error",
            ),
        ]
    )

    assert len(state.items) == 1
    assert state.items[0].subagent is not None
    assert state.items[0].subagent.status == "cancelled"


def _trace_update(*, tool_call_id: str = "task-trace") -> ToolExecutionUpdateEvent:
    return ToolExecutionUpdateEvent(
        tool_call_id=tool_call_id,
        message="",
        data={
            "kind": "subagent_trace",
            "version": 1,
            "agent": "scout",
            "items": [
                {"kind": "human", "text": "Inspect auth"},
                {"kind": "tool_call", "tool": "read", "text": "Calling read"},
                {"kind": "tool_result", "tool": "read", "status": "ok"},
                {"kind": "assistant", "text": "Auth looks healthy."},
            ],
            "truncated": False,
            "input_tokens": 4200,
            "output_tokens": 730,
            "total_tokens": 4930,
        },
    )


def test_tui_adapter_attaches_trace_in_place_with_usage() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="task-trace",
                name="task",
                arguments={"agent": "scout", "instruction": "Inspect auth"},
            )
        )
    )

    adapter.apply(_trace_update())

    assert len(state.items) == 1
    display = state.items[0].subagent
    assert display is not None
    assert display.trace_available is True
    assert len(display.trace_items) == 4
    assert display.tool_calls == 1
    assert display.total_tokens == 4930


def test_tui_adapter_keeps_live_trace_stats_when_failed_artifact_has_no_stats() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="task-trace-failed",
                name="task",
                arguments={"agent": "scout", "instruction": "Inspect auth"},
            )
        )
    )

    adapter.apply(_trace_update(tool_call_id="task-trace-failed"))
    adapter.apply(
        ToolExecutionEndEvent(
            result=AgentToolResult(
                tool_call_id="task-trace-failed",
                name="task",
                ok=False,
                content="child failed",
                data={
                    "kind": "subagent_run",
                    "version": 2,
                    "agent": "scout",
                    "status": "failed",
                    "instruction": "Inspect auth",
                    "final_output": "",
                    "model_calls": 1,
                    "tool_calls": 0,
                    "queued_ms": 0,
                    "duration_ms": 1200,
                    "truncated": False,
                    "error": "child failed",
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                },
            )
        )
    )

    display = state.items[0].subagent
    assert display is not None
    assert display.status == "failed"
    assert display.trace_available is True
    assert len(display.trace_items) == 4
    assert display.tool_calls == 1
    assert display.input_tokens == 4200
    assert display.output_tokens == 730
    assert display.total_tokens == 4930


def test_tui_adapter_ignores_malformed_trace_without_a_panel() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="task-bad-trace",
                name="task",
                arguments={"agent": "scout", "instruction": "Inspect auth"},
            )
        )
    )
    malformed = _trace_update(tool_call_id="task-bad-trace")
    assert malformed.data is not None
    malformed.data["version"] = 99

    adapter.apply(malformed)

    display = state.items[0].subagent
    assert display is not None
    assert display.trace_available is False
    assert display.trace_items == ()


def test_tui_adapter_ignores_trace_with_extreme_usage_integer() -> None:
    state = TuiState()
    adapter = TuiEventAdapter(state)
    adapter.apply(
        ToolExecutionStartEvent(
            tool_call=ToolCall(
                id="task-huge-trace",
                name="task",
                arguments={"agent": "scout", "instruction": "Inspect auth"},
            )
        )
    )
    update = _trace_update(tool_call_id="task-huge-trace")
    assert update.data is not None
    update.data["total_tokens"] = 1 << 20_000

    adapter.apply(update)

    display = state.items[0].subagent
    assert display is not None
    assert display.trace_available is False


def test_tui_state_load_messages_restores_trace_index_without_model_messages() -> None:
    from langchain_core.messages import ToolMessage

    state = TuiState()
    state.load_messages(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "task-history-trace",
                        "name": "task",
                        "args": {"agent": "scout", "instruction": "Inspect auth"},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                tool_call_id="task-history-trace",
                name="task",
                content="Auth looks healthy.",
                artifact={
                    "kind": "subagent_run",
                    "version": 2,
                    "agent": "scout",
                    "status": "completed",
                    "instruction": "Inspect auth",
                    "final_output": "Auth looks healthy.",
                    "model_calls": 1,
                    "tool_calls": 1,
                    "queued_ms": 0,
                    "duration_ms": 1200,
                    "truncated": False,
                    "error": None,
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
            ),
        ],
        subagent_traces={
            "task-history-trace": {
                "version": 1,
                "tool_call_id": "task-history-trace",
                "agent": "scout",
                "items": [{"kind": "assistant", "text": "Auth looks healthy."}],
                "truncated": False,
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }
        },
    )

    assert len(state.items) == 1
    display = state.items[0].subagent
    assert display is not None
    assert display.trace_available is True
    assert display.final_output == "Auth looks healthy."
    assert display.total_tokens == 30
