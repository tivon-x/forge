from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from forge_agent.session import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    JsonlSessionStorage,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionJsonlError,
    SessionState,
    SessionTreeError,
    entry_from_json_line,
    entry_to_json_line,
    path_to_entry,
)


def test_session_entry_round_trips_jsonl() -> None:
    entry = MessageEntry(id="entry-1", message=HumanMessage(content="Hello"))

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert parsed == entry


def test_tool_message_metadata_round_trips_jsonl() -> None:
    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="edit",
            content="Successfully replaced 1 block.",
            artifact={
                "data": {"patch": "--- a.py\n+++ a.py\n@@\n-old\n+new"},
                "details": {"first_changed_line": 12},
            },
        ),
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert parsed == entry


def test_compaction_entry_round_trips_jsonl() -> None:
    entry = CompactionEntry(
        id="compact",
        summary="The user asked about session replay.",
        replaces_entry_ids=["user", "assistant"],
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert parsed == entry


def test_invalid_jsonl_line_raises_useful_error() -> None:
    with pytest.raises(SessionJsonlError, match="Invalid session entry on line 3"):
        entry_from_json_line('{"type":"unknown"}', line_number=3)


@pytest.mark.anyio
async def test_jsonl_storage_appends_and_reads_entries(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "sessions" / "one.jsonl")
    first = MessageEntry(id="one", message=HumanMessage(content="Hi"))
    second = LabelEntry(id="two", label="Greeting")

    await storage.append(first)
    await storage.append(second)

    assert await storage.read_all() == [first, second]


@pytest.mark.anyio
async def test_jsonl_storage_missing_file_is_empty(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "missing.jsonl")

    assert await storage.read_all() == []


def test_session_state_replays_linear_entries() -> None:
    entries = [
        MessageEntry(id="user", message=HumanMessage(content="Hi")),
        ModelChangeEntry(id="model", model="fake-model"),
        MessageEntry(id="assistant", message=AIMessage(content="Hello")),
        LabelEntry(id="label", label="Greeting"),
        CustomEntry(id="custom", namespace="test", data={"ok": True}),
        LeafEntry(id="leaf", entry_id="assistant"),
    ]

    state = SessionState.from_entries(entries)

    assert state.messages == (HumanMessage(content="Hi"), AIMessage(content="Hello"))
    assert state.model == "fake-model"
    assert state.label == "Greeting"
    assert state.active_leaf_id == "assistant"
    assert state.custom_entries == (entries[4],)
    assert state.context_entry_ids == ("user", "assistant")


def test_session_state_can_replay_explicit_empty_leaf() -> None:
    root = MessageEntry(id="root", message=HumanMessage(content="Hi"))

    state = SessionState.from_entries([root], leaf_id=None)

    assert state.messages == ()
    assert state.active_leaf_id is None
    assert state.context_entry_ids == ()


def test_session_state_replays_compaction_as_context_summary() -> None:
    user = MessageEntry(id="user", message=HumanMessage(content="Explain sessions."))
    assistant = MessageEntry(
        id="assistant",
        parent_id="user",
        message=AIMessage(content="Sessions are append-only."),
    )
    compaction = CompactionEntry(
        id="compact",
        parent_id="assistant",
        summary="The user asked about sessions. The assistant explained append-only replay.",
        replaces_entry_ids=["user", "assistant"],
    )
    followup = MessageEntry(
        id="followup",
        parent_id="compact",
        message=HumanMessage(content="Continue."),
    )

    state = SessionState.from_entries([user, assistant, compaction, followup])

    assert state.messages == (
        HumanMessage(
            content=(
                "Previous conversation summary:\n"
                "The user asked about sessions. The assistant explained append-only replay."
            )
        ),
        HumanMessage(content="Continue."),
    )
    assert state.compaction_entries == (compaction,)
    assert state.context_entry_ids == ("compact", "followup")


def test_session_state_inserts_partial_compaction_before_retained_messages() -> None:
    old_user = MessageEntry(id="old-user", message=HumanMessage(content="Old request"))
    old_assistant = MessageEntry(
        id="old-assistant",
        parent_id="old-user",
        message=AIMessage(content="Old answer"),
    )
    recent_user = MessageEntry(
        id="recent-user",
        parent_id="old-assistant",
        message=HumanMessage(content="Recent request"),
    )
    recent_assistant = MessageEntry(
        id="recent-assistant",
        parent_id="recent-user",
        message=AIMessage(content="Recent answer"),
    )
    compaction = CompactionEntry(
        id="compact",
        parent_id="recent-assistant",
        summary="Older work was summarized.",
        replaces_entry_ids=["old-user", "old-assistant"],
    )

    state = SessionState.from_entries(
        [old_user, old_assistant, recent_user, recent_assistant, compaction]
    )

    assert state.messages == (
        HumanMessage(content="Previous conversation summary:\nOlder work was summarized."),
        HumanMessage(content="Recent request"),
        AIMessage(content="Recent answer"),
    )
    assert state.context_entry_ids == ("compact", "recent-user", "recent-assistant")


def test_session_state_replays_branch_summary_as_context_summary() -> None:
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    summary = BranchSummaryEntry(
        id="branch-summary",
        parent_id="root",
        branch_root_id="root",
        summary="The abandoned branch explored an alternate implementation.",
    )

    state = SessionState.from_entries([root, summary], leaf_id="branch-summary")

    assert state.messages == (
        HumanMessage(content="Root"),
        HumanMessage(
            content=(
                "The following is a summary of a branch that this conversation came back from:\n"
                "<summary>\n"
                "The abandoned branch explored an alternate implementation.\n"
                "</summary>"
            )
        ),
    )
    assert state.context_entry_ids == ("root", "branch-summary")


def test_path_to_entry_returns_root_to_leaf_branch() -> None:
    root = MessageEntry(id="root", message=HumanMessage(content="Hi"))
    left = MessageEntry(id="left", parent_id="root", message=AIMessage(content="Left"))
    right = MessageEntry(id="right", parent_id="root", message=AIMessage(content="Right"))

    assert path_to_entry([root, left, right], "right") == [root, right]


def test_session_state_can_replay_one_branch() -> None:
    root = MessageEntry(id="root", message=HumanMessage(content="Hi"))
    left = MessageEntry(id="left", parent_id="root", message=AIMessage(content="Left"))
    right = MessageEntry(id="right", parent_id="root", message=AIMessage(content="Right"))

    state = SessionState.from_entries([root, left, right], leaf_id="right")

    assert state.messages == (HumanMessage(content="Hi"), AIMessage(content="Right"))
    assert state.active_leaf_id == "right"
    assert state.entries == (root, right)


def test_session_state_replays_compaction_on_active_branch() -> None:
    root = MessageEntry(id="root", message=HumanMessage(content="Root"))
    left = MessageEntry(id="left", parent_id="root", message=AIMessage(content="Left"))
    compact = CompactionEntry(
        id="compact",
        parent_id="left",
        summary="Root and left branch summary.",
        replaces_entry_ids=["root", "left"],
    )
    right = MessageEntry(id="right", parent_id="root", message=AIMessage(content="Right"))

    state = SessionState.from_entries([root, left, compact, right], leaf_id="compact")

    assert state.messages == (
        HumanMessage(content="Previous conversation summary:\nRoot and left branch summary."),
    )
    assert state.entries == (root, left, compact)


def test_path_to_entry_rejects_missing_parent() -> None:
    entry = MessageEntry(id="child", parent_id="missing", message=HumanMessage(content="Hi"))

    with pytest.raises(SessionTreeError, match="Missing session entry"):
        path_to_entry([entry], "child")
