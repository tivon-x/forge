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


# --------------------------------------------------------------------------- #
# Torn JSONL tail recovery: only a trailing incomplete JSON fragment is
# recoverable; middle corruption and complete-but-invalid lines stay errors.
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_read_all_ignores_torn_tail_line(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first = MessageEntry(id="one", message=HumanMessage(content="Hi"))
    storage.path.write_text(
        entry_to_json_line(first) + '{"type": "message", "mess',
        encoding="utf-8",
    )

    assert await storage.read_all() == [first]


@pytest.mark.anyio
async def test_read_all_tolerates_single_torn_line_file(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    storage.path.write_text('{"type": "message", "mess', encoding="utf-8")

    assert await storage.read_all() == []


@pytest.mark.anyio
async def test_read_all_empty_file_is_empty_session(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    storage.path.write_text("", encoding="utf-8")

    assert await storage.read_all() == []


@pytest.mark.anyio
async def test_read_all_normal_trailing_newline_reads_all_entries(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first = MessageEntry(id="one", message=HumanMessage(content="Hi"))
    second = LabelEntry(id="two", label="Greeting")
    storage.path.write_text(
        entry_to_json_line(first) + entry_to_json_line(second),
        encoding="utf-8",
    )

    assert await storage.read_all() == [first, second]


@pytest.mark.anyio
async def test_read_all_keeps_complete_line_without_trailing_newline(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first = MessageEntry(id="one", message=HumanMessage(content="Hi"))
    storage.path.write_text(entry_to_json_line(first).rstrip("\n"), encoding="utf-8")

    assert await storage.read_all() == [first]


@pytest.mark.anyio
async def test_read_all_rejects_complete_but_invalid_last_line(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    storage.path.write_text('{"type":"unknown"}', encoding="utf-8")

    with pytest.raises(SessionJsonlError, match="Invalid session entry on line 1"):
        await storage.read_all()


@pytest.mark.anyio
async def test_read_all_rejects_middle_corruption(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    storage.path.write_text(
        entry_to_json_line(MessageEntry(id="one", message=HumanMessage(content="Hi")))
        + "not json at all\n"
        + entry_to_json_line(LabelEntry(id="two", label="Greeting")),
        encoding="utf-8",
    )

    with pytest.raises(SessionJsonlError, match="line 2"):
        await storage.read_all()


@pytest.mark.anyio
async def test_read_all_ignores_torn_tail_splitting_utf8_character(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first = MessageEntry(id="one", message=HumanMessage(content="中文记录"))
    # Torn tail splits the middle of a multi-byte UTF-8 character; the raw
    # bytes are not decodable, but read_all must still return the record.
    torn_tail = b'{"type": "message", "mess' + "中".encode()[:1]
    storage.path.write_bytes(entry_to_json_line(first).encode("utf-8") + torn_tail)

    assert await storage.read_all() == [first]


@pytest.mark.anyio
async def test_append_truncates_utf8_torn_tail_without_corrupting_record(
    tmp_path: Path,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first = MessageEntry(id="one", message=HumanMessage(content="中文记录"))
    torn_tail = b'{"type": "message", "mess' + "中".encode()[:1]
    storage.path.write_bytes(entry_to_json_line(first).encode("utf-8") + torn_tail)

    second = LabelEntry(id="two", label="后续")
    await storage.append(second)

    assert await storage.read_all() == [first, second]
    # The Chinese record survives byte-exact.
    raw = storage.path.read_bytes()
    assert raw.decode("utf-8") == entry_to_json_line(first) + entry_to_json_line(second)


@pytest.mark.anyio
async def test_append_truncates_torn_tail_then_continues(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first = MessageEntry(id="one", message=HumanMessage(content="Hi"))
    storage.path.write_text(
        entry_to_json_line(first) + '{"type": "message", "mess',
        encoding="utf-8",
    )

    second = LabelEntry(id="two", label="Greeting")
    await storage.append(second)

    assert await storage.read_all() == [first, second]


@pytest.mark.anyio
async def test_append_separates_complete_tail_without_newline(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    first = MessageEntry(id="one", message=HumanMessage(content="Hi"))
    storage.path.write_text(entry_to_json_line(first).rstrip("\n"), encoding="utf-8")

    second = LabelEntry(id="two", label="Greeting")
    await storage.append(second)

    assert await storage.read_all() == [first, second]


# --------------------------------------------------------------------------- #
# ToolMessage artifact persistence degrades safely: JSON-compatible artifacts
# round-trip unchanged; arbitrary objects and bytes become a stable placeholder
# that never leaks repr()/raw bytes, while content/tool_call_id/name/status and
# response/usage metadata survive.
# --------------------------------------------------------------------------- #
def test_tool_artifact_json_compatible_round_trips(tmp_path: Path) -> None:
    artifact = {
        "data": {"patch": "--- a\n+++ b"},
        "extra": {"nested": {"flag": False}},
        "count": 3,
        "ok": True,
        "nothing": None,
        "items": ["a", "b"],
    }
    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="edit",
            content="done",
            status="success",
            artifact=artifact,
            response_metadata={"headers": {"x": "y"}},
            usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        ),
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    assert parsed.message.content == "done"
    assert parsed.message.tool_call_id == "call-1"
    assert parsed.message.name == "edit"
    assert parsed.message.status == "success"
    assert parsed.message.response_metadata == {"headers": {"x": "y"}}
    assert parsed.message.usage_metadata == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }
    assert (parsed.message.artifact or {}).get("data") == {"patch": "--- a\n+++ b"}
    assert (parsed.message.artifact or {}).get("extra") == {"nested": {"flag": False}}
    assert (parsed.message.artifact or {}).get("count") == 3
    assert (parsed.message.artifact or {}).get("items") == ["a", "b"]


def test_tool_artifact_arbitrary_object_becomes_omission_placeholder() -> None:
    class Opaque:
        secret = "must-not-leak"

    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="third_party",
            content="kept",
            artifact=Opaque(),
        ),
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    assert parsed.message.content == "kept"
    assert parsed.message.tool_call_id == "call-1"
    assert parsed.message.name == "third_party"
    expected_type = f"{type(Opaque()).__module__}.{type(Opaque()).__qualname__}"
    assert parsed.message.artifact == {
        "forge_serialization": {"status": "omitted", "python_type": expected_type}
    }
    assert "must-not-leak" not in line


def test_tool_artifact_utf8_bytes_becomes_omission_placeholder() -> None:
    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="third_party",
            content="kept",
            artifact={"payload": b"hello-utf8"},
        ),
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    assert parsed.message.artifact == {
        "forge_serialization": {"status": "omitted", "python_type": "builtins.dict"}
    }
    assert "hello-utf8" not in line


def test_tool_artifact_non_utf8_bytes_becomes_omission_placeholder() -> None:
    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="third_party",
            content="kept",
            artifact=b"\xff\xfe\x00binary",
        ),
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    assert parsed.message.artifact == {
        "forge_serialization": {"status": "omitted", "python_type": "builtins.bytes"}
    }
    assert line.count("\xff") == 0


def test_tool_artifact_placeholder_preserves_tool_call_pairing(tmp_path: Path) -> None:
    from forge_agent.session import entry_from_json_line

    tool_message = ToolMessage(
        tool_call_id="call-7",
        name="read",
        content="file contents",
        status="success",
        artifact=object(),
    )
    entry = MessageEntry(id="entry-1", message=tool_message)
    line = entry_to_json_line(entry)

    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    assert parsed.message.tool_call_id == "call-7"
    assert parsed.message.name == "read"
    assert parsed.message.content == "file contents"
    assert parsed.message.status == "success"


def test_tool_artifact_pydantic_model_with_bytes_field_is_omitted() -> None:
    from pydantic import BaseModel

    class Payload(BaseModel):
        name: str = "x"
        blob: bytes = b"utf8-bytes"

    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="third_party",
            content="kept",
            artifact=Payload(),
        ),
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    expected_type = f"{type(Payload()).__module__}.{type(Payload()).__qualname__}"
    assert parsed.message.artifact == {
        "forge_serialization": {"status": "omitted", "python_type": expected_type}
    }
    assert "utf8-bytes" not in line


def test_tool_artifact_self_referencing_dict_is_omitted() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="third_party",
            content="kept",
            artifact=cyclic,
        ),
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    assert parsed.message.artifact == {
        "forge_serialization": {"status": "omitted", "python_type": "builtins.dict"}
    }


def test_tool_artifact_self_referencing_list_is_omitted() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="third_party",
            content="kept",
            artifact=cyclic,
        ),
    )

    line = entry_to_json_line(entry)

    parsed = entry_from_json_line(line)
    assert isinstance(parsed, MessageEntry)
    assert parsed.message.artifact == {
        "forge_serialization": {"status": "omitted", "python_type": "builtins.list"}
    }


def test_tool_artifact_self_referencing_pydantic_model_is_omitted() -> None:
    from pydantic import BaseModel

    class Node(BaseModel):
        name: str = "root"
        child: object = None

    node = Node()
    node.child = node  # type: ignore[assignment]

    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="third_party",
            content="kept",
            artifact=node,
        ),
    )

    line = entry_to_json_line(entry)  # must not raise RecursionError
    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    assert parsed.message.artifact == {
        "forge_serialization": {
            "status": "omitted",
            "python_type": f"{type(node).__module__}.{type(node).__qualname__}",
        }
    }


def test_tool_artifact_dataclass_with_bytes_is_omitted() -> None:
    from dataclasses import dataclass

    @dataclass
    class Payload:
        name: str = "x"
        payload: bytes = b"hello-dataclass-bytes"

    entry = MessageEntry(
        id="entry-1",
        message=ToolMessage(
            tool_call_id="call-1",
            name="third_party",
            content="kept",
            artifact=Payload(),
        ),
    )

    line = entry_to_json_line(entry)
    parsed = entry_from_json_line(line)

    assert isinstance(parsed, MessageEntry)
    # bytes anywhere must be omitted -- never serialized to plaintext.
    assert "hello-dataclass-bytes" not in line
    assert parsed.message.artifact == {
        "forge_serialization": {
            "status": "omitted",
            "python_type": f"{type(Payload()).__module__}.{type(Payload()).__qualname__}",
        }
    }
