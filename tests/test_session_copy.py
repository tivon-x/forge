from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from forge_agent.session import (
    CustomEntry,
    JsonlSessionStorage,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
)
from forge_coding.sessions.copying import (
    SESSION_ORIGIN_NAMESPACE,
    SessionCopyError,
    copy_active_branch,
)
from forge_coding.sessions.manager import CodingSessionRecord


def _record(path: Path, *, session_id: str, cwd: Path) -> CodingSessionRecord:
    return CodingSessionRecord(
        id=session_id,
        path=path,
        cwd=cwd,
        model="fake",
        title="Copied session",
        created_at=20.0,
        updated_at=20.0,
    )


@pytest.mark.anyio
async def test_copy_active_branch_keeps_only_path_and_replaces_session_info(
    tmp_path: Path,
) -> None:
    source_record = _record(tmp_path / "source.jsonl", session_id="source", cwd=tmp_path)
    destination_record = _record(
        tmp_path / "destination.jsonl",
        session_id="destination",
        cwd=tmp_path / "project",
    )
    destination_record.cwd.mkdir()
    info = SessionInfoEntry(id="info", created_at=1.0, cwd=str(tmp_path), title="Original")
    model = ModelChangeEntry(id="model", parent_id=info.id, model="old-model")
    user = MessageEntry(id="user", parent_id=model.id, message=HumanMessage(content="Keep"))
    assistant = MessageEntry(
        id="assistant",
        parent_id=user.id,
        message=AIMessage(content="Answer"),
    )
    custom = CustomEntry(id="custom", parent_id=assistant.id, namespace="test", data={"ok": True})
    sibling = MessageEntry(
        id="sibling",
        parent_id=user.id,
        message=AIMessage(content="Drop me"),
    )
    entries: list[SessionEntry] = [
        info,
        model,
        user,
        assistant,
        custom,
        sibling,
        LeafEntry(id="old-leaf", parent_id=custom.id, entry_id=custom.id),
        LeafEntry(id="sibling-leaf", parent_id=sibling.id, entry_id=sibling.id),
    ]
    source_storage = JsonlSessionStorage(source_record.path)
    for entry in entries:
        await source_storage.append(entry)
    source_before = source_record.path.read_bytes()

    result = await copy_active_branch(source_record, custom.id, destination_record)

    assert result == destination_record
    assert source_record.path.read_bytes() == source_before
    copied = await JsonlSessionStorage(destination_record.path).read_all()
    assert [entry.id for entry in copied[:6]] == [
        copied[0].id,
        copied[1].id,
        model.id,
        user.id,
        assistant.id,
        custom.id,
    ]
    assert isinstance(copied[0], SessionInfoEntry)
    assert copied[0] != info
    assert copied[0].cwd == str(destination_record.cwd)
    assert copied[0].title == destination_record.title
    assert sibling.id not in {entry.id for entry in copied}
    assert all(
        not isinstance(entry, LeafEntry) or entry.id not in {"old-leaf", "sibling-leaf"}
        for entry in copied
    )
    origin = copied[1]
    assert isinstance(origin, CustomEntry)
    assert origin.namespace == SESSION_ORIGIN_NAMESPACE
    assert origin.data == {"source_session_id": source_record.id, "source_leaf_id": custom.id}
    assert origin.parent_id == copied[0].id
    assert copied[2].parent_id == origin.id
    leaf = copied[-1]
    assert isinstance(leaf, LeafEntry)
    assert leaf.entry_id == custom.id
    assert leaf.parent_id == custom.id


@pytest.mark.anyio
async def test_copy_active_branch_rejects_conflict_without_changing_source(tmp_path: Path) -> None:
    source = _record(tmp_path / "source.jsonl", session_id="source", cwd=tmp_path)
    destination = _record(tmp_path / "destination.jsonl", session_id="destination", cwd=tmp_path)
    info = SessionInfoEntry(id="info", cwd=str(tmp_path))
    user = MessageEntry(id="user", parent_id=info.id, message=HumanMessage(content="Hi"))
    await JsonlSessionStorage(source.path).append(info)
    await JsonlSessionStorage(source.path).append(user)
    source_before = source.path.read_bytes()
    destination.path.write_bytes(b"keep me")

    with pytest.raises(SessionCopyError, match="already exists"):
        await copy_active_branch(source, user.id, destination)

    assert source.path.read_bytes() == source_before
    assert destination.path.read_bytes() == b"keep me"


@pytest.mark.anyio
async def test_copy_active_branch_failure_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _record(tmp_path / "source.jsonl", session_id="source", cwd=tmp_path)
    destination = _record(tmp_path / "destination.jsonl", session_id="destination", cwd=tmp_path)
    info = SessionInfoEntry(id="info", cwd=str(tmp_path))
    user = MessageEntry(id="user", parent_id=info.id, message=HumanMessage(content="Hi"))
    await JsonlSessionStorage(source.path).append(info)
    await JsonlSessionStorage(source.path).append(user)
    source_before = source.path.read_bytes()

    def fail_link(_source: Path, _destination: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(SessionCopyError, match="Cannot create the destination"):
        await copy_active_branch(source, user.id, destination)

    assert not destination.path.exists()
    assert source.path.read_bytes() == source_before
    assert list(tmp_path.glob(f".{destination.path.name}.*.tmp")) == []
