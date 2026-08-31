from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from fake_models import ScriptedChatModel
from forge_agent.session import JsonlSessionStorage
from forge_coding import CodingSession, CodingSessionConfig, ForgePaths, SessionManager
from forge_coding.sessions import session as session_module
from forge_coding.sessions.clipboard import ClipboardResult, ClipboardStatus
from forge_coding.sessions.copying import SessionCopyError


async def _managed_session(
    tmp_path: Path, response: str = "Answer"
) -> tuple[
    CodingSession,
    SessionManager,
    Path,
]:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake", title="Original")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel([AIMessage(content=response)]),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(record.path),
            cwd=tmp_path,
            session_id=record.id,
            session_manager=manager,
        )
    )
    async for _event in session.prompt("User request"):
        pass
    return session, manager, record.path.read_bytes()


@pytest.mark.anyio
async def test_clone_switches_to_copy_and_keeps_source_unchanged(tmp_path: Path) -> None:
    session, manager, source_before = await _managed_session(tmp_path)

    message = await session.clone_current_session()

    assert message.startswith("Cloned session: ")
    assert session.session_id is not None
    records = manager.list_sessions()
    assert len(records) == 2
    copied = manager.get_session(session.session_id)
    assert copied is not None
    assert copied.title == "Original (copy)"
    source = next(record for record in records if record.id != session.session_id)
    assert source.path.read_bytes() == source_before
    assert copied.path.read_bytes()


@pytest.mark.anyio
async def test_clone_load_failure_removes_unindexed_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, manager, source_before = await _managed_session(tmp_path)
    source = manager.get_session(session.session_id or "")
    assert source is not None

    async def fail_load(*_args: object, **_kwargs: object) -> CodingSession:
        raise RuntimeError("load failed")

    monkeypatch.setattr(session, "_load_copied_replacement", fail_load)

    with pytest.raises(RuntimeError, match="load failed"):
        await session.clone_current_session()

    assert manager.list_sessions() == [source]
    session_files = [
        path for path in source.path.parent.glob("*.jsonl") if path.name != "index.jsonl"
    ]
    assert session_files == [source.path]
    assert source.path.read_bytes() == source_before


@pytest.mark.anyio
async def test_clone_conflict_never_deletes_existing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, manager, source_before = await _managed_session(tmp_path)
    source = manager.get_session(session.session_id or "")
    assert source is not None
    monkeypatch.setattr(manager, "prepare_session", lambda **_kwargs: source)

    with pytest.raises(SessionCopyError, match="paths must differ"):
        await session.clone_current_session()

    assert source.path.read_bytes() == source_before
    assert manager.get_session(source.id) == source


@pytest.mark.anyio
async def test_clone_initializes_an_empty_session_before_copying(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake", title="Empty")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel([]),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(record.path),
            cwd=tmp_path,
            session_id=record.id,
            session_manager=manager,
        )
    )
    assert not record.path.exists()

    await session.clone_current_session()

    assert record.path.exists()
    assert len(manager.list_sessions()) == 2


@pytest.mark.anyio
async def test_fork_copies_parent_and_prefills_selected_human_message(
    tmp_path: Path,
) -> None:
    session, manager, source_before = await _managed_session(tmp_path)
    entries = await session.storage.read_all()
    selected = next(entry for entry in entries if entry.type == "message")
    assert isinstance(selected.message, HumanMessage)

    choices = await session.fork_choices()
    assert [choice.entry_id for choice in choices] == [selected.id]

    result = await session.fork_from_entry(selected.id)

    assert result.input_prefill == "User request"
    assert session.messages == ()
    copied = manager.get_session(session.session_id or "")
    assert copied is not None
    assert copied.title == "Original (fork)"
    source = next(record for record in manager.list_sessions() if record.id != session.session_id)
    assert source.path.read_bytes() == source_before


@pytest.mark.anyio
async def test_copy_last_assistant_uses_active_unicode_text_and_bounded_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _manager, _source_before = await _managed_session(tmp_path, response="回答 ✅")
    copied: list[str] = []

    def fake_copy(text: str) -> ClipboardResult:
        copied.append(text)
        return ClipboardResult(ClipboardStatus.COPIED, command="fake")

    monkeypatch.setattr(session_module, "copy_to_clipboard", fake_copy)

    result = await session.copy_last_assistant()

    assert copied == ["回答 ✅"]
    assert "Copied" in result


@pytest.mark.anyio
async def test_copy_last_assistant_without_response_does_not_invoke_clipboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake", title="Empty")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel([]),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(record.path),
            cwd=tmp_path,
            session_id=record.id,
            session_manager=manager,
        )
    )
    monkeypatch.setattr(
        session_module,
        "copy_to_clipboard",
        lambda _text: pytest.fail("clipboard should not be called"),
    )

    result = await session.copy_last_assistant()

    assert "No assistant response" in result
