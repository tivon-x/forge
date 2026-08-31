from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from forge_agent.session import (
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    entry_to_json_line,
)
from forge_coding.paths import ForgePaths
from forge_coding.resources import ForgeResourcePaths, TrustStore
from forge_coding.sessions.importing import (
    SessionImportDenied,
    SessionImportError,
    commit_session_import,
    parse_import_session,
    prepare_session_import,
)
from forge_coding.sessions.manager import SessionManager


def _write_session(path: Path, entries: list[object], *, suffix: str = "") -> None:
    path.write_text(
        "".join(entry_to_json_line(entry) for entry in entries) + suffix,
        encoding="utf-8",
    )


def _valid_entries(cwd: Path) -> list[object]:
    info = SessionInfoEntry(id="info", cwd=str(cwd))
    model = ModelChangeEntry(id="model", parent_id=info.id, model="fake")
    user = MessageEntry(id="user", parent_id=model.id, message=HumanMessage(content="Hello"))
    leaf = LeafEntry(id="leaf", parent_id=user.id, entry_id=user.id)
    return [info, model, user, leaf]


def test_parse_import_session_streams_valid_entries_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    entries = _valid_entries(tmp_path)
    _write_session(source, entries)

    parsed = parse_import_session(source)

    assert parsed.source_path == source.resolve()
    assert parsed.entries == tuple(entries)
    assert parsed.session_info == entries[0]
    assert parsed.cwd == tmp_path.resolve()
    assert parsed.active_leaf == entries[-1]
    assert parsed.source_leaf_id == "user"


def test_parse_import_session_tolerates_only_final_torn_tail_without_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    entries = _valid_entries(tmp_path)
    _write_session(source, entries, suffix='{"type":"message","mess')
    before = source.read_bytes()

    parsed = parse_import_session(source)

    assert parsed.entries == tuple(entries)
    assert source.read_bytes() == before


def test_parse_import_session_rejects_middle_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    entries = _valid_entries(tmp_path)
    source.write_text(
        entry_to_json_line(entries[0]) + "not json\n" + entry_to_json_line(entries[1]),
        encoding="utf-8",
    )

    with pytest.raises(SessionImportError, match="line 2"):
        parse_import_session(source)


def test_parse_import_session_rejects_complete_but_invalid_final_record(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_session(source, _valid_entries(tmp_path), suffix='{"type":"unknown"}')

    with pytest.raises(SessionImportError, match="line 5"):
        parse_import_session(source)


def test_parse_import_session_requires_one_session_info(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_session(source, [LabelEntry(id="label", label="missing info")])

    with pytest.raises(SessionImportError, match="missing SessionInfo"):
        parse_import_session(source)


def test_parse_import_session_rejects_duplicate_ids_and_dangling_parents(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    _write_session(
        duplicate,
        [
            SessionInfoEntry(id="same", cwd=str(tmp_path)),
            LabelEntry(id="same", label="duplicate"),
        ],
    )
    with pytest.raises(SessionImportError, match="duplicate entry ids"):
        parse_import_session(duplicate)

    dangling = tmp_path / "dangling.jsonl"
    _write_session(
        dangling,
        [
            SessionInfoEntry(id="info", cwd=str(tmp_path)),
            LabelEntry(id="child", parent_id="missing", label="dangling"),
        ],
    )
    with pytest.raises(SessionImportError, match="dangling parent"):
        parse_import_session(dangling)


def test_parse_import_session_rejects_parent_cycles(tmp_path: Path) -> None:
    source = tmp_path / "cycle.jsonl"
    _write_session(
        source,
        [
            SessionInfoEntry(id="info", cwd=str(tmp_path)),
            LabelEntry(id="left", parent_id="right", label="left"),
            LabelEntry(id="right", parent_id="left", label="right"),
        ],
    )

    with pytest.raises(SessionImportError, match="parent cycle"):
        parse_import_session(source)


def test_parse_import_session_rejects_non_root_info_and_leaf_ancestor(
    tmp_path: Path,
) -> None:
    non_root = tmp_path / "non-root.jsonl"
    root = LabelEntry(id="root", label="root")
    info = SessionInfoEntry(id="info", parent_id=root.id, cwd=str(tmp_path))
    _write_session(non_root, [root, info])
    with pytest.raises(SessionImportError, match="root entry"):
        parse_import_session(non_root)

    leaf_ancestor = tmp_path / "leaf-ancestor.jsonl"
    info = SessionInfoEntry(id="info", cwd=str(tmp_path))
    user = MessageEntry(
        id="user",
        parent_id=info.id,
        message=HumanMessage(content="first"),
    )
    old_leaf = LeafEntry(id="old-leaf", parent_id=user.id, entry_id=user.id)
    later = MessageEntry(
        id="later",
        parent_id=old_leaf.id,
        message=HumanMessage(content="later"),
    )
    active_leaf = LeafEntry(id="active-leaf", parent_id=later.id, entry_id=later.id)
    _write_session(leaf_ancestor, [info, user, old_leaf, later, active_leaf])
    with pytest.raises(SessionImportError, match="invalid active path"):
        parse_import_session(leaf_ancestor)


@pytest.mark.parametrize("cwd_kind", ["missing", "file", "none", "relative"])
def test_parse_import_session_rejects_invalid_cwd(tmp_path: Path, cwd_kind: str) -> None:
    source = tmp_path / f"{cwd_kind}.jsonl"
    if cwd_kind == "missing":
        cwd: str | None = str(tmp_path / "does-not-exist")
    elif cwd_kind == "file":
        file_path = tmp_path / "not-a-directory"
        file_path.write_text("", encoding="utf-8")
        cwd = str(file_path)
    elif cwd_kind == "relative":
        cwd = "."
    else:
        cwd = None
    _write_session(source, [SessionInfoEntry(id="info", cwd=cwd)])

    with pytest.raises(SessionImportError, match="cwd"):
        parse_import_session(source)


@pytest.mark.anyio
async def test_import_trust_deny_writes_no_destination_and_allow_reuses_validated_entries(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".forge" / "skills" / "unsafe").mkdir(parents=True)
    (project / ".forge" / "skills" / "unsafe" / "SKILL.md").write_text(
        "# project skill", encoding="utf-8"
    )
    source = tmp_path / "source.jsonl"
    entries = _valid_entries(project)
    _write_session(source, entries)
    store = TrustStore(tmp_path / "trust.json")
    paths = ForgeResourcePaths(cwd=project)
    prepared = await prepare_session_import(source, paths=paths, store=store)
    assert prepared.cwd == project.resolve()
    assert prepared.trust_required is True

    manager = SessionManager(ForgePaths(home=tmp_path / "home", agents_home=tmp_path / "agents"))
    with pytest.raises(SessionImportDenied):
        await commit_session_import(prepared, "deny", manager=manager, model="fake")
    assert manager.list_sessions() == []
    assert not manager.paths.sessions_dir.exists()

    # The commit must use the validated immutable entries rather than reading
    # a changed source file again.
    source.write_text("corrupt now\n", encoding="utf-8")
    copied = await commit_session_import(prepared, "once", manager=manager, model="fake")
    assert copied.path.exists()
    assert manager.get_session(copied.id) == copied


@pytest.mark.anyio
async def test_import_cancel_stays_cancelled_if_resources_disappear(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    skill = project / ".forge" / "skills" / "unsafe" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# project skill", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    _write_session(source, _valid_entries(project))
    prepared = await prepare_session_import(
        source,
        paths=ForgeResourcePaths(cwd=project),
        store=TrustStore(tmp_path / "trust.json"),
    )
    manager = SessionManager(ForgePaths(home=tmp_path / "home", agents_home=tmp_path / "agents"))
    skill.unlink()

    with pytest.raises(SessionImportDenied):
        await commit_session_import(prepared, "deny", manager=manager, model="fake")

    assert manager.list_sessions() == []
    assert not manager.paths.sessions_dir.exists()


@pytest.mark.anyio
async def test_import_rejects_cwd_removed_after_preflight_without_writing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source.jsonl"
    _write_session(source, _valid_entries(project))
    prepared = await prepare_session_import(source)
    manager = SessionManager(ForgePaths(home=tmp_path / "home", agents_home=tmp_path / "agents"))
    project.rmdir()

    with pytest.raises(SessionImportError, match="cwd no longer exists"):
        await commit_session_import(prepared, manager=manager, model="fake")

    assert manager.list_sessions() == []
    assert not manager.paths.sessions_dir.exists()


@pytest.mark.anyio
async def test_import_destination_conflict_preserves_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_session(source, _valid_entries(tmp_path))
    prepared = await prepare_session_import(source)
    manager = SessionManager(ForgePaths(home=tmp_path / "home", agents_home=tmp_path / "agents"))
    destination = manager.prepare_session(cwd=tmp_path, model="fake")
    destination.path.write_bytes(b"existing")
    manager.prepare_session = lambda **_kwargs: destination  # type: ignore[method-assign]

    with pytest.raises(SessionImportError, match="already exists"):
        await commit_session_import(prepared, manager=manager, model="fake")

    assert destination.path.read_bytes() == b"existing"


@pytest.mark.anyio
async def test_prepare_import_explicit_policy_deny_fails_without_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".forge" / "prompts").mkdir(parents=True)
    (project / ".forge" / "prompts" / "unsafe.md").write_text("# prompt", encoding="utf-8")
    _write_session(source, _valid_entries(project))

    with pytest.raises(SessionImportDenied):
        await prepare_session_import(
            source,
            paths=ForgeResourcePaths(cwd=project),
            store=TrustStore(tmp_path / "trust.json"),
            env={"FORGE_TRUST": "never"},
        )
