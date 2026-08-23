from __future__ import annotations

import io
import json
import multiprocessing
import os
from pathlib import Path
from threading import Barrier, Thread

import pytest

from forge_coding import ForgePaths, ForgeResourcePaths
from forge_coding.resources import (
    TrustStore,
    canonical_path,
    resolve_project_trust,
)
from forge_coding.resources.discovery import discover_project_context_with_diagnostics
from forge_coding.resources.prompt_templates import load_prompt_templates_with_diagnostics
from forge_coding.resources.skills import load_skills_with_diagnostics


def _store_process_worker(store_path: str, cwd: str) -> None:
    TrustStore(Path(store_path)).set(Path(cwd), "allow")


def _paths(tmp_path: Path, cwd: Path, *, allowed: bool = True) -> ForgeResourcePaths:
    forge_home = tmp_path / "home" / ".forge"
    agents_home = tmp_path / "home" / ".agents"
    return ForgeResourcePaths(
        root=forge_home,
        agents_root=agents_home,
        cwd=cwd,
        paths=ForgePaths(home=forge_home, agents_home=agents_home),
        project_resources_allowed=allowed,
    )


def _project_with_prompt(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    prompts = project / ".forge" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "unsafe.md").write_text("project content", encoding="utf-8")
    return project


def test_preflight_only_checks_project_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project_with_prompt(tmp_path)
    store = TrustStore(tmp_path / "home" / ".forge" / "trust.json")
    original = Path.read_text
    reads: list[Path] = []

    def spy(path: Path, *args: object, **kwargs: object) -> str:
        if project in path.parents or path == project:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    result = resolve_project_trust(project, paths=_paths(tmp_path, project), store=store)

    assert result.project_resources_present is True
    assert result.project_resources_allowed is False
    assert reads == []


def test_precedence_cli_then_environment_then_session_then_store(tmp_path: Path) -> None:
    project = _project_with_prompt(tmp_path)
    store = TrustStore(tmp_path / "home" / ".forge" / "trust.json")
    store.set(project, "allow")
    paths = _paths(tmp_path, project)

    assert resolve_project_trust(
        project,
        paths=paths,
        store=store,
        cli_override="no",
        env={"FORGE_TRUST": "always"},
    ).project_resources_allowed is False
    assert resolve_project_trust(
        project,
        paths=paths,
        store=store,
        env={"FORGE_TRUST": "never"},
    ).project_resources_allowed is False
    assert resolve_project_trust(
        project,
        paths=paths,
        store=store,
        session_decision="allow",
    ).project_resources_allowed is True
    assert resolve_project_trust(
        project,
        paths=paths,
        store=store,
    ).project_resources_allowed is True


def test_store_uses_canonical_paths_and_nearest_ancestor(tmp_path: Path) -> None:
    project = _project_with_prompt(tmp_path)
    nested = project / "src"
    nested.mkdir()
    (nested / ".forge" / "prompts").mkdir(parents=True)
    (nested / ".forge" / "prompts" / "nested.md").write_text("nested", encoding="utf-8")
    store = TrustStore(tmp_path / "home" / ".forge" / "trust.json")
    store.set(project, "allow")

    result = resolve_project_trust(nested, paths=_paths(tmp_path, nested), store=store)
    assert result.project_resources_allowed is True
    assert result.source == "trust store"

    if hasattr(Path, "symlink_to"):
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(project, target_is_directory=True)
        except OSError:
            pass
        else:
            assert canonical_path(alias) == canonical_path(project)
            assert resolve_project_trust(
                alias,
                paths=_paths(tmp_path, alias),
                store=store,
            ).project_resources_allowed is True


def test_trusting_parent_replaces_conflicting_exact_decision(tmp_path: Path) -> None:
    project = _project_with_prompt(tmp_path)
    store = TrustStore(tmp_path / "home" / ".forge" / "trust.json")
    store.set(project, "deny")

    store.set(project, "allow", scope="parent")

    resolved = store.lookup(project)
    assert resolved is not None
    record, matched_path = resolved
    assert record.decision == "allow"
    assert matched_path == canonical_path(project).parent


def test_corrupt_store_fails_closed_and_is_not_overwritten(tmp_path: Path) -> None:
    project = _project_with_prompt(tmp_path)
    path = tmp_path / "home" / ".forge" / "trust.json"
    path.parent.mkdir(parents=True)
    original = "{not json\n"
    path.write_text(original, encoding="utf-8")
    store = TrustStore(path)

    result = resolve_project_trust(project, paths=_paths(tmp_path, project), store=store)
    assert result.project_resources_allowed is False
    assert path.read_text(encoding="utf-8") == original
    with pytest.raises(ValueError, match="corrupt"):
        store.set(project, "allow")
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({"decisions": {}}, "version"),
        ({"version": 2, "decisions": {}}, "version"),
        (
            {"version": 1, "decisions": {"relative": {"decision": "allow", "scope": "folder"}}},
            "key",
        ),
        ({"version": 1, "decisions": {}, "extra": True}, "fields"),
    ],
)
def test_store_schema_unknown_or_missing_fields_fails_closed(
    tmp_path: Path,
    payload: dict[str, object],
    needle: str,
) -> None:
    project = _project_with_prompt(tmp_path)
    path = tmp_path / "home" / ".forge" / "trust.json"
    path.parent.mkdir(parents=True)
    original = json.dumps(payload)
    path.write_text(original, encoding="utf-8")
    store = TrustStore(path)

    result = resolve_project_trust(project, paths=_paths(tmp_path, project), store=store)
    assert result.project_resources_allowed is False
    with pytest.raises(ValueError, match="corrupt"):
        store.set(project, "allow")
    assert path.read_text(encoding="utf-8") == original
    assert needle


def test_empty_project_result_is_fail_closed_when_resources_are_added(tmp_path: Path) -> None:
    project = tmp_path / "empty-project"
    project.mkdir()
    paths = _paths(tmp_path, project)
    result = resolve_project_trust(project, paths=paths)
    assert result.project_resources_present is False
    assert result.project_resources_allowed is False

    prompts = project / ".forge" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "new.md").write_text("new", encoding="utf-8")
    assert result.project_resources_allowed is False
    refreshed = resolve_project_trust(project, paths=paths)
    assert refreshed.project_resources_present is True
    assert refreshed.project_resources_allowed is False


def test_lock_pid_write_failure_cleans_owned_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "home" / ".forge" / "trust.json"
    lock_path = path.with_name(f".{path.name}.lock")
    monkeypatch.setattr(os, "write", lambda _fd, _data: 0)
    with pytest.raises(ValueError, match="lock"):
        TrustStore(path).set(tmp_path / "project", "allow")
    assert not lock_path.exists()


def test_nonblocking_store_update_reports_lock_contention(tmp_path: Path) -> None:
    path = tmp_path / "home" / ".forge" / "trust.json"
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(f"{os.getpid()}\n", encoding="ascii")

    with pytest.raises(ValueError, match="Timed out waiting"):
        TrustStore(path).set(
            tmp_path / "project",
            "allow",
            lock_timeout_seconds=0.0,
        )


def test_store_write_os_error_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "home" / ".forge" / "trust.json"

    def fail_replace(_source: object, _target: object) -> None:
        raise PermissionError

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(ValueError, match="Could not update trust store"):
        TrustStore(path).set(tmp_path / "project", "allow")


def test_multiprocess_store_updates_preserve_both_decisions(tmp_path: Path) -> None:
    path = tmp_path / "home" / ".forge" / "trust.json"
    first = tmp_path / "one"
    second = tmp_path / "two"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_store_process_worker, args=(str(path), str(first))),
        context.Process(target=_store_process_worker, args=(str(path), str(second))),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
    assert all(process.exitcode == 0 for process in processes)
    store = TrustStore(path)
    assert store.lookup(first) is not None
    assert store.lookup(second) is not None


def test_concurrent_store_updates_are_atomic(tmp_path: Path) -> None:
    path = tmp_path / "home" / ".forge" / "trust.json"
    store = TrustStore(path)
    first = tmp_path / "one"
    second = tmp_path / "two"
    barrier = Barrier(2)
    errors: list[BaseException] = []

    def update(cwd: Path) -> None:
        try:
            barrier.wait()
            TrustStore(path).set(cwd, "allow")
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    threads = [Thread(target=update, args=(first,)), Thread(target=update, args=(second,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.lookup(first) is not None
    assert store.lookup(second) is not None
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_denied_project_keeps_user_resources(tmp_path: Path) -> None:
    project = _project_with_prompt(tmp_path)
    (project / "AGENTS.md").write_text("project instructions", encoding="utf-8")
    user_skill = tmp_path / "home" / ".forge" / "skills" / "safe"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("# Safe", encoding="utf-8")
    project_skill = project / ".forge" / "skills" / "unsafe"
    project_skill.mkdir(parents=True)
    (project_skill / "SKILL.md").write_text("# Unsafe", encoding="utf-8")

    paths = _paths(tmp_path, project, allowed=False)
    skills, _ = load_skills_with_diagnostics(paths)
    assert [skill.name for skill in skills] == ["safe"]
    assert paths.prompts_dirs == (paths.prompts_dir, paths.agents_root / "prompts")
    assert paths.subagents_dirs == (paths._paths().user_agents_dir,)
    context, _ = discover_project_context_with_diagnostics(paths)
    assert context == ()

    allowed_paths = _paths(tmp_path, project, allowed=True)
    allowed_skills, _ = load_skills_with_diagnostics(allowed_paths)
    assert [skill.name for skill in allowed_skills] == ["safe", "unsafe"]


def test_no_project_resources_do_not_prompt_or_warn(tmp_path: Path) -> None:
    project = tmp_path / "empty-project"
    project.mkdir()
    stderr = io.StringIO()

    result = resolve_project_trust(
        project,
        paths=_paths(tmp_path, project),
        stderr=stderr,
    )

    assert result.project_resources_present is False
    assert result.project_resources_allowed is False
    assert stderr.getvalue() == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cli_override": "invalid"},
        {"env": {"FORGE_TRUST": "invalid"}},
    ],
)
def test_invalid_override_is_rejected_without_project_resources(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    project = tmp_path / "empty-project"
    project.mkdir()

    with pytest.raises(ValueError, match="Trust must be yes, no, or ask"):
        resolve_project_trust(
            project,
            paths=_paths(tmp_path, project),
            **kwargs,  # type: ignore[arg-type]
        )


def test_project_symlink_resources_cannot_escape_boundary(tmp_path: Path) -> None:
    project = _project_with_prompt(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("# Escaped", encoding="utf-8")
    (outside / "escape.md").write_text("escaped", encoding="utf-8")

    skills_root = project / ".forge" / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    try:
        (skills_root / "escape").symlink_to(outside, target_is_directory=True)
        prompts_root = project / ".forge" / "prompts"
        (prompts_root / "escape.md").unlink()
        (prompts_root / "escape.md").symlink_to(outside / "escape.md")
        (project / "AGENTS.md").symlink_to(outside / "escape.md")
    except OSError:
        pytest.skip("symlinks are unavailable")

    paths = _paths(tmp_path, project)
    skills, _ = load_skills_with_diagnostics(paths)
    prompts, _ = load_prompt_templates_with_diagnostics(paths)
    context, _ = discover_project_context_with_diagnostics(paths)
    assert all(skill.name != "escape" for skill in skills)
    assert all(template.name != "escape" for template in prompts)
    assert all(Path(item.path).resolve() != outside / "escape.md" for item in context)
