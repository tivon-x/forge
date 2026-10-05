"""D1 durable-journal acceptance contract.

These are deliberately process-level tests.  A test that needs a new journal
operation should fail here before the implementation grows a compatibility
shim around the old JSONL append path.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from durable_d1_driver import (
    close,
    commit,
    commit_sample,
    read_all,
    sample_document_change,
    sample_document_write,
    sample_entries,
    sample_task_write,
)
from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent import MessageEndEvent, ToolExecutionEndEvent
from forge_agent.session import (
    BranchSummaryEntry,
    CustomEntry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    SessionInfoEntry,
    entry_to_json_line,
)
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.journal import (
    Journal,
    JournalClosedError,
    JournalCorruptionError,
    JournalPoisonedError,
)
from forge_coding.sessions.storage import (
    JsonlSessionStorage,
    SessionStorageError,
    copy_session_storage,
    export_legacy,
    import_legacy,
)

PI_DURABLE_COMMIT = "4812cb268efc3f0846337cdfb8f00ce51b7b8d5a"
FAILURE_MATRIX: tuple[dict[str, str], ...] = (
    {
        "id": "task-final-write",
        "expected": "GC follows final task write; reopen preserves live task",
    },
    {
        "id": "document-action-merge",
        "expected": "retire/create orders share candidate and recovery",
    },
    {"id": "export-source-guard", "expected": "all export entries reject source files and aliases"},
    {
        "id": "cancel-admitted",
        "expected": "admitted writes settle before caller cancellation returns",
    },
    {
        "id": "fake-process-effect",
        "expected": "reopened fake session does not repeat model or tool effect",
    },
    {"id": "sidecar-partial", "expected": "poison; reopen ignores unconfirmed sidecar"},
    {"id": "utf8-tear", "expected": "byte truncate at last newline; reuse seq"},
    {"id": "flush-failure", "expected": "sidecar flush poisons; no marker"},
    {"id": "missing-sidecar", "expected": "confirmed sidecar loss rejects open"},
    {"id": "sequence", "expected": "non-increasing main seq rejects open"},
    {"id": "reclaim-interrupted", "expected": "published marker survives gc interruption"},
    {"id": "concurrent-commit", "expected": "one ordered sequence per commit"},
    {"id": "close-write", "expected": "closed storage rejects late commit"},
    {"id": "close-admitted", "expected": "admitted commits settle before close"},
    {"id": "migration-interrupted", "expected": "index remains on legacy until switch"},
    {"id": "migration-retry", "expected": "retry reuses validated import"},
    {"id": "branch-import-export-copy", "expected": "IDs, parents, leaf, and all files survive"},
    {"id": "atomic-projections", "expected": "message/usage/goal/todo/receipt commit together"},
    {"id": "restart-no-run", "expected": "D1 reopen does not invoke fake model/tool"},
    {"id": "append-before-sidecar", "expected": "poison and discard unconfirmed sidecar"},
    {"id": "append-after-sidecar", "expected": "poison and discard unconfirmed sidecar"},
    {"id": "append-short-sidecar", "expected": "poison and truncate byte tail"},
    {"id": "append-before-main", "expected": "poison and discard unconfirmed marker"},
    {"id": "append-after-main", "expected": "poison then recover confirmed marker"},
    {"id": "append-short-main", "expected": "poison and truncate marker tail"},
    {"id": "flush-before-sidecar", "expected": "poison without marker"},
    {"id": "flush-after-sidecar", "expected": "poison without marker"},
    {"id": "flush-before-gc-main", "expected": "published state survives best-effort GC"},
    {"id": "flush-after-gc-main", "expected": "published state survives best-effort GC"},
    {"id": "gc-write", "expected": "reclaim write interruption retries after reopen"},
    {"id": "gc-flush", "expected": "reclaim flush interruption retries after reopen"},
    {"id": "gc-rename", "expected": "reclaim rename interruption retries after reopen"},
    {"id": "gc-remove", "expected": "stale reclaim removal retries after reopen"},
    {"id": "prepare-noio", "expected": "invalid preparation performs no I/O and remains usable"},
    {"id": "duplicate-noio", "expected": "duplicate IDs do not consume sequence"},
    {"id": "transition-validation", "expected": "document transition errors do not publish"},
    {"id": "document-create-pair", "expected": "same commit creates distinct document addresses"},
    {
        "id": "document-replacement-order",
        "expected": "both same-address retire/create orders preserve one current incarnation",
    },
    {"id": "queued-failure", "expected": "queued commit cannot publish after poison"},
    {"id": "session-ai-usage", "expected": "assistant and usage share one commit"},
    {"id": "session-todo-tool", "expected": "Todo tool result and snapshot share one commit"},
    {"id": "session-goal-tool", "expected": "Goal tool result and snapshot share one commit"},
    {
        "id": "session-goal-todo-orders",
        "expected": "both model call orders pair Goal/Todo tool results with snapshots",
    },
    {"id": "session-background", "expected": "receipt precedes acceptance and terminal usage"},
    {"id": "session-marker-error", "expected": "uncertain marker stops next model call"},
    {"id": "session-refresh-error", "expected": "read-back failure invalidates session"},
    {"id": "session-close-cancel", "expected": "cancelled close still settles once"},
)


def _run_driver(
    action: str,
    session: Path,
    artifact_dir: Path,
    *,
    fsync: bool = False,
    exit_code: int = 17,
    fault_operation: str | None = None,
    fault_target: str | None = None,
    fault_mode: str | None = None,
    result_name: str = "driver-result.json",
    source: Path | None = None,
    destination: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("durable_d1_driver.py")),
        action,
        str(session),
    ]
    command.extend(("--result", str(session / result_name)))
    if source is not None:
        command.extend(("--source", str(source)))
    if destination is not None:
        command.extend(("--destination", str(destination)))
    if fsync:
        command.append("--fsync")
    if fault_operation is not None:
        command.extend(("--fault-operation", fault_operation))
    if fault_target is not None:
        command.extend(("--fault-target", fault_target))
    if fault_mode is not None:
        command.extend(("--fault-mode", fault_mode))
    command.extend(("--exit-code", str(exit_code)))
    env = dict(os.environ)
    env["HOME"] = str(artifact_dir / "home")
    env["USERPROFILE"] = str(artifact_dir / "home")
    env["FORGE_D1_ARTIFACT_DIR"] = str(artifact_dir)
    completed = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    payload: dict[str, Any] = {}
    result_path = session / result_name
    if result_path.exists():
        try:
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                payload = candidate
        except json.JSONDecodeError:
            payload = {"result": "malformed"}
    model_calls = payload.get("model_calls")
    _record(
        artifact_dir,
        "driver",
        command=command,
        faultpoint={
            "operation": fault_operation,
            "target": fault_target,
            "mode": fault_mode,
        },
        returncode=completed.returncode,
        pid=payload.get("pid"),
        model_call_count=(len(model_calls) if isinstance(model_calls, list) else model_calls),
        sequences=payload.get("seqs", [payload["seq"]] if "seq" in payload else []),
    )
    return completed


def _driver_payload(session: Path, result_name: str = "driver-result.json") -> dict[str, Any]:
    payload = json.loads((session / result_name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _record(root: Path, scenario: str, **data: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"scenarios": [], "assertions": [], "pids": [], "sequences": [], "pairings": []}
    report.setdefault("scenarios", []).append({"scenario": scenario, **data})
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _snapshot_artifacts(
    root: Path, session: Path, scenario: str, *, pairings: tuple[str, ...] = ()
) -> dict[str, str]:
    destination = root / "effects" / scenario
    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for source in sorted(session.glob("*.jsonl")):
        content = source.read_bytes()
        target = destination / source.name
        target.write_bytes(content)
        hashes[source.name] = hashlib.sha256(content).hexdigest()
    _record(
        root,
        scenario,
        hashes=hashes,
        assertions=["reopen/replay"],
        pairings=list(pairings),
    )
    return hashes


@pytest.fixture
def artifact_dir(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    configured = os.environ.get("FORGE_D1_ARTIFACT_DIR")
    artifact_root = Path(configured) if configured else tmp_path / "d1-artifacts"
    node_hash = hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:16]
    # Keep evidence paths below Windows MAX_PATH; the full pytest node id is
    # already retained in the manifest and does not belong in every filename.
    root = artifact_root / "cases" / node_hash
    root.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("langchain-core", "pydantic", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    manifest = {
        "forge_head": _git_head(),
        "pi_durable_commit": PI_DURABLE_COMMIT,
        "python": sys.version,
        "dependencies": versions,
        "command": "uv run pytest tests/test_durable_recovery_e2e.py",
        "random_seed": "deterministic-none",
        "failure_matrix": list(FAILURE_MATRIX),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    yield root
    hashes = {}
    for source in sorted(tmp_path.rglob("*")):
        if not source.is_file() or source.is_relative_to(artifact_root):
            continue
        if source.suffix not in {".jsonl", ".json", ".txt", ".log"}:
            continue
        relative = source.relative_to(tmp_path)
        target = root / "evidence" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        target.write_bytes(payload)
        hashes[str(relative)] = hashlib.sha256(payload).hexdigest()
    _record(root, "fixture", matrix_count=len(FAILURE_MATRIX), evidence_sha256=hashes)


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _main(session: Path) -> Path:
    return session / "main.jsonl"


def _sidecars(session: Path) -> list[Path]:
    return sorted(path for path in session.glob("*.jsonl") if path.name != "main.jsonl")


# ---------------------------------------------------------------------------
# D2 durable task acceptance
# ---------------------------------------------------------------------------

D2_PI_DURABLE_COMMIT = "4812cb268efc3f0846337cdfb8f00ce51b7b8d5a"


def _run_d2_driver(
    action: str,
    session: Path,
    artifact_dir: Path,
    *,
    result_name: str = "d2-driver-result.json",
    exit_code: int = 17,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    """Run one D2 action in a fresh interpreter and record its process evidence."""

    command = [
        sys.executable,
        str(Path(__file__).with_name("durable_d2_driver.py")),
        action,
        str(session),
        "--result",
        str(session / result_name),
        "--exit-code",
        str(exit_code),
    ]
    env = dict(os.environ)
    env["HOME"] = str(artifact_dir / "home")
    env["USERPROFILE"] = str(artifact_dir / "home")
    env["FORGE_D2_ARTIFACT_DIR"] = str(artifact_dir)
    completed = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    payload: dict[str, Any] = {}
    result_path = session / result_name
    if result_path.exists():
        try:
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                payload = candidate
        except json.JSONDecodeError:
            payload = {"error": "malformed-result"}
    _record_d2(
        artifact_dir,
        "driver",
        action=action,
        command=command,
        returncode=completed.returncode,
        pid=payload.get("pid"),
        task_id=payload.get("task_id"),
        effects=payload.get("effects", []),
        payload=payload,
        stdout=completed.stdout[-4000:],
        stderr=completed.stderr[-4000:],
    )
    return completed


def _d2_payload(session: Path, result_name: str = "d2-driver-result.json") -> dict[str, Any]:
    payload = json.loads((session / result_name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _record_d2(root: Path, scenario: str, **data: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"stage": "D2", "scenarios": [], "pids": [], "calls": [], "pairings": []}
    report.setdefault("scenarios", []).append({"scenario": scenario, **data})
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _d2_state(record: Any) -> Any:
    if isinstance(record, Mapping):
        return record.get("state", record)
    return getattr(record, "state", record)


def _d2_status(record: Any) -> str | None:
    state = _d2_state(record)
    value = state.get("status") if isinstance(state, Mapping) else getattr(state, "status", None)
    return value if isinstance(value, str) else None


def _d2_outcome(record: Any) -> Any:
    state = _d2_state(record)
    if isinstance(state, Mapping):
        return state.get("outcome")
    return getattr(state, "outcome", None)


def _d2_record_id(record: Any) -> Any:
    if isinstance(record, Mapping):
        return record.get("id", record.get("task_id"))
    return getattr(record, "id", record)


def _snapshot_d2_evidence(root: Path, session_root: Path) -> list[dict[str, str]]:
    """Copy short, hashed evidence names so Windows path limits stay bounded."""

    evidence_root = root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for source in sorted(session_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in {".json", ".jsonl", ".log", ".txt"}:
            continue
        relative = source.relative_to(session_root).as_posix()
        digest = hashlib.sha256(relative.encode("utf-8") + source.read_bytes()).hexdigest()[:12]
        stem = "".join(character if character.isalnum() else "-" for character in source.stem)[:40]
        target = evidence_root / f"{digest}-{stem}{source.suffix.lower()}"
        target.write_bytes(source.read_bytes())
        copied.append(
            {
                "source": relative,
                "path": str(target),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    return copied


@pytest.fixture
def d2_artifact_dir(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    configured = os.environ.get("FORGE_D2_ARTIFACT_DIR")
    artifact_root = Path(configured) if configured else tmp_path / "d2-artifacts"
    node_hash = hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:12]
    root = artifact_root / "cases" / node_hash
    root.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("langchain-core", "langgraph", "pydantic", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    failure_matrix_path = Path(__file__).parents[1] / ".artifacts" / "d2" / "failure-matrix.json"
    matrix = (
        json.loads(failure_matrix_path.read_text(encoding="utf-8"))
        if failure_matrix_path.exists()
        else {}
    )
    manifest = {
        "stage": "D2",
        "forge_head": _git_head(),
        "pi_durable_commit": D2_PI_DURABLE_COMMIT,
        "python": sys.version,
        "dependencies": versions,
        "command": "uv run pytest tests/test_durable_recovery_e2e.py -k d2",
        "random_seed": "deterministic-none",
        "failure_matrix": matrix.get("scenarios", []),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    yield root
    evidence = _snapshot_d2_evidence(root, tmp_path)
    _record_d2(
        root,
        "fixture",
        matrix_count=len(manifest["failure_matrix"]),
        evidence=evidence,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", ["paused-abort", "abort-waiting", "close-tail"])
async def test_d2_review_handler_lifecycle(
    d2_artifact_dir: Path, tmp_path: Path, scenario: str
) -> None:
    from forge_coding.features.tasks import TaskCoordinator, TaskDefinition, TaskRuntime

    path = tmp_path / scenario
    calls: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    handlers: list[asyncio.Task[Any]] = []

    async def run(record: dict[str, Any], runtime: TaskRuntime) -> None:
        calls.append("run")
        if scenario == "close-tail":
            handler = asyncio.current_task()
            assert handler is not None
            handlers.append(handler)
            await runtime.complete()
            started.set()
            await release.wait()
            calls.append("after-close-effect")
        else:
            await release.wait()

    async def abort(record: dict[str, Any], runtime: TaskRuntime) -> None:
        calls.append("abort")
        if scenario == "abort-waiting":
            await runtime.commit(
                {
                    "status": "waiting",
                    "on": [],
                    "policy": "allSettled",
                    "checkpoint": {"phase": "run"},
                }
            )
        else:
            await runtime.commit({"status": "terminal", "outcome": {"status": "aborted"}})

    definition = TaskDefinition("review", run=run, abort=abort)
    journal = await Journal.open(path)
    coordinator = await TaskCoordinator.open(journal, [definition])
    task_id = await coordinator.create("review", None)
    try:
        if scenario == "paused-abort":
            await coordinator.close()
            await journal.close()
            journal = await Journal.open(path)
            coordinator = await TaskCoordinator.open(journal, [definition])
            assert (await coordinator.inspect())["scheduling"] == "paused"
            await asyncio.wait_for(coordinator.abort_conversation(0), 2)
            settled = await coordinator.wait(task_id, timeout=2)
            assert settled["state"]["outcome"]["status"] == "aborted"
            assert calls == ["abort"]
        elif scenario == "abort-waiting":
            await coordinator.resume()
            await coordinator.abort(task_id)
            settled = await coordinator.wait(task_id, timeout=2)
            assert settled["state"]["outcome"]["status"] == "faulted"
            assert "cannot wait" in str(settled["state"]["outcome"])
            assert calls.count("abort") == 1
        else:
            await coordinator.resume()
            await asyncio.wait_for(started.wait(), 2)
            await coordinator.close()
            assert handlers[0].done()
            release.set()
            await asyncio.sleep(0)
            assert "after-close-effect" not in calls
        _record_d2(d2_artifact_dir, scenario, calls=calls, inspection=await coordinator.inspect())
    finally:
        release.set()
        await coordinator.close()
        await journal.close()


@pytest.mark.anyio
async def test_d2_review_close_drains_accepted_publication(
    d2_artifact_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forge_coding.features.tasks import TaskCoordinator, TaskDefinition, TaskRuntime

    calls: list[str] = []

    async def run(record: dict[str, Any], runtime: TaskRuntime) -> None:
        calls.append("run")
        await runtime.complete()

    journal = await Journal.open(tmp_path / "close-drain")
    coordinator = await TaskCoordinator.open(journal, [TaskDefinition("review", run=run)])
    await coordinator.create("review", None)
    original = journal.commit_records
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked(writes: Any, entries: Any = ()) -> int:
        entered.set()
        await release.wait()
        return await original(writes, entries)

    monkeypatch.setattr(journal, "commit_records", blocked)
    resumed = asyncio.create_task(coordinator.resume())
    closed: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        closed = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0.05)
        assert not closed.done(), "close returned before accepted publication settled"
        release.set()
        await asyncio.wait_for(closed, 2)
        assert resumed.done()
        assert calls == []
        sequence = (await journal.read_records())["sequence"]
        await asyncio.sleep(0)
        assert (await journal.read_records())["sequence"] == sequence
        _record_d2(d2_artifact_dir, "close-drains-publication", calls=calls, sequence=sequence)
    finally:
        release.set()
        await resumed
        if closed is not None:
            await closed
        await coordinator.close()
        await journal.close()


@pytest.mark.anyio
async def test_d2_review_open_failure_closes_owned_resources(
    d2_artifact_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    model = ScriptedChatModel([])
    storage = JsonlSessionStorage(tmp_path / "open-failure")
    configure = CodingSession._configure_tool_features

    def configure_owned_client(session: CodingSession) -> None:
        configure(session)
        # Inject a session-owned client, rather than assuming caller ownership.
        session._owned_providers.append(model)

    monkeypatch.setattr(CodingSession, "_configure_tool_features", configure_owned_client)

    async def fail_read() -> dict[str, Any]:
        raise OSError("controlled D2 open failure")

    monkeypatch.setattr(storage, "read_records", fail_read)
    try:
        with pytest.raises(OSError, match="controlled D2 open failure"):
            await CodingSession.load(
                CodingSessionConfig(
                    provider=model,
                    model="fake",
                    system="offline",
                    cwd=tmp_path,
                    storage=storage,
                    enable_subagents=False,
                    auto_compact_enabled=False,
                )
            )
        assert model.closed
        assert model.calls == []
        _record_d2(d2_artifact_dir, "open-failure-cleanup", provider_closed=model.closed)
    finally:
        await storage.close()
        await model.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("outcome", ["success", "failure", "abort"])
async def test_d2_review_continuation_owns_abort_and_final_queue(
    d2_artifact_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockModel(ScriptedChatModel):
        async def _agenerate(self, messages, **kwargs):  # type: ignore[override]
            if not self.calls:
                started.set()
                await release.wait()
                if outcome == "failure":
                    self._record_call(messages)
                    raise RuntimeError("controlled continuation failure")
            return await super()._agenerate(messages, **kwargs)

    model = BlockModel([AIMessage(content="continued"), AIMessage(content="follow-up answer")])
    storage = JsonlSessionStorage(tmp_path / "continuation")
    await storage.append(MessageEntry(message=HumanMessage(content="existing history")))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            cwd=tmp_path,
            storage=storage,
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )

    async def consume() -> None:
        async for _ in session.continue_():
            pass

    turn = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), 3)
        assert [
            message.content for message in session.messages if isinstance(message, HumanMessage)
        ] == ["existing history"]
        queued = await session.submit(
            "follow-up during continuation", request_id="continuation-follow-up"
        )
        assert (await queued.status())["status"] == "queued"
        if outcome == "abort":
            await asyncio.wait_for(session.abort(), 3)
            assert not session.is_running
            settled = await queued.wait(timeout=3)
            assert settled["status"] == "unanswered" and settled["reason"] == "aborted"
        else:
            release.set()
            await asyncio.wait_for(turn, 3)
            if outcome == "success":
                assert (await queued.wait(timeout=3))["status"] == "done"
                assert len(model.calls) == 2
            else:
                assert (await queued.status())["status"] == "queued"
                assert len(model.calls) == 1
        _record_d2(
            d2_artifact_dir,
            "continuation-" + outcome,
            model_calls=len(model.calls),
            queued=await queued.status(),
            inspection=await session.inspect(),
        )
    finally:
        release.set()
        await asyncio.gather(turn, return_exceptions=True)
        await session.aclose()


@pytest.mark.anyio
async def test_d2_review_external_cancel_stops_goal_replacement(
    d2_artifact_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forge_coding import GoalCommandAction
    from test_goal_session import DelayedCancelChatModel, _persist_active_goal

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    model = DelayedCancelChatModel()
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            cwd=tmp_path,
            storage=JsonlSessionStorage(tmp_path / "cancel-replacement"),
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    old_id = await _persist_active_goal(session, "old objective")

    async def consume(stream: Any) -> None:
        async for _ in stream:
            pass

    old = asyncio.create_task(consume(session.continue_()))
    replacing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(model.started.wait(), 3)
        replacing = asyncio.create_task(
            consume(
                session.apply_goal_action(
                    GoalCommandAction(
                        action="start", objective="must not start", goal_id=old_id, replace=True
                    )
                )
            )
        )
        await asyncio.wait_for(model.cancellation_seen.wait(), 3)
        session.cancel()
        await asyncio.sleep(0)
        assert replacing.cancelling() > 0
        model.release.set()
        results = await asyncio.wait_for(asyncio.gather(old, replacing, return_exceptions=True), 3)
        assert isinstance(results[1], asyncio.CancelledError)
        assert session.goal is not None and session.goal.id == old_id
        assert model.calls == 1
        _record_d2(
            d2_artifact_dir,
            "external-cancel-replacement",
            model_calls=model.calls,
            goal_id=session.goal.id,
            inspection=await session.inspect(),
        )
    finally:
        model.release.set()
        await asyncio.gather(
            old, *([replacing] if replacing is not None else []), return_exceptions=True
        )
        await session.aclose()


@pytest.mark.anyio
async def test_d2_no_progress_is_faulted_and_not_restarted(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-no-progress"
    completed = _run_d2_driver("no-progress", session, d2_artifact_dir)
    assert completed.returncode == 0, completed.stderr
    payload = _d2_payload(session)
    outcome = payload["state"].get("outcome", {})
    assert payload["state"]["status"] == "terminal"
    assert outcome["status"] == "faulted"
    assert "durable progress" in outcome.get("error", {}).get("message", "")
    assert [item["event"] for item in payload["effects"]] == ["run"]
    _record_d2(
        d2_artifact_dir,
        "no-progress-fault",
        task_id=payload["task_id"],
        run_calls=sum(item["event"] == "run" for item in payload["effects"]),
        outcome=outcome,
    )


@pytest.mark.anyio
async def test_d2_publication_close_and_blocked_abort_boundaries(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-boundary-safety"
    completed = _run_d2_driver("boundary-safety", session, d2_artifact_dir)
    assert completed.returncode == 0, completed.stderr
    payload = _d2_payload(session)
    _record_d2(d2_artifact_dir, "boundary-safety", **payload)
    assert payload["failures"] == []


@pytest.mark.anyio
async def test_d2_question_resume_failure_retains_queue_and_never_reuses_old_answer(
    d2_artifact_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))

    class QuestionModel(ScriptedChatModel):
        async def _agenerate(self, messages, **kwargs):  # type: ignore[override]
            if len(self.calls) >= 2:
                self._record_call(messages)
                raise RuntimeError("offline questionnaire resume failure")
            return await super()._agenerate(messages, **kwargs)

    model = QuestionModel(
        [
            AIMessage(content="older answer"),
            tool_call_ai(
                "d2-question",
                "ask_user_question",
                {
                    "questions": [
                        {
                            "header": "Mode",
                            "question": "Choose mode",
                            "multi_select": False,
                            "options": [
                                {"label": "Fast", "description": "Speed"},
                                {"label": "Careful", "description": "Review"},
                            ],
                        }
                    ]
                },
            ),
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            cwd=tmp_path,
            storage=JsonlSessionStorage(tmp_path / "question-session"),
            interactive=True,
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    failures: list[str] = []
    queued = None
    try:
        old = await session.submit("old question")
        await old.wait(timeout=5)
        receipt = await session.submit("ask now")
        async with asyncio.timeout(5):
            while not session.is_waiting_for_input or session.is_running:
                await asyncio.sleep(0.005)
        try:
            queued = await session.submit("retain after failed answer")
        except RuntimeError:
            failures.append("follow-up-rejected-during-questionnaire")
        _events = [event async for event in session.respond_to_human_input("Fast")]
        settled = await receipt.wait(timeout=5)
        if settled["status"] != "unanswered" or settled.get("reason") != "failed":
            failures.append("failed-questionnaire-used-old-answer")
        if queued is not None and (await queued.status())["status"] != "queued":
            failures.append("failed-questionnaire-consumed-follow-up")
        _record_d2(
            d2_artifact_dir,
            "questionnaire-resume-failure",
            failures=failures,
            submission=settled,
            model_calls=len(model.calls),
        )
        assert failures == []
    finally:
        await session.aclose()


@pytest.mark.anyio
async def test_d2_open_does_not_dispatch_and_resume_reuses_checkpoint(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-restart"
    crashed = _run_d2_driver("crash-after-checkpoint", session, d2_artifact_dir)
    assert crashed.returncode == 17, crashed.stderr
    opened = _run_d2_driver("reopen-inspect", session, d2_artifact_dir)
    assert opened.returncode == 0, opened.stderr
    before = _d2_payload(session)
    assert before["record"]["state"]["status"] == "pending"
    assert [item["event"] for item in before["effects"]] == [
        "run-start",
        "checkpoint-committed",
    ]
    resumed = _run_d2_driver("resume-after-restart", session, d2_artifact_dir)
    assert resumed.returncode == 0, resumed.stderr
    after = _d2_payload(session)
    assert after["record"]["state"]["status"] == "terminal"
    assert after["record"]["state"]["outcome"]["status"] == "completed"
    events = [item["event"] for item in after["effects"]]
    assert events == ["run-start", "checkpoint-committed", "resume-run"]
    # Terminal records intentionally drop the live checkpoint; the reopen
    # assertion above proves the committed checkpoint before resume.
    _record_d2(
        d2_artifact_dir,
        "open-resume-checkpoint",
        task_id=after["task_id"],
        processes=[before.get("pid"), after.get("pid")],
        calls={"run": events.count("resume-run") + events.count("run-start")},
        effects=events,
    )


@pytest.mark.anyio
async def test_d2_abort_handler_restarts_after_process_exit(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-abort-restart"
    crashed = _run_d2_driver("abort-crash", session, d2_artifact_dir)
    assert crashed.returncode == 17, crashed.stderr
    resumed = _run_d2_driver("reopen-abort", session, d2_artifact_dir)
    assert resumed.returncode == 0, resumed.stderr
    payload = _d2_payload(session)
    assert payload["record"]["state"]["status"] == "terminal"
    assert payload["record"]["state"]["outcome"]["status"] == "completed"
    assert payload["record"]["state"]["outcome"]["result"].get("reason") == "recovered"
    events = [item["event"] for item in payload["effects"]]
    assert events.count("run") == 1
    assert events.count("abort-start") == 1
    assert events.count("abort-restart") == 1
    assert "unexpected-run" not in events
    _record_d2(
        d2_artifact_dir,
        "abort-restart",
        task_id=payload["task_id"],
        calls={
            "run": events.count("run"),
            "abort": events.count("abort-start") + events.count("abort-restart"),
        },
        effects=events,
    )


@pytest.mark.anyio
async def test_d2_abort_restarts_child_before_parent_after_cascade_crash(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-abort-tree"
    crashed = _run_d2_driver("abort-tree-crash", session, d2_artifact_dir)
    assert crashed.returncode == 17, crashed.stderr
    resumed = _run_d2_driver("abort-tree-reopen", session, d2_artifact_dir)
    assert resumed.returncode == 0, resumed.stderr
    payload = _d2_payload(session)
    events = [item["event"] for item in payload["effects"]]
    assert events.index("child-abort-restart") < events.index("owner-abort-restart")
    assert "unexpected-owner-run" not in events
    assert "unexpected-child-run" not in events
    assert payload["child"]["state"]["status"] == "terminal"
    assert payload["owner"]["state"]["status"] == "terminal"
    _record_d2(
        d2_artifact_dir,
        "abort-cascade-restart",
        pids=sorted({item["pid"] for item in payload["effects"]}),
        order=events,
    )


@pytest.mark.anyio
async def test_d2_completing_outcome_survives_restart_until_child_settles(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-completing-restart"
    crashed = _run_d2_driver("completing-crash", session, d2_artifact_dir)
    assert crashed.returncode == 17, crashed.stderr
    resumed = _run_d2_driver("completing-reopen", session, d2_artifact_dir)
    assert resumed.returncode == 0, resumed.stderr
    payload = _d2_payload(session)
    events = [item["event"] for item in payload["effects"]]
    assert events.count("owner-completing") == 1
    assert events.count("child-resume") == 1
    assert "unexpected-owner-rerun" not in events
    assert payload["owner"]["state"]["status"] == "terminal"
    assert payload["owner"]["state"]["outcome"]["result"] == {"owner": "held"}
    assert payload["child"]["state"]["status"] == "terminal"
    _record_d2(
        d2_artifact_dir,
        "completing-restart",
        pids=sorted({item["pid"] for item in payload["effects"]}),
        effects=events,
        held_outcome=payload["owner"]["state"]["outcome"],
    )


@pytest.mark.anyio
async def test_d2_migration_failure_preserves_record_and_new_definition_retries(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-migration"
    prepared = _run_d2_driver("migration-prepare", session, d2_artifact_dir)
    assert prepared.returncode == 0, prepared.stderr
    failed = _run_d2_driver("migration-fail", session, d2_artifact_dir)
    assert failed.returncode == 0, failed.stderr
    blocked = _d2_payload(session)
    assert blocked["record"]["version"] == 1
    assert blocked["record"]["state"]["status"] == "pending"
    assert blocked["migration_count"] == 1
    inspection = json.dumps(blocked["inspection"], ensure_ascii=False)
    assert "migration_failed" in inspection
    assert [item["event"] for item in blocked["effects"]] == [
        "migration",
    ]
    succeeded = _run_d2_driver("migration-success", session, d2_artifact_dir)
    assert succeeded.returncode == 0, succeeded.stderr
    settled = _d2_payload(session)
    assert settled["record"]["version"] == 2
    assert settled["record"]["state"]["status"] == "terminal"
    assert settled["record"]["state"]["outcome"]["status"] == "completed"
    events = [item["event"] for item in settled["effects"]]
    assert events.count("migration") == 2
    assert events.count("new-run") == 1
    _record_d2(
        d2_artifact_dir,
        "migration-failure-retry",
        task_id=settled["task_id"],
        migration_calls=events.count("migration"),
        effects=events,
    )


@pytest.mark.anyio
async def test_d2_missing_definition_is_inspectable_then_orphaned_on_abort(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-orphan"
    prepared = _run_d2_driver("orphan-prepare", session, d2_artifact_dir)
    assert prepared.returncode == 0, prepared.stderr
    orphaned = _run_d2_driver("orphan-abort", session, d2_artifact_dir)
    assert orphaned.returncode == 0, orphaned.stderr
    payload = _d2_payload(session)
    blocked = json.dumps(payload["blocked"], ensure_ascii=False)
    assert "missing_task" in blocked
    assert payload["record"]["state"]["status"] == "terminal"
    assert payload["record"]["state"]["outcome"] == {
        "reason": "missing_task",
        "status": "orphaned",
    }
    assert payload["effects"] == []
    _record_d2(
        d2_artifact_dir,
        "missing-definition-orphan",
        task_id=payload["task_id"],
        blocked=payload["blocked"],
        outcome=payload["record"]["state"]["outcome"],
    )


@pytest.mark.anyio
async def test_d2_close_rejects_stale_runtime_handle(d2_artifact_dir: Path, tmp_path: Path) -> None:
    session = tmp_path / "d2-stale-handle"
    completed = _run_d2_driver("stale-handle", session, d2_artifact_dir)
    assert completed.returncode == 0, completed.stderr
    payload = _d2_payload(session)
    events = [item["event"] for item in payload["effects"]]
    assert "handle-created" in events
    rejected = [item for item in payload["effects"] if item["event"] == "stale-rejected"]
    assert len(rejected) == 1
    _record_d2(
        d2_artifact_dir,
        "stale-runtime-handle",
        task_id=payload["task_id"],
        rejected=rejected,
    )


@pytest.mark.anyio
async def test_d2_subprocess_loop_shutdown_releases_storage_waiter(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    """A cancelled storage waiter must not keep a fresh subprocess alive."""

    session = tmp_path / "d2-loop-shutdown"
    completed = _run_d2_driver(
        "loop-shutdown",
        session,
        d2_artifact_dir,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    payload = _d2_payload(session)
    assert payload["lock_held"] is True
    assert payload["pending"] is True
    _record_d2(
        d2_artifact_dir,
        "storage-loop-shutdown",
        pid=payload["pid"],
        pending=payload["pending"],
        subprocess_returncode=completed.returncode,
    )


@pytest.mark.anyio
async def test_d2_session_submit_deduplicates_queues_and_retains_failed_input(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "d2-sdk-submissions"
    completed = _run_d2_driver("sdk-submissions", session, d2_artifact_dir)
    assert completed.returncode == 0, completed.stderr
    payload = _d2_payload(session)
    assert payload["duplicate_id"] == payload["first_id"]
    assert payload["type_conflict"] is True
    assert payload["busy_reject"] is True
    assert payload["observer_timeout"] is True
    assert payload["observer_cancelled"] is True
    assert payload["queued_before"]["status"] == "queued"
    assert payload["placed_after_first"]["status"] == "placed"
    assert payload["placed_after_first"]["type"] == "input"
    assert payload["queued_after_first"]["status"] == "placed"
    assert payload["queued_after_second"]["status"] == "placed"
    assert payload["placed_after_second"]["status"] == "done"
    assert payload["placed_after_third"]["status"] == "placed"
    assert all(value["status"] == "done" for value in payload["settled"].values())
    assert payload["reopened_receipt"]["status"] == "done"
    failure = payload["failure"]
    assert failure["failed"]["status"] == "unanswered"
    assert failure["retained"]["status"] == "queued"
    retained_id = failure["retained"]["id"]
    assert any(
        item.get("id") == retained_id
        for snapshot in failure["inbox"]
        for item in snapshot.get("items", [])
    )
    write_idle = payload["write_idle"]
    assert write_idle["model_calls"] == 0
    assert write_idle["submission"]["status"] == "done"
    assert write_idle["submission"]["type"] == "write"
    tool_boundary = payload["tool_boundary"]
    assert tool_boundary["effect_calls"] == 1
    assert tool_boundary["model_calls"] == 3
    assert tool_boundary["paired_tool_call_ids"] is True
    assert tool_boundary["post_tools"]["status"] == "placed"
    assert tool_boundary["final_queued"]["status"] in {"queued", "placed"}
    assert tool_boundary["final_placed"]["status"] == "placed"
    _record_d2(
        d2_artifact_dir,
        "session-submissions",
        calls=payload["model_calls"],
        deduplicated=payload["duplicate_id"] == payload["first_id"],
        queue=[
            payload["placed_after_first"]["id"],
            payload["placed_after_second"]["id"],
            payload["placed_after_third"]["id"],
        ],
        failed_retained=retained_id,
        write_model_calls=write_idle["model_calls"],
        tool_effect_calls=tool_boundary["effect_calls"],
        tool_pairing=tool_boundary["paired_tool_call_ids"],
    )


@pytest.mark.anyio
async def test_d2_steer_after_background_tool_persists_receipt_before_dispatch(
    d2_artifact_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-round steer must adopt the child receipt before it can run."""

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    model = _BackgroundAcceptanceModel()
    storage = JsonlSessionStorage(tmp_path / "d2-background-boundary" / "main.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="D2 background boundary",
            storage=storage,
            cwd=tmp_path,
            tools=[],
            interactive=True,
            enable_subagents=True,
            auto_compact_enabled=False,
        )
    )
    steering: Any | None = None
    seen: list[str] = []

    async def consume() -> None:
        nonlocal steering
        assert session._durable_inputs is not None  # noqa: SLF001
        async for event in session._durable_inputs.events(  # noqa: SLF001
            "launch background", when_busy="reject"
        ):
            seen.append(type(event).__name__)
            if (
                isinstance(event, ToolExecutionEndEvent)
                and event.result.name == "task"
                and steering is None
            ):
                steering = await session.submit(
                    "steer after tools",
                    request_id="same-tool-round-steer",
                    when_busy="steer",
                )

    try:
        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(model.started.wait(), timeout=5)
        model.gate.set()
        await asyncio.wait_for(consumer, timeout=10)
        assert steering is not None
        assert (await steering.status())["status"] == "done"
        entries = await storage.read_all()
        tool_index = next(
            index
            for index, entry in enumerate(entries)
            if isinstance(entry, MessageEntry)
            and isinstance(entry.message, ToolMessage)
            and entry.message.tool_call_id == "submit-d1"
        )
        receipt_indexes = [
            index
            for index, entry in enumerate(entries)
            if isinstance(entry, CustomEntry)
            and entry.namespace == "forge.subagent_run.v1"
            and entry.data.get("status") == "queued"
        ]
        assert receipt_indexes and receipt_indexes[0] < tool_index
        tool_calls = {
            call["id"]
            for message in session.messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }
        tool_results = {
            message.tool_call_id for message in session.messages if isinstance(message, ToolMessage)
        }
        assert "submit-d1" in tool_calls and "submit-d1" in tool_results
        assert model.calls and len(model.calls) >= 2
    except BaseException:
        call_summary = []
        for call in model.calls:
            call_summary.append(
                {
                    "messages": [type(message).__name__ for message in call],
                    "last_human": next(
                        (
                            str(message.content)
                            for message in reversed(call)
                            if isinstance(message, HumanMessage)
                        ),
                        None,
                    ),
                    "last_type": type(call[-1]).__name__ if call else None,
                }
            )
        _record_d2(
            d2_artifact_dir,
            "background-receipt-adopt-before-dispatch-failure",
            model_calls=call_summary,
            background=session.background_subagents,
            entries=[type(entry).__name__ for entry in await storage.read_all()],
        )
        raise
    finally:
        await session.aclose()
        await storage.close()
    _record_d2(
        d2_artifact_dir,
        "background-receipt-adopt-before-dispatch",
        model_calls=len(model.calls),
        child_started=model.started.is_set(),
        events=seen,
        pairings=["submit-d1:receipt", "submit-d1:tool-message"],
    )


@pytest.mark.anyio
async def test_d2_waiting_policies_and_repeated_resume_are_durable(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    """Exercise allSettled/failFast without polling handlers or duplicate wakes."""

    from durable_d2_driver import (
        _call_create,
        _close,
        _definition,
        _inspect,
        _maybe_await,
        _open,
        _resume,
        _wait,
    )

    session = tmp_path / "d2-waiting"
    all_gate = asyncio.Event()
    fail_fast_gate = asyncio.Event()
    runs: dict[str, int] = {}

    def count(name: str) -> None:
        runs[name] = runs.get(name, 0) + 1

    async def child_run(task: Any, runtime: Any) -> None:
        name = str(task["input"]["name"] if isinstance(task, Mapping) else task.input["name"])
        count(name)
        gate = all_gate if name.startswith("all-") else fail_fast_gate
        gate_wait = asyncio.create_task(gate.wait())
        signal_wait = asyncio.create_task(runtime.signal.wait())
        done, pending = await asyncio.wait(
            {gate_wait, signal_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        for waiter in pending:
            waiter.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if signal_wait in done:
            return
        await _maybe_await(runtime.complete({"name": name}))

    async def failed_run(task: Any, runtime: Any) -> None:
        count("fail")
        await _maybe_await(runtime.fail({"message": "child failed"}))

    async def child_abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"status": "aborted"}))

    async def all_waiter(task: Any, runtime: Any) -> None:
        state = task.get("state", {}) if isinstance(task, Mapping) else task.state
        checkpoint = state.get("checkpoint", {}) if isinstance(state, Mapping) else state.checkpoint
        if checkpoint.get("phase") == "wait":
            await _maybe_await(
                runtime.waiting(
                    task["input"]["on"] if isinstance(task, Mapping) else task.input["on"],
                    policy="allSettled",
                    checkpoint={"phase": "done"},
                )
            )
            return
        count("all-waiter-done")
        await _maybe_await(runtime.complete({"policy": "allSettled"}))

    async def fail_fast_waiter(task: Any, runtime: Any) -> None:
        state = task.get("state", {}) if isinstance(task, Mapping) else task.state
        checkpoint = state.get("checkpoint", {}) if isinstance(state, Mapping) else state.checkpoint
        input_value = task["input"] if isinstance(task, Mapping) else task.input
        if checkpoint.get("phase") == "wait":
            await _maybe_await(
                runtime.waiting(
                    input_value["on"],
                    policy="failFast",
                    checkpoint={"phase": "done"},
                )
            )
            return
        count("fail-fast-done")
        await _maybe_await(runtime.fail({"message": "failFast dependency"}))

    definitions = [
        _definition("d2.wait-child", 1, child_run, child_abort),
        _definition("d2.wait-fail", 1, failed_run, child_abort),
        _definition("d2.wait-all", 1, all_waiter, child_abort),
        _definition("d2.wait-fast", 1, fail_fast_waiter, child_abort),
    ]
    storage, coordinator = await _open(session, definitions)
    all_one = _call_create(coordinator, "d2.wait-child", {"name": "all-one"}, {"phase": "run"})
    all_two = _call_create(coordinator, "d2.wait-child", {"name": "all-two"}, {"phase": "run"})
    all_ids = [
        (await _maybe_await(all_one)),
        (await _maybe_await(all_two)),
    ]
    all_wait = await _call_create(
        coordinator,
        "d2.wait-all",
        {"on": [_d2_record_id(value) for value in all_ids]},
        {"phase": "wait"},
    )
    fast_child = await _call_create(coordinator, "d2.wait-fail", {"name": "fail"}, {"phase": "run"})
    fast_sibling = await _call_create(
        coordinator, "d2.wait-child", {"name": "fast-sibling"}, {"phase": "run"}
    )
    fast_wait = await _call_create(
        coordinator,
        "d2.wait-fast",
        {"on": [_d2_record_id(fast_child), _d2_record_id(fast_sibling)]},
        {"phase": "wait"},
    )
    await _resume(coordinator)
    for _ in range(10):
        await _resume(coordinator)
        await asyncio.sleep(0)
    all_wait_id = _d2_record_id(all_wait)
    all_state = await _inspect(coordinator)
    assert "waiting" in json.dumps(all_state, ensure_ascii=False)
    all_gate.set()
    all_result = await _wait(coordinator, all_wait_id, timeout=5)
    try:
        fast_result = await _wait(coordinator, _d2_record_id(fast_wait), timeout=5)
    except TimeoutError:
        from durable_d2_driver import _jsonable

        _record_d2(
            d2_artifact_dir,
            "waiting-timeout-debug",
            inspection=_jsonable(await _inspect(coordinator)),
            records=_jsonable(await storage.read_records()),
            calls=runs,
        )
        raise
    assert _d2_status(all_result) == "terminal"
    assert _d2_status(fast_result) == "terminal"
    assert runs["all-one"] == 1 and runs["all-two"] == 1
    assert runs["all-waiter-done"] == 1
    assert runs["fail"] == 1
    assert runs.get("fast-sibling", 0) <= 1
    await _close(coordinator, storage)
    _record_d2(
        d2_artifact_dir,
        "waiting-policies",
        calls=runs,
        all_settled=_d2_outcome(all_result),
        fail_fast=_d2_outcome(fast_result),
    )


@pytest.mark.anyio
async def test_d2_completing_and_background_abort_boundary(
    d2_artifact_dir: Path, tmp_path: Path
) -> None:
    from durable_d2_driver import (
        _call_create,
        _close,
        _definition,
        _inspect,
        _maybe_await,
        _open,
        _resume,
        _wait,
    )

    session = tmp_path / "d2-ownership"
    foreground_gate = asyncio.Event()
    background_gate = asyncio.Event()

    async def owner_run(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"owner": "done"}))

    async def child_run(task: Any, runtime: Any) -> None:
        gate = foreground_gate if task["input"]["name"] == "foreground" else background_gate
        gate_wait = asyncio.create_task(gate.wait())
        signal_wait = asyncio.create_task(runtime.signal.wait())
        done, pending = await asyncio.wait(
            {gate_wait, signal_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        for waiter in pending:
            waiter.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if signal_wait in done:
            return
        await _maybe_await(runtime.complete({"name": task["input"]["name"]}))

    async def child_abort(task: Any, runtime: Any) -> None:
        await _maybe_await(runtime.complete({"status": "aborted"}))

    definitions = [
        _definition("d2.owner", 1, owner_run, child_abort),
        _definition("d2.owned-child", 1, child_run, child_abort),
    ]
    storage, coordinator = await _open(session, definitions)
    owner = await _call_create(coordinator, "d2.owner", {}, {"phase": "run"})
    owner_id = _d2_record_id(owner)
    foreground = await _call_create(
        coordinator,
        "d2.owned-child",
        {"name": "foreground"},
        {"phase": "run"},
        owner=owner_id,
    )
    background = await _call_create(
        coordinator,
        "d2.owned-child",
        {"name": "background"},
        {"phase": "run"},
        owner=owner_id,
        background=True,
    )
    await _resume(coordinator)
    for _ in range(100):
        inspection = await _inspect(coordinator)
        if "completing" in json.dumps(inspection, ensure_ascii=False):
            break
        await asyncio.sleep(0.01)
    inspection = await _inspect(coordinator)
    assert "completing" in json.dumps(inspection, ensure_ascii=False)
    foreground_gate.set()
    owner_result = await _wait(coordinator, owner_id, timeout=5)
    assert _d2_status(owner_result) == "terminal"
    # Background work is outside ordinary owner completion/abort traversal.
    background_id = _d2_record_id(background)
    background_state = await _inspect(coordinator)
    assert str(background_id) in json.dumps(background_state, ensure_ascii=False)
    await _maybe_await(coordinator.abort(background_id, include_background=True))
    background_result = await _wait(coordinator, background_id, timeout=5)
    assert _d2_status(background_result) == "terminal"
    assert _d2_record_id(foreground) != background_id
    await _close(coordinator, storage)
    _record_d2(
        d2_artifact_dir,
        "completing-background-boundary",
        owner=owner_id,
        foreground=_d2_record_id(foreground),
        background=background_id,
        owner_outcome=_d2_outcome(owner_result),
        background_outcome=_d2_outcome(background_result),
    )


def _unconfirmed_record(sidecar: Path, *, seq: int = 2, ordinal: int = 99) -> bytes:
    """Copy a valid sidecar payload while moving it beyond the confirmed marker."""

    record = json.loads(sidecar.read_bytes().splitlines(keepends=True)[0])
    record["seq"] = seq
    record["ordinal"] = ordinal
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


async def _new_session(path: Path) -> Any:
    return await Journal.open(path, fsync=False)


class _RecordingStorage:
    """SessionStorage seam that records exact atomic batches for assertions."""

    def __init__(
        self,
        path: Path,
        *,
        fail_on_message_commit: int | None = None,
        fail_read_after_message_commit: bool = False,
    ) -> None:
        self.inner = JsonlSessionStorage(path)
        self.path = self.inner.path
        self.batches: list[tuple[Any, ...]] = []
        self.fail_on_message_commit = fail_on_message_commit
        self.fail_read_after_message_commit = fail_read_after_message_commit
        self._commit_count = 0
        self._message_commit_count = 0
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned or self.inner.poisoned

    async def read_all(self) -> list[Any]:
        if self.fail_read_after_message_commit and self._message_commit_count > 0:
            raise OSError("D1 persisted-state refresh failure")
        return await self.inner.read_all()

    async def commit(self, entries: Any) -> int:
        batch = tuple(entries)
        self.batches.append(batch)
        self._commit_count += 1
        message_batch = any(isinstance(entry, MessageEntry) for entry in batch)
        if message_batch:
            self._message_commit_count += 1
        sequence = await self.inner.commit(batch)
        if message_batch and self.fail_on_message_commit == self._message_commit_count:
            self._poisoned = True
            raise OSError("D1 marker publication result is uncertain")
        return sequence

    async def append(self, entry: Any) -> int:
        return await self.commit((entry,))

    async def close(self) -> None:
        await self.inner.close()


def _batch_with(batch_entries: tuple[Any, ...], predicate: object) -> bool:
    return any(predicate(entry) for entry in batch_entries)  # type: ignore[operator]


def _entry_names(batch: tuple[Any, ...]) -> set[str]:
    return {getattr(entry, "id", "") for entry in batch}


async def _load_recording_session(
    tmp_path: Path,
    model: BaseChatModel,
    storage: _RecordingStorage,
    *,
    interactive: bool = False,
    enable_subagents: bool = False,
) -> CodingSession:
    return await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="Forge D1 acceptance",
            storage=storage,
            cwd=tmp_path,
            tools=[],
            interactive=interactive,
            enable_subagents=enable_subagents,
            auto_compact_enabled=False,
        )
    )


async def _persist_active_goal_for_test(session: CodingSession, objective: str) -> str:
    session._goal_controller.start(objective)  # noqa: SLF001
    session._goal_dirty = True  # noqa: SLF001
    await session._persist_goal_update()  # noqa: SLF001
    assert session.goal is not None
    return session.goal.id


def _paired_goal_todo_response(order: tuple[str, str], goal_id: str) -> AIMessage:
    tool_calls: list[dict[str, object]] = []
    for index, name in enumerate(order, start=1):
        if name == "todo":
            arguments: dict[str, object] = {
                "todos": [{"content": "paired control", "status": "in_progress"}]
            }
        else:
            arguments = {"goal_id": goal_id, "summary": "paired controls verified"}
        tool_calls.append(
            {
                "id": f"{name}-paired-{index}",
                "name": "write_todos" if name == "todo" else "goal_complete",
                "args": arguments,
                "type": "tool_call",
            }
        )
    return AIMessage(content="", tool_calls=tool_calls)


class _BackgroundAcceptanceModel(ScriptedChatModel):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "started", asyncio.Event())
        object.__setattr__(self, "gate", asyncio.Event())

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        del stop, run_manager, kwargs
        self._record_call(messages)
        human = next(message for message in reversed(messages) if isinstance(message, HumanMessage))
        if str(human.content).startswith("child"):
            self.started.set()
            await self.gate.wait()
            response = AIMessage(
                content="CHILD_RESULT",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            )
        elif str(human.content).startswith("launch") and not isinstance(messages[-1], ToolMessage):
            response = tool_call_ai(
                "submit-d1",
                "task",
                {
                    "agent": "scout",
                    "instruction": "child one",
                    "mode": "background",
                    "context": "fresh",
                },
            )
        else:
            response = AIMessage(content="PARENT_READY")
        return ChatResult(generations=[ChatGeneration(message=response)])


@pytest.mark.anyio
async def test_coding_session_commits_ai_and_usage_as_one_batch(
    artifact_dir: Path, tmp_path: Path
) -> None:
    model = ScriptedChatModel(
        [
            AIMessage(
                content="done",
                usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            )
        ]
    )
    storage = _RecordingStorage(tmp_path / "session.jsonl")
    session = await _load_recording_session(tmp_path, model, storage)
    try:
        _events = [event async for event in session.prompt("hello")]
        assert len(model.calls) == 1
        batches = storage.batches
        usage_batch = next(
            batch
            for batch in batches
            if any(
                isinstance(entry, CustomEntry) and entry.namespace == "forge.usage.v1"
                for entry in batch
            )
        )
        assert any(
            isinstance(entry, MessageEntry)
            and isinstance(entry.message, AIMessage)
            and entry.message.content == "done"
            for entry in usage_batch
        )
        usage = next(
            entry
            for entry in usage_batch
            if isinstance(entry, CustomEntry) and entry.namespace == "forge.usage.v1"
        )
        assert usage.data["total_tokens"] == 6
        assert any(isinstance(entry, LeafEntry) for entry in usage_batch)
    finally:
        await session.aclose()
        await storage.close()
    _record(
        artifact_dir, "session-ai-usage", model_calls=len(model.calls), pairings=["assistant:usage"]
    )


@pytest.mark.anyio
async def test_coding_session_commits_todo_tool_result_and_snapshot_as_one_batch(
    artifact_dir: Path, tmp_path: Path
) -> None:
    model = ScriptedChatModel(
        [
            tool_call_ai(
                "todo-d1",
                "write_todos",
                {"todos": [{"content": "Inspect journal", "status": "in_progress"}]},
            ),
            AIMessage(content="plan recorded"),
        ]
    )
    storage = _RecordingStorage(tmp_path / "session.jsonl")
    session = await _load_recording_session(tmp_path, model, storage)
    try:
        _events = [event async for event in session.prompt("make a plan")]
        assert len(model.calls) == 2
        todo_batch = next(
            batch
            for batch in storage.batches
            if any(
                isinstance(entry, CustomEntry) and entry.namespace == "forge.todo.v1"
                for entry in batch
            )
        )
        assert any(
            isinstance(entry, MessageEntry)
            and isinstance(entry.message, ToolMessage)
            and entry.message.tool_call_id == "todo-d1"
            for entry in todo_batch
        )
        snapshot = next(
            entry
            for entry in todo_batch
            if isinstance(entry, CustomEntry) and entry.namespace == "forge.todo.v1"
        )
        assert snapshot.data["todos"][0]["content"] == "Inspect journal"
    finally:
        await session.aclose()
        await storage.close()
    _record(
        artifact_dir, "session-todo-tool", model_calls=len(model.calls), pairings=["todo-d1:todo"]
    )


@pytest.mark.anyio
async def test_coding_session_commits_goal_tool_result_and_snapshot_as_one_batch(
    artifact_dir: Path, tmp_path: Path
) -> None:
    model = ScriptedChatModel()
    storage = _RecordingStorage(tmp_path / "session.jsonl")
    session = await _load_recording_session(tmp_path, model, storage)
    try:
        goal_id = await _persist_active_goal_for_test(session, "Ship durable journal")
        model.responses = [
            tool_call_ai("goal-d1", "goal_complete", {"goal_id": goal_id, "summary": "verified"}),
            AIMessage(content="goal complete"),
        ]
        _events = [event async for event in session.continue_()]
        assert len(model.calls) == 1
        assert session.goal is not None and session.goal.status == "complete"
        goal_batch = next(
            batch
            for batch in storage.batches
            if any(
                isinstance(entry, CustomEntry) and entry.namespace == "forge.goal.v1"
                for entry in batch
            )
            and any(
                isinstance(entry, MessageEntry)
                and isinstance(entry.message, ToolMessage)
                and entry.message.tool_call_id == "goal-d1"
                for entry in batch
            )
        )
        snapshot = next(
            entry
            for entry in goal_batch
            if isinstance(entry, CustomEntry) and entry.namespace == "forge.goal.v1"
        )
        assert snapshot.data["goal"]["status"] == "complete"
    finally:
        await session.aclose()
        await storage.close()
    _record(
        artifact_dir, "session-goal-tool", model_calls=len(model.calls), pairings=["goal-d1:goal"]
    )


@pytest.mark.anyio
@pytest.mark.parametrize("order", [("todo", "goal"), ("goal", "todo")])
async def test_coding_session_pairs_goal_and_todo_for_both_tool_orders(
    artifact_dir: Path, tmp_path: Path, order: tuple[str, str]
) -> None:
    model = ScriptedChatModel()
    storage = _RecordingStorage(tmp_path / f"session-controls-{'-'.join(order)}")
    session = await _load_recording_session(tmp_path, model, storage)
    try:
        goal_id = await _persist_active_goal_for_test(session, "Ship paired controls")
        model.responses = [_paired_goal_todo_response(order, goal_id)]
        _events = [event async for event in session.prompt("run paired controls")]
        assert len(model.calls) == 1
        for name in order:
            tool_call_id = f"{name}-paired-{order.index(name) + 1}"
            namespace = "forge.todo.v1" if name == "todo" else "forge.goal.v1"
            paired_batches = [
                batch
                for batch in storage.batches
                if any(
                    isinstance(entry, MessageEntry)
                    and isinstance(entry.message, ToolMessage)
                    and entry.message.tool_call_id == tool_call_id
                    for entry in batch
                )
                and any(
                    isinstance(entry, CustomEntry) and entry.namespace == namespace
                    for entry in batch
                )
            ]
            assert len(paired_batches) == 1
            snapshot = next(
                entry
                for entry in paired_batches[0]
                if isinstance(entry, CustomEntry) and entry.namespace == namespace
            )
            if name == "todo":
                assert snapshot.data["todos"][0]["content"] == "paired control"
            else:
                assert snapshot.data["goal"]["status"] == "complete"
        assert {
            message.tool_call_id for message in session.messages if isinstance(message, ToolMessage)
        } >= {
            "todo-paired-1" if order[0] == "todo" else "todo-paired-2",
            "goal-paired-1" if order[0] == "goal" else "goal-paired-2",
        }
    finally:
        await session.aclose()
        await storage.close()
    _record(
        artifact_dir,
        "session-goal-todo-orders",
        model_calls=len(model.calls),
        pairings=[f"{name}:snapshot" for name in order],
        order=list(order),
    )


@pytest.mark.anyio
async def test_coding_session_commits_background_receipt_before_accept_and_terminal_usage(
    artifact_dir: Path, tmp_path: Path
) -> None:
    model = _BackgroundAcceptanceModel()
    storage = _RecordingStorage(tmp_path / "session.jsonl")
    session = await _load_recording_session(
        tmp_path,
        model,
        storage,
        interactive=True,
        enable_subagents=True,
    )
    try:
        _events = [event async for event in session.prompt("launch background")]
        await asyncio.wait_for(model.started.wait(), 5)
        run_id = session.background_subagents[0]["run_id"]
        model.gate.set()
        await asyncio.wait_for(session._subagent_runs.control("wait", run_id, 5), 6)  # noqa: SLF001
        await session.flush_background_subagents()
        run_batches = [
            batch
            for batch in storage.batches
            if any(
                isinstance(entry, MessageEntry)
                and isinstance(entry.message, ToolMessage)
                and entry.message.tool_call_id == "submit-d1"
                for entry in batch
            )
        ]
        assert run_batches
        receipt_batch = run_batches[0]
        assert any(
            isinstance(entry, CustomEntry)
            and entry.namespace == "forge.subagent_run.v1"
            and entry.data["status"] == "queued"
            for entry in receipt_batch
        )
        terminal_batch = next(
            batch
            for batch in storage.batches
            if any(
                isinstance(entry, CustomEntry)
                and entry.namespace == "forge.subagent_run.v1"
                and entry.data["status"] == "completed"
                for entry in batch
            )
        )
        assert any(
            isinstance(entry, CustomEntry)
            and entry.namespace == "forge.usage.v1"
            and entry.data.get("purpose") == "subagent"
            for entry in terminal_batch
        )
        assert len(model.calls) >= 2
    finally:
        await session.aclose()
        await storage.close()
    _record(
        artifact_dir,
        "session-background",
        model_calls=len(model.calls),
        pairings=["submit-d1:queued", "run:usage"],
    )


@pytest.mark.anyio
async def test_marker_error_poison_stops_next_prompt_without_success_refresh(
    artifact_dir: Path, tmp_path: Path
) -> None:
    model = ScriptedChatModel([AIMessage(content="must not be published")])
    storage = _RecordingStorage(tmp_path / "session.jsonl", fail_on_message_commit=2)
    session = await _load_recording_session(tmp_path, model, storage)
    first_events: list[object] = []
    try:
        with pytest.raises((OSError, RuntimeError), match="uncertain|poison|publication"):
            async for event in session.prompt("trigger marker failure"):
                first_events.append(event)
        assert len(model.calls) == 1
        assert any(
            isinstance(event, MessageEndEvent)
            and isinstance(event.message, HumanMessage)
            and event.message.content == "trigger marker failure"
            for event in first_events
        )
        assert not any(
            isinstance(event, MessageEndEvent)
            and isinstance(event.message, AIMessage)
            and event.message.content == "must not be published"
            for event in first_events
        )
        message_batches = [
            batch for batch in storage.batches if any(isinstance(e, MessageEntry) for e in batch)
        ]
        confirmed_batch = message_batches[0]
        assert any(
            isinstance(entry, MessageEntry)
            and isinstance(entry.message, HumanMessage)
            and entry.message.content == "trigger marker failure"
            for entry in confirmed_batch
        )
        uncertain_batch = message_batches[1]
        assert any(
            isinstance(entry, MessageEntry)
            and isinstance(entry.message, AIMessage)
            and entry.message.content == "must not be published"
            for entry in uncertain_batch
        )
        assert any(
            isinstance(entry, CustomEntry) and entry.namespace == "forge.usage.v1"
            for entry in uncertain_batch
        )
        with pytest.raises(RuntimeError, match="uncertain|reopen|poison"):
            async for _event in session.prompt("must not call model"):
                pass
        assert len(model.calls) == 1
        assert any(
            isinstance(message, HumanMessage) and message.content == "trigger marker failure"
            for message in session.messages
        )
        assert not any(
            isinstance(message, AIMessage) and message.content == "must not be published"
            for message in session.messages
        )
    finally:
        await session.aclose()
        await storage.close()
    _record(artifact_dir, "session-marker-error", model_calls=len(model.calls), pairings=[])


@pytest.mark.anyio
async def test_session_refresh_failure_invalidates_future_prompts(
    artifact_dir: Path, tmp_path: Path
) -> None:
    """A committed marker whose read-back fails makes the session unusable."""

    model = ScriptedChatModel([AIMessage(content="committed")])
    storage = _RecordingStorage(tmp_path / "session-refresh", fail_read_after_message_commit=True)
    session = await _load_recording_session(tmp_path, model, storage)
    try:
        with pytest.raises((OSError, RuntimeError), match="refresh|storage|persist"):
            async for _event in session.prompt("trigger refresh failure"):
                pass
        assert len(model.calls) == 0
        assert any(
            isinstance(entry, MessageEntry)
            and isinstance(entry.message, HumanMessage)
            and entry.message.content == "trigger refresh failure"
            for batch in storage.batches
            for entry in batch
        )
        assert session._storage_failure is not None  # noqa: SLF001
        with pytest.raises(RuntimeError, match="storage|reopen|uncertain"):
            async for _event in session.prompt("must not call model"):
                pass
        assert len(model.calls) == 0
    finally:
        await session.aclose()
        await storage.close()
    _record(artifact_dir, "session-refresh-error", model_calls=len(model.calls), pairings=[])


@pytest.mark.anyio
async def test_session_close_cancellation_waits_for_one_settled_close(
    artifact_dir: Path, tmp_path: Path
) -> None:
    model = ScriptedChatModel()
    storage = JsonlSessionStorage(tmp_path / "session-close")
    session = await _load_recording_session(tmp_path, model, storage)  # type: ignore[arg-type]
    original_close = storage.close
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_calls = 0

    async def blocked_close() -> None:
        nonlocal close_calls
        close_calls += 1
        close_started.set()
        await release_close.wait()
        await original_close()

    storage.close = blocked_close  # type: ignore[method-assign]
    first = asyncio.create_task(session.aclose())
    await asyncio.wait_for(close_started.wait(), 5)
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done()
    second = asyncio.create_task(session.aclose())
    await asyncio.sleep(0)
    assert not second.done()
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert close_calls == 1
    assert session._closed  # noqa: SLF001
    with pytest.raises(RuntimeError, match="closed|closing"):
        async for _event in session.prompt("must not call model after close"):
            pass
    assert model.calls == []
    _record(artifact_dir, "session-close-cancel", model_calls=len(model.calls), pairings=[])


@pytest.mark.anyio
async def test_atomic_commit_replay_and_append_compat(artifact_dir: Path, tmp_path: Path) -> None:
    session = tmp_path / "atomic"
    storage = await _new_session(session)
    entries = sample_entries()
    sequence = await commit_sample(storage)
    assert sequence >= 1
    assert [getattr(entry, "id", None) for entry in await read_all(storage)] == [
        getattr(entry, "id", None) for entry in entries
    ]
    assert len(_main(session).read_bytes().splitlines()) == 1
    assert {path.name for path in _sidecars(session)} == {"task-1.jsonl", "doc-2.jsonl"}
    await close(storage)

    reopened = await _new_session(session)
    assert [getattr(entry, "id", None) for entry in await read_all(reopened)] == [
        getattr(entry, "id", None) for entry in entries
    ]
    task = await reopened.task(1)
    document = await reopened.document(2)
    assert task is not None and task["state"]["status"] == "pending"
    assert document is not None and document["value"]["branch"] == "main"
    assert await reopened.append(LabelEntry(id="compat", label="append")) is None
    assert any(getattr(entry, "id", None) == "compat" for entry in await read_all(reopened))
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "atomic-projections")


@pytest.mark.anyio
async def test_restart_after_real_process_exit_does_not_call_fake_model_or_tool(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "restart"
    result = _run_driver("commit-and-exit", session, artifact_dir)
    assert result.returncode == 17, result.stderr
    reader = _run_driver("read", session, artifact_dir, result_name="restart-reader.json")
    assert reader.returncode == 0, reader.stderr
    assert {"human", "usage", "leaf"} <= set(_driver_payload(session, "restart-reader.json")["ids"])
    reopened = await _new_session(session)
    entries = await read_all(reopened)
    assert {getattr(entry, "id", None) for entry in entries} >= set(
        ("human", "assistant", "tool", "usage")
    )
    assert not (session / "fake-model-calls.jsonl").exists()
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "restart-no-run")


@pytest.mark.anyio
async def test_fake_session_process_restart_preserves_effect_without_reexecution(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "fake-process"
    crashed = _run_driver("session-and-exit", session, artifact_dir, result_name="writer.json")
    assert crashed.returncode == 17, crashed.stderr
    writer = _driver_payload(session, "writer.json")
    assert writer["model_calls"] == 2
    assert writer["tool_calls"] == 1
    effect_before = (session / "workspace" / "effect.txt").read_bytes()
    reader = _run_driver("session-reopen", session, artifact_dir, result_name="reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = _driver_payload(session, "reader.json")
    assert reopened["model_calls"] == 0
    assert reopened["tool_calls"] == 0
    assert reopened["paired_call_ids"] == ["effect-d1"]
    assert writer["pid"] != reopened["pid"]
    assert (session / "workspace" / "effect.txt").read_bytes() == effect_before
    _record(
        artifact_dir,
        "fake-process-restart",
        writer=writer,
        reader=reopened,
        effect_sha256=hashlib.sha256(effect_before).hexdigest(),
    )


@pytest.mark.anyio
async def test_torn_sidecar_and_utf8_tail_reuse_unconfirmed_sequence(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "utf8"
    result = _run_driver("write", session, artifact_dir)
    assert result.returncode == 0, result.stderr
    sidecar = _sidecars(session)[0]
    main_before = _main(session).read_bytes()
    sidecar_before = sidecar.read_bytes()
    torn = "区€".encode()[:-1]
    sidecar.write_bytes(sidecar_before + torn)
    _main(session).write_bytes(main_before + torn)
    reader = _run_driver("read", session, artifact_dir, result_name="utf8-reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    assert sidecar.read_bytes() == sidecar_before
    assert _main(session).read_bytes() == main_before
    sequence = await commit(reopened, [LabelEntry(id="after-torn", label="after")])
    assert sequence == 2
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "utf8-tear")


@pytest.mark.anyio
async def test_complete_main_json_without_newline_is_discarded_and_sequence_reused(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "main-no-newline"
    assert _run_driver("write", session, artifact_dir).returncode == 0
    journal = await _new_session(session)
    assert await commit(journal, [LabelEntry(id="second", label="second")]) == 2
    await close(journal)
    main = _main(session)
    raw = main.read_bytes()
    assert raw.endswith(b"\n")
    main.write_bytes(raw[:-1])
    reader = _run_driver("read", session, artifact_dir, result_name="main-no-newline-reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    ids = {getattr(entry, "id", None) for entry in await read_all(reopened)}
    assert "human" in ids and "second" not in ids
    assert await commit(reopened, [LabelEntry(id="reused", label="reused")]) == 2
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "main-no-newline")


@pytest.mark.anyio
async def test_complete_unconfirmed_sidecar_tail_is_removed_without_resurrection(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "partial"
    result = _run_driver("write", session, artifact_dir)
    assert result.returncode == 0, result.stderr
    sidecar = _sidecars(session)[0]
    sidecar.write_bytes(sidecar.read_bytes() + _unconfirmed_record(sidecar, ordinal=99))
    reader = _run_driver("read", session, artifact_dir, result_name="partial-reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    assert '"seq":2' not in sidecar.read_text(encoding="utf-8")
    assert {getattr(entry, "id", None) for entry in await read_all(reopened)} >= set(
        ("human", "usage")
    )
    assert await commit(reopened, [LabelEntry(id="after-partial", label="after")]) == 2
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "sidecar-partial")


@pytest.mark.anyio
async def test_missing_confirmed_sidecar_and_bad_sequence_fail_closed(
    artifact_dir: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-sidecar"
    assert _run_driver("write", missing, artifact_dir).returncode == 0
    _sidecars(missing)[0].unlink()
    missing_reader = _run_driver("read", missing, artifact_dir, result_name="missing-reader.json")
    assert missing_reader.returncode == 2
    assert _driver_payload(missing, "missing-reader.json")["error"] == "JournalCorruptionError"
    with pytest.raises(JournalCorruptionError, match="(?i)(missing|confirmed|sidecar|corrupt)"):
        await _new_session(missing)

    sequence = tmp_path / "sequence"
    assert _run_driver("write", sequence, artifact_dir).returncode == 0
    main = _main(sequence)
    first = main.read_bytes().splitlines(keepends=True)[0]
    main.write_bytes(main.read_bytes() + first)
    sequence_reader = _run_driver(
        "read", sequence, artifact_dir, result_name="sequence-reader.json"
    )
    assert sequence_reader.returncode == 2
    assert _driver_payload(sequence, "sequence-reader.json")["error"] == "JournalCorruptionError"
    with pytest.raises(JournalCorruptionError, match="(?i)(sequence|increas|corrupt)"):
        await _new_session(sequence)
    _snapshot_artifacts(artifact_dir, missing, "missing-sidecar")
    _snapshot_artifacts(artifact_dir, sequence, "sequence")


@pytest.mark.anyio
async def test_confirmed_record_after_unconfirmed_tail_is_corruption(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "confirmed-after-tail"
    assert _run_driver("write", session, artifact_dir).returncode == 0
    sidecar = _sidecars(session)[0]
    lines = sidecar.read_bytes().splitlines(keepends=True)
    assert lines
    sidecar.write_bytes(lines[0] + _unconfirmed_record(sidecar) + lines[0])
    reader = _run_driver("read", session, artifact_dir, result_name="confirmed-tail-reader.json")
    assert reader.returncode == 2
    assert (
        _driver_payload(session, "confirmed-tail-reader.json")["error"] == "JournalCorruptionError"
    )
    with pytest.raises(JournalCorruptionError, match="(?i)(order|unconfirmed|corrupt)"):
        await _new_session(session)
    _snapshot_artifacts(artifact_dir, session, "confirmed-after-tail")


@pytest.mark.anyio
async def test_flush_failure_poison_does_not_publish_marker(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "flush-failure"
    result = _run_driver("flush-failure", session, artifact_dir, fsync=True)
    assert result.returncode == 17
    assert _main(session).exists() is False or not _main(session).read_bytes().strip()
    reader = _run_driver("read", session, artifact_dir, result_name="flush-reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    assert await read_all(reopened) == []
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "flush-failure")


@pytest.mark.anyio
@pytest.mark.parametrize("target", ("sidecar", "main"))
@pytest.mark.parametrize("mode", ("before", "after", "short"))
async def test_append_failure_matrix_is_poisoned_and_reopen_uses_marker_only(
    artifact_dir: Path, tmp_path: Path, target: str, mode: str
) -> None:
    session = tmp_path / f"append-{target}-{mode}"
    result = _run_driver(
        "fault-commit",
        session,
        artifact_dir,
        fault_operation="append",
        fault_target=target,
        fault_mode=mode,
    )
    assert result.returncode == 17, result.stderr
    payload = _driver_payload(session)
    assert payload["error"] == "JournalPoisonedError"
    assert payload["model_calls"] == []
    reader = _run_driver("read", session, artifact_dir, result_name="fault-reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    entries = await read_all(reopened)
    if target == "main" and mode == "after":
        assert {getattr(entry, "id", None) for entry in entries} >= {"human", "usage", "leaf"}
        assert await reopened.task(1) is not None
        assert await reopened.document(2) is not None
    else:
        assert entries == []
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, f"append-{target}-{mode}")


@pytest.mark.anyio
@pytest.mark.parametrize("target", ("sidecar", "main"))
@pytest.mark.parametrize("mode", ("before", "after"))
async def test_flush_failure_matrix_poison_or_best_effort_gc(
    artifact_dir: Path, tmp_path: Path, target: str, mode: str
) -> None:
    session = tmp_path / f"flush-{target}-{mode}"
    if target == "sidecar":
        result = _run_driver(
            "fault-commit",
            session,
            artifact_dir,
            fsync=True,
            fault_operation="flush",
            fault_target=target,
            fault_mode=mode,
        )
        assert result.returncode == 17
        payload = _driver_payload(session)
        assert payload["error"] == "JournalPoisonedError"
        reader = _run_driver("read", session, artifact_dir, result_name="flush-matrix-reader.json")
        assert reader.returncode == 0, reader.stderr
    else:
        result = _run_driver(
            "gc-fault",
            session,
            artifact_dir,
            fsync=True,
            fault_operation="flush",
            fault_target="main",
            fault_mode=mode,
        )
        assert result.returncode == 0
        payload = _driver_payload(session)
        assert payload["model_calls"] == []
        reader = _run_driver("read", session, artifact_dir, result_name="gc-flush-reader.json")
        assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    entries = await read_all(reopened)
    if target == "sidecar":
        assert entries == []
    else:
        assert {getattr(entry, "id", None) for entry in entries} >= {"human", "usage", "leaf"}
        assert await reopened.document(2) is not None
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, f"flush-{target}-{mode}")


@pytest.mark.anyio
async def test_reclaim_interruption_keeps_published_state_and_recovers_temp(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "reclaim"
    assert _run_driver("write", session, artifact_dir).returncode == 0
    reclaim = session / "doc-2.jsonl.reclaim"
    reclaim.write_text("partial reclaim", encoding="utf-8")
    reader = _run_driver("read", session, artifact_dir, result_name="reclaim-reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    assert {getattr(entry, "id", None) for entry in await read_all(reopened)} >= {"human", "usage"}
    await close(reopened)
    assert not reclaim.exists()
    _snapshot_artifacts(artifact_dir, session, "reclaim-interrupted")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("operation", "mode"),
    (
        ("write", "before"),
        ("write", "after"),
        ("write", "short"),
        ("rename", "before"),
        ("rename", "after"),
    ),
)
async def test_gc_interruption_keeps_published_state(
    artifact_dir: Path, tmp_path: Path, operation: str, mode: str
) -> None:
    session = tmp_path / f"gc-{operation}-{mode}"
    result = _run_driver(
        "gc-fault",
        session,
        artifact_dir,
        fsync=True,
        fault_operation=operation,
        fault_mode=mode,
    )
    assert result.returncode == 0, result.stderr
    payload = _driver_payload(session)
    assert payload["model_calls"] == []
    reader = _run_driver("read", session, artifact_dir, result_name="gc-reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    entries = await read_all(reopened)
    assert {getattr(entry, "id", None) for entry in entries} >= {"human", "usage", "leaf"}
    document = await reopened.document(2)
    assert document is not None and document["value"]["checkpoint"] == "replacement"
    await close(reopened)
    assert not any(path.name.endswith(".reclaim") for path in session.iterdir())
    _snapshot_artifacts(artifact_dir, session, f"gc-{operation}-{mode}")


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ("before", "after"))
async def test_gc_remove_interruption_is_retried_on_next_open(
    artifact_dir: Path, tmp_path: Path, mode: str
) -> None:
    session = tmp_path / f"gc-remove-{mode}"
    result = _run_driver(
        "gc-remove",
        session,
        artifact_dir,
        fault_mode=mode,
    )
    assert result.returncode == 0, result.stderr
    assert (session / "doc-2.jsonl.reclaim").exists() is (mode == "before")
    reader = _run_driver("read", session, artifact_dir, result_name="gc-remove-reader.json")
    assert reader.returncode == 0, reader.stderr
    reopened = await _new_session(session)
    assert {getattr(entry, "id", None) for entry in await read_all(reopened)} >= {
        "human",
        "usage",
        "leaf",
    }
    await close(reopened)
    assert not (session / "doc-2.jsonl.reclaim").exists()
    _snapshot_artifacts(artifact_dir, session, f"gc-remove-{mode}")


@pytest.mark.anyio
async def test_concurrent_commits_have_strictly_increasing_sequences(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "concurrent"
    result = _run_driver("concurrent", session, artifact_dir)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout or "{}") if result.stdout.strip().startswith("{") else {}
    reopened = await _new_session(session)
    entries = await read_all(reopened)
    assert {getattr(entry, "id", None) for entry in entries} >= {
        "label-0",
        "label-1",
        "label-2",
        "label-3",
    }
    lines = [json.loads(line) for line in _main(session).read_text(encoding="utf-8").splitlines()]
    seqs = [int(line["seq"]) for line in lines]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    _record(artifact_dir, "concurrent-commit", child_payload=payload, sequences=seqs)
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "concurrent-commit")


@pytest.mark.anyio
async def test_queued_commit_after_failure_cannot_publish_or_refresh_cache(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "queued-failure"
    journal = await _new_session(session)
    original = journal._append_file
    first = True

    def fail_first(path: Path, data: bytes) -> None:
        nonlocal first
        if first:
            first = False
            raise OSError("D1 first commit failure")
        original(path, data)

    journal._append_file = fail_first  # type: ignore[method-assign]
    results = await asyncio.gather(
        commit(journal, [LabelEntry(id="first", label="failed")]),
        commit(journal, [LabelEntry(id="second", label="must-not-publish")]),
        return_exceptions=True,
    )
    assert all(isinstance(value, JournalPoisonedError) for value in results)
    assert not _main(session).exists() or not _main(session).read_bytes().strip()
    await close(journal)
    reopened = await _new_session(session)
    assert await read_all(reopened) == []
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "queued-failure")


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", [False, True])
async def test_commit_cancellation_cannot_interrupt_admitted_storage_settlement(
    artifact_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter: bool
) -> None:
    directory = tmp_path / "cancel-admitted"
    storage = JsonlSessionStorage(directory) if adapter else await Journal.open(directory)
    entered, release = threading.Event(), threading.Event()
    append = Journal._append_file
    blocked = False

    def blocked_marker(owner: Journal, path: Path, data: bytes) -> None:
        nonlocal blocked
        if path.name == "main.jsonl" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)
        append(owner, path, data)

    monkeypatch.setattr(Journal, "_append_file", blocked_marker)
    first_entry = MessageEntry(id="cancel-first", message=HumanMessage(content="first"))
    second_entry = MessageEntry(
        id="cancel-second", parent_id=first_entry.id, message=HumanMessage(content="second")
    )
    first_writes = (
        [first_entry]
        if adapter
        else [{"type": "entry", "value": json.loads(entry_to_json_line(first_entry))}]
    )
    second_writes = (
        [second_entry]
        if adapter
        else [{"type": "entry", "value": json.loads(entry_to_json_line(second_entry))}]
    )
    first = asyncio.create_task(storage.commit(first_writes))
    assert await asyncio.to_thread(entered.wait, 10)
    second = asyncio.create_task(storage.commit(second_writes))
    await asyncio.sleep(0)
    second.cancel()
    release.set()
    await first
    with pytest.raises(asyncio.CancelledError):
        await second
    await storage.close()
    reader = _run_driver("read", directory, artifact_dir, result_name="cancel-reader.json")
    assert reader.returncode == 0, reader.stderr
    assert _driver_payload(directory, "cancel-reader.json")["ids"] == [
        "cancel-first",
        "cancel-second",
    ]
    _record(artifact_dir, "cancel-admitted", adapter=adapter, sequences=[1, 2])


@pytest.mark.anyio
async def test_close_rejects_write_after_close(artifact_dir: Path, tmp_path: Path) -> None:
    session = tmp_path / "close"
    result = _run_driver("close-during-write", session, artifact_dir)
    assert result.returncode == 0, result.stderr
    payload = json.loads((session / "driver-result.json").read_text(encoding="utf-8"))
    assert payload["outcomes"] == [JournalClosedError.__name__]
    _snapshot_artifacts(artifact_dir, session, "close-write")


@pytest.mark.anyio
async def test_close_settles_commits_admitted_before_close(
    artifact_dir: Path, tmp_path: Path
) -> None:
    """Close waits for operations that passed admission before it started."""

    session = tmp_path / "close-admitted"
    journal = await _new_session(session)
    original_append = journal._append_file
    first_started = threading.Event()
    release_first = threading.Event()
    first_append = True

    def block_first(path: Path, data: bytes) -> None:
        nonlocal first_append
        if first_append:
            first_append = False
            first_started.set()
            if not release_first.wait(5):
                raise OSError("D1 close admission test timed out")
        original_append(path, data)

    journal._append_file = block_first  # type: ignore[method-assign]
    original_admit = journal._assert_admitting
    admitted = 0
    second_admitted = asyncio.Event()

    def observe_admission() -> None:
        nonlocal admitted
        admitted += 1
        if admitted == 2:
            second_admitted.set()
        original_admit()

    journal._assert_admitting = observe_admission  # type: ignore[method-assign]
    first_task = asyncio.create_task(commit(journal, [LabelEntry(id="first", label="one")]))
    await asyncio.to_thread(first_started.wait, 5)
    second_task = asyncio.create_task(commit(journal, [LabelEntry(id="second", label="two")]))
    await asyncio.wait_for(second_admitted.wait(), 5)
    close_task = asyncio.create_task(journal.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    release_first.set()
    outcomes = await asyncio.gather(first_task, second_task, close_task, return_exceptions=True)
    assert outcomes[:2] == [1, 2]
    assert outcomes[2] is None
    with pytest.raises(JournalClosedError):
        await commit(journal, [LabelEntry(id="late", label="rejected")])
    reopened = await _new_session(session)
    assert [getattr(entry, "id", None) for entry in await read_all(reopened)] == [
        "first",
        "second",
    ]
    await close(reopened)
    _record(artifact_dir, "close-admitted", sequences=outcomes[:2])
    _snapshot_artifacts(artifact_dir, session, "close-admitted")


@pytest.mark.anyio
async def test_prepare_validation_does_no_io_and_does_not_poison_journal(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "prepare"
    journal = await _new_session(session)
    with pytest.raises(TypeError, match="unsupported|JSON|object"):
        await journal.commit([{"type": "entry", "value": {"id": "bad", "value": object()}}])
    assert not list(session.glob("*.jsonl"))
    assert await commit(journal, [LabelEntry(id="after-prepare", label="usable")]) == 1
    assert [getattr(entry, "id", None) for entry in await read_all(journal)] == ["after-prepare"]
    await close(journal)
    _snapshot_artifacts(artifact_dir, session, "prepare-noio")


@pytest.mark.anyio
async def test_prepare_rejects_same_commit_duplicates_without_consuming_sequence(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "duplicate"
    journal = await _new_session(session)
    duplicate = [LabelEntry(id="duplicate", label="one"), LabelEntry(id="duplicate", label="two")]
    with pytest.raises(ValueError, match="already exists|duplicate"):
        await commit(journal, duplicate)
    assert not list(session.glob("*.jsonl"))
    assert await commit(journal, [LabelEntry(id="after-duplicate", label="usable")]) == 1
    await close(journal)
    _snapshot_artifacts(artifact_dir, session, "duplicate-noio")


@pytest.mark.anyio
async def test_task_and_document_validation_rejects_invalid_transitions_before_io(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "transitions"
    journal = await _new_session(session)
    task = sample_task_write()
    document = sample_document_write()
    same_document = sample_document_write()
    with pytest.raises(ValueError, match="already belongs|already exists|document"):
        await journal.commit([task, document, same_document])
    assert not list(session.glob("*.jsonl"))
    await journal.commit([task, document])
    invalid_delta = {
        "type": "document.change",
        "id": 2,
        "content": {"kind": "delta", "version": 2, "ops": []},
    }
    with pytest.raises(ValueError, match="version|base"):
        await journal.commit([invalid_delta])
    assert len(_main(session).read_bytes().splitlines()) == 1
    await journal.commit([sample_document_change()])
    await close(journal)
    _snapshot_artifacts(artifact_dir, session, "transition-validation")


def _document_create(document_id: int, key: str, checkpoint: str) -> dict[str, Any]:
    return {
        "type": "document.create",
        "record": {
            "id": document_id,
            "kind": "d1.validation",
            "key": key,
            "scope": {"kind": "session"},
        },
        "content": {
            "kind": "base",
            "version": 1,
            "value": {"checkpoint": checkpoint},
        },
    }


@pytest.mark.anyio
async def test_same_commit_creates_two_distinct_document_addresses(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "document-create-pair"
    journal = await _new_session(session)
    assert (
        await journal.commit(
            [
                _document_create(10, "first", "one"),
                _document_create(11, "second", "two"),
            ]
        )
        == 1
    )
    first = await journal.document(10)
    second = await journal.document(11)
    assert first is not None and first["value"]["checkpoint"] == "one"
    assert second is not None and second["value"]["checkpoint"] == "two"
    assert len(_main(session).read_bytes().splitlines()) == 1
    await close(journal)
    _snapshot_artifacts(artifact_dir, session, "document-create-pair")


@pytest.mark.anyio
async def test_same_address_replacement_accepts_both_write_orders(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "document-replacement-order"
    journal = await _new_session(session)
    await journal.commit([_document_create(20, "same-address", "old")])
    reversed_writes = [
        _document_create(21, "same-address", "new"),
        {"type": "document.retire", "id": 20},
    ]
    assert await journal.commit(reversed_writes) == 2
    assert await journal.document(20) is None
    replacement = await journal.document(21)
    assert replacement is not None and replacement["value"]["checkpoint"] == "new"
    await journal.commit([_document_create(30, "forward-address", "old-forward")])
    assert (
        await journal.commit(
            [
                {"type": "document.retire", "id": 30},
                _document_create(31, "forward-address", "new-forward"),
            ]
        )
        == 4
    )
    assert await journal.document(30) is None
    forward_replacement = await journal.document(31)
    assert forward_replacement is not None
    assert forward_replacement["value"]["checkpoint"] == "new-forward"
    await close(journal)
    reopened = await _new_session(session)
    assert await reopened.document(20) is None
    reopened_reverse = await reopened.document(21)
    assert reopened_reverse is not None
    assert reopened_reverse["value"]["checkpoint"] == "new"
    assert await reopened.document(30) is None
    reopened_forward = await reopened.document(31)
    assert reopened_forward is not None
    assert reopened_forward["value"]["checkpoint"] == "new-forward"
    await close(reopened)
    _snapshot_artifacts(artifact_dir, session, "document-replacement-order")


@pytest.mark.anyio
async def test_conversation_duplicate_id_is_rejected_before_io(
    artifact_dir: Path, tmp_path: Path
) -> None:
    session = tmp_path / "conversation-duplicate"
    journal = await _new_session(session)
    value = {"id": 3, "conversationId": "conversation-3", "title": "offline"}
    with pytest.raises(ValueError, match="already exists|duplicate|conversation"):
        await journal.commit(
            [
                {"type": "conversation", "value": value},
                {"type": "conversation", "value": value},
            ]
        )
    assert not list(session.glob("*.jsonl"))
    await close(journal)
    _snapshot_artifacts(artifact_dir, session, "conversation-duplicate")


def _legacy_source(path: Path) -> list[object]:
    entries = [
        SessionInfoEntry(id="legacy-info", cwd=str(path.parent)),
        MessageEntry(
            id="legacy-user", parent_id="legacy-info", message=HumanMessage(content="legacy 分支")
        ),
        BranchSummaryEntry(
            id="legacy-branch",
            parent_id="legacy-user",
            summary="legacy branch summary",
            branch_root_id="legacy-user",
        ),
        LeafEntry(id="legacy-leaf", parent_id="legacy-branch", entry_id="legacy-branch"),
        CustomEntry(
            id="legacy-todo", parent_id="legacy-user", namespace="forge.todo.v1", data={"todos": []}
        ),
    ]
    path.write_text("".join(entry_to_json_line(entry) for entry in entries), encoding="utf-8")
    return entries


@pytest.mark.anyio
async def test_legacy_import_is_no_overwrite_and_retryable(
    artifact_dir: Path, tmp_path: Path
) -> None:
    source = tmp_path / "legacy.jsonl"
    entries = _legacy_source(source)
    original = source.read_bytes()
    assert original.endswith(b"\n")
    source.write_bytes(original[:-1])
    original = source.read_bytes()
    interrupted = tmp_path / "migration-interrupted"
    interrupted_result = _run_driver(
        "migrate-fault",
        interrupted,
        artifact_dir,
        source=source,
        destination=interrupted,
        fault_mode="after",
        result_name="migration-fault-result.json",
    )
    assert interrupted_result.returncode == 17, interrupted_result.stderr
    assert _driver_payload(interrupted, "migration-fault-result.json")["error"] == (
        "JournalPoisonedError"
    )
    interrupted_reader = _run_driver(
        "read", interrupted, artifact_dir, result_name="migration-fault-reader.json"
    )
    assert interrupted_reader.returncode == 0, interrupted_reader.stderr
    await import_legacy(source, interrupted)
    interrupted_journal = await _new_session(interrupted)
    assert [getattr(entry, "id", None) for entry in await read_all(interrupted_journal)] == [
        getattr(entry, "id", None) for entry in entries
    ]
    await close(interrupted_journal)
    assert source.read_bytes() == original
    _snapshot_artifacts(artifact_dir, interrupted, "migration-interrupted")
    destination = tmp_path / "migrated"
    migrated = _run_driver(
        "migrate",
        destination,
        artifact_dir,
        source=source,
        destination=destination,
        result_name="migrate-result.json",
    )
    assert migrated.returncode == 0, migrated.stderr
    reader = _run_driver("read", destination, artifact_dir, result_name="migrate-reader.json")
    assert reader.returncode == 0, reader.stderr
    assert _driver_payload(destination, "migrate-reader.json")["ids"]
    await import_legacy(source, destination)
    assert source.read_bytes() == original
    assert destination.exists()
    reopened = await _new_session(destination)
    restored = await read_all(reopened)
    assert [getattr(entry, "id", None) for entry in restored] == [
        getattr(entry, "id", None) for entry in entries
    ]
    branch = next(entry for entry in restored if isinstance(entry, BranchSummaryEntry))
    leaf = next(entry for entry in restored if isinstance(entry, LeafEntry))
    assert branch.branch_root_id == "legacy-user" and leaf.entry_id == "legacy-branch"
    await close(reopened)
    mismatch = tmp_path / "mismatch"
    mismatch_journal = await _new_session(mismatch)
    await commit(mismatch_journal, [LabelEntry(id="different", label="different")])
    await close(mismatch_journal)
    with pytest.raises(SessionStorageError, match="prefix|match|source"):
        await import_legacy(source, mismatch)
    assert source.read_bytes() == original
    _snapshot_artifacts(artifact_dir, destination, "migration-retry")


@pytest.mark.anyio
async def test_legacy_export_and_full_copy_keep_branch_and_storage_files(
    artifact_dir: Path, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    storage = await _new_session(source)
    entries = sample_entries()
    await commit_sample(storage)
    await close(storage)
    legacy = tmp_path / "legacy-export.jsonl"
    copied = tmp_path / "copy"
    await export_legacy(source, legacy)
    await copy_session_storage(source, copied)
    legacy_text = legacy.read_text(encoding="utf-8")
    assert "recovery" not in legacy_text.lower()
    assert _main(copied).exists()
    assert {path.name for path in copied.iterdir()} >= {"main.jsonl", "task-1.jsonl", "doc-2.jsonl"}
    assert {path.name for path in copied.iterdir()} == {path.name for path in source.iterdir()}
    copied_storage = await _new_session(copied)
    assert [getattr(entry, "id", None) for entry in await read_all(copied_storage)] == [
        getattr(entry, "id", None) for entry in entries
    ]
    await close(copied_storage)
    _snapshot_artifacts(artifact_dir, copied, "branch-import-export-copy")


@pytest.mark.anyio
@pytest.mark.parametrize("final_status", ["pending", "terminal"])
@pytest.mark.parametrize("fsync", [False, True])
async def test_review_task_reclamation_follows_final_write(
    artifact_dir: Path, tmp_path: Path, final_status: str, fsync: bool
) -> None:
    session = tmp_path / "final-task"
    journal = await Journal.open(session, fsync=fsync)
    live = sample_task_write()
    terminal = copy.deepcopy(live)
    terminal["value"]["state"] = {"status": "terminal", "outcome": {"type": "completed"}}
    await journal.commit([live])
    writes = [terminal, live] if final_status == "pending" else [live, terminal]
    assert await journal.commit(writes) == 2
    task = await journal.task(1)
    assert task is not None and task["state"]["status"] == final_status
    assert (session / "task-1.jsonl").exists() == (final_status == "pending")
    await journal.close()
    reader = _run_driver("read", session, artifact_dir, fsync=fsync)
    assert reader.returncode == 0, reader.stderr
    payload = _driver_payload(session)
    assert payload["task"]["state"]["status"] == final_status
    assert payload["seq"] == 2
    _record(
        artifact_dir, "task-final-write", final_status=final_status, fsync=fsync, reader=payload
    )


@pytest.mark.anyio
@pytest.mark.parametrize("retire_first", [False, True])
@pytest.mark.parametrize("foreign_marker", [False, True])
async def test_review_document_actions_merge_before_adoption_and_recovery(
    artifact_dir: Path, tmp_path: Path, retire_first: bool, foreign_marker: bool
) -> None:
    session = tmp_path / "merged-document"
    create = sample_document_write()
    retire = {"type": "document.retire", "id": 2}
    writes = [retire, create] if retire_first else [create, retire]
    if foreign_marker:
        session.mkdir()
        encoded_create = {"type": "document.create", "record": create["record"], "ordinal": 0}
        encoded = [retire, encoded_create] if retire_first else [encoded_create, retire]
        _main(session).write_text(
            json.dumps({"format": 1, "type": "commit", "seq": 1, "writes": encoded}) + "\n",
            encoding="utf-8",
        )
    journal = await Journal.open(session)
    if not foreign_marker:
        assert await journal.commit(writes) == 1
    assert journal.sequence == 1
    assert await journal.document(2) is None
    assert not (session / "doc-2.jsonl").exists()
    # A retired transient incarnation does not conflict with a live address.
    other = copy.deepcopy(create)
    other["record"]["id"] = 3
    await journal.commit([other])
    transient = copy.deepcopy(create)
    transient["record"]["id"] = 4
    transient_retire = {"type": "document.retire", "id": 4}
    transient_writes = (
        [transient_retire, transient] if retire_first else [transient, transient_retire]
    )
    assert await journal.commit(transient_writes) == 3
    assert await journal.document(3) is not None
    assert await journal.document(4) is None
    await journal.close()
    reader = _run_driver("read", session, artifact_dir)
    assert reader.returncode == 0, reader.stderr
    assert _driver_payload(session)["document"] is None
    reopened = await Journal.open(session)
    assert await reopened.document(3) is not None
    assert await reopened.document(4) is None
    await reopened.close()
    _record(
        artifact_dir,
        "document-action-merge",
        retire_first=retire_first,
        foreign_marker=foreign_marker,
        sequences=[1, 2, 3],
    )


@pytest.mark.anyio
@pytest.mark.parametrize("entrypoint", ["legacy", "cli", "session"])
async def test_review_export_rejects_source_storage_and_aliases(
    artifact_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    from conftest import isolate_home
    from forge_cli.cli import export_session_command
    from forge_cli.tui.app import ForgeTuiApp
    from forge_cli.tui.prompt import PromptInput

    isolate_home(monkeypatch, tmp_path / "home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = tmp_path / "source.jsonl"
    original.write_text(
        entry_to_json_line(SessionInfoEntry(id="export-info", cwd=str(workspace))), encoding="utf-8"
    )
    adapter = JsonlSessionStorage(original)
    await adapter.append(LabelEntry(id="export-label", parent_id="export-info", label="preserve"))
    source = adapter.path.parent
    journal = await Journal.open(source)
    await journal.commit([sample_task_write(), sample_document_write()])
    await journal.close()
    model = ScriptedChatModel([])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            storage=adapter,
            cwd=workspace,
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    protected = [original, _main(source), source / "task-1.jsonl", source / "doc-2.jsonl"]
    before = {str(p): p.read_bytes() for p in protected}
    targets = [
        *protected,
        source / "unrelated-export.jsonl",
        source / ".." / source.name / "main.jsonl",
    ]
    for index, protected_path in enumerate(protected):
        alias = tmp_path / f"hardlink-{index}.jsonl"
        os.link(protected_path, alias)
        targets.append(alias)
    rejected = []
    try:
        for format in ["jsonl"] if entrypoint == "legacy" else ["jsonl", "html"]:
            for target in targets:
                with pytest.raises((SessionStorageError, ValueError), match="source storage"):
                    if entrypoint == "legacy":
                        await export_legacy(adapter.path, target)
                    elif entrypoint == "cli":
                        await export_session_command(str(adapter.path), target, format)
                    else:
                        await session.export(target, format=format)
                rejected.append({"target": str(target), "format": format})
                assert {str(p): p.read_bytes() for p in protected} == before
        if entrypoint == "session":
            app = ForgeTuiApp(session)
            notices: list[str] = []
            monkeypatch.setattr(app, "_notify", lambda message, **kwargs: notices.append(message))
            async with app.run_test(size=(100, 24)) as pilot:
                app.query_one(
                    "#prompt", PromptInput
                ).text = f"/export --format jsonl {_main(source).as_posix()}"
                await app.action_submit_prompt()
                await pilot.pause()
                assert any("source storage" in message for message in notices), notices
                assert not any(message.startswith("Exported") for message in notices)
        else:
            await session.aclose()
        output = tmp_path / "outside-export.jsonl"
        if entrypoint == "legacy":
            await export_legacy(adapter.path, output)
        elif entrypoint == "cli":
            await export_session_command(str(adapter.path), output, "jsonl")
        else:
            # The TUI closes its session on exit; use a new instance for valid exports.
            session = await CodingSession.load(
                CodingSessionConfig(
                    provider=model,
                    model="fake",
                    storage=JsonlSessionStorage(adapter.path),
                    cwd=workspace,
                    enable_subagents=False,
                    auto_compact_enabled=False,
                )
            )
            await session.export(output, format="jsonl")
        assert "export-label" in output.read_text(encoding="utf-8")
        assert {str(p): p.read_bytes() for p in protected} == before
        assert model.calls == []
        reader = _run_driver("read", source, artifact_dir)
        assert reader.returncode == 0, reader.stderr
        _record(
            artifact_dir,
            "export-source-guard",
            entrypoint=entrypoint,
            rejected=rejected,
            source_sha256={p: hashlib.sha256(data).hexdigest() for p, data in before.items()},
            model_calls=len(model.calls),
            reader=_driver_payload(source),
        )
    finally:
        await session.aclose()
