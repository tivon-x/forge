"""Subprocess driver for the D1 durable-journal acceptance cases.

The driver deliberately exits with ``os._exit`` for crash points.  It never
opens a user's configured session: every path is supplied by the parent test.
The journal adapter is intentionally tiny; the production implementation owns
the public names and the Pi commit protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from fake_models import ScriptedChatModel, tool_call_ai
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
from forge_coding.sessions.journal import Journal
from forge_coding.sessions.storage import JsonlSessionStorage, import_legacy

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IDS = (
    "info",
    "human",
    "assistant",
    "tool",
    "usage",
    "todo",
    "goal",
    "run",
    "branch",
    "leaf",
)


def sample_task_write() -> dict[str, Any]:
    """Return a live task write that must use a task sidecar."""

    return {
        "type": "task",
        "value": {
            "id": 1,
            "conversationId": "conversation-1",
            "kind": "d1.acceptance",
            "version": 1,
            "input": {"prompt": "offline acceptance"},
            "background": False,
            "abortRequested": False,
            "state": {"status": "pending", "checkpoint": {"phase": "start"}},
        },
    }


def sample_document_write() -> dict[str, Any]:
    """Return a session document create that must use a document sidecar."""

    return {
        "type": "document.create",
        "record": {"id": 2, "kind": "d1.checkpoint", "scope": {"kind": "session"}},
        "content": {
            "kind": "base",
            "version": 1,
            "value": {"branch": "main", "checkpoint": "durable"},
        },
    }


def sample_document_change() -> dict[str, Any]:
    """Return a new base that triggers current-only document reclamation."""

    return {
        "type": "document.change",
        "id": 2,
        "content": {
            "kind": "base",
            "version": 1,
            "value": {"branch": "main", "checkpoint": "replacement"},
        },
    }


def sample_entries() -> list[object]:
    """Return one atomic commit containing every D1 product projection."""

    return [
        SessionInfoEntry(id="info", cwd=str(ROOT)),
        MessageEntry(id="human", parent_id="info", message=HumanMessage(content="你好 🦄")),
        MessageEntry(
            id="assistant",
            parent_id="human",
            message=AIMessage(content="已完成", response_metadata={"request_id": "req-1"}),
        ),
        MessageEntry(
            id="tool",
            parent_id="assistant",
            message=ToolMessage(content="tool-output", tool_call_id="call-1"),
        ),
        CustomEntry(
            id="usage",
            parent_id="tool",
            namespace="forge.usage.v1",
            data={"call_id": "call-1", "input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        ),
        CustomEntry(
            id="todo",
            parent_id="tool",
            namespace="forge.todo.v1",
            data={"todos": [{"id": "todo-1", "content": "verify", "status": "completed"}]},
        ),
        CustomEntry(
            id="goal",
            parent_id="tool",
            namespace="forge.goal.v1",
            data={"goal": {"objective": "durable D1", "status": "active"}},
        ),
        CustomEntry(
            id="run",
            parent_id="tool",
            namespace="forge.subagent_run.v1",
            data={
                "run_id": "run-1",
                "kind": "subagent_receipt",
                "status": "completed",
                "tool_call_id": "call-1",
            },
        ),
        BranchSummaryEntry(
            id="branch",
            parent_id="tool",
            summary="D1 branch summary",
            branch_root_id="human",
        ),
        LeafEntry(id="leaf", parent_id="branch", entry_id="branch"),
    ]


async def open_storage(path: Path, **kwargs: Any) -> Journal:
    """Open the concrete D1 Journal contract."""

    return await Journal.open(path, fsync=bool(kwargs.get("fsync", False)))


async def commit(storage: Journal, entries: Sequence[object]) -> int:
    writes = [
        {"type": "entry", "value": json.loads(entry_to_json_line(entry))} for entry in entries
    ]
    value = await storage.commit(writes)
    if not isinstance(value, int):
        raise RuntimeError(f"commit returned non-sequence value: {value!r}")
    return value


async def commit_sample(storage: Any) -> int:
    """Commit entries, task, and document under one main marker."""

    writes = [
        {"type": "entry", "value": json.loads(entry_to_json_line(entry))}
        for entry in sample_entries()
    ]
    writes.extend((sample_task_write(), sample_document_write()))
    value = await storage.commit(writes)
    if not isinstance(value, int):
        raise RuntimeError(f"commit returned non-sequence value: {value!r}")
    return value


async def read_all(storage: Journal) -> list[Any]:
    return await storage.read_all()


async def close(storage: Journal) -> None:
    await storage.close()


def _fault_matches(path: Path, target: str) -> bool:
    if target == "main":
        return path.name == "main.jsonl"
    if target == "reclaim":
        return path.name.endswith(".reclaim")
    return path.name != "main.jsonl"


def install_append_fault(storage: Journal, target: str, mode: str) -> None:
    """Inject one append failure while retaining the Pi write ordering."""

    original = storage._append_file
    hit = False

    def faulty(path: Path, data: bytes) -> None:
        nonlocal hit
        if hit or not _fault_matches(path, target):
            original(path, data)
            return
        hit = True
        if mode == "before":
            raise OSError(f"D1 append {target} before")
        if mode == "short":
            original(path, data[: max(1, len(data) // 2)])
            raise OSError(f"D1 append {target} short")
        original(path, data)
        raise OSError(f"D1 append {target} after")

    storage._append_file = faulty  # type: ignore[method-assign]


def install_flush_fault(storage: Journal, target: str, mode: str) -> None:
    """Inject one flush failure for sidecar or post-marker main flush."""

    original = storage._flush_file
    hit = False

    def faulty(path: Path) -> None:
        nonlocal hit
        if hit or not _fault_matches(path, target):
            original(path)
            return
        hit = True
        if mode == "before":
            raise OSError(f"D1 flush {target} before")
        original(path)
        raise OSError(f"D1 flush {target} after")

    storage._flush_file = faulty  # type: ignore[method-assign]


def install_gc_fault(storage: Journal, operation: str, mode: str) -> None:
    """Inject write/flush/rename failure in the best-effort GC path."""

    if operation == "write":
        original = storage._replace_sidecar

        def faulty(name: str, content: bytes | None) -> None:
            if content is None:
                original(name, content)
                return
            temporary = storage.directory / f"{name}.reclaim"
            if mode == "before":
                raise OSError("D1 gc write before")
            temporary.write_bytes(
                content[: max(1, len(content) // 2)] if mode == "short" else content
            )
            if mode in {"short", "after"}:
                if mode == "after":
                    raise OSError("D1 gc write after")
                raise OSError("D1 gc write short")
            os.replace(temporary, storage.directory / name)

        storage._replace_sidecar = faulty  # type: ignore[method-assign]
        return

    if operation == "flush":
        install_flush_fault(storage, "reclaim", mode)
        return

    if operation == "rename":
        original = os.replace
        hit = False

        def faulty(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
            nonlocal hit
            source_path = Path(source)
            if hit or not source_path.name.endswith(".reclaim"):
                original(source, destination)
                return
            hit = True
            if mode == "before":
                raise OSError("D1 gc rename before")
            original(source, destination)
            raise OSError("D1 gc rename after")

        os.replace = faulty  # type: ignore[assignment]
        return

    raise ValueError(f"unknown gc operation {operation!r}")


def _write_result(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


async def run_fake_session(args: argparse.Namespace, session: Path, result_path: Path) -> int:
    workspace = session / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    tool_calls = 0

    @tool
    async def effect() -> str:
        """Record exactly one controlled offline effect."""
        nonlocal tool_calls
        tool_calls += 1
        with (workspace / "effect.txt").open("a", encoding="utf-8") as output:
            output.write("effect-d1\n")
        return "effect recorded"

    provider = ScriptedChatModel(
        [tool_call_ai("effect-d1", "effect", {}), AIMessage(content="completed")]
    )
    coding = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="D1 offline process acceptance",
            storage=JsonlSessionStorage(session / "native"),
            cwd=workspace,
            tools=[effect],
            enable_subagents=False,
            auto_compact_enabled=False,
        )
    )
    if args.action == "session-and-exit":
        _events = [event async for event in coding.prompt("record one effect")]
    ai_calls = [
        call["id"]
        for message in coding.messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    results = [
        message.tool_call_id for message in coding.messages if isinstance(message, ToolMessage)
    ]
    assert sorted(ai_calls) == sorted(results)
    _write_result(
        result_path,
        action=args.action,
        pid=os.getpid(),
        model_calls=len(provider.calls),
        tool_calls=tool_calls,
        paired_call_ids=sorted(results),
    )
    if args.action == "session-and-exit":
        os._exit(args.exit_code)
    await coding.aclose()
    return 0


async def run(args: argparse.Namespace) -> int:
    session = Path(args.session)
    result_path = Path(args.result) if args.result else session / "driver-result.json"
    if args.action in {"session-and-exit", "session-reopen"}:
        return await run_fake_session(args, session, result_path)
    storage = await open_storage(session, fsync=args.fsync)

    if args.action == "migrate":
        if args.source is None:
            raise ValueError("migrate requires --source")
        await close(storage)
        destination = Path(args.destination) if args.destination else session
        migrated = await import_legacy(Path(args.source), destination)
        _write_result(
            result_path,
            action=args.action,
            destination=str(migrated),
            pid=os.getpid(),
            model_calls=[],
        )
        return 0

    if args.action == "migrate-fault":
        if args.source is None:
            raise ValueError("migrate-fault requires --source")
        original_append = Journal._append_file
        hit = False

        def faulty_append(owner: Journal, path: Path, data: bytes) -> None:
            nonlocal hit
            if hit or path.name != "main.jsonl":
                original_append(owner, path, data)
                return
            hit = True
            if args.fault_mode == "before":
                raise OSError("D1 migration append before")
            if args.fault_mode == "short":
                original_append(owner, path, data[: max(1, len(data) // 2)])
                raise OSError("D1 migration append short")
            original_append(owner, path, data)
            raise OSError("D1 migration append after")

        Journal._append_file = faulty_append  # type: ignore[assignment]
        destination = Path(args.destination) if args.destination else session
        try:
            await import_legacy(Path(args.source), destination)
        except BaseException as exc:
            _write_result(
                result_path,
                action=args.action,
                error=type(exc).__name__,
                pid=os.getpid(),
                model_calls=[],
            )
            os._exit(args.exit_code)
        _write_result(
            result_path,
            action=args.action,
            error="fault-not-triggered",
            pid=os.getpid(),
            model_calls=[],
        )
        os._exit(2)

    if args.action == "commit-and-exit":
        seq = await commit_sample(storage)
        _write_result(result_path, action=args.action, seq=seq, pid=os.getpid(), model_calls=[])
        os._exit(args.exit_code)

    if args.action == "flush-failure":

        def fail_fsync(fd: int) -> None:
            del fd
            raise OSError("D1 injected flush failure")

        os.fsync = fail_fsync  # type: ignore[assignment]
        try:
            await commit_sample(storage)
        except BaseException as exc:  # record the poison before the crash
            _write_result(
                result_path,
                action=args.action,
                error=type(exc).__name__,
                pid=os.getpid(),
                model_calls=[],
            )
            os._exit(args.exit_code)
        _write_result(result_path, action=args.action, error=None, pid=os.getpid(), model_calls=[])
        os._exit(args.exit_code)

    if args.action == "fault-commit":
        if args.fault_operation == "append":
            install_append_fault(storage, args.fault_target, args.fault_mode)
        elif args.fault_operation == "flush":
            install_flush_fault(storage, args.fault_target, args.fault_mode)
        else:
            raise ValueError("fault-commit supports append or flush")
        try:
            await commit_sample(storage)
        except BaseException as exc:
            _write_result(
                result_path,
                action=args.action,
                operation=args.fault_operation,
                target=args.fault_target,
                mode=args.fault_mode,
                error=type(exc).__name__,
                pid=os.getpid(),
                model_calls=[],
            )
            os._exit(args.exit_code)
        _write_result(
            result_path,
            action=args.action,
            error="fault-not-triggered",
            pid=os.getpid(),
            model_calls=[],
        )
        os._exit(2)

    if args.action == "gc-fault":
        await commit_sample(storage)
        if args.fault_operation == "flush" and args.fault_target == "main":
            install_flush_fault(storage, "main", args.fault_mode)
        else:
            install_gc_fault(storage, args.fault_operation, args.fault_mode)
        try:
            seq = await storage.commit([sample_document_change()])
        except BaseException as exc:
            _write_result(
                result_path,
                action=args.action,
                operation=args.fault_operation,
                mode=args.fault_mode,
                error=type(exc).__name__,
                pid=os.getpid(),
                model_calls=[],
            )
            os._exit(args.exit_code)
        _write_result(
            result_path,
            action=args.action,
            operation=args.fault_operation,
            mode=args.fault_mode,
            seq=seq,
            pid=os.getpid(),
            model_calls=[],
        )
        return 0

    if args.action == "gc-remove":
        await commit_sample(storage)
        await close(storage)
        reclaim = session / "doc-2.jsonl.reclaim"
        reclaim.write_bytes(b"stale reclaim")
        original_unlink = Path.unlink
        hit = False

        def faulty_unlink(path: Path, *unlink_args: Any, **unlink_kwargs: Any) -> None:
            nonlocal hit
            if path.name.endswith(".reclaim") and not hit:
                hit = True
                if args.fault_mode == "before":
                    raise OSError("D1 gc remove before")
                original_unlink(path, *unlink_args, **unlink_kwargs)
                raise OSError("D1 gc remove after")
            original_unlink(path, *unlink_args, **unlink_kwargs)

        Path.unlink = faulty_unlink  # type: ignore[assignment]
        reopened = await Journal.open(session, fsync=args.fsync)
        await close(reopened)
        _write_result(
            result_path,
            action=args.action,
            mode=args.fault_mode,
            pid=os.getpid(),
            model_calls=[],
        )
        return 0

    if args.action == "close-during-write":
        await close(storage)
        outcomes = []
        try:
            outcomes.append(await commit(storage, sample_entries()))
        except BaseException as exc:  # close must reject late writes
            outcomes.append(exc)
        _write_result(
            result_path,
            action=args.action,
            outcomes=[
                type(value).__name__ if isinstance(value, BaseException) else value
                for value in outcomes
            ],
            pid=os.getpid(),
            model_calls=[],
        )
        return 0

    if args.action == "concurrent":
        results = await asyncio.gather(
            *(
                commit(storage, [LabelEntry(id=f"label-{index}", label=f"row-{index}")])
                for index in range(4)
            )
        )
        _write_result(
            result_path, action=args.action, seqs=results, pid=os.getpid(), model_calls=[]
        )
        return 0

    if args.action == "write":
        seq = await commit_sample(storage)
        _write_result(result_path, action=args.action, seq=seq, pid=os.getpid(), model_calls=[])
        return 0

    if args.action == "read":
        entries = await read_all(storage)
        _write_result(
            result_path,
            action=args.action,
            ids=[getattr(entry, "id", None) for entry in entries],
            seq=storage.sequence,
            task=await storage.task(1),
            document=await storage.document(2),
            pid=os.getpid(),
            model_calls=[],
        )
        return 0

    raise RuntimeError(f"unknown driver action {args.action!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "write",
            "read",
            "migrate",
            "migrate-fault",
            "commit-and-exit",
            "session-and-exit",
            "session-reopen",
            "flush-failure",
            "fault-commit",
            "gc-fault",
            "gc-remove",
            "close-during-write",
            "concurrent",
        ),
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--fsync", action="store_true")
    parser.add_argument("--exit-code", type=int, default=17)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--fault-operation", choices=("append", "flush", "write", "rename"), default="append"
    )
    parser.add_argument("--fault-target", choices=("sidecar", "main"), default="sidecar")
    parser.add_argument("--fault-mode", choices=("before", "after", "short"), default="before")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except BaseException as exc:
        result = Path(args.result) if args.result else Path(args.session) / "driver-error.json"
        _write_result(result, error=type(exc).__name__, message=str(exc), pid=os.getpid())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
