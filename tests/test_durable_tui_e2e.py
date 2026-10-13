"""D3 UI acceptance scenarios written before the snapshot projection changes.

These cases deliberately use a small session double.  The durable session and
its storage are covered by the sibling process tests; this file verifies that
mounted frontends consume the committed view and can join it late.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from conftest import isolate_home
from fake_models import StreamingScriptedChatModel
from forge_agent import CommittedSnapshotEvent
from forge_cli.rendering import FinalTextRenderer, JsonEventRenderer, TranscriptRenderer
from forge_cli.tui.app import ForgeTuiApp
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.storage import JsonlSessionStorage


def _run_progress_driver(root: Path, action: str) -> dict[str, object]:
    """Run the real exit/reopen driver and return its durable artifacts."""
    driver = Path(__file__).with_name("durable_progress_driver.py")
    expected_exit = 17 if action in {"partial", "background"} else 0
    result = subprocess.run(
        [sys.executable, str(driver), action, str(root)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == expected_exit, (
        f"driver {action} failed with {result.returncode}:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    artifact = root / ("confirmed.json" if action != "reopen" else "reopened.json")
    return json.loads(artifact.read_text(encoding="utf-8"))


def _live_value(snapshot: dict[str, object]) -> dict[str, object]:
    for document in snapshot.get("documents", ()):
        if not isinstance(document, dict):
            continue
        record = document.get("record")
        if isinstance(record, dict) and record.get("kind") == "forge.live.v1":
            value = document.get("value")
            if isinstance(value, dict):
                return value
    return {}


def _entry(entry_id: str, message: object, parent_id: str | None = None) -> dict[str, object]:
    return {
        "type": "message",
        "id": entry_id,
        "parent_id": parent_id,
        "timestamp": 1.0,
        "message": message.model_dump(mode="json"),  # type: ignore[attr-defined]
    }


def _snapshot(*, assistant: str = "partial answer", sequence: int = 7) -> dict[str, object]:
    return {
        "sequence": sequence,
        "entries": [
            _entry("u1", HumanMessage(content="already committed")),
            _entry("a1", AIMessage(content="confirmed answer"), parent_id="u1"),
        ],
        "tasks": [
            {
                "id": 1,
                "owner": 0,
                "background": False,
                "state": {"status": "waiting"},
            },
            {
                "id": 2,
                "owner": 1,
                "background": True,
                "state": {"status": "completing"},
            },
        ],
        "documents": [
            {
                "record": {"id": 1, "kind": "forge.live.v1"},
                "version": 1,
                "value": {
                    "assistant": assistant,
                    "thinking": "thinking so far",
                    "tools": {
                        "call-1": {
                            "start": {
                                "type": "tool_execution_start",
                                "tool_call": {
                                    "id": "call-1",
                                    "name": "read",
                                    "arguments": {"path": "README.md"},
                                },
                            },
                            "updates": [
                                {
                                    "type": "tool_execution_update",
                                    "tool_call_id": "call-1",
                                    "message": "reading README.md",
                                }
                            ],
                        }
                    },
                    "background": [
                        {"run_id": "task-background", "status": "running", "agent": "worker"}
                    ],
                    "running": True,
                },
            }
        ],
    }


class SnapshotSession:
    """Minimal session surface needed by the mounted Forge TUI."""

    def __init__(self, snapshot: dict[str, object]) -> None:
        self._snapshot = snapshot
        self.messages = ()
        self.cwd = Path("/workspace/project")
        self.provider_name = "offline"
        self.model = "fake"
        self.available_models = ("fake",)
        self.available_providers = ("offline",)
        self.available_thinking_levels = ("off", "medium")
        self.thinking_level = "medium"
        self.skills = ()
        self.prompt_templates = ()
        self.tools = ()
        self.resource_diagnostics = ()
        self.context_files = ()
        self.context_token_estimate = 0
        self.auto_compact_token_threshold = 100_000
        self.context_window_tokens = 100_000
        self.system_prompt = "offline"
        self.session_manager = None
        self.queued_steering_messages = ()
        self.queued_follow_up_messages = ()

    @property
    def session_title(self) -> str | None:
        return None

    def snapshot(self) -> dict[str, object]:
        return self._snapshot

    @property
    def background_subagents(self) -> tuple[dict[str, object], ...]:
        documents = self._snapshot.get("documents", ())
        for document in documents:
            if (
                isinstance(document, dict)
                and document.get("record", {}).get("kind") == "forge.live.v1"
            ):
                value = document.get("value", {})
                if isinstance(value, dict):
                    background = value.get("background", ())
                    if isinstance(background, list):
                        return tuple(item for item in background if isinstance(item, dict))
        return ()

    def queue_update_event(self):
        from forge_agent import QueueUpdateEvent

        return QueueUpdateEvent()

    async def prompt(
        self, text: str, *, streaming_behavior: str | None = None
    ) -> AsyncIterator[object]:
        del text, streaming_behavior
        if False:
            yield None


@pytest.mark.anyio
async def test_mounted_tui_rehydrates_late_snapshot_without_duplicate_messages() -> None:
    session = SnapshotSession(_snapshot())
    app = ForgeTuiApp(session)  # type: ignore[arg-type]

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert "already committed" in "\n".join(item.text for item in app.state.items)
        assert "confirmed answer" in "\n".join(item.text for item in app.state.items)
        assert app.state.assistant_buffer == "partial answer"
        assert "thinking so far" in "\n".join(item.text for item in app.state.items)
        assert "reading README.md" in "\n".join(item.text for item in app.state.items)
        assert "task 1" in str(app.query_one("#task-graph").render())
        assert "waiting" in str(app.query_one("#task-graph").render())
        assert "background" in str(app.query_one("#task-graph").render())

        app.adapter.apply(
            CommittedSnapshotEvent(snapshot=_snapshot(assistant="partial answer!", sequence=8))
        )
        await pilot.pause()

        all_text = "\n".join(item.text for item in app.state.items)
        assert all_text.count("already committed") == 1
        assert all_text.count("confirmed answer") == 1
        assert app.state.assistant_buffer == "partial answer!"


@pytest.mark.anyio
async def test_mounted_tui_restores_failure_waiting_and_inbox_facts() -> None:
    snapshot = _snapshot()
    live = snapshot["documents"][0]["value"]  # type: ignore[index]
    assert isinstance(live, dict)
    live.update(
        {
            "error": {"type": "error", "message": "committed failure"},
            "retry": {"type": "retry", "message": "retrying"},
            "question": {"type": "human_input_requested", "questions": []},
        }
    )
    snapshot["documents"].append(  # type: ignore[union-attr]
        {
            "record": {"id": 2, "kind": "forge.inbox.v1"},
            "version": 1,
            "value": {
                "items": [
                    {"id": 1, "mode": "steer", "content": "steer me"},
                    {"id": 2, "mode": "follow_up", "content": "follow me"},
                ]
            },
        }
    )
    app = ForgeTuiApp(SnapshotSession(snapshot))  # type: ignore[arg-type]

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.state.error == "committed failure"
        assert app.state.queued_steering == ("steer me",)
        assert app.state.queued_follow_up == ("follow me",)
        visible = "\n".join(item.text for item in app.state.items)
        assert "Error: committed failure" in visible
        assert "Waiting for user input" in visible
        assert "retrying" in visible
        assert "questionnaire" not in type(app.screen).__name__.lower()


@pytest.mark.anyio
async def test_mounted_tui_uses_real_session_committed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_home(monkeypatch, tmp_path)
    storage = JsonlSessionStorage(tmp_path / "session")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=StreamingScriptedChatModel([AIMessage(content="real committed answer")]),
            model="fake",
            system="offline",
            storage=storage,
            cwd=tmp_path,
            enable_subagents=False,
        )
    )
    try:
        _ = [event async for event in session.prompt("real committed prompt")]
        app = ForgeTuiApp(session)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            visible = "\n".join(item.text for item in app.state.items)
            assert "real committed prompt" in visible
            assert "real committed answer" in visible
            assert app.state.snapshot_mode is True
            assert app.state.snapshot_sequence == session.snapshot()["sequence"]
    finally:
        await session.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["partial", "background"])
async def test_mounted_tui_reopens_last_confirmed_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    """A real process exit must reopen the exact confirmed live projection."""
    isolate_home(monkeypatch, tmp_path)
    confirmed = _run_progress_driver(tmp_path, action)
    confirmed_live = _live_value(confirmed)
    # Mount the first reopened session; closing an intermediate session is a
    # new committed lifecycle transition and would change the reference state.
    reopened_live = confirmed_live

    model = StreamingScriptedChatModel([AIMessage(content="unused")])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            storage=JsonlSessionStorage(tmp_path / "session"),
            cwd=tmp_path,
            enable_subagents=action == "background",
            interactive=action == "background",
        )
    )
    try:
        app = ForgeTuiApp(session)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.state.snapshot_sequence == session.snapshot().get("sequence")
            session_live = _live_value(session.snapshot())
            assert session_live.get("assistant") == confirmed_live.get("assistant")
            assert session_live.get("thinking") == confirmed_live.get("thinking")
            assert app.state.live_activity == reopened_live
            assert model.calls == []
            if action == "partial":
                partial = reopened_live.get("assistant")
                assert isinstance(partial, str) and partial
                assert app.state.assistant_buffer == partial
                transcript = app.query_one("#transcript")
                assert any(partial in line.text for line in transcript.lines)
                assert "不会显示" not in "\n".join(line.text for line in transcript.lines)
            else:
                background = reopened_live.get("background")
                assert isinstance(background, list) and background
                output = background[0].get("output")
                assert isinstance(output, str) and output
                assert any(item.get("output") == output for item in app._fleet_snapshots)
    finally:
        await session.aclose()


def test_renderers_accept_committed_snapshot_frames(capsys: pytest.CaptureFixture[str]) -> None:
    event = CommittedSnapshotEvent(snapshot=_snapshot())

    plain = FinalTextRenderer()
    plain.render(event)
    assert plain.finish() is True
    assert "partial answer" in capsys.readouterr().out

    transcript = TranscriptRenderer()
    transcript.render(event)
    assert transcript.finish() is True
    assert "partial answer" in capsys.readouterr().out

    json_renderer = JsonEventRenderer()
    json_renderer.render(event)
    output = capsys.readouterr().out
    assert '"type":"committed_snapshot"' in output
    assert "confirmed answer" in output


def test_print_renderers_fail_on_snapshot_error_and_hide_aborted_tail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    error_snapshot = _snapshot(assistant="must not print")
    error_value = error_snapshot["documents"][0]["value"]  # type: ignore[index]
    assert isinstance(error_value, dict)
    error_value.update({"error": {"type": "error", "message": "committed failure"}})
    plain = FinalTextRenderer()
    plain.render(CommittedSnapshotEvent(snapshot=error_snapshot))
    assert plain.finish() is False
    captured = capsys.readouterr()
    assert "committed failure" in captured.err
    assert "must not print" not in captured.out

    transcript = TranscriptRenderer()
    transcript.render(CommittedSnapshotEvent(snapshot=error_snapshot))
    assert transcript.finish() is False
    captured = capsys.readouterr()
    assert "committed failure" in captured.err
    assert "must not print" not in captured.out

    aborted_snapshot = _snapshot(assistant="aborted partial")
    aborted_value = aborted_snapshot["documents"][0]["value"]  # type: ignore[index]
    assert isinstance(aborted_value, dict)
    aborted_value["aborted"] = True
    plain = FinalTextRenderer()
    plain.render(CommittedSnapshotEvent(snapshot=aborted_snapshot))
    assert plain.finish() is True
    captured = capsys.readouterr()
    assert "aborted partial" not in captured.out

    transcript = TranscriptRenderer()
    transcript.render(CommittedSnapshotEvent(snapshot=aborted_snapshot))
    assert transcript.finish() is True
    captured = capsys.readouterr()
    assert "aborted partial" not in captured.out


def test_print_snapshot_respects_selected_branch(capsys: pytest.CaptureFixture[str]) -> None:
    snapshot = _snapshot(assistant="")
    snapshot["entries"].extend(  # type: ignore[union-attr]
        [
            _entry("other", AIMessage(content="other branch"), parent_id="u1"),
            {"type": "leaf", "id": "leaf", "parent_id": "a1", "timestamp": 2.0, "entry_id": "a1"},
        ]
    )
    for renderer in (FinalTextRenderer(), TranscriptRenderer()):
        renderer.render(CommittedSnapshotEvent(snapshot=snapshot))
        assert renderer.finish() is True
        output = capsys.readouterr().out
        assert "confirmed answer" in output
        assert "other branch" not in output


@pytest.mark.anyio
async def test_mounted_slow_consumer_does_not_replay_snapshot_commit(
    tmp_path: Path,
) -> None:
    """Overflow absorbs the complete commit, including deltas arriving later."""
    import os

    from forge_agent import MessageDeltaEvent
    from forge_coding.features.observation import CommitEventQueue

    snapshot = _snapshot(assistant="AB", sequence=101)
    queue = CommitEventQueue(lambda: snapshot)  # type: ignore[arg-type]
    for sequence in range(1, 101):
        queue.put_nowait((7, MessageDeltaEvent(delta="x"), sequence))
    queue.put_nowait((7, MessageDeltaEvent(delta="A"), 101))
    queue.put_nowait((7, MessageDeltaEvent(delta="B"), 101))
    app = ForgeTuiApp(SnapshotSession(_snapshot(assistant="", sequence=0)))  # type: ignore[arg-type]
    events: list[str] = []
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        while not queue.empty():
            _, event = queue.get_nowait()
            assert event is not None and not isinstance(event, BaseException)
            events.append(event.type)
            app.adapter.apply(event)
        await pilot.pause()
        assert app.state.assistant_buffer == "AB"
        # A late delta from the replaced commit stays suppressed after dequeue.
        queue.put_nowait((7, MessageDeltaEvent(delta="B"), 101))
        queue.put_nowait((7, MessageDeltaEvent(delta="C"), 102))
        while not queue.empty():
            _, event = queue.get_nowait()
            assert event is not None and not isinstance(event, BaseException)
            events.append(event.type)
            app.adapter.apply(event)
        await pilot.pause()
        assert app.state.assistant_buffer == "ABC"
    artifact = Path(os.environ.get("FORGE_D3_ARTIFACT_DIR", str(tmp_path))) / "overflow-mounted"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "result.json").write_text(
        json.dumps({"events": events, "visible": "ABC", "expected": "ABC"}), encoding="utf-8"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["abort", "close"])
async def test_question_teardown_reopens_with_committed_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    from fake_models import ScriptedChatModel, tool_call_ai

    isolate_home(monkeypatch, tmp_path)
    model = ScriptedChatModel(
        [
            tool_call_ai(
                "question",
                "ask_user_question",
                {
                    "questions": [
                        {
                            "header": "Mode",
                            "question": "Choose",
                            "multi_select": False,
                            "options": [
                                {"label": "Fast", "description": "Speed"},
                                {"label": "Careful", "description": "Review"},
                            ],
                        }
                    ]
                },
            )
        ]
    )
    path = tmp_path / "session"
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=model,
            model="fake",
            system="offline",
            cwd=tmp_path,
            storage=JsonlSessionStorage(path),
            interactive=True,
            enable_subagents=False,
        )
    )
    try:
        _ = [event async for event in session.prompt("ask")]
        assert session.is_waiting_for_input
        if action == "abort":
            await session.abort()
            assert session.live_activity["question"] is None
        await session.aclose()
        unused = ScriptedChatModel([AIMessage(content="answered")] if action == "close" else [])
        reopened = await CodingSession.load(
            CodingSessionConfig(
                provider=unused,
                model="fake",
                system="offline",
                cwd=tmp_path,
                storage=JsonlSessionStorage(path),
                interactive=True,
                enable_subagents=False,
            )
        )
        try:
            assert unused.calls == []
            if action == "close":
                assert reopened.live_activity["question"] is not None
                assert reopened.live_activity["aborted"] is False
                _ = [event async for event in reopened.respond_to_human_input("Fast")]
                assert len(unused.calls) == 1
            else:
                assert reopened.live_activity["question"] is None
                assert reopened.live_activity["aborted"] is True
            assert not reopened.is_waiting_for_input
            app = ForgeTuiApp(reopened)
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                assert not any("Waiting for user input" in item.text for item in app.state.items)
            calls = [
                call["id"]
                for message in reopened.messages
                if isinstance(message, AIMessage)
                for call in message.tool_calls
            ]
            from langchain_core.messages import ToolMessage

            results = [
                message.tool_call_id
                for message in reopened.messages
                if isinstance(message, ToolMessage)
            ]
            assert calls == results == ["question"]
            assert len(unused.calls) == (1 if action == "close" else 0)
        finally:
            await reopened.aclose()
    finally:
        await session.aclose()
