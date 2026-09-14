import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from textual.app import App, ComposeResult
from textual.widgets import Label, TextArea

from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent import (
    AgentHarness,
    AgentHarnessConfig,
    HumanInputRequestedEvent,
    TodoItem,
    TodoUpdateEvent,
)
from forge_agent.session import CustomEntry, JsonlSessionStorage
from forge_agent.tool_execution import (
    SEQUENTIAL_TOOL_EXECUTION_MODE,
    TOOL_EXECUTION_MODE_METADATA_KEY,
)
from forge_cli.tui import TuiState
from forge_cli.tui.questionnaire import (
    AskUserQuestionScreen,
    QuestionnaireDrafts,
    QuestionnaireResult,
)
from forge_cli.tui.todos import render_todos, visible_todos
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.features.human_input import (
    AskOption,
    AskQuestion,
    AskUserQuestionInput,
    create_ask_user_question_tool,
    create_human_input_middleware,
    serialize_answers,
)
from forge_coding.features.planning import (
    TODO_NAMESPACE,
    create_todo_middleware,
    latest_todo_snapshot,
)


def _question(
    *,
    multi_select: bool = False,
    header: str = "Mode",
    question: str = "Which mode should Forge use?",
) -> AskQuestion:
    return AskQuestion(
        header=header,
        question=question,
        options=(
            AskOption(label="Fast", description="Prioritize speed."),
            AskOption(label="Careful", description="Prioritize review."),
        ),
        multi_select=multi_select,
    )


def test_ask_schema_rejects_invalid_bounds_and_reserved_labels() -> None:
    with pytest.raises(ValueError):
        AskUserQuestionInput(questions=())
    with pytest.raises(ValueError):
        AskOption(label="Other", description="Reserved")
    with pytest.raises(ValueError):
        AskUserQuestionInput(
            questions=(
                _question(),
                _question(),
            )
        )


def test_serialize_answers_is_stable_and_supports_custom_multi_and_cancelled() -> None:
    questions = (
        _question(),
        _question(
            multi_select=True,
            header="Options",
            question="Which options should Forge use?",
        ),
    )
    payload = serialize_answers(
        questions,
        (
            {"answer": "my custom answer", "notes": "because"},
            {"selected": ["Fast", "Careful"]},
        ),
    )
    rows = json.loads(payload)
    assert [row["kind"] for row in rows] == ["custom", "multi"]
    assert rows[0]["notes"] == "because"
    assert rows[1]["selected"] == ["Fast", "Careful"]
    cancelled = json.loads(serialize_answers(questions, cancelled=True))
    assert all(row["cancelled"] for row in cancelled)


def test_questionnaire_drafts_keep_answers_indexed_and_do_not_select_cursor() -> None:
    questions = (
        _question(),
        _question(
            multi_select=True,
            header="Options",
            question="Which options should Forge use?",
        ),
    )
    drafts = QuestionnaireDrafts(questions)

    drafts.set_text(0, "custom answer")
    drafts.toggle(1, "Careful")

    assert drafts.answer(0) == {
        "answer": "custom answer",
        "selected": [],
        "notes": "custom answer",
    }
    assert drafts.answer(1) == {
        "answer": ["Careful"],
        "selected": ["Careful"],
        "notes": "",
    }


def test_questionnaire_drafts_preserve_text_when_navigating() -> None:
    drafts = QuestionnaireDrafts((_question(), _question(header="Other", question="Other?")))
    drafts.set_text(0, "first draft")
    drafts.set_text(1, "second draft")

    assert drafts.text(0) == "first draft"
    assert drafts.text(1) == "second draft"


class _QuestionnaireApp(App[None]):
    def __init__(self, questions: tuple[AskQuestion, ...]) -> None:
        super().__init__()
        self.result: QuestionnaireResult | None = None
        self._event = HumanInputRequestedEvent(
            interrupt_id="interrupt",
            tool_call_id="ask",
            tool_name="ask_user_question",
            arguments={"questions": [question.model_dump(mode="json") for question in questions]},
        )

    def compose(self) -> ComposeResult:
        yield Label("transcript")

    def on_mount(self) -> None:
        self.push_screen(AskUserQuestionScreen(self._event), callback=self._save_result)

    def _save_result(self, result: QuestionnaireResult | None) -> None:
        self.result = result


@pytest.mark.anyio
async def test_questionnaire_multiselect_submits_only_checked_options() -> None:
    app = _QuestionnaireApp((_question(multi_select=True),))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("down", "space", "enter")

    assert app.result is not None
    rows = json.loads(app.result.message)
    assert rows[0]["selected"] == ["Careful"]


@pytest.mark.anyio
async def test_questionnaire_tab_preserves_indexed_custom_drafts() -> None:
    app = _QuestionnaireApp((_question(), _question(header="Other", question="Other question?")))
    async with app.run_test(size=(100, 30)) as pilot:
        screen = app.screen
        notes = screen.query_one("#ask-user-question-notes", TextArea)
        notes.text = "first custom"
        await pilot.press("tab")
        notes.text = "second custom"
        await pilot.press("shift+tab")
        assert notes.text == "first custom"
        await pilot.press("tab", "enter")

    assert app.result is not None
    rows = json.loads(app.result.message)
    assert [row["answer"] for row in rows] == ["first custom", "second custom"]


@pytest.mark.anyio
async def test_questionnaire_renders_model_text_as_literals() -> None:
    question = AskQuestion(
        header="[b]Mode[/b]",
        question="[link=https://example.com]Choose[/link]",
        options=(
            AskOption(label="[red]Fast[/red]", description="[dim]Speed[/dim]"),
            AskOption(label="Careful", description="Review"),
        ),
    )
    app = _QuestionnaireApp((question,))
    async with app.run_test(size=(100, 30)):
        body = app.screen.query_one("#ask-user-question-body", Label)
        assert body.render().plain == "[link=https://example.com]Choose[/link]"


def test_todo_replay_ignores_malformed_entries_and_keeps_last_valid_snapshot() -> None:
    entries = [
        CustomEntry(
            namespace=TODO_NAMESPACE,
            data={"todos": [{"content": "first", "status": "pending"}]},
        ),
        CustomEntry(namespace=TODO_NAMESPACE, data={"todos": [{"content": "bad"}]}),
        CustomEntry(
            namespace=TODO_NAMESPACE,
            data={"todos": [{"content": "done", "status": "completed"}]},
        ),
    ]
    assert latest_todo_snapshot(entries) == (TodoItem(content="done", status="completed"),)


def test_todo_panel_applies_line_budget_and_delayed_completed_hide() -> None:
    todos = tuple(
        TodoItem(content=f"task-{index}", status="completed" if index == 0 else "pending")
        for index in range(15)
    )
    visible, omitted = visible_todos(todos, max_lines=4)
    assert [item.content for item in visible] == ["task-1", "task-2"]
    assert omitted == 13
    assert "Ctrl+Shift+T" in render_todos(todos, collapsed=True).plain

    state = TuiState()
    state.update_todos((TodoItem(content="done", status="completed"),))
    assert state.todos
    state.add_user_message("next turn")
    assert state.todos == ()


@pytest.mark.anyio
async def test_todo_middleware_projects_updates_without_changing_tool_loop() -> None:
    todo_middleware = create_todo_middleware()
    assert todo_middleware.tools[0].metadata == {
        TOOL_EXECUTION_MODE_METADATA_KEY: SEQUENTIAL_TOOL_EXECUTION_MODE
    }
    model = ScriptedChatModel(
        responses=[
            tool_call_ai(
                "todo-call",
                "write_todos",
                {"todos": [{"content": "Inspect", "status": "in_progress"}]},
            ),
            AIMessage(content="done"),
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=model,
            model="fake",
            system="You are Forge.",
            middleware=(todo_middleware,),
        )
    )
    events = [event async for event in harness.prompt("plan this")]
    todo_events = [event for event in events if isinstance(event, TodoUpdateEvent)]
    assert todo_events and todo_events[-1].todos[0].content == "Inspect"
    assert any(isinstance(message, ToolMessage) for message in harness.messages)


@pytest.mark.anyio
async def test_human_input_interrupt_resumes_with_paired_tool_message() -> None:
    ask_tool = create_ask_user_question_tool()
    assert ask_tool.metadata == {TOOL_EXECUTION_MODE_METADATA_KEY: SEQUENTIAL_TOOL_EXECUTION_MODE}
    model = ScriptedChatModel(
        responses=[
            tool_call_ai(
                "ask-call",
                "ask_user_question",
                {"questions": [_question().model_dump(mode="json")]},
            ),
            AIMessage(content="continued"),
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=model,
            model="fake",
            system="You are Forge.",
            tools=(ask_tool,),
            middleware=(create_human_input_middleware(),),
            interactive=True,
        )
    )
    first_events = [event async for event in harness.prompt("ask me")]
    assert any(event.type == "human_input_requested" for event in first_events)
    assert harness.is_waiting_for_input is True
    assert not any(isinstance(message, ToolMessage) for message in harness.messages)

    resumed = [event async for event in harness.respond_to_human_input("[answer]")]
    assert harness.is_waiting_for_input is False
    assert any(event.type == "message_end" for event in resumed)
    tool_messages = [message for message in harness.messages if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "[answer]"


@pytest.mark.anyio
async def test_human_input_preserves_multiple_decision_order() -> None:
    first = _question(header="First", question="First question?").model_dump(mode="json")
    second = _question(header="Second", question="Second question?").model_dump(mode="json")
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "ask-1",
                        "name": "ask_user_question",
                        "args": {"questions": [first]},
                        "type": "tool_call",
                    },
                    {
                        "id": "ask-2",
                        "name": "ask_user_question",
                        "args": {"questions": [second]},
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="continued"),
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=model,
            model="fake",
            system="You are Forge.",
            tools=(create_ask_user_question_tool(),),
            middleware=(create_human_input_middleware(),),
            interactive=True,
        )
    )
    _first_events = [event async for event in harness.prompt("ask twice")]
    requests = harness.pending_human_input[0].requests
    assert [request.tool_call_id for request in requests] == ["ask-1", "ask-2"]

    _resumed_events = [
        event
        async for event in harness.respond_to_human_input(
            [
                {"type": "respond", "message": "first answer"},
                {"type": "respond", "message": "second answer"},
            ]
        )
    ]
    tool_messages = [message for message in harness.messages if isinstance(message, ToolMessage)]
    assert [(message.tool_call_id, message.content) for message in tool_messages] == [
        ("ask-1", "first answer"),
        ("ask-2", "second answer"),
    ]


@pytest.mark.anyio
async def test_session_persists_todo_snapshot_after_tool_messages(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(
                responses=[
                    tool_call_ai(
                        "todo-call",
                        "write_todos",
                        {"todos": [{"content": "Ship", "status": "pending"}]},
                    ),
                    AIMessage(content="ready"),
                ]
            ),
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            interactive=False,
        )
    )
    _events = [event async for event in session.prompt("plan")]
    assert session.todos == (TodoItem(content="Ship", status="pending"),)
    entries = await storage.read_all()
    assert any(entry.type == "custom" and entry.namespace == TODO_NAMESPACE for entry in entries)


@pytest.mark.anyio
async def test_noninteractive_session_does_not_register_ask_tool(tmp_path: Path) -> None:
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="fake",
            system="You are Forge.",
            storage=JsonlSessionStorage(tmp_path / "print.jsonl"),
            cwd=tmp_path,
            interactive=False,
            enable_subagents=False,
        )
    )
    assert "ask_user_question" not in {tool.name for tool in session.tools}
