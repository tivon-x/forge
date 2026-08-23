"""Pi-style ask-user questionnaire screen used by the Forge TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static, TextArea

from forge_agent import HumanInputRequest, HumanInputRequestedEvent
from forge_coding.features.human_input import AskQuestion, AskUserQuestionInput, serialize_answers

_CUSTOM_OPTION = "Type something."


@dataclass(frozen=True, slots=True)
class QuestionnaireResult:
    """Result returned by :class:`AskUserQuestionScreen`."""

    message: str
    cancelled: bool = False


class QuestionnaireDrafts:
    """Per-question draft state, independent from the visible list cursor."""

    def __init__(self, questions: tuple[AskQuestion, ...]) -> None:
        self._questions = questions
        self._selected = [set[str]() for _ in questions]
        self._text = ["" for _ in questions]

    def selected(self, index: int) -> frozenset[str]:
        return frozenset(self._selected[index])

    def toggle(self, index: int, label: str) -> None:
        question = self._questions[index]
        selected = self._selected[index]
        if question.multi_select:
            if label in selected:
                selected.remove(label)
            else:
                selected.add(label)
        else:
            self._selected[index] = {label}
            self._text[index] = ""

    def set_text(self, index: int, value: str) -> None:
        self._text[index] = value
        if value.strip() and not self._questions[index].multi_select:
            self._selected[index].clear()

    def text(self, index: int) -> str:
        return self._text[index]

    def answer(self, index: int) -> dict[str, Any]:
        question = self._questions[index]
        selected = sorted(self._selected[index])
        text = self._text[index].strip()
        answer: object = selected if question.multi_select else selected[0] if selected else text
        return {"answer": answer, "selected": selected, "notes": text}

    def answers(self) -> list[dict[str, Any]]:
        return [self.answer(index) for index in range(len(self._questions))]


class AskUserQuestionScreen(ModalScreen[QuestionnaireResult | None]):
    """Small modal questionnaire that keeps the transcript visible behind it."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("space", "toggle_option", "Toggle", show=False),
        Binding("enter", "submit", "Submit", priority=True),
        Binding("n", "focus_notes", "Notes", show=False),
        Binding("ctrl+]", "toggle_collapse", "Collapse", show=False),
    ]

    def __init__(
        self,
        event: HumanInputRequestedEvent,
        *,
        theme: object | None = None,
    ) -> None:
        super().__init__()
        self.event = event
        self.theme = theme
        self._index = 0
        self._requests = event.requests or (
            HumanInputRequest(
                interrupt_id=event.interrupt_id,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                arguments=event.arguments,
                allowed_decisions=event.allowed_decisions,
                description=event.description,
            ),
        )
        self._request_index = 0
        self._request_messages: list[str] = []
        self._collapsed = False
        self.questions = self._questions_for_request(self._requests[0])
        self._drafts = QuestionnaireDrafts(self.questions)

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-user-question"):
            yield Static("Forge needs your answer", id="ask-user-question-title")
            yield Static("", id="ask-user-question-progress")
            yield Label("", id="ask-user-question-body", markup=False)
            yield ListView(id="ask-user-question-options")
            yield Static("", id="ask-user-question-preview", markup=False)
            yield TextArea("", id="ask-user-question-notes")
            yield Static(
                "↑/↓ choose · Enter next · Space multi-select · n notes · Esc cancel",
                id="ask-user-question-help",
            )

    def on_mount(self) -> None:
        self._render_question()
        self.query_one("#ask-user-question-options", ListView).focus()

    def on_key(self, event: Key) -> None:
        if event.key == "tab":
            event.stop()
            self._save_text()
            self._index = (self._index + 1) % max(len(self.questions), 1)
            self._render_question()
        elif event.key in {"shift+tab", "backtab"}:
            event.stop()
            self._save_text()
            self._index = (self._index - 1) % max(len(self.questions), 1)
            self._render_question()

    def action_cursor_up(self) -> None:
        self.query_one("#ask-user-question-options", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#ask-user-question-options", ListView).action_cursor_down()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Show the highlighted option preview without interpreting markup."""

        if not self.questions:
            return
        index = event.list_view.index
        question = self.questions[self._index]
        preview = ""
        if index is not None and 0 <= index < len(question.options):
            preview = question.options[index].preview or ""
        self.query_one("#ask-user-question-preview", Static).update(Text(preview))

    def action_toggle_option(self) -> None:
        if not self.questions:
            return
        selected = self.query_one("#ask-user-question-options", ListView).index
        if selected is None or selected < 0:
            return
        question = self.questions[self._index]
        if selected == len(question.options):
            self.query_one("#ask-user-question-notes", TextArea).focus()
            return
        if selected > len(question.options):
            return
        label = question.options[selected].label
        self._drafts.toggle(self._index, label)
        self._render_question()

    def action_submit(self) -> None:
        if not self.questions:
            cancelled = [
                serialize_answers(self._questions_for_request(request), (), cancelled=True)
                for request in self._requests
            ]
            self.dismiss(
                QuestionnaireResult(message=self._respond_message(cancelled), cancelled=True)
            )
            return
        list_view = self.query_one("#ask-user-question-options", ListView)
        question = self.questions[self._index]
        selected_index = list_view.index
        self._save_text()
        if (
            not question.multi_select
            and not self._drafts.selected(self._index)
            and not self._drafts.text(self._index).strip()
            and selected_index is not None
        ):
            if selected_index == len(question.options):
                self.query_one("#ask-user-question-notes", TextArea).focus()
                return
            if 0 <= selected_index < len(question.options):
                self._drafts.toggle(self._index, question.options[selected_index].label)
        if self._index + 1 < len(self.questions):
            self._index += 1
            self._render_question()
            return
        self._request_messages.append(serialize_answers(self.questions, self._drafts.answers()))
        if self._request_index + 1 < len(self._requests):
            self._request_index += 1
            self._index = 0
            self.questions = self._questions_for_request(self._requests[self._request_index])
            self._drafts = QuestionnaireDrafts(self.questions)
            self._render_question()
            return
        self.dismiss(QuestionnaireResult(message=self._respond_message(), cancelled=False))

    def action_focus_notes(self) -> None:
        self.query_one("#ask-user-question-notes", TextArea).focus()

    def action_toggle_collapse(self) -> None:
        """Collapse the questionnaire body while preserving draft selections."""

        self._collapsed = not self._collapsed
        for widget_id in (
            "#ask-user-question-progress",
            "#ask-user-question-body",
            "#ask-user-question-options",
            "#ask-user-question-preview",
            "#ask-user-question-notes",
            "#ask-user-question-help",
        ):
            self.query_one(widget_id).display = not self._collapsed

    def action_cancel(self) -> None:
        cancelled = [
            serialize_answers(self._questions_for_request(request), (), cancelled=True)
            for request in self._requests
        ]
        self.dismiss(
            QuestionnaireResult(
                message=self._respond_message(cancelled),
                cancelled=True,
            )
        )

    def _questions_for_request(self, request: HumanInputRequest) -> tuple[Any, ...]:
        try:
            return AskUserQuestionInput.model_validate(request.arguments).questions
        except ValueError:
            return ()

    def _respond_message(self, messages: list[str] | None = None) -> str:
        values = list(self._request_messages if messages is None else messages)
        if len(values) == 1:
            return values[0]
        return json.dumps(
            {"decisions": [{"type": "respond", "message": message} for message in values]},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _render_question(self) -> None:
        if not self.questions:
            self.query_one("#ask-user-question-body", Label).update("Question payload is invalid.")
            return
        question = self.questions[self._index]
        self.query_one("#ask-user-question-progress", Static).update(
            Text(f"{self._index + 1}/{len(self.questions)} · {question.header}")
        )
        self.query_one("#ask-user-question-body", Label).update(Text(question.question))
        options = self.query_one("#ask-user-question-options", ListView)
        options.clear()
        selected = self._drafts.selected(self._index)
        for option in question.options:
            marker = "☑" if option.label in selected else "☐" if question.multi_select else "○"
            options.append(
                ListItem(
                    Label(Text(f"{marker} {option.label} — {option.description}"), markup=False)
                )
            )
        options.append(ListItem(Label(Text(f"✎ {_CUSTOM_OPTION}"), markup=False)))
        options.index = 0
        notes = self.query_one("#ask-user-question-notes", TextArea)
        notes.text = self._drafts.text(self._index)

    def _save_text(self) -> None:
        if self.questions:
            self._drafts.set_text(
                self._index,
                self.query_one("#ask-user-question-notes", TextArea).text,
            )


__all__ = ["AskUserQuestionScreen", "QuestionnaireDrafts", "QuestionnaireResult"]
