"""Structured ask-user input and the official LangChain HITL adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from forge_agent.tool_execution import (
    SEQUENTIAL_TOOL_EXECUTION_MODE,
    TOOL_EXECUTION_MODE_METADATA_KEY,
)
from forge_agent.types import JSONValue, stripped_text

_RESERVED_OPTION_LABELS = frozenset({"other", "type something.", "next"})
AnswerKind = Literal["option", "custom", "multi"]


class AskOption(BaseModel):
    """One selectable option shown by the TUI questionnaire."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, max_length=60)
    description: str = Field(min_length=1, max_length=500)
    preview: str | None = Field(default=None, max_length=8_000)

    @field_validator("label", "description")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        return stripped_text(value)

    @field_validator("label")
    @classmethod
    def _reject_reserved_label(cls, value: str) -> str:
        if value.casefold() in _RESERVED_OPTION_LABELS:
            raise ValueError(f"reserved option label: {value}")
        return value


class AskQuestion(BaseModel):
    """A single structured question accepted by ``ask_user_question``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    header: str = Field(min_length=1, max_length=16)
    question: str = Field(min_length=1, max_length=2_000)
    options: tuple[AskOption, ...] = Field(min_length=2, max_length=4)
    multi_select: bool = False

    @field_validator("header", "question")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        return stripped_text(value)

    @model_validator(mode="after")
    def _unique_options(self) -> AskQuestion:
        labels = [option.label.casefold() for option in self.options]
        if len(labels) != len(set(labels)):
            raise ValueError("option labels must be unique")
        return self


class AskUserQuestionInput(BaseModel):
    """Tool input: one to four questions in a single model turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: tuple[AskQuestion, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _unique_questions(self) -> AskUserQuestionInput:
        headers = [question.header.casefold() for question in self.questions]
        prompts = [question.question.casefold() for question in self.questions]
        if len(headers) != len(set(headers)):
            raise ValueError("question headers must be unique")
        if len(prompts) != len(set(prompts)):
            raise ValueError("questions must be unique")
        return self


def validate_questions(value: object) -> AskUserQuestionInput:
    """Validate an ask-user payload and normalize it to immutable models."""

    return AskUserQuestionInput.model_validate(value)


def _ask_user_question(questions: list[dict[str, Any]]) -> str:
    del questions
    raise ToolException("ask_user_question must be handled by Forge's human-input UI")


async def _aask_user_question(questions: list[dict[str, Any]]) -> str:
    return _ask_user_question(questions)


def create_ask_user_question_tool() -> StructuredTool:
    """Create the placeholder tool intercepted by HITL middleware."""

    return StructuredTool.from_function(
        name="ask_user_question",
        description=(
            "Ask the user one to four structured questions when a missing answer "
            "is required to continue. Combine related questions into one call and "
            "use this tool alone in a model turn."
        ),
        func=_ask_user_question,
        coroutine=_aask_user_question,
        args_schema=AskUserQuestionInput,
        infer_schema=False,
        metadata={TOOL_EXECUTION_MODE_METADATA_KEY: SEQUENTIAL_TOOL_EXECUTION_MODE},
    )


def create_human_input_middleware() -> HumanInTheLoopMiddleware:
    """Create the ask-user-only HITL policy.

    ``respond`` is intentional: answering a question is not approval or
    rejection of a side effect, and the placeholder tool must never execute.
    """

    return HumanInTheLoopMiddleware(
        {"ask_user_question": {"allowed_decisions": ["respond"]}},
        description_prefix="Forge needs your answer",
    )


def serialize_answers(
    questions: Sequence[AskQuestion | Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any] | str | Sequence[str] | None] = (),
    *,
    cancelled: bool = False,
) -> str:
    """Serialize questionnaire answers as stable JSON for a ToolMessage."""

    if not questions:
        return "[]"
    normalized_questions = validate_questions({"questions": list(questions)}).questions
    rows: list[dict[str, JSONValue]] = []
    for index, question in enumerate(normalized_questions):
        raw = answers[index] if index < len(answers) else None
        selected: list[str] = []
        notes: str | None = None
        answer: str | list[str] = ""
        if isinstance(raw, Mapping):
            raw_selected = raw.get("selected")
            if isinstance(raw_selected, Sequence) and not isinstance(raw_selected, (str, bytes)):
                selected = [str(item) for item in raw_selected]
            raw_answer = raw.get("answer")
            if isinstance(raw_answer, Sequence) and not isinstance(raw_answer, (str, bytes)):
                answer = [str(item) for item in raw_answer]
            elif raw_answer is not None:
                answer = str(raw_answer)
            raw_notes = raw.get("notes")
            if raw_notes is not None:
                notes = str(raw_notes)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            selected = [str(item) for item in raw]
            answer = selected
        elif raw is not None:
            answer = str(raw)

        if question.multi_select:
            if not selected and isinstance(answer, list):
                selected = list(answer)
            kind: AnswerKind = "multi"
            if not isinstance(answer, list):
                answer = selected
        elif (
            isinstance(answer, str)
            and answer
            and answer not in {option.label for option in question.options}
        ):
            kind = "custom"
        else:
            kind = "option"

        row: dict[str, JSONValue] = {
            "question_index": index,
            "question": question.question,
            "kind": kind,
            "answer": cast(JSONValue, answer),
            "selected": cast(JSONValue, selected),
            "notes": notes,
            "cancelled": cancelled,
        }
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_respond_decision(message: str) -> dict[str, JSONValue]:
    """Build the exact decision shape consumed by ``respond`` HITL."""

    return {"type": "respond", "message": message}


__all__ = [
    "AskOption",
    "AskQuestion",
    "AskUserQuestionInput",
    "AnswerKind",
    "build_respond_decision",
    "create_ask_user_question_tool",
    "create_human_input_middleware",
    "serialize_answers",
    "validate_questions",
]
