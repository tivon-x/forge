"""Display state for Forge's Textual TUI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from forge_agent.message_codec import message_text
from forge_agent.tools import AgentToolResult, ToolCall
from forge_cli.formatting import (
    _string_argument,
    format_tool_call_block,
    format_tool_result_block,
    format_tool_result_summary,
)
from forge_coding.skills import Skill, parse_skill_invocation

ChatItemRole = Literal[
    "user",
    "assistant",
    "tool",
    "subagent",
    "error",
    "status",
    "thinking",
    "skill",
    "branch_summary",
    "compaction_summary",
]

SubagentStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


@dataclass(slots=True)
class SubagentDisplay:
    """Durable display facts for one inline ``task`` tool call.

    Activity is deliberately a single presentation string.  Nested child
    messages are not part of the parent transcript and are never retained here.
    The final artifact carries the stable counters and output used by history
    restore.
    """

    tool_call_id: str
    agent: str
    instruction: str
    status: SubagentStatus = "queued"
    activity: str = ""
    tool_calls: int = 0
    queued_ms: int = 0
    duration_ms: int = 0
    final_output: str | None = None
    truncated: bool = False
    error: str | None = None


@dataclass(slots=True)
class ChatItem:
    """One rendered item in the TUI transcript."""

    role: ChatItemRole
    text: str
    tool_call_id: str | None = None
    tool_result_text: str | None = None
    always_show_tool_result: bool = False
    subagent: SubagentDisplay | None = None


@dataclass(slots=True)
class TuiState:
    """Mutable display state for the interactive TUI."""

    items: list[ChatItem] = field(default_factory=list)
    assistant_buffer: str = ""
    running: bool = False
    error: str | None = None
    show_tool_results: bool = False
    show_thinking: bool = False
    queued_steering: tuple[str, ...] = ()
    queued_follow_up: tuple[str, ...] = ()
    skills: tuple[Skill, ...] = ()

    def add_item(
        self,
        role: ChatItemRole,
        text: str,
        *,
        tool_call_id: str | None = None,
        tool_result_text: str | None = None,
        always_show_tool_result: bool = False,
        subagent: SubagentDisplay | None = None,
    ) -> None:
        """Append a transcript item."""
        self.items.append(
            ChatItem(
                role=role,
                text=text,
                tool_call_id=tool_call_id,
                tool_result_text=tool_result_text,
                always_show_tool_result=always_show_tool_result,
                subagent=subagent,
            )
        )

    def add_subagent_task(self, tool_call: ToolCall) -> bool:
        """Append a task as an inline subagent block when its arguments are valid.

        ``task`` is the only tool with this specialized display.  A malformed
        call is intentionally left as a normal tool row so older/third-party
        task payloads remain readable.
        """
        if tool_call.name != "task":
            self._add_plain_tool_call(tool_call)
            return False
        arguments = tool_call.arguments
        agent = _string_argument(arguments, "agent")
        instruction = _string_argument(arguments, "instruction")
        if agent is None or instruction is None:
            self._add_plain_tool_call(tool_call)
            return False
        display = SubagentDisplay(
            tool_call_id=tool_call.id,
            agent=agent,
            instruction=instruction,
        )
        self.add_item(
            "subagent",
            format_tool_call_block(tool_call),
            tool_call_id=tool_call.id,
            subagent=display,
        )
        return True

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Append a collapsed tool-call item."""
        if tool_call.name == "task":
            self.add_subagent_task(tool_call)
            return
        self._add_plain_tool_call(tool_call)

    def _add_plain_tool_call(self, tool_call: ToolCall) -> None:
        """Append a normal tool row without task specialization."""
        skill_name = self._read_skill_name(tool_call)
        if skill_name is not None:
            self.add_item(
                "skill",
                f"Loading skill: {skill_name}",
                tool_call_id=tool_call.id,
            )
            return
        self.add_item(
            "tool",
            format_tool_call_block(tool_call),
            tool_call_id=tool_call.id,
        )

    def add_user_message(self, content: str) -> None:
        """Append a user-authored message, compacting skill and summary messages."""
        branch_summary = _parse_branch_summary_message(content)
        if branch_summary is not None:
            self.add_item(
                "branch_summary",
                "Branch summary (Ctrl+O to expand)",
                tool_result_text=branch_summary,
            )
            return

        compaction_summary = _parse_compaction_summary_message(content)
        if compaction_summary is not None:
            self.add_item(
                "compaction_summary",
                "Compaction summary (Ctrl+O to expand)",
                tool_result_text=compaction_summary,
            )
            return

        skill_invocation = parse_skill_invocation(content)
        if skill_invocation is None:
            self.add_item("user", content)
            return
        self.add_item("skill", f"Using skill: {skill_invocation.name}")
        if skill_invocation.additional_instructions:
            self.add_item("user", skill_invocation.additional_instructions)

    def add_thinking_delta(self, delta: str) -> None:
        """Append a thinking/reasoning fragment to the current thinking block."""
        if self.items and self.items[-1].role == "thinking":
            self.items[-1].text += delta
            return
        self.add_item("thinking", delta)

    def record_tool_result(self, result: AgentToolResult) -> None:
        """Attach a tool result to its matching call, or append an orphan result."""
        if result.name == "task":
            self.finish_subagent_task(result)
            return
        result_text = format_tool_result_block(
            name=result.name,
            ok=result.ok,
            content=result.content,
            data=result.data,
        )
        for item in reversed(self.items):
            if item.role in {"tool", "skill"} and item.tool_call_id == result.tool_call_id:
                item.tool_result_text = result_text
                return
        self.add_item(
            "tool",
            format_tool_result_summary(name=result.name, ok=result.ok),
            tool_call_id=result.tool_call_id,
            tool_result_text=result_text,
        )

    def update_subagent_activity(self, event: Any) -> bool:
        """Update a task block in place from a ``subagent_activity`` event."""
        data = getattr(event, "data", None)
        if not isinstance(data, Mapping) or data.get("kind") != "subagent_activity":
            return False
        tool_call_id = str(getattr(event, "tool_call_id", ""))
        item = self._find_subagent_item(tool_call_id)
        if item is None or item.subagent is None:
            return False
        display = item.subagent
        status = data.get("status")
        if status in {"queued", "running", "completed", "failed", "cancelled"}:
            display.status = cast(SubagentStatus, status)
        else:
            display.status = "running"
        activity = data.get("activity")
        summary = ""
        if isinstance(activity, Mapping):
            raw_summary = activity.get("summary")
            if isinstance(raw_summary, str):
                summary = raw_summary
            phase = activity.get("phase")
            tool = activity.get("tool")
            if not summary and isinstance(phase, str):
                summary = phase
            raw_calls = data.get("tool_calls")
            if isinstance(raw_calls, int) and raw_calls >= 0:
                display.tool_calls = raw_calls
            elif isinstance(tool, str) and phase in {"started", "tool_started"}:
                display.tool_calls += 1
        elif isinstance(activity, str):
            summary = activity
        message = getattr(event, "message", "")
        if not summary and isinstance(message, str):
            summary = message
        if summary:
            display.activity = summary
        return True

    def finish_subagent_task(self, result: AgentToolResult) -> bool:
        """Finish a task block from its stable v1 artifact.

        Invalid, missing, or unknown artifacts are converted to ordinary tool
        rows.  This keeps history replay tolerant of third-party artifacts and
        future schema versions.
        """
        item = self._find_subagent_item(result.tool_call_id)
        artifact = result.data
        if _is_interrupted_task_result(result):
            if item is not None and item.subagent is not None:
                item.subagent.status = "cancelled"
                item.subagent.activity = "cancelled"
                return True
            return False
        if not _is_subagent_artifact(artifact):
            self._fallback_subagent_result(result, item=item)
            return False

        assert isinstance(artifact, Mapping)
        status = artifact.get("status")
        if status not in {"completed", "failed"}:
            self._fallback_subagent_result(result, item=item)
            return False
        if item is None:
            display = SubagentDisplay(
                tool_call_id=result.tool_call_id,
                agent=_artifact_string(artifact, "agent") or "subagent",
                instruction=_artifact_string(artifact, "instruction") or "",
            )
            self.add_item(
                "subagent",
                f"→ task {_artifact_string(artifact, 'agent') or 'subagent'}",
                tool_call_id=result.tool_call_id,
                subagent=display,
            )
            item = self.items[-1]
        current_display = item.subagent
        if current_display is None:
            self._fallback_subagent_result(result, item=item)
            return False
        current_display.status = cast(SubagentStatus, status)
        current_display.agent = _artifact_string(artifact, "agent") or current_display.agent
        current_display.instruction = (
            _artifact_string(artifact, "instruction") or current_display.instruction
        )
        current_display.final_output = _artifact_string(artifact, "final_output")
        current_display.activity = "completed" if status == "completed" else "failed"
        current_display.tool_calls = _artifact_int(
            artifact, "tool_calls", current_display.tool_calls
        )
        current_display.queued_ms = _artifact_int(artifact, "queued_ms", current_display.queued_ms)
        current_display.duration_ms = _artifact_int(
            artifact, "duration_ms", current_display.duration_ms
        )
        current_display.truncated = bool(artifact.get("truncated", False))
        current_display.error = _artifact_string(artifact, "error")
        return True

    def cancel_subagent_tasks(self) -> None:
        """Converge visible queued/running tasks to ``cancelled``."""
        for item in self.items:
            display = item.subagent
            if display is None or display.status not in {"queued", "running"}:
                continue
            display.status = "cancelled"
            display.activity = "cancelled"

    def has_subagent_task(self, tool_call_id: str) -> bool:
        """Return whether a visible item belongs to the given task call."""
        return self._find_subagent_item(tool_call_id) is not None

    def _find_subagent_item(self, tool_call_id: str) -> ChatItem | None:
        for item in reversed(self.items):
            if (
                item.role == "subagent"
                and item.tool_call_id == tool_call_id
                and item.subagent is not None
            ):
                return item
        return None

    def _fallback_subagent_result(
        self,
        result: AgentToolResult,
        *,
        item: ChatItem | None,
    ) -> None:
        result_text = format_tool_result_block(
            name=result.name,
            ok=result.ok,
            content=result.content,
            data=result.data,
        )
        if item is not None:
            item.role = "tool"
            item.subagent = None
            item.tool_result_text = result_text
            return
        self.add_item(
            "tool",
            format_tool_result_summary(name=result.name, ok=result.ok),
            tool_call_id=result.tool_call_id,
            tool_result_text=result_text,
        )

    def toggle_tool_results(self) -> bool:
        """Toggle expanded display for tool results and return the new state."""
        self.show_tool_results = not self.show_tool_results
        return self.show_tool_results

    def toggle_thinking(self) -> bool:
        """Toggle thinking-token display and return the new state."""
        self.show_thinking = not self.show_thinking
        return self.show_thinking

    def update_queue(self, *, steering: tuple[str, ...], follow_up: tuple[str, ...]) -> None:
        """Replace visible queued-message state."""
        self.queued_steering = steering
        self.queued_follow_up = follow_up

    @property
    def queued_message_count(self) -> int:
        """Return the total number of pending queued messages."""
        return len(self.queued_steering) + len(self.queued_follow_up)

    def clear(self) -> None:
        """Clear visible transcript state without modifying durable session history."""
        self.items.clear()
        self.assistant_buffer = ""
        self.error = None

    def set_skills(self, skills: Iterable[Skill]) -> None:
        """Replace loaded skill metadata used for presentation-only path matching."""
        self.skills = tuple(skills)

    def load_messages(self, messages: Iterable[AnyMessage]) -> None:
        """Populate the transcript from restored session messages."""
        for message in messages:
            if isinstance(message, HumanMessage) or getattr(message, "role", None) == "user":
                self.add_user_message(message_text(message))
            elif isinstance(message, AIMessage):
                if message_text(message):
                    self.add_item("assistant", message_text(message))
                for raw_call in message.tool_calls:
                    tool_call = ToolCall(
                        id=str(raw_call.get("id") or ""),
                        name=str(raw_call.get("name") or "tool"),
                        arguments=raw_call.get("args", {}),
                    )
                    if tool_call.name == "task":
                        self.add_subagent_task(tool_call)
                    else:
                        self.add_tool_call(tool_call)
            elif isinstance(message, ToolMessage):
                artifact = message.artifact
                stored = None
                if isinstance(artifact, dict):
                    try:
                        stored = AgentToolResult.model_validate(artifact)
                    except ValueError as exc:  # noqa: PERF401 - third-party artifact
                        # A native BaseTool may attach an arbitrary business
                        # artifact that is not a Forge AgentToolResult; never let
                        # session restore crash over it.
                        del exc
                if stored is not None:
                    self.record_tool_result(stored)
                    continue
                if _is_subagent_artifact(artifact) and (
                    str(message.name or "") == "task"
                    or self.has_subagent_task(str(message.tool_call_id))
                ):
                    self.finish_subagent_task(
                        AgentToolResult(
                            tool_call_id=str(message.tool_call_id),
                            name="task",
                            ok=getattr(message, "status", "success") != "error",
                            content=message_text(message),
                            data=cast(dict[str, Any], artifact),
                        )
                    )
                    continue
                self.record_tool_result(
                    AgentToolResult(
                        tool_call_id=str(message.tool_call_id),
                        name=str(message.name or "tool"),
                        ok=getattr(message, "status", "success") != "error",
                        content=message_text(message),
                    )
                )
            elif getattr(message, "role", None) == "assistant":
                legacy_message = cast(Any, message)
                if legacy_message.content:
                    self.add_item("assistant", str(legacy_message.content))
                for tool_call in legacy_message.tool_calls:
                    if tool_call.name == "task":
                        self.add_subagent_task(tool_call)
                    else:
                        self.add_tool_call(tool_call)
            elif getattr(message, "role", None) == "tool":
                legacy_message = cast(Any, message)
                self.record_tool_result(
                    AgentToolResult(
                        tool_call_id=str(legacy_message.tool_call_id),
                        name=str(legacy_message.name),
                        ok=bool(legacy_message.ok),
                        content=str(legacy_message.content),
                        data=legacy_message.data,
                        details=legacy_message.details,
                        error=legacy_message.error,
                    )
                )

    def _read_skill_name(self, tool_call: ToolCall) -> str | None:
        if tool_call.name != "read":
            return None
        path = _string_argument(tool_call.arguments, "path")
        if path is None:
            return None
        read_path = _normalized_path(path)
        for skill in self.skills:
            if _normalized_path(skill.path) == read_path:
                return skill.name
        return None


def _parse_branch_summary_message(content: str) -> str | None:
    prefix = (
        "The following is a summary of a branch that this conversation came back from:\n<summary>\n"
    )
    suffix = "\n</summary>"
    if content.startswith(prefix) and content.endswith(suffix):
        return content.removeprefix(prefix).removesuffix(suffix)
    return None


def _parse_compaction_summary_message(content: str) -> str | None:
    prefix = "Previous conversation summary:\n"
    if content.startswith(prefix):
        return content.removeprefix(prefix)
    return None


def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_subagent_artifact(data: object) -> bool:
    return (
        isinstance(data, Mapping)
        and data.get("kind") == "subagent_run"
        and isinstance(data.get("version"), int)
        and not isinstance(data.get("version"), bool)
        and data.get("version") == 1
    )


def _is_interrupted_task_result(result: AgentToolResult) -> bool:
    return (
        result.name == "task" and result.content.strip().lower() == "tool call interrupted by user"
    )


def _artifact_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _artifact_int(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default
