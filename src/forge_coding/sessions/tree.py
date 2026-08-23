"""Helpers for reading and ordering Forge session trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from forge_agent.message_codec import message_text
from forge_agent.session import LeafEntry, SessionState, SessionTreeError
from forge_agent.session.entries import SessionEntry
from forge_agent.session.tree import path_to_entry
from forge_agent.tools import ToolCall


@dataclass(frozen=True, slots=True)
class SessionTreeChoice:
    """One branchable entry in the active session tree."""

    entry_id: str
    label: str
    active: bool = False
    is_tool_call: bool = False


@dataclass(frozen=True, slots=True)
class SessionTreeBranchResult:
    """Result of moving the active session tree leaf."""

    message: str
    input_prefill: str | None = None


def _detach_missing_parents(entries: list[SessionEntry]) -> list[SessionEntry]:
    """Return entries with dangling parent pointers detached from external history."""
    entry_ids = {entry.id for entry in entries}
    return [
        entry.model_copy(update={"parent_id": None})
        if entry.parent_id is not None and entry.parent_id not in entry_ids
        else entry
        for entry in entries
    ]


def _last_parent_id_from_state(state: SessionState) -> str | None:
    if state.active_leaf_id is not None:
        return state.active_leaf_id
    if state.entries:
        return state.entries[-1].id
    return None


def _latest_leaf_entry(entries: list[SessionEntry]) -> LeafEntry | None:
    for entry in reversed(entries):
        if isinstance(entry, LeafEntry):
            return entry
    return None


def _is_branchable_tree_entry(entry: SessionEntry) -> bool:
    if entry.type in {"compaction", "branch_summary"}:
        return True
    if entry.type != "message":
        return False
    return isinstance(
        entry.message,
        HumanMessage | AIMessage,
    )


def _tree_choice_label(entry: SessionEntry, *, branch_indent: int = 0) -> str:
    prefix = "  " * branch_indent
    return f"{prefix}{_tree_entry_title(entry)}"


def _tree_branch_indents(entries: list[SessionEntry]) -> dict[str, int]:
    children_by_parent: dict[str | None, list[str]] = {}
    for entry in entries:
        if entry.type != "leaf":
            children_by_parent.setdefault(entry.parent_id, []).append(entry.id)

    sibling_indexes = {
        child_id: index
        for children in children_by_parent.values()
        for index, child_id in enumerate(children)
    }
    indents: dict[str, int] = {}
    for entry in entries:
        if entry.type == "leaf":
            continue
        parent_indent = indents.get(entry.parent_id, 0) if entry.parent_id is not None else 0
        sibling_index = sibling_indexes.get(entry.id, 0)
        indents[entry.id] = parent_indent + (1 if sibling_index > 0 else 0)
    return indents


def _ordered_tree_entries(entries: list[SessionEntry]) -> tuple[SessionEntry, ...]:
    children_by_parent: dict[str | None, list[SessionEntry]] = {}
    for entry in entries:
        if entry.type != "leaf":
            children_by_parent.setdefault(entry.parent_id, []).append(entry)

    ordered: list[SessionEntry] = []
    seen: set[str] = set()
    expanded: set[str | None] = set()

    def append_descendants(root_parent_id: str | None) -> None:
        # Iterative depth-first walk rather than recursion so a long session (a
        # deep root-to-leaf entry chain) cannot exceed Python's recursion limit.
        # `expanded` also makes a malformed parent cycle terminate instead of
        # recursing forever. Emitting a node's direct children before descending,
        # and pushing them reversed so the first child is processed next,
        # preserves the original traversal order.
        stack: list[str | None] = [root_parent_id]
        while stack:
            parent_id = stack.pop()
            if parent_id in expanded:
                continue
            expanded.add(parent_id)
            children = children_by_parent.get(parent_id, [])
            for child in children:
                if child.id not in seen:
                    ordered.append(child)
                    seen.add(child.id)
            for child in reversed(children):
                stack.append(child.id)

    append_descendants(None)
    for entry in entries:
        if entry.type != "leaf" and entry.id not in seen:
            ordered.append(entry)
            seen.add(entry.id)
            append_descendants(entry.id)
    return tuple(ordered)


def _is_tool_call_tree_entry(entry: SessionEntry) -> bool:
    if entry.type != "message":
        return False
    message = entry.message
    if isinstance(message, AIMessage):
        return bool(message.tool_calls)
    return False


def _tree_entry_title(entry: SessionEntry) -> str:
    match entry.type:
        case "message":
            message = entry.message
            if isinstance(message, AIMessage) and message.tool_calls and not message_text(message):
                calls = message.tool_calls
                tool_names = ", ".join(
                    call.name if isinstance(call, ToolCall) else str(call.get("name", "tool"))
                    for call in calls
                )
                return f"tool call: {tool_names}"
            return f"{_message_role(message)}: {_message_text_preview(message)}"
        case "compaction":
            return f"compaction summary: {_short_preview(entry.summary)}"
        case "branch_summary":
            return f"branch summary: {_short_preview(entry.summary)}"
        case _:
            return entry.type


def _message_text_preview(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return _short_preview(content)
    return _short_preview(str(content))


def _message_role(message: Any) -> str:
    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(
        str(getattr(message, "type", "")), "message"
    )


def _short_preview(text: str, *, limit: int = 72) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized or "(empty)"
    return f"{normalized[: limit - 1]}..."


def _messages_after_entry_on_active_path(
    entries: list[SessionEntry],
    entry_id: str,
    active_leaf_id: str | None,
) -> tuple[Any, ...]:
    if active_leaf_id is None:
        return ()
    try:
        active_path = path_to_entry(entries, active_leaf_id)
    except SessionTreeError:
        return ()
    try:
        target_index = next(
            index for index, entry in enumerate(active_path) if entry.id == entry_id
        )
    except StopIteration:
        return ()
    return tuple(
        entry.message for entry in active_path[target_index + 1 :] if entry.type == "message"
    )
