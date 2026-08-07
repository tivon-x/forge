"""Model-assisted summaries for abandoned session-tree branches."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from forge_agent.message_codec import message_text, to_langchain_message
from forge_agent.messages import AgentMessage, AssistantMessage, ToolResultMessage, UserMessage
from forge_agent.provider import ModelProvider
from forge_coding.compat import stream_legacy_provider_final

BRANCH_SUMMARY_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary "
    "following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

BRANCH_SUMMARY_PREAMBLE = (
    "The user explored a different conversation branch before returning here.\n"
    "Summary of that exploration:\n\n"
)

BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context
when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

MAX_SUMMARY_SOURCE_MESSAGE_CHARS = 4_000
MAX_SUMMARY_SOURCE_TOTAL_CHARS = 60_000
TOOL_RESULT_MAX_CHARS = 2_000


async def summarize_branch_messages_with_model(
    *,
    provider: ModelProvider | BaseChatModel,
    model: str,
    messages: Sequence[AgentMessage],
    custom_instructions: str | None = None,
    replace_instructions: bool = False,
) -> str | None:
    """Return a model-generated branch summary, or None when generation fails."""
    if not messages:
        return None

    if isinstance(provider, BaseChatModel):
        prompt = _branch_summary_prompt(
            messages,
            custom_instructions=custom_instructions,
            replace_instructions=replace_instructions,
        )
        chunks: list[str] = []
        async for chunk in provider.astream(
            [
                SystemMessage(content=BRANCH_SUMMARY_SYSTEM_PROMPT),
                to_langchain_message(UserMessage(content=prompt)),
            ]
        ):
            chunks.append(message_text(chunk))
        summary = "".join(chunks).strip()
        return _add_branch_summary_context(summary, messages) if summary else None

    response = await stream_legacy_provider_final(
        provider,
        model=model,
        system=BRANCH_SUMMARY_SYSTEM_PROMPT,
        messages=[
            UserMessage(
                content=_branch_summary_prompt(
                    messages,
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                )
            )
        ],
    )

    if response is None:
        return None
    summary = response.content.strip()
    if not summary:
        return None
    return _add_branch_summary_context(summary, messages)


def _branch_summary_prompt(
    messages: Sequence[AgentMessage],
    *,
    custom_instructions: str | None = None,
    replace_instructions: bool = False,
) -> str:
    conversation = _serialize_branch_conversation(messages)
    if replace_instructions and custom_instructions:
        instructions = custom_instructions
    elif custom_instructions:
        instructions = f"{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}"
    else:
        instructions = BRANCH_SUMMARY_PROMPT
    return f"<conversation>\n{conversation}\n</conversation>\n\n{instructions}"


def _serialize_branch_conversation(messages: Sequence[AgentMessage]) -> str:
    parts: list[str] = []
    remaining_chars = MAX_SUMMARY_SOURCE_TOTAL_CHARS
    omitted_count = 0

    for index, message in enumerate(messages, start=1):
        rendered = _format_summary_source_message(message)
        if len(rendered) > remaining_chars:
            omitted_count = len(messages) - index + 1
            break
        parts.append(rendered)
        remaining_chars -= len(rendered)

    if omitted_count:
        parts.append(f"[... {omitted_count} message(s) omitted because the branch was too long]")

    return "\n\n".join(parts)


def _format_summary_source_message(message: AgentMessage) -> str:
    # Native LangChain messages are the production transcript format; render
    # them through the same summary-source contract as the legacy rows.
    if isinstance(message, HumanMessage):
        return f"[User]: {_trim_summary_source_text(message_text(message))}"
    if isinstance(message, AIMessage):
        return _format_native_assistant_summary_source(message)
    if isinstance(message, ToolMessage):
        status = "ok" if str(getattr(message, "status", "success")) != "error" else "failed"
        content = _trim_summary_source_text(message_text(message), max_chars=TOOL_RESULT_MAX_CHARS)
        return f"[Tool result: {message.name or 'tool'} ({status})]: {content}"
    match message:
        case UserMessage():
            return f"[User]: {_trim_summary_source_text(message.content)}"
        case AssistantMessage():
            return _format_assistant_summary_source(message)
        case ToolResultMessage():
            status = "ok" if message.ok else "failed"
            content = _trim_summary_source_text(message.content, max_chars=TOOL_RESULT_MAX_CHARS)
            return f"[Tool result: {message.name} ({status})]: {content}"


def _format_assistant_summary_source(message: AssistantMessage) -> str:
    parts: list[str] = []
    content = _trim_summary_source_text(message.content)
    if content != "(empty)":
        parts.append(f"[Assistant]: {content}")
    if message.tool_calls:
        calls = [
            f"{call.name}({_format_tool_call_arguments(call.arguments)})"
            for call in message.tool_calls
        ]
        parts.append(f"[Assistant tool calls]: {'; '.join(calls)}")
    return "\n".join(parts) if parts else "[Assistant]: (empty)"


def _format_native_assistant_summary_source(message: AIMessage) -> str:
    parts: list[str] = []
    content = _trim_summary_source_text(message_text(message))
    if content != "(empty)":
        parts.append(f"[Assistant]: {content}")
    if message.tool_calls:
        calls = []
        for call in message.tool_calls:
            if not isinstance(call, Mapping):
                continue
            raw_args = call.get("args")
            arguments = raw_args if isinstance(raw_args, Mapping) else {}
            calls.append(f"{call.get('name', 'tool')}({_format_tool_call_arguments(arguments)})")
        if calls:
            parts.append(f"[Assistant tool calls]: {'; '.join(calls)}")
    return "\n".join(parts) if parts else "[Assistant]: (empty)"


def _format_tool_call_arguments(arguments: Mapping[str, object]) -> str:
    return ", ".join(
        f"{key}={json.dumps(value, sort_keys=True)}" for key, value in sorted(arguments.items())
    )


def _trim_summary_source_text(
    text: str,
    *,
    max_chars: int = MAX_SUMMARY_SOURCE_MESSAGE_CHARS,
) -> str:
    normalized = text.strip() or "(empty)"
    if len(normalized) <= max_chars:
        return normalized
    truncated_chars = len(normalized) - max_chars
    return f"{normalized[:max_chars].rstrip()}\n\n[... {truncated_chars} more characters truncated]"


def _add_branch_summary_context(summary: str, messages: Sequence[AgentMessage]) -> str:
    read_files, modified_files = _branch_file_operations(messages)
    sections = [BRANCH_SUMMARY_PREAMBLE + summary]
    if read_files:
        read_file_text = "\n".join(read_files)
        sections.append(f"<read-files>\n{read_file_text}\n</read-files>")
    if modified_files:
        modified_file_text = "\n".join(modified_files)
        sections.append(f"<modified-files>\n{modified_file_text}\n</modified-files>")
    return "\n\n".join(sections)


def _branch_file_operations(messages: Sequence[AgentMessage]) -> tuple[list[str], list[str]]:
    read: set[str] = set()
    modified: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if not isinstance(call, Mapping):
                    continue
                raw_args = call.get("args")
                arguments = raw_args if isinstance(raw_args, Mapping) else {}
                path = arguments.get("path")
                if not isinstance(path, str) or not path:
                    continue
                name = str(call.get("name") or "tool")
                if name == "read":
                    read.add(path)
                elif name in {"edit", "write"}:
                    modified.add(path)
            continue
        if not isinstance(message, AssistantMessage):
            continue
        for call in message.tool_calls:
            path = call.arguments.get("path")
            if not isinstance(path, str) or not path:
                continue
            if call.name == "read":
                read.add(path)
            elif call.name in {"edit", "write"}:
                modified.add(path)
    read_only = sorted(path for path in read if path not in modified)
    return read_only, sorted(modified)
