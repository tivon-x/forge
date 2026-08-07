from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from forge_coding.context_window import (
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    build_compaction_summary_prompt,
    estimate_context_tokens,
    estimate_context_usage,
    estimate_message_tokens,
    estimate_text_tokens,
    serialize_messages_for_compaction,
    summarize_messages_for_compaction,
)
from forge_coding.tools import create_coding_tools


def _ai_with_tool_call(content: str) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {"id": "call-1", "name": "read", "args": {"path": "README.md"}, "type": "tool_call"}
        ],
    )


def test_text_token_estimate_is_deterministic() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("a") == 1
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_message_token_estimate_counts_roles_and_tool_calls() -> None:
    user_tokens = estimate_message_tokens(HumanMessage(content="hello"))
    assistant_tokens = estimate_message_tokens(_ai_with_tool_call("using tool"))
    tool_tokens = estimate_message_tokens(
        ToolMessage(content="contents", tool_call_id="call-1", name="read")
    )

    assert user_tokens > estimate_text_tokens("hello")
    assert assistant_tokens > user_tokens
    assert tool_tokens > estimate_text_tokens("contents")


def test_context_token_estimate_includes_system_messages_and_tools(tmp_path: Path) -> None:
    tools = tuple(create_coding_tools(cwd=tmp_path))

    estimate = estimate_context_tokens(
        system="You are Forge.",
        messages=(HumanMessage(content="hello"), AIMessage(content="hi")),
        tools=tools,
    )

    assert estimate > estimate_text_tokens("You are Forge.hellohi")


def test_auto_compaction_threshold_keeps_pi_style_reserve() -> None:
    assert auto_compaction_threshold_for_context_window(128_000) == 111_616
    assert auto_compaction_threshold_for_context_window(16_384) == 1
    assert auto_compaction_threshold_for_context_window(0) is None


def test_context_usage_estimate_reports_breakdown(tmp_path: Path) -> None:
    tools = tuple(create_coding_tools(cwd=tmp_path))
    messages = (HumanMessage(content="hello"), AIMessage(content="hi"))

    usage = estimate_context_usage(system="You are Forge.", messages=messages, tools=tools)

    assert isinstance(usage, ContextUsageEstimate)
    assert usage.message_count == 2
    assert usage.tool_count == len(tools)
    assert usage.system_tokens == estimate_text_tokens("You are Forge.")
    assert usage.message_tokens == sum(estimate_message_tokens(message) for message in messages)
    assert usage.total_tokens == usage.system_tokens + usage.message_tokens + usage.tool_tokens
    assert estimate_context_tokens(system="You are Forge.", messages=messages, tools=tools) == (
        usage.total_tokens
    )


def test_summarize_messages_for_compaction_is_deterministic() -> None:
    summary = summarize_messages_for_compaction(
        (
            HumanMessage(content="Read README.md"),
            _ai_with_tool_call("I'll inspect it."),
            ToolMessage(content="README contents", tool_call_id="call-1", name="read"),
        )
    )

    assert summary == "\n".join(
        [
            "Automatically compacted 3 prior message(s).",
            "1. human: Read README.md",
            "2. ai: I'll inspect it. [tool calls: read]",
            "3. tool: read ok: README contents",
        ]
    )


def test_compaction_summary_prompt_uses_pi_format_and_custom_instructions() -> None:
    prompt = build_compaction_summary_prompt(
        (
            HumanMessage(content="Refactor src/app.py"),
            AIMessage(content="Updated src/app.py"),
        ),
        custom_instructions="Focus on files changed.",
    )

    assert "<conversation>" in prompt
    assert "Use this EXACT format:" in prompt
    assert "## Goal" in prompt
    assert "Preserve exact file paths" in prompt
    assert "Additional focus: Focus on files changed." in prompt
    assert "Refactor src/app.py" in prompt


def test_compaction_summary_prompt_updates_previous_summary() -> None:
    prompt = build_compaction_summary_prompt(
        (
            HumanMessage(content="Previous conversation summary:\n## Goal\nShip compaction."),
            HumanMessage(content="Now add tests."),
        )
    )

    assert "<previous-summary>\n## Goal\nShip compaction.\n</previous-summary>" in prompt
    assert "NEW conversation messages" in prompt
    assert "Now add tests." in prompt
    assert "Previous conversation summary" not in serialize_messages_for_compaction(
        (HumanMessage(content="Now add tests."),)
    )
