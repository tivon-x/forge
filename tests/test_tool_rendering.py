from __future__ import annotations

from forge_agent.tools import ToolCall
from forge_cli.tool_rendering import (
    TOOL_VIEW_REGISTRY,
    ToolViewRegistry,
    format_tool_call_block,
    format_tool_result_block,
)


def test_builtin_registry_formats_calls_and_redacts_task_prompt() -> None:
    assert format_tool_call_block(ToolCall(id="r", name="read", arguments={"path": "a.py"})) == (
        "→ read a.py"
    )
    task = ToolCall(
        id="t",
        name="task",
        arguments={"agent": "scout", "instruction": "secret child prompt"},
    )
    rendered = format_tool_call_block(task)
    assert rendered == "→ task scout"
    assert "secret child prompt" not in rendered


def test_task_result_projection_does_not_expose_child_output() -> None:
    rendered = format_tool_result_block(
        name="task",
        ok=True,
        content="secret child output",
        data={"result": "secret"},
    )
    assert rendered == "✓ task"
    assert "secret" not in rendered


def test_task_role_projection_is_bounded() -> None:
    rendered = format_tool_call_block(
        ToolCall(
            id="t",
            name="task",
            arguments={"agent": "x" * 100_000, "instruction": "hidden"},
        )
    )

    assert len(rendered) <= 64
    assert "hidden" not in rendered


def test_unknown_tool_uses_bounded_generic_formatter() -> None:
    rendered = format_tool_result_block(name="custom", ok=False, content="failed")
    assert rendered == "✗ custom\nfailed"


def test_unknown_tool_call_preview_is_bounded() -> None:
    rendered = format_tool_call_block(
        ToolCall(id="1", name="custom", arguments={"value": "x" * 4_000})
    )
    assert len(rendered) <= 2_200
    assert "Preview only" in rendered


def test_registry_allows_pure_custom_formatter_without_ui_objects() -> None:
    registry = ToolViewRegistry()
    registry.register("echo", call=lambda call: f"echo {call.arguments.get('value')}")
    assert registry.format_call(ToolCall(id="1", name="echo", arguments={"value": "ok"})) == (
        "echo ok"
    )
    assert TOOL_VIEW_REGISTRY.formatter("read") is not None
