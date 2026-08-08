from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from forge_agent import (
    CompactionEntry,
    LeafEntry,
    MessageEntry,
)
from forge_coding.session_export import export_session_html, render_session_html


def _ai_with_tool_call(content: str) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {"id": "call-1", "name": "read", "args": {"path": "README.md"}, "type": "tool_call"}
        ],
    )


def test_render_session_html_preserves_branch_tree() -> None:
    entries = [
        MessageEntry(id="root", message=HumanMessage(content="Start <session>")),
        MessageEntry(
            id="left",
            parent_id="root",
            message=AIMessage(content="Left branch"),
        ),
        MessageEntry(
            id="right",
            parent_id="root",
            message=_ai_with_tool_call("Right branch"),
        ),
        MessageEntry(
            id="tool",
            parent_id="right",
            message=ToolMessage(
                content="File contents",
                tool_call_id="call-1",
                name="read",
                artifact={"data": {"bytes": 13}},
            ),
        ),
        CompactionEntry(
            id="compact",
            parent_id="tool",
            summary="The right branch was compacted.",
            replaces_entry_ids=["root", "right", "tool"],
        ),
        LeafEntry(id="leaf", parent_id="compact", entry_id="compact"),
    ]

    html = render_session_html(entries, title="Test Export", source="/tmp/session.jsonl")

    assert "<title>Test Export</title>" in html
    assert "Source: <code>/tmp/session.jsonl</code>" in html
    assert 'id="entry-root"' in html
    assert 'id="entry-left"' in html
    assert 'id="entry-right"' in html
    assert 'id="entry-compact"' in html
    assert "Start &lt;session&gt;" in html
    assert "Right branch [read]" in html
    assert "active-path" in html
    assert "active-leaf" in html
    assert "Replaces entries" in html


def test_render_session_html_uses_static_document_layout() -> None:
    entries = [MessageEntry(id="root", message=HumanMessage(content="Export layout"))]

    html = render_session_html(entries, title="Layout Export")

    assert '<p class="eyebrow">Forge session export</p>' in html
    assert '<main class="session-shell">' in html
    assert '<aside class="tree-rail">' in html
    assert '<section class="entry-stream" aria-label="Session entries">' in html
    assert 'class="entry-card active-entry"' in html
    assert "Session" in html
    assert "Transcript" in html
    assert "border-right: 1px solid var(--line);" in html
    assert 'id="themeToggle"' in html
    assert "<link" not in html.lower()
    assert "http://" not in html and "https://" not in html


def test_render_session_html_syntax_highlights_tool_call_arguments() -> None:
    entries = [
        MessageEntry(
            id="root",
            message=_ai_with_tool_call("Reading a file"),
        ),
    ]

    html = render_session_html(entries, title="Highlight Export")

    assert 'class="highlight"' in html
    assert '<span class="nt">' in html or '<span class="s2">' in html


def test_render_session_html_includes_theme_toggle_script() -> None:
    entries = [MessageEntry(id="root", message=HumanMessage(content="Hello"))]

    html = render_session_html(entries, title="Toggle Export")

    assert 'id="themeToggle"' in html
    assert "localStorage" in html
    assert "data-theme" in html


def test_export_session_html_writes_file(tmp_path: Path) -> None:
    entries = [MessageEntry(id="root", message=HumanMessage(content="Hello"))]
    output_path = tmp_path / "session.html"

    result = export_session_html(entries, output_path, title="Session")

    assert result == output_path
    assert output_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_export_session_jsonl_projects_foreign_artifacts(tmp_path: Path) -> None:
    """JSONL export must not fail on arbitrary tool artifacts."""
    from forge_coding.session_export import export_session_jsonl

    entries = [
        MessageEntry(
            id="root",
            message=ToolMessage(
                content="done",
                tool_call_id="call-1",
                name="third_party",
                artifact=object(),
            ),
        ),
    ]
    output_path = tmp_path / "session.jsonl"

    result = export_session_jsonl(entries, output_path)

    assert result == output_path
    raw = output_path.read_text(encoding="utf-8")
    assert "forge_serialization" in raw
    assert raw.endswith("\n")


def test_export_session_jsonl_keeps_json_compatible_artifacts(tmp_path: Path) -> None:
    from forge_coding.session_export import export_session_jsonl

    entries = [
        MessageEntry(
            id="root",
            message=ToolMessage(
                content="done",
                tool_call_id="call-1",
                name="third_party",
                artifact={"data": {"bytes": 13}},
            ),
        ),
    ]
    output_path = tmp_path / "session.jsonl"

    export_session_jsonl(entries, output_path)

    raw = output_path.read_text(encoding="utf-8")
    assert '"bytes":13' in raw
    assert "forge_serialization" not in raw
