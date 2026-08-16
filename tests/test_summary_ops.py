"""Tests for file-operation extraction shared by compaction and branch summaries."""

from langchain_core.messages import AIMessage, HumanMessage

from forge_coding.summary_ops import (
    FileOperations,
    compute_file_lists,
    details_from_file_operations,
    extract_file_operations,
    file_operations_from_details,
    format_file_operations,
    merge_file_operations,
)


def _tool_call_ai(call_id: str, name: str, path: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": {"path": path}, "type": "tool_call"}],
    )


def test_extract_file_operations_collects_read_edit_write() -> None:
    operations = extract_file_operations(
        (
            _tool_call_ai("c1", "read", "src/app.py"),
            _tool_call_ai("c2", "edit", "src/app.py"),
            _tool_call_ai("c3", "write", "src/new.py"),
            HumanMessage(content="no tool"),
        )
    )

    assert operations.read == {"src/app.py"}
    assert operations.edited == {"src/app.py"}
    assert operations.written == {"src/new.py"}
    assert compute_file_lists(operations) == ([], ["src/app.py", "src/new.py"])


def test_extract_file_operations_skips_missing_and_foreign_paths() -> None:
    operations = extract_file_operations(
        (
            _tool_call_ai("c1", "read", "src/app.py"),
            _tool_call_ai("c2", "read", ""),
            _tool_call_ai("c3", "edit", ""),
            _tool_call_ai("c4", "grep", "src/other.py"),
        )
    )

    assert operations.read == {"src/app.py"}
    assert operations.edited == set()
    assert operations.written == set()


def test_merge_file_operations_keeps_modified_files_modified() -> None:
    merged = merge_file_operations(
        FileOperations(read={"a.py", "b.py"}, edited={"b.py"}),
        FileOperations(read={"b.py", "c.py"}, written={"c.py"}),
    )

    # b.py and c.py were modified at some point, so later reads stay modified.
    assert compute_file_lists(merged) == (["a.py"], ["b.py", "c.py"])


def test_file_operations_details_round_trip_and_format() -> None:
    operations = FileOperations(read={"a.py"}, edited={"b.py"})

    details = details_from_file_operations(operations)
    assert details == {"read_files": ["a.py"], "modified_files": ["b.py"]}

    rebuilt = file_operations_from_details(details)
    assert compute_file_lists(rebuilt) == (["a.py"], ["b.py"])

    formatted = format_file_operations(details["read_files"], details["modified_files"])
    assert "<read-files>\na.py\n</read-files>" in formatted
    assert "<modified-files>\nb.py\n</modified-files>" in formatted


def test_file_operations_from_details_ignores_malformed_values() -> None:
    assert file_operations_from_details(None).read == set()

    operations = file_operations_from_details(
        {"read_files": ["a.py", 3], "modified_files": "not-a-list"}
    )

    assert operations.read == {"a.py"}
    assert operations.edited == set()


def test_format_file_operations_is_empty_without_files() -> None:
    assert format_file_operations([], []) == ""
