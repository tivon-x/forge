"""Forge coding tools organized by capability."""

# Every adapter preserves the direct executor contract: signal=None, context=context.

from __future__ import annotations

from forge_coding.tools.base import (
    ForgeStructuredTool,
    ToolInputError,
    _call_executor,
    _tool_result_text,
)
from forge_coding.tools.bash import (
    BashToolInput,
    create_bash_tool,
    create_bash_tool_definition,
)
from forge_coding.tools.common import (
    _edits_arg,
    _optional_float_arg,
    _optional_int_arg,
    _path_arg,
    _prepare_edit_arguments,
    _shell_command_prefix,
    _str_arg,
    _workspace_root,
)
from forge_coding.tools.definition import ToolDefinition
from forge_coding.tools.edit import (
    EditItemInput,
    EditToolInput,
    create_edit_tool,
    create_edit_tool_definition,
)
from forge_coding.tools.edit_diff import (
    _count_occurrences,
    _duplicate_error,
    _empty_old_text_error,
    _no_change_error,
    _not_found_error,
    _strip_bom,
    _validate_non_overlapping,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
)
from forge_coding.tools.file_operation_queue import FileOperationQueue, file_operation_queue
from forge_coding.tools.read import (
    SUPPORTED_IMAGE_MIME_TYPES,
    ReadToolInput,
    _base64_text,
    _detect_supported_image_mime_type,
    create_read_tool,
    create_read_tool_definition,
)
from forge_coding.tools.registry import create_coding_tool_set, create_coding_tools
from forge_coding.tools.shell import (
    _communicate_with_cancellation,
    _kill_process_tree,
    _prefixed_shell_command,
    _shell_output_encodings,
    _wait_for_cancel,
    _windows_bash_path,
    _write_temp_output,
)
from forge_coding.tools.shell import (
    _decode_shell_output as _decode_shell_output_impl,
)
from forge_coding.tools.tool_set import ToolSet
from forge_coding.tools.truncation import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    TruncationResult,
    _split_lines_for_counting,
    _truncate_string_to_bytes_from_end,
    _truncation_result,
    append_status_block,
    format_size,
    truncate_head,
    truncate_tail,
)
from forge_coding.tools.write import WriteToolInput, create_write_tool, create_write_tool_definition


def _decode_shell_output(data: bytes) -> str:
    """Decode shell output while preserving the legacy module-level hook.

    Before the tools were split into modules, callers and tests could
    monkeypatch ``forge_coding.tools._shell_output_encodings`` directly.  Keep
    that compatibility surface on the package facade while the bash tool uses
    the implementation module directly.
    """

    return _decode_shell_output_impl(data, encodings=_shell_output_encodings())


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_OUTPUT_LINES",
    "ForgeStructuredTool",
    "BashToolInput",
    "EditItemInput",
    "EditToolInput",
    "ReadToolInput",
    "SUPPORTED_IMAGE_MIME_TYPES",
    "ToolDefinition",
    "ToolSet",
    "FileOperationQueue",
    "file_operation_queue",
    "ToolInputError",
    "WriteToolInput",
    "TruncationResult",
    "create_bash_tool",
    "create_bash_tool_definition",
    "create_coding_tools",
    "create_coding_tool_set",
    "create_edit_tool",
    "create_edit_tool_definition",
    "create_read_tool",
    "create_read_tool_definition",
    "create_write_tool",
    "create_write_tool_definition",
    "append_status_block",
    "apply_edits_to_normalized_content",
    "detect_line_ending",
    "format_size",
    "generate_diff_string",
    "generate_unified_patch",
    "normalize_to_lf",
    "restore_line_endings",
    "truncate_head",
    "truncate_tail",
    "_decode_shell_output",
    "_base64_text",
    "_call_executor",
    "_communicate_with_cancellation",
    "_count_occurrences",
    "_detect_supported_image_mime_type",
    "_duplicate_error",
    "_edits_arg",
    "_empty_old_text_error",
    "_kill_process_tree",
    "_no_change_error",
    "_not_found_error",
    "_optional_float_arg",
    "_optional_int_arg",
    "_path_arg",
    "_prefixed_shell_command",
    "_prepare_edit_arguments",
    "_shell_command_prefix",
    "_shell_output_encodings",
    "_str_arg",
    "_strip_bom",
    "_tool_result_text",
    "_truncation_result",
    "_truncate_string_to_bytes_from_end",
    "_split_lines_for_counting",
    "_validate_non_overlapping",
    "_wait_for_cancel",
    "_windows_bash_path",
    "_workspace_root",
    "_write_temp_output",
]
