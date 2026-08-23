"""Markdown resource discovery, parsing, and prompt-asset loading."""

from forge_coding.resources.base import (
    ForgeResourcePaths,
    ResourceDiagnostic,
    ResourceError,
    derive_description,
    format_subagent_path,
    metadata_to_json,
    parse_markdown_resource,
    parse_strict_markdown_frontmatter,
    read_resource_text,
    resource_paths_with_cwd,
)
from forge_coding.resources.trust import (
    TrustError,
    TrustRecord,
    TrustResult,
    TrustStore,
    canonical_path,
    find_project_root,
    project_path_is_safe,
    project_resources_present,
    resolve_project_trust,
)

__all__ = [
    "ForgeResourcePaths",
    "ResourceDiagnostic",
    "ResourceError",
    "derive_description",
    "format_subagent_path",
    "metadata_to_json",
    "parse_markdown_resource",
    "parse_strict_markdown_frontmatter",
    "read_resource_text",
    "resource_paths_with_cwd",
    "TrustError",
    "TrustRecord",
    "TrustResult",
    "TrustStore",
    "canonical_path",
    "find_project_root",
    "project_resources_present",
    "project_path_is_safe",
    "resolve_project_trust",
]
