"""Declarative ``AGENT.md`` profiles for coding subagents.

This module owns discovery and validation only.  Execution remains in
``forge_agent.SubagentRunner``; a profile never carries a filesystem path or a
provider/runtime object beyond its safe source token.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from forge_coding.resources import (
    ForgeResourcePaths,
    ResourceDiagnostic,
    ResourceError,
    format_subagent_path,
    parse_strict_markdown_frontmatter,
)

SubagentProfileSource = Literal["builtin", "user", "project"]

PROFILE_MAX_FILE_BYTES = 20 * 1024
PROFILE_MAX_BODY_BYTES = 16 * 1024
PROFILE_MAX_DESCRIPTION_BYTES = 300
PROFILE_MAX_COUNT = 16
TASK_REGISTRY_MAX_DESCRIPTION_BYTES = 8 * 1024
PROFILE_DEFAULT_MAX_MODEL_CALLS = 8
PROFILE_DEFAULT_MAX_RESULT_BYTES = 50 * 1024
PROFILE_ALLOWED_FRONTMATTER = frozenset(
    {"description", "tools", "max-model-calls", "max-result-bytes"}
)
_PROFILE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_INTEGER = re.compile(r"^[0-9]+$")


@dataclass(frozen=True, slots=True)
class CodingSubagentProfile:
    """Validated, immutable role definition used by the coding layer."""

    name: str
    description: str
    prompt: str
    tool_names: tuple[str, ...] | None
    max_model_calls: int
    max_result_bytes: int
    source: SubagentProfileSource


@dataclass(frozen=True, slots=True)
class LoadedSubagentProfiles:
    """Atomically replaceable profile registry and its diagnostics."""

    profiles: tuple[CodingSubagentProfile, ...]
    diagnostics: tuple[ResourceDiagnostic, ...]


def builtin_subagent_profiles() -> tuple[CodingSubagentProfile, ...]:
    """Return the built-in roles in their stable registry order."""
    return (
        CodingSubagentProfile(
            name="scout",
            description="Investigate code and collect evidence without editing files.",
            prompt=(
                "You are Forge's scout subagent. Investigate only; do not modify files. "
                "Locate the relevant files, symbols, and call paths, support conclusions "
                "with concrete evidence, and report remaining uncertainty. Your bash "
                "access is a convenience, not a read-only security sandbox, so do not "
                "run commands that change the workspace."
            ),
            tool_names=("read", "find", "grep", "ls", "bash"),
            max_model_calls=PROFILE_DEFAULT_MAX_MODEL_CALLS,
            max_result_bytes=PROFILE_DEFAULT_MAX_RESULT_BYTES,
            source="builtin",
        ),
        CodingSubagentProfile(
            name="worker",
            description="Implement one clearly bounded coding change and verify it.",
            prompt=(
                "You are Forge's worker subagent. Implement only the explicitly assigned "
                "scope. Reuse the existing design, make the smallest coherent change, "
                "run targeted validation, and report the files changed, checks run, and "
                "any remaining risk. Do not broaden the task."
            ),
            tool_names=None,
            max_model_calls=PROFILE_DEFAULT_MAX_MODEL_CALLS,
            max_result_bytes=PROFILE_DEFAULT_MAX_RESULT_BYTES,
            source="builtin",
        ),
        CodingSubagentProfile(
            name="reviewer",
            description=(
                "Independently review existing code or a proposed change without editing files."
            ),
            prompt=(
                "You are Forge's reviewer subagent. Review only; do not modify files. "
                "Report actionable findings first with file and symbol locations, then "
                "state remaining risks or missing validation. Your bash access is a "
                "convenience, not a read-only security sandbox, so do not run commands "
                "that change the workspace."
            ),
            tool_names=("read", "find", "grep", "ls", "bash"),
            max_model_calls=PROFILE_DEFAULT_MAX_MODEL_CALLS,
            max_result_bytes=PROFILE_DEFAULT_MAX_RESULT_BYTES,
            source="builtin",
        ),
    )


def format_profile_source(profile: CodingSubagentProfile) -> str:
    """Return the stable path/source label used by ``/agents`` output."""
    return format_subagent_path(source=profile.source, name=profile.name)


def load_subagent_profiles(
    paths: ForgeResourcePaths | None = None,
    *,
    available_tool_names: Iterable[str] | None = None,
    include_builtins: bool = True,
) -> LoadedSubagentProfiles:
    """Discover and merge built-in, user, and project profile definitions.

    A valid higher-precedence profile replaces the complete lower-precedence
    profile.  Invalid profiles are skipped, so a malformed project file cannot
    hide a safe user or built-in role.
    """
    resource_paths = paths or ForgeResourcePaths()
    available = None if available_tool_names is None else frozenset(available_tool_names)
    profiles: dict[str, CodingSubagentProfile] = {}
    diagnostics: list[ResourceDiagnostic] = []

    if include_builtins:
        for profile in builtin_subagent_profiles():
            profiles[profile.name] = profile

    scopes: tuple[tuple[SubagentProfileSource, Path, Path], ...] = ()
    roots = resource_paths.subagents_dirs
    if roots:
        user_home = (
            resource_paths.paths.home if resource_paths.paths is not None else resource_paths.root
        )
        user_boundary = user_home.parent
        scopes = (("user", roots[0], user_boundary),)
        if len(roots) > 1:
            assert resource_paths.cwd is not None
            scopes += (("project", roots[1], resource_paths.cwd),)

    for source, root, boundary in scopes:
        discovered, scope_diagnostics = _load_scope(
            root,
            boundary=boundary,
            source=source,
            available_tool_names=available,
        )
        diagnostics.extend(scope_diagnostics)
        for profile in discovered:
            previous = profiles.get(profile.name)
            if previous is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="subagent",
                        name=profile.name,
                        path=root / profile.name / "AGENT.md",
                        source=profile.source,
                        message=(f"overrides lower-precedence {previous.source} subagent profile"),
                        severity="warning",
                    )
                )
            profiles[profile.name] = profile

    ordered = _sort_profiles(profiles.values(), include_builtins=include_builtins)
    ordered, budget_diagnostic = _apply_registry_budget(ordered)
    if budget_diagnostic is not None:
        diagnostics.append(budget_diagnostic)
    return LoadedSubagentProfiles(tuple(ordered), tuple(diagnostics))


def load_custom_subagent_profiles(
    paths: ForgeResourcePaths | None = None,
    *,
    available_tool_names: Iterable[str] | None = None,
) -> LoadedSubagentProfiles:
    """Discover only user/project profiles, without built-ins."""
    return load_subagent_profiles(
        paths,
        available_tool_names=available_tool_names,
        include_builtins=False,
    )


def _sort_profiles(
    profiles: Iterable[CodingSubagentProfile],
    *,
    include_builtins: bool,
) -> list[CodingSubagentProfile]:
    builtin_order = {name: index for index, name in enumerate(("scout", "worker", "reviewer"))}
    return sorted(
        profiles,
        key=lambda profile: (
            0 if include_builtins and profile.name in builtin_order else 1,
            builtin_order.get(profile.name, 0),
            profile.name,
        ),
    )


def _apply_registry_budget(
    profiles: list[CodingSubagentProfile],
) -> tuple[list[CodingSubagentProfile], ResourceDiagnostic | None]:
    """Keep one deterministic, model-safe registry and report skipped roles."""
    kept: list[CodingSubagentProfile] = []
    description_bytes = len(_registry_description_prefix().encode("utf-8"))
    for profile in profiles:
        line_bytes = len(f"\n- {profile.name}: {profile.description}".encode())
        if len(kept) >= PROFILE_MAX_COUNT or (
            description_bytes + line_bytes > TASK_REGISTRY_MAX_DESCRIPTION_BYTES
        ):
            continue
        kept.append(profile)
        description_bytes += line_bytes
    skipped = len(profiles) - len(kept)
    if skipped == 0:
        return kept, None
    return kept, ResourceDiagnostic(
        kind="subagent",
        message=(
            f"skipped {skipped} profile(s) because the registry exceeds "
            f"{PROFILE_MAX_COUNT} roles or {TASK_REGISTRY_MAX_DESCRIPTION_BYTES} UTF-8 bytes"
        ),
        severity="error",
    )


def _registry_description_prefix() -> str:
    return (
        "Delegate one bounded task to a fresh, isolated coding subagent. "
        "Each call accepts exactly one task and returns only the subagent's final result."
        "\n\nAvailable subagents:"
    )


def _load_scope(
    root: Path,
    *,
    boundary: Path,
    source: SubagentProfileSource,
    available_tool_names: frozenset[str] | None,
) -> tuple[list[CodingSubagentProfile], list[ResourceDiagnostic]]:
    diagnostics: list[ResourceDiagnostic] = []
    root = root.expanduser()
    boundary = boundary.expanduser()
    if _has_link_like_component(root, boundary):
        diagnostics.append(
            _diagnostic(
                source,
                root,
                None,
                "agents directory and its resource parents must not be symlinks",
            )
        )
        return [], diagnostics
    if not root.exists():
        return [], diagnostics
    if not root.is_dir():
        diagnostics.append(_diagnostic(source, root, None, "agents path is not a directory"))
        return [], diagnostics
    try:
        resolved_boundary = boundary.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
        if not _within_resolved(resolved_root, resolved_boundary):
            raise ResourceError("agents directory escapes its configured resource boundary")
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except (OSError, RuntimeError, ResourceError) as exc:
        diagnostics.append(
            _diagnostic(
                source,
                root,
                None,
                f"could not scan agents directory: {_safe_profile_error(exc)}",
            )
        )
        return [], diagnostics

    profiles: list[CodingSubagentProfile] = []
    for entry in entries:
        if _is_link_like(entry):
            if entry.is_dir():
                diagnostics.append(
                    _diagnostic(
                        source,
                        entry,
                        entry.name,
                        "agent directory must not be a symlink",
                    )
                )
            continue
        if not entry.is_dir():
            continue
        name = entry.name
        if _PROFILE_NAME.fullmatch(name) is None:
            diagnostics.append(_diagnostic(source, entry, name, "invalid subagent name"))
            continue
        if not _within(entry, root) or not _within_resolved(entry, resolved_root):
            diagnostics.append(
                _diagnostic(source, entry, name, "agent directory escapes its resource root")
            )
            continue
        path = entry / "AGENT.md"
        if _is_link_like(path):
            diagnostics.append(_diagnostic(source, path, name, "AGENT.md must not be a symlink"))
            continue
        if not path.exists() or not path.is_file():
            diagnostics.append(_diagnostic(source, path, name, "missing AGENT.md"))
            continue
        if not _within(path, root) or not _within_resolved(path, resolved_root):
            diagnostics.append(
                _diagnostic(source, path, name, "AGENT.md escapes its resource root")
            )
            continue
        try:
            raw = _read_bounded(path, resolved_root=resolved_root)
            text = raw.decode("utf-8")
            profile = _parse_profile(
                name,
                text,
                source=source,
                available_tool_names=available_tool_names,
            )
        except (OSError, UnicodeDecodeError, ResourceError, ValueError) as exc:
            diagnostics.append(_diagnostic(source, path, name, _safe_profile_error(exc)))
            continue
        profiles.append(profile)
    return profiles, diagnostics


def _read_bounded(path: Path, *, resolved_root: Path) -> bytes:
    """Read one stable regular-file handle after validating its final identity."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(current.st_mode):
            raise ResourceError("AGENT.md must be a regular file")
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ResourceError("AGENT.md changed while it was being opened")
        if _is_link_like(path) or not _within_resolved(path, resolved_root):
            raise ResourceError("AGENT.md escapes its resource root")
        if opened.st_size > PROFILE_MAX_FILE_BYTES:
            raise ResourceError(f"AGENT.md exceeds {PROFILE_MAX_FILE_BYTES} UTF-8 bytes")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(PROFILE_MAX_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > PROFILE_MAX_FILE_BYTES:
        raise ResourceError(f"AGENT.md exceeds {PROFILE_MAX_FILE_BYTES} UTF-8 bytes")
    return raw


def _has_link_like_component(path: Path, boundary: Path) -> bool:
    """Return whether a descendant component below a trusted boundary is a link."""
    try:
        relative = path.absolute().relative_to(boundary.absolute())
    except ValueError:
        return True
    current = boundary
    for part in relative.parts:
        current /= part
        if current.exists() and _is_link_like(current):
            return True
    return False


def _parse_profile(
    name: str,
    text: str,
    *,
    source: SubagentProfileSource,
    available_tool_names: frozenset[str] | None,
) -> CodingSubagentProfile:
    metadata, body = parse_strict_markdown_frontmatter(
        text,
        allowed_keys=PROFILE_ALLOWED_FRONTMATTER,
    )
    description = metadata.get("description", "").strip()
    if not description:
        raise ResourceError("description is required")
    if "\n" in description or "\r" in description:
        raise ResourceError("description must be a single line")
    if len(description.encode("utf-8")) > PROFILE_MAX_DESCRIPTION_BYTES:
        raise ResourceError(f"description exceeds {PROFILE_MAX_DESCRIPTION_BYTES} UTF-8 bytes")

    if len(body.encode("utf-8")) > PROFILE_MAX_BODY_BYTES:
        raise ResourceError(f"AGENT.md body exceeds {PROFILE_MAX_BODY_BYTES} UTF-8 bytes")
    prompt = body.strip()
    if not prompt:
        raise ResourceError("AGENT.md body must not be empty")

    tool_names = _parse_tools(metadata.get("tools"), available_tool_names)
    max_model_calls = _parse_integer(
        metadata.get("max-model-calls"),
        field_name="max-model-calls",
        minimum=1,
        maximum=8,
        default=PROFILE_DEFAULT_MAX_MODEL_CALLS,
    )
    max_result_bytes = _parse_integer(
        metadata.get("max-result-bytes"),
        field_name="max-result-bytes",
        minimum=1024,
        maximum=PROFILE_DEFAULT_MAX_RESULT_BYTES,
        default=PROFILE_DEFAULT_MAX_RESULT_BYTES,
    )
    return CodingSubagentProfile(
        name=name,
        description=description,
        prompt=prompt,
        tool_names=tool_names,
        max_model_calls=max_model_calls,
        max_result_bytes=max_result_bytes,
        source=source,
    )


def _parse_tools(
    value: str | None,
    available_tool_names: frozenset[str] | None,
) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    names = tuple(part.strip() for part in value.split(","))
    if any(not name for name in names):
        raise ResourceError("tools must be a comma-separated allowlist")
    if len(set(names)) != len(names):
        raise ResourceError("tools must not contain duplicates")
    if "task" in names:
        raise ResourceError("tools may not include task")
    if available_tool_names is not None:
        unknown = sorted(set(names) - available_tool_names)
        if unknown:
            raise ResourceError(f"unknown tool(s): {', '.join(unknown)}")
    return names


def _parse_integer(
    value: str | None,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
    default: int,
) -> int:
    if value is None or not value.strip():
        if value is None:
            return default
        raise ResourceError(f"{field_name} must be an integer")
    normalized = value.strip()
    if _INTEGER.fullmatch(normalized) is None:
        raise ResourceError(f"{field_name} must be an integer")
    parsed = int(normalized)
    if not minimum <= parsed <= maximum:
        raise ResourceError(f"{field_name} must be between {minimum} and {maximum}")
    return parsed


def _diagnostic(
    source: SubagentProfileSource,
    path: Path,
    name: str | None,
    message: str,
) -> ResourceDiagnostic:
    return ResourceDiagnostic(
        kind="subagent",
        message=message,
        path=path,
        name=name,
        severity="error",
        source=source,
    )


def _safe_profile_error(exc: BaseException) -> str:
    """Return a bounded diagnostic message without filesystem paths."""
    if isinstance(exc, OSError):
        message = exc.strerror or exc.__class__.__name__
    else:
        message = str(exc) or exc.__class__.__name__
    encoded = message.encode("utf-8")
    if len(encoded) <= 300:
        return message
    return encoded[:300].decode("utf-8", errors="ignore")


def _within(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def _is_link_like(path: Path) -> bool:
    """Reject symbolic links and Windows directory junctions."""
    return path.is_symlink() or path.is_junction()


def _within_resolved(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


__all__ = [
    "CodingSubagentProfile",
    "LoadedSubagentProfiles",
    "PROFILE_ALLOWED_FRONTMATTER",
    "PROFILE_DEFAULT_MAX_MODEL_CALLS",
    "PROFILE_DEFAULT_MAX_RESULT_BYTES",
    "PROFILE_MAX_BODY_BYTES",
    "PROFILE_MAX_DESCRIPTION_BYTES",
    "PROFILE_MAX_FILE_BYTES",
    "PROFILE_MAX_COUNT",
    "TASK_REGISTRY_MAX_DESCRIPTION_BYTES",
    "SubagentProfileSource",
    "builtin_subagent_profiles",
    "format_profile_source",
    "load_custom_subagent_profiles",
    "load_subagent_profiles",
]
