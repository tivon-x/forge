"""Markdown resource path and frontmatter helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from forge_agent.types import JSONValue
from forge_coding.paths import ForgePaths


class ResourceError(ValueError):
    """Raised when Forge resources are invalid or cannot be expanded."""


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    """A non-fatal resource discovery problem or precedence note."""

    kind: str
    message: str
    path: Path | None = None
    name: str | None = None
    severity: str = "warning"
    source: str | None = None

    def format(self) -> str:
        """Return a concise human-readable diagnostic line."""
        parts = [self.severity, self.kind]
        if self.name is not None:
            parts.append(self.name)
        label = " ".join(parts)
        if self.path is None:
            return f"{label}: {self.message}"
        return f"{label}: {self.message} ({self.path})"

    def format_safe(self, *, cwd: Path | None = None) -> str:
        """Format a diagnostic without exposing profile filesystem paths."""
        parts = [self.severity, self.kind]
        if self.name is not None:
            parts.append(self.name)
        label = " ".join(parts)
        if self.kind != "subagent":
            return self.format()
        if self.source is None:
            return f"{label}: {self.message}"
        location = format_subagent_path(
            source=self.source,
            name=self.name,
            cwd=cwd,
        )
        return f"{label}: {self.message} ({location})"


@dataclass(frozen=True, slots=True)
class ForgeResourcePaths:
    """Filesystem locations for Forge markdown resources.

    By default Forge loads both Forge-native resources and `.agents` resources from
    the user home directory. When a cwd is provided, project-local `.forge` and
    `.agents` resources are loaded automatically as well.
    """

    root: Path = field(default_factory=lambda: Path.home() / ".forge")
    cwd: Path | None = None
    agents_root: Path | None = field(default_factory=lambda: Path.home() / ".agents")
    paths: ForgePaths | None = None

    @property
    def skills_dir(self) -> Path:
        """Return the primary Forge skills directory."""
        return self.root / "skills"

    @property
    def prompts_dir(self) -> Path:
        """Return the primary Forge prompt templates directory."""
        return self.root / "prompts"

    @property
    def skills_dirs(self) -> tuple[Path, ...]:
        """Return skill directories in increasing precedence order.

        Only the ``skills`` subdirectory of an ``.agents`` root is scanned,
        never the root ``.agents`` directory itself (which may contain
        ``README.md``, ``AGENTS.md``, etc.).
        """
        paths = self._paths()
        dirs = [self.skills_dir]
        if self.agents_root is not None:
            dirs.append(self.agents_root / "skills")
        if self.cwd is not None:
            dirs.extend(
                [
                    paths.project_skills_dir(self.cwd),
                    paths.project_agents_skills_dir(self.cwd),
                ]
            )
        return tuple(_dedupe_paths(dirs))

    @property
    def prompts_dirs(self) -> tuple[Path, ...]:
        """Return prompt template directories in increasing precedence order."""
        paths = self._paths()
        dirs = [self.prompts_dir]
        if self.agents_root is not None:
            dirs.append(self.agents_root / "prompts")
        if self.cwd is not None:
            dirs.extend(
                [
                    paths.project_prompts_dir(self.cwd),
                    paths.project_agents_prompts_dir(self.cwd),
                ]
            )
        return tuple(_dedupe_paths(dirs))

    @property
    def subagents_dirs(self) -> tuple[Path, ...]:
        """Return user/project declarative subagent roots in precedence order."""
        paths = self._paths()
        dirs = [paths.user_agents_dir]
        if self.cwd is not None:
            dirs.append(paths.project_forge_agents_dir(self.cwd))
        return tuple(_dedupe_paths(dirs))

    def _paths(self) -> ForgePaths:
        agents_home = self.agents_root or Path.home() / ".agents"
        return self.paths or ForgePaths(home=self.root, agents_home=agents_home)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def resource_paths_with_cwd(
    paths: ForgeResourcePaths | None,
    cwd: Path,
) -> ForgeResourcePaths:
    """Return resource paths with a cwd available for project-local discovery."""
    if paths is None:
        return ForgeResourcePaths(cwd=cwd)
    if paths.cwd is not None:
        return paths
    return ForgeResourcePaths(
        root=paths.root,
        cwd=cwd,
        agents_root=paths.agents_root,
        paths=paths.paths,
    )


def parse_markdown_resource(text: str) -> tuple[dict[str, str], str]:
    """Parse minimal YAML-like frontmatter from a markdown resource.

    Only simple `key: value` pairs are supported. This keeps resource parsing
    dependency-free and avoids evaluating arbitrary code.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized

    end = normalized.find("\n---", 4)
    if end == -1:
        return {}, normalized

    raw_frontmatter = normalized[4:end]
    body = normalized[end + len("\n---") :]
    if body.startswith("\n"):
        body = body[1:]

    metadata: dict[str, str] = {}
    for line in raw_frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body


def parse_strict_markdown_frontmatter(
    text: str,
    *,
    allowed_keys: Iterable[str],
) -> tuple[dict[str, str], str]:
    """Parse a small, duplicate-aware frontmatter dialect.

    Profile files intentionally use a stricter parser than legacy skills and
    prompts. Unknown fields, malformed lines, duplicate keys and unterminated
    frontmatter are rejected before values are converted into a mapping.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    allowed = frozenset(allowed_keys)
    if not normalized.startswith("---\n"):
        return {}, normalized

    lines = normalized.split("\n")
    end_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index] == "---":
            end_index = index
            break
    if end_index is None:
        raise ResourceError("unterminated frontmatter")

    metadata: dict[str, str] = {}
    for raw_line in lines[1:end_index]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        key = key.strip()
        if not separator or not key:
            raise ResourceError("frontmatter entries must use key: value syntax")
        if key not in allowed:
            raise ResourceError(f"unknown frontmatter field: {key}")
        if key in metadata:
            raise ResourceError(f"duplicate frontmatter field: {key}")
        metadata[key] = value.strip().strip("\"'")

    body = "\n".join(lines[end_index + 1 :])
    if body.startswith("\n"):
        body = body[1:]
    return metadata, body


def format_subagent_path(*, source: str, name: str | None, cwd: Path | None = None) -> str:
    """Return the stable, non-sensitive path label used by agent displays."""
    del cwd
    if source == "builtin":
        return "builtin"
    if source == "user":
        return "~/.forge/agents" if name is None else f"~/.forge/agents/{name}/AGENT.md"
    if source == "project":
        return ".forge/agents" if name is None else f".forge/agents/{name}/AGENT.md"
    return "unknown"


def derive_description(content: str) -> str | None:
    """Derive a short description from markdown content."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        return stripped
    return None


def metadata_to_json(metadata: dict[str, str]) -> dict[str, JSONValue]:
    """Convert string metadata into JSON-like values."""
    return dict(metadata)
