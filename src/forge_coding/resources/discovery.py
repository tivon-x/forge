"""Project instruction discovery for Forge coding sessions."""

from __future__ import annotations

from pathlib import Path

from forge_coding.resources import (
    ForgeResourcePaths,
    ResourceDiagnostic,
    ResourceError,
    read_resource_text,
)
from forge_coding.resources.system_prompt import ProjectContextFile
from forge_coding.resources.trust import ancestor_agents_files, find_project_root


def discover_project_context(
    paths: ForgeResourcePaths | None = None,
) -> tuple[ProjectContextFile, ...]:
    """Discover project instruction files for system prompt context."""
    context_files, _diagnostics = discover_project_context_with_diagnostics(paths)
    return context_files


def discover_project_context_with_diagnostics(
    paths: ForgeResourcePaths | None = None,
) -> tuple[tuple[ProjectContextFile, ...], tuple[ResourceDiagnostic, ...]]:
    """Discover project instruction files and return non-fatal diagnostics."""
    resource_paths = paths or ForgeResourcePaths()
    context_files: list[ProjectContextFile] = []
    diagnostics: list[ResourceDiagnostic] = []
    for path in _context_file_candidates(resource_paths):
        try:
            content = read_resource_text(
                path,
                project_root=resource_paths.project_root_for_path(path),
            )
        except (OSError, ResourceError, UnicodeDecodeError) as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="context",
                    path=path,
                    message=f"could not read context file: {exc}",
                )
            )
            continue
        context_files.append(ProjectContextFile(path=str(path), content=content))
    return tuple(context_files), tuple(diagnostics)


def _context_file_candidates(paths: ForgeResourcePaths) -> tuple[Path, ...]:
    candidates: list[Path] = [paths.root / "AGENTS.md"]
    if paths.agents_root is not None:
        candidates.append(paths.agents_root / "AGENTS.md")

    if paths.cwd is not None:
        cwd = paths.cwd.expanduser().resolve()
        project_root = find_project_root(cwd)
        if paths.project_resources_allowed:
            candidates.extend(ancestor_agents_files(project_root, cwd))
            forge_paths = paths._paths()
            candidates.extend(
                [
                    forge_paths.project_forge_dir(cwd) / "AGENTS.md",
                    forge_paths.project_agents_dir(cwd) / "AGENTS.md",
                ]
            )

    existing = [path for path in candidates if path.is_file() and paths.is_project_path_safe(path)]
    return tuple(_dedupe_resolved_paths(existing))


def _dedupe_resolved_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path.expanduser())
    return deduped
