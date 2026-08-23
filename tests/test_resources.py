import os
from pathlib import Path

import pytest

import forge_coding.resources.base as resources_base
from forge_coding import ForgePaths, ForgeResourcePaths
from forge_coding.resources import (
    ResourceError,
    derive_description,
    parse_markdown_resource,
    read_resource_text,
)


def test_resource_paths_use_tau_subdirectories(tmp_path: Path) -> None:
    paths = ForgeResourcePaths(root=tmp_path, agents_root=None)

    assert paths.skills_dir == tmp_path / "skills"
    assert paths.prompts_dir == tmp_path / "prompts"
    assert paths.skills_dirs == (tmp_path / "skills",)
    assert paths.prompts_dirs == (tmp_path / "prompts",)


def test_resource_paths_include_agents_and_project_directories(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    forge_home = tmp_path / "home" / ".forge"
    agents_home = tmp_path / "home" / ".agents"
    paths = ForgeResourcePaths(
        root=forge_home,
        agents_root=agents_home,
        cwd=cwd,
        paths=ForgePaths(home=forge_home, agents_home=agents_home),
        project_resources_allowed=True,
    )

    assert paths.skills_dirs == (
        forge_home / "skills",
        agents_home / "skills",
        cwd / ".forge" / "skills",
        cwd / ".agents" / "skills",
    )
    assert paths.prompts_dirs == (
        forge_home / "prompts",
        agents_home / "prompts",
        cwd / ".forge" / "prompts",
        cwd / ".agents" / "prompts",
    )


def test_parse_frontmatter_description() -> None:
    metadata, body = parse_markdown_resource(
        "---\ndescription: Write tests\n---\n# Testing\nUse pytest."
    )

    assert metadata == {"description": "Write tests"}
    assert body == "# Testing\nUse pytest."


def test_parse_frontmatter_normalizes_crlf_line_endings() -> None:
    metadata, body = parse_markdown_resource(
        "---\r\ndescription: Write tests\r\n---\r\n# Testing\r\nUse pytest."
    )

    assert metadata == {"description": "Write tests"}
    assert body == "# Testing\nUse pytest."


def test_derive_description_uses_first_heading_or_paragraph() -> None:
    assert derive_description("\n# Title\nBody") == "Title"
    assert derive_description("\nFirst paragraph\nMore") == "First paragraph"


def test_project_resource_read_rejects_handle_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    resource = project / "AGENTS.md"
    resource.write_text("safe", encoding="utf-8")
    real_fstat = resources_base.os.fstat

    def mismatched_fstat(descriptor: int) -> os.stat_result:
        opened = real_fstat(descriptor)
        values = list(opened)
        values[1] = opened.st_ino + 1
        return os.stat_result(values)

    monkeypatch.setattr(resources_base.os, "fstat", mismatched_fstat)

    with pytest.raises(ResourceError, match="changed while it was being opened"):
        read_resource_text(resource, project_root=project)
