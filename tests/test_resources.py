from pathlib import Path

from forge_coding import ForgePaths, ForgeResourcePaths
from forge_coding.resources import derive_description, parse_markdown_resource


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
