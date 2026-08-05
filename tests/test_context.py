from pathlib import Path

from forge_coding.context import discover_project_context
from forge_coding.paths import ForgePaths
from forge_coding.resources import ForgeResourcePaths


def test_discovers_user_project_and_agents_context_files(tmp_path: Path) -> None:
    forge_home = tmp_path / "home" / ".forge"
    agents_home = tmp_path / "home" / ".agents"
    project = tmp_path / "project"
    nested = project / "pkg"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (forge_home).mkdir(parents=True)
    (agents_home).mkdir(parents=True)
    (project / ".forge").mkdir()
    (project / ".agents").mkdir()

    (forge_home / "AGENTS.md").write_text("User Forge instructions", encoding="utf-8")
    (agents_home / "AGENTS.md").write_text("User agents instructions", encoding="utf-8")
    (project / "AGENTS.md").write_text("Project instructions", encoding="utf-8")
    (nested / "AGENTS.md").write_text("Nested instructions", encoding="utf-8")
    (nested / ".forge").mkdir()
    (nested / ".agents").mkdir()
    (nested / ".forge" / "AGENTS.md").write_text("Project Forge instructions", encoding="utf-8")
    (nested / ".agents" / "AGENTS.md").write_text("Project agents instructions", encoding="utf-8")

    context_files = discover_project_context(
        ForgeResourcePaths(
            root=forge_home,
            agents_root=agents_home,
            cwd=nested,
            paths=ForgePaths(home=forge_home, agents_home=agents_home),
        )
    )

    assert [Path(context_file.path) for context_file in context_files] == [
        forge_home / "AGENTS.md",
        agents_home / "AGENTS.md",
        project / "AGENTS.md",
        nested / "AGENTS.md",
        nested / ".forge" / "AGENTS.md",
        nested / ".agents" / "AGENTS.md",
    ]
    assert [context_file.content for context_file in context_files] == [
        "User Forge instructions",
        "User agents instructions",
        "Project instructions",
        "Nested instructions",
        "Project Forge instructions",
        "Project agents instructions",
    ]
