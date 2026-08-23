from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import forge_coding.resources.subagent_profiles as subagent_profiles_module
from forge_agent import SubagentRunner, SubagentRuntime
from forge_coding.features.subagents import create_coding_subagent_specs, create_task_tool
from forge_coding.paths import ForgePaths
from forge_coding.resources import (
    ForgeResourcePaths,
    ResourceError,
    parse_strict_markdown_frontmatter,
)
from forge_coding.resources.subagent_profiles import (
    PROFILE_MAX_COUNT,
    TASK_REGISTRY_MAX_DESCRIPTION_BYTES,
    CodingSubagentProfile,
    builtin_subagent_profiles,
    load_subagent_profiles,
)
from forge_coding.tools import create_bash_tool, create_read_tool, create_write_tool


def _paths(tmp_path: Path) -> ForgeResourcePaths:
    forge_home = tmp_path / "forge-home"
    return ForgeResourcePaths(
        root=forge_home,
        cwd=tmp_path / "project",
        agents_root=None,
        paths=ForgePaths(home=forge_home),
        project_resources_allowed=True,
    )


def _write_agent(root: Path, name: str, body: str, *, frontmatter: str | None = None) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    metadata = frontmatter or "description: Test role\n"
    content = f"---\n{metadata}---\n\n{body}"
    path = directory / "AGENT.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_profiles_merge_project_over_user_and_builtin(tmp_path: Path) -> None:
    resource_paths = _paths(tmp_path)
    user_root = resource_paths.paths.user_agents_dir  # type: ignore[union-attr]
    project_root = resource_paths.paths.project_forge_agents_dir(tmp_path / "project")  # type: ignore[union-attr]
    _write_agent(
        user_root,
        "oracle",
        "User oracle.",
        frontmatter="description: User oracle\ntools: read\nmax-model-calls: 2\n",
    )
    _write_agent(
        project_root,
        "oracle",
        "Project oracle.",
        frontmatter="description: Project oracle\ntools: bash\nmax-result-bytes: 2048\n",
    )
    result = load_subagent_profiles(
        resource_paths,
        available_tool_names={"read", "bash", "write"},
    )

    oracle = next(profile for profile in result.profiles if profile.name == "oracle")
    assert oracle.source == "project"
    assert oracle.prompt == "Project oracle."
    assert oracle.tool_names == ("bash",)
    assert oracle.max_result_bytes == 2048
    assert any(
        diagnostic.name == "oracle"
        and diagnostic.source == "project"
        and "overrides lower-precedence user" in diagnostic.message
        for diagnostic in result.diagnostics
    )
    assert [profile.name for profile in result.profiles[:3]] == ["scout", "worker", "reviewer"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max-model-calls", "0"),
        ("max-model-calls", "9"),
        ("max-model-calls", "nope"),
        ("max-result-bytes", "1023"),
        ("max-result-bytes", "51201"),
        ("max-result-bytes", "nope"),
    ],
)
def test_invalid_profile_fields_are_diagnostic_and_do_not_hide_builtins(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    resource_paths = _paths(tmp_path)
    root = resource_paths.paths.project_forge_agents_dir(tmp_path / "project")  # type: ignore[union-attr]
    _write_agent(root, "scout", "Bad profile.", frontmatter=f"description: Bad\n{field}: {value}\n")

    result = load_subagent_profiles(resource_paths)
    scout = next(profile for profile in result.profiles if profile.name == "scout")
    assert scout.source == "builtin"
    assert any(
        diagnostic.name == "scout" and diagnostic.severity == "error"
        for diagnostic in result.diagnostics
    )


def test_profile_parser_rejects_duplicate_and_unknown_frontmatter() -> None:
    with pytest.raises(ResourceError, match="duplicate"):
        parse_strict_markdown_frontmatter(
            "---\ndescription: one\ndescription: two\n---\nbody",
            allowed_keys={"description"},
        )
    with pytest.raises(ResourceError, match="unknown"):
        parse_strict_markdown_frontmatter(
            "---\ndescription: one\nmodel: fast\n---\nbody",
            allowed_keys={"description"},
        )


def test_profile_tools_reject_duplicates_unknown_and_task(tmp_path: Path) -> None:
    resource_paths = _paths(tmp_path)
    root = resource_paths.paths.project_forge_agents_dir(tmp_path / "project")  # type: ignore[union-attr]
    _write_agent(
        root,
        "duplicate",
        "Body.",
        frontmatter="description: Duplicate\ntools: read, read\n",
    )
    _write_agent(root, "unknown", "Body.", frontmatter="description: Unknown\ntools: mystery\n")
    _write_agent(root, "recursive", "Body.", frontmatter="description: Recursive\ntools: task\n")

    result = load_subagent_profiles(resource_paths, available_tool_names={"read", "bash", "task"})
    names = {profile.name for profile in result.profiles}
    assert not {"duplicate", "unknown", "recursive"} & names
    assert {diagnostic.name for diagnostic in result.diagnostics} >= {
        "duplicate",
        "unknown",
        "recursive",
    }


def test_explicit_empty_capability_rejects_custom_tools_without_hiding_builtins(
    tmp_path: Path,
) -> None:
    resource_paths = _paths(tmp_path)
    root = resource_paths.paths.project_forge_agents_dir(tmp_path / "project")  # type: ignore[union-attr]
    _write_agent(
        root,
        "oracle",
        "Review only.",
        frontmatter="description: Review architecture\ntools: read\n",
    )

    result = load_subagent_profiles(resource_paths, available_tool_names=())

    assert "oracle" not in {profile.name for profile in result.profiles}
    assert {profile.name for profile in result.profiles} == {"scout", "worker", "reviewer"}
    assert any("unknown tool(s): read" in item.message for item in result.diagnostics)


def test_profile_size_limits_and_utf8_boundaries(tmp_path: Path) -> None:
    resource_paths = _paths(tmp_path)
    root = resource_paths.paths.project_forge_agents_dir(tmp_path / "project")  # type: ignore[union-attr]
    _write_agent(root, "too-long", "x" * (16 * 1024 + 1))
    path = root / "oversized" / "AGENT.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (20 * 1024 + 1))
    result = load_subagent_profiles(resource_paths)
    assert not {profile.name for profile in result.profiles} & {"too-long", "oversized"}
    assert any("exceeds" in diagnostic.message for diagnostic in result.diagnostics)


def test_profile_symlinks_are_rejected_without_reading_target(tmp_path: Path) -> None:
    resource_paths = _paths(tmp_path)
    root = resource_paths.paths.project_forge_agents_dir(tmp_path / "project")  # type: ignore[union-attr]
    target = tmp_path / "outside"
    _write_agent(target, "escaped", "Do not load.")
    role_link = root / "escaped"
    root.mkdir(parents=True, exist_ok=True)
    try:
        role_link.symlink_to(target / "escaped", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this Windows environment")
    result = load_subagent_profiles(resource_paths)
    assert "escaped" not in {profile.name for profile in result.profiles}
    assert any(diagnostic.name == "escaped" for diagnostic in result.diagnostics)


def test_profile_resource_parent_links_are_rejected(tmp_path: Path) -> None:
    resource_paths = _paths(tmp_path)
    project = tmp_path / "project"
    project.mkdir(parents=True)
    outside_forge = tmp_path / "outside-forge"
    _write_agent(outside_forge / "agents", "oracle", "Do not load.")
    project_forge = project / ".forge"
    try:
        if os.name == "nt":
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(project_forge), str(outside_forge)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                pytest.skip("directory junctions unavailable")
        else:
            project_forge.symlink_to(outside_forge, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory links unavailable")

    result = load_subagent_profiles(resource_paths)

    assert "oracle" not in {profile.name for profile in result.profiles}
    assert any(
        diagnostic.source == "project" and "must not be symlinks" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_profile_read_rejects_handle_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resource_paths = _paths(tmp_path)
    root = resource_paths.paths.project_forge_agents_dir(tmp_path / "project")  # type: ignore[union-attr]
    _write_agent(root, "oracle", "Safe body.")
    real_fstat = subagent_profiles_module.os.fstat

    def mismatched_fstat(descriptor: int) -> os.stat_result:
        opened = real_fstat(descriptor)
        values = list(opened)
        values[1] = opened.st_ino + 1
        return os.stat_result(values)

    monkeypatch.setattr(subagent_profiles_module.os, "fstat", mismatched_fstat)

    result = load_subagent_profiles(resource_paths)

    assert "oracle" not in {profile.name for profile in result.profiles}
    assert any("changed while it was being opened" in item.message for item in result.diagnostics)


def test_compiler_enforces_capability_ceiling_and_dynamic_task_description(tmp_path: Path) -> None:
    tools = [
        create_write_tool(cwd=tmp_path),
        create_read_tool(cwd=tmp_path),
        create_bash_tool(cwd=tmp_path),
    ]
    profile = CodingSubagentProfile(
        name="oracle",
        description="Challenge assumptions.",
        prompt="Review only.",
        tool_names=("read", "bash"),
        max_model_calls=3,
        max_result_bytes=2048,
        source="project",
    )
    specs = create_coding_subagent_specs(
        cwd=tmp_path,
        tools=tools,
        skills=(),
        context_files=(),
        system="You are Forge.",
        custom_system_prompt=None,
        append_system_prompt=None,
        profiles=(*builtin_subagent_profiles(), profile),
    )
    oracle = next(spec for spec in specs if spec.name == "oracle")
    assert [tool.name for tool in oracle.tools] == ["read", "bash"]
    assert oracle.max_model_calls == 3
    assert oracle.max_result_bytes == 2048
    assert '<subagent_role source="project">' in oracle.system_prompt

    runner = SubagentRunner(lambda: SubagentRuntime(provider=None))  # type: ignore[arg-type]
    runner.replace_specs(specs)
    task = create_task_tool(runner)
    assert "- oracle: Challenge assumptions." in task.description
    schema = task.args_schema.model_json_schema()  # type: ignore[union-attr]
    assert "enum" not in schema["properties"]["agent"]


def test_profile_registry_budget_is_deterministic_and_diagnostic(tmp_path: Path) -> None:
    resource_paths = _paths(tmp_path)
    root = resource_paths.paths.project_forge_agents_dir(tmp_path / "project")  # type: ignore[union-attr]
    for index in range(PROFILE_MAX_COUNT + 5):
        _write_agent(root, f"role-{index:02d}", "Review only.")

    result = load_subagent_profiles(resource_paths)

    assert len(result.profiles) == PROFILE_MAX_COUNT
    assert [profile.name for profile in result.profiles[:3]] == ["scout", "worker", "reviewer"]
    assert any("skipped 8 profile(s)" in item.message for item in result.diagnostics)


def test_task_registry_defensively_rejects_count_and_description_budgets(tmp_path: Path) -> None:
    profiles = tuple(
        CodingSubagentProfile(
            name=f"role-{index:02d}",
            description="Review only.",
            prompt="Review only.",
            tool_names=(),
            max_model_calls=1,
            max_result_bytes=1024,
            source="project",
        )
        for index in range(PROFILE_MAX_COUNT + 1)
    )
    with pytest.raises(ValueError, match="at most"):
        create_coding_subagent_specs(
            cwd=tmp_path,
            tools=(),
            skills=(),
            context_files=(),
            system="You are Forge.",
            custom_system_prompt=None,
            append_system_prompt=None,
            profiles=profiles,
        )

    specs = create_coding_subagent_specs(
        cwd=tmp_path,
        tools=(),
        skills=(),
        context_files=(),
        system="You are Forge.",
        custom_system_prompt=None,
        append_system_prompt=None,
        profiles=profiles[:PROFILE_MAX_COUNT],
    )
    oversized = tuple(
        spec.__class__(
            name=spec.name,
            description="x" * TASK_REGISTRY_MAX_DESCRIPTION_BYTES,
            system_prompt=spec.system_prompt,
            tools=spec.tools,
            max_model_calls=spec.max_model_calls,
            max_result_bytes=spec.max_result_bytes,
        )
        for spec in specs[:1]
    )
    runner = SubagentRunner(lambda: SubagentRuntime(provider=None))  # type: ignore[arg-type]
    runner.replace_specs(oversized)
    with pytest.raises(ValueError, match="description exceeds"):
        create_task_tool(runner)
