from pathlib import Path

from forge_agent import TodoItem
from forge_coding.commands import CommandRegistry, SlashCommand, create_default_command_registry
from forge_coding.paths import ForgePaths
from forge_coding.reload import CodingReloadSummary, ReloadCategorySummary
from forge_coding.resources.skills import Skill
from forge_coding.resources.subagent_profiles import CodingSubagentProfile
from forge_coding.resources.system_prompt import ProjectContextFile
from forge_coding.session import ModelChoice
from forge_coding.session_manager import SessionManager
from forge_coding.tools import create_coding_tools


class FakeSession:
    def __init__(self, tmp_path: Path, manager: SessionManager | None = None) -> None:
        self.cwd = tmp_path
        self.provider_name = "openai"
        self.model = "fake-model"
        self.available_models: tuple[str, ...] = ("fake-model", "other-model")
        self.available_model_choices = (
            ModelChoice(provider_name="openai", model="fake-model"),
            ModelChoice(provider_name="openai", model="other-model"),
            ModelChoice(provider_name="local", model="local-model"),
        )
        self.available_providers = ("openai", "local")
        self.tools = tuple(create_coding_tools(cwd=tmp_path))
        self.skills = (
            Skill(
                name="review",
                path=tmp_path / "review.md",
                content="Review code",
                description="Review code",
            ),
        )
        self.prompt_templates = ()
        self.context_files = (
            ProjectContextFile(path=str(tmp_path / "AGENTS.md"), content="Follow instructions."),
        )
        self.context_token_estimate = 123
        self.auto_compact_token_threshold = 200
        self.context_window_tokens = 584
        self.thinking_level = "medium"
        self.available_thinking_levels: tuple[str, ...] = (
            "off",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        )
        self.thinking_unavailable_reason: str | None = None
        self.tui_theme = "forge-dark"
        self.resource_diagnostics = ()
        self.agents = (
            CodingSubagentProfile(
                name="oracle",
                description="Challenge assumptions.",
                prompt="Review only.",
                tool_names=("read",),
                max_model_calls=3,
                max_result_bytes=2048,
                source="project",
            ),
        )
        self.system_prompt = "You are Forge.\nFollow project instructions."
        self.session_id = "session-1"
        self.session_title: str | None = None
        self.session_manager: SessionManager | None = manager
        self.ensure_session_indexed_called = False
        self.reload_called = False
        self.provider_reload_called = False

    def ensure_session_indexed(self) -> None:
        self.ensure_session_indexed_called = True
        if self.session_manager is not None:
            self.session_manager.create_session(
                cwd=self.cwd,
                model=self.model,
                provider_name=self.provider_name,
                session_id=self.session_id,
            )

    def set_model(self, model: str) -> None:
        self.model = model

    def set_provider(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.model = "local-model"
        self.available_models = ("local-model",)

    def reload(self) -> CodingReloadSummary:
        self.reload_called = True
        return CodingReloadSummary(
            skills=ReloadCategorySummary(before=0, after=len(self.skills), changed=True),
            prompt_templates=ReloadCategorySummary(
                before=0,
                after=len(self.prompt_templates),
                changed=False,
            ),
            context_files=ReloadCategorySummary(
                before=0,
                after=len(self.context_files),
                changed=True,
            ),
            diagnostics=ReloadCategorySummary(
                before=0,
                after=len(self.resource_diagnostics),
                changed=False,
            ),
            system_prompt_rebuilt=True,
        )

    def reload_provider_settings(self) -> None:
        self.provider_reload_called = True


def test_registry_ignores_ordinary_prompts_and_skill_expansion(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    assert registry.execute(session, "hello").handled is False
    assert registry.execute(session, "/skill:review fix this").handled is False


def test_registered_commands_are_pi_aligned(tmp_path: Path) -> None:
    commands = create_default_command_registry().list_commands()

    assert [command.name for command in commands] == [
        "agents",
        "compact",
        "export",
        "goal",
        "hotkeys",
        "login",
        "logout",
        "model",
        "name",
        "new",
        "quit",
        "reload",
        "resources",
        "resume",
        "scoped-models",
        "session",
        "skill",
        "system",
        "theme",
        "todos",
        "tree",
    ]


def test_system_command_returns_active_prompt(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    result = registry.execute(session, "/system")

    assert result.handled is True
    assert result.message == "You are Forge.\nFollow project instructions."
    assert registry.execute(session, "/system extra").message == "Usage: /system"


def test_quit_and_new_return_control_flags(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    assert registry.execute(session, "/quit").exit_requested is True
    assert registry.execute(session, "/exit").exit_requested is True
    assert registry.execute(session, "/q").message == "Unknown command: /q"
    assert registry.execute(session, "/new").new_session_requested is True
    assert registry.execute(session, "/clear").message == "Unknown command: /clear"


def test_compact_command_accepts_optional_instructions(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    default = registry.execute(session, "/compact")
    requested = registry.execute(session, "/compact Summary of prior work.")

    assert default.compact_summary == ""
    assert requested.compact_summary == "Summary of prior work."


def test_todos_command_shows_current_plan(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    session.todos = (TodoItem(content="Ship it", status="in_progress"),)

    result = create_default_command_registry().execute(session, "/todos")

    assert result.handled is True
    assert result.message == "Todos (0/1):\n◐ Ship it"


def test_tree_command_requests_picker(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    result = registry.execute(session, "/tree")
    with_args = registry.execute(session, "/tree root")

    assert result.handled is True
    assert result.tree_picker_requested is True
    assert with_args.message == "Usage: /tree"


def test_export_command_requests_default_export(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/export")

    assert result.handled is True
    assert result.export_requested is True
    assert result.export_destination is None
    assert result.export_format is None


def test_export_command_parses_format_and_destination(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(
        FakeSession(tmp_path),
        "/export --format jsonl exports/session.jsonl",
    )

    assert result.export_requested is True
    assert result.export_format == "jsonl"
    assert result.export_destination == Path("exports/session.jsonl")


def test_goal_command_parses_manager_and_all_actions(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    manager = registry.execute(session, "/goal")
    start = registry.execute(session, "/goal Ship and verify the parser")
    status = registry.execute(session, "/goal status")
    pause = registry.execute(session, "/goal pause")
    resume = registry.execute(session, "/goal resume")
    edit = registry.execute(session, "/goal edit Refine the parser")
    clear = registry.execute(session, "/goal clear")

    assert manager.handled is True
    assert manager.goal_manager_requested is True
    assert manager.goal_action is None

    assert start.goal_action is not None
    assert start.goal_action.action == "start"
    assert start.goal_action.objective == "Ship and verify the parser"
    assert status.goal_action is not None and status.goal_action.action == "status"
    assert pause.goal_action is not None and pause.goal_action.action == "pause"
    assert resume.goal_action is not None and resume.goal_action.action == "resume"
    assert edit.goal_action is not None
    assert edit.goal_action.action == "edit"
    assert edit.goal_action.objective == "Refine the parser"
    assert clear.goal_action is not None and clear.goal_action.action == "clear"
    assert all(result.message is None for result in (start, status, pause, resume, edit, clear))


def test_goal_command_rejects_invalid_action_syntax_and_long_objectives(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)
    usage = "Usage: /goal [objective|status|pause|resume|edit <objective>|clear]"

    for text in ("/goal edit", "/goal pause now", "/goal status extra", "/goal --unknown"):
        result = registry.execute(session, text)
        assert result.handled is True
        assert result.message == usage
        assert result.goal_action is None

    result = registry.execute(session, f"/goal {'x' * 4001}")
    assert result.message == "Goal objective must be 4000 characters or fewer."
    assert result.goal_action is None


def test_session_command_includes_session_details(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/session")

    assert result.message is not None
    assert "Model: fake-model" in result.message
    assert f"CWD: {tmp_path}" in result.message
    assert "Tools: 4" in result.message
    assert "Skills: 1" in result.message
    assert "Context files: 1" in result.message
    assert "Estimated context tokens: 123" in result.message
    assert "Context window: 584" in result.message
    assert "Thinking mode: medium" in result.message
    assert "Auto compact threshold: 200" in result.message
    assert "Resource diagnostics: 0" in result.message
    assert "Session: session-1" in result.message
    assert "Session name:" not in result.message
    assert (
        create_default_command_registry().execute(FakeSession(tmp_path), "/status").message
        == "Unknown command: /status"
    )


def test_session_command_includes_named_session_title(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    session.session_title = "Customer bugfix"

    result = create_default_command_registry().execute(session, "/session")

    assert result.message is not None
    assert "Session: session-1" in result.message
    assert "Session name: Customer bugfix" in result.message


def test_session_command_explains_unavailable_thinking_controls(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    session.available_thinking_levels = ()
    session.thinking_unavailable_reason = "Provider local does not declare thinking_levels"

    result = create_default_command_registry().execute(session, "/session")

    assert result.message is not None
    assert "Thinking mode: unavailable" in result.message
    assert "Thinking unavailable: Provider local does not declare thinking_levels" in result.message
    assert "Thinking mode: medium" not in result.message


def test_hotkeys_command_lists_common_tui_shortcuts(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/hotkeys")

    assert result.message is not None
    assert "Common keyboard shortcuts" in result.message
    assert "Ctrl+K: open slash-command completions" in result.message
    assert "Ctrl+R: open session picker" in result.message
    assert "Shift+Tab: cycle thinking mode" in result.message


def test_model_command_requests_picker_and_switches_models(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    registry = create_default_command_registry()

    list_result = registry.execute(session, "/model")
    switch_result = registry.execute(session, "/model other-model")

    assert list_result.model_picker_requested is True
    assert switch_result.message == "Current model: other-model"
    assert session.model == "other-model"
    assert session.provider_reload_called is True


def test_scoped_models_command_requests_scoped_picker(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    registry = create_default_command_registry()

    dashed_result = registry.execute(session, "/scoped-models")
    pi_style_result = registry.execute(session, "/scoped models")

    assert dashed_result.scoped_models_picker_requested is True
    assert pi_style_result.scoped_models_picker_requested is True
    assert session.provider_reload_called is True


def test_model_command_rejects_unknown_model(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)

    result = create_default_command_registry().execute(session, "/model missing")

    assert result.message is not None
    assert "Unknown model for provider openai: missing" in result.message
    assert session.model == "fake-model"


def test_model_command_reports_provider_refresh_failure(tmp_path: Path) -> None:
    class FailingRefreshSession(FakeSession):
        def reload_provider_settings(self) -> None:
            raise ValueError("providers.json is invalid")

    result = create_default_command_registry().execute(FailingRefreshSession(tmp_path), "/model")

    assert result.message == "Could not refresh provider settings: providers.json is invalid"
    assert result.model_picker_requested is False


def test_theme_command_requests_picker_and_sets_theme(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)
    registry = create_default_command_registry()

    list_result = registry.execute(session, "/theme")
    switch_result = registry.execute(session, "/theme forge-light")
    unknown_result = registry.execute(session, "/theme solarized")

    assert list_result.theme_picker_requested is True
    assert switch_result.theme == "forge-light"
    assert unknown_result.message is not None
    assert "Unknown theme: solarized" in unknown_result.message


def test_non_pi_commands_are_not_registered(tmp_path: Path) -> None:
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    for command in ("/provider", "/skills", "/context", "/help"):
        result = registry.execute(session, command)
        assert result.handled is True
        assert result.message == f"Unknown command: {command}"


def test_agents_command_lists_profiles_and_safe_sources(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/agents")

    assert result.handled is True
    assert result.message is not None
    assert "Available subagents:" in result.message
    assert "oracle: Challenge assumptions." in result.message
    assert "tools: read" in result.message
    assert "source: .forge/agents/oracle/AGENT.md" in result.message
    assert str(tmp_path) not in result.message
    invalid = create_default_command_registry().execute(FakeSession(tmp_path), "/agents extra")
    assert invalid.message == "Usage: /agents"


def test_resources_command_includes_subagents(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/resources")

    assert result.handled is True
    assert result.message is not None
    assert "Subagents: 1" in result.message


def test_login_command_requests_provider_picker(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/login")

    assert result.handled is True
    assert result.login_picker_requested is True


def test_login_command_requests_provider_login(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/login openai")

    assert result.handled is True
    assert result.login_provider == "openai"


def test_login_command_requests_custom_provider_login(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/login custom")

    assert result.handled is True
    assert result.custom_provider_login_requested is True


def test_logout_command_requests_provider_picker(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/logout")

    assert result.handled is True
    assert result.logout_picker_requested is True


def test_logout_command_requests_provider_logout(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/logout openai")

    assert result.handled is True
    assert result.logout_provider == "openai"


def test_logout_command_rejects_unknown_provider(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/logout local")

    assert result.handled is True
    assert result.message is not None
    assert "Unknown logout provider: local" in result.message


def test_reload_command_refreshes_session_resources(tmp_path: Path) -> None:
    session = FakeSession(tmp_path)

    result = create_default_command_registry().execute(session, "/reload")

    assert result.message is not None
    assert "Reloaded local coding resources and project context." in result.message
    assert "Resources:" in result.message
    assert "Skills: 1 total (changed, +1)" in result.message
    assert "Prompt templates: 0 total (unchanged)" in result.message
    assert "Project context files: 1 total (changed, +1)" in result.message
    assert "Next-turn system prompt: rebuilt" in result.message
    assert "Provider config:" in result.message
    assert "Not refreshed by /reload" in result.message
    assert session.reload_called is True
    assert session.provider_reload_called is False


def test_resume_without_argument_requests_picker(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    session = FakeSession(tmp_path, manager=manager)

    result = create_default_command_registry().execute(session, "/resume")

    assert result.resume_picker_requested is True
    assert result.message is None
    assert create_default_command_registry().execute(session, "/sessions").message == (
        "Unknown command: /sessions"
    )


def test_resume_command_requests_indexed_session(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake-model", title="Test session")
    session = FakeSession(tmp_path, manager=manager)

    result = create_default_command_registry().execute(session, f"/resume {record.id}")

    assert result.resume_session_id == record.id
    assert result.message is None


def test_resume_command_rejects_missing_or_unknown_session(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    session = FakeSession(tmp_path, manager=manager)

    unknown = create_default_command_registry().execute(session, "/resume missing")

    assert unknown.message == "Unknown session: missing"


def test_name_command_shows_current_name_and_usage(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake-model", title="Test session")
    session = FakeSession(tmp_path, manager=manager)
    session.session_id = record.id

    result = create_default_command_registry().execute(session, "/name")

    assert result.message == "Current session name: Test session\nUsage: /name <new name>"


def test_name_command_renames_current_session(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake-model", title="Old name")
    session = FakeSession(tmp_path, manager=manager)
    session.session_id = record.id

    result = create_default_command_registry().execute(session, "/name Customer bugfix")

    assert result.message == "Session renamed: Customer bugfix"
    renamed = manager.get_session(record.id)
    assert renamed is not None
    assert renamed.title == "Customer bugfix"
    assert renamed.model == "fake-model"
    assert renamed.updated_at >= record.updated_at


def test_name_command_indexes_pending_session_before_renaming(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    session = FakeSession(tmp_path, manager=manager)
    session.session_id = "pending-session"

    result = create_default_command_registry().execute(session, "/name Customer bugfix")

    assert result.message == "Session renamed: Customer bugfix"
    assert session.ensure_session_indexed_called is True
    record = manager.get_session("pending-session")
    assert record is not None
    assert record.title == "Customer bugfix"


def test_name_command_reports_missing_session_manager(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/name Work")

    assert result.message == "Session manager is not available."


def test_name_command_rejects_multiline_name(tmp_path: Path) -> None:
    manager = SessionManager(ForgePaths(home=tmp_path / ".forge", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake-model")
    session = FakeSession(tmp_path, manager=manager)
    session.session_id = record.id

    result = create_default_command_registry().execute(session, "/name Bad\nName")

    assert result.message == "Session name must be a single line."
    assert manager.get_session(record.id) == record


def test_unknown_command_returns_message(tmp_path: Path) -> None:
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/missing")

    assert result.handled is True
    assert result.message == "Unknown command: /missing"


def test_registry_rejects_duplicate_commands_and_aliases() -> None:
    registry = CommandRegistry()
    command = SlashCommand(
        name="test",
        usage="/test",
        description="Test",
        handler=lambda context: create_default_command_registry().execute(
            context.session, "/session"
        ),
    )
    registry.register(command)

    try:
        registry.register(command)
    except ValueError as exc:
        assert "Duplicate slash command" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected duplicate command to fail")
