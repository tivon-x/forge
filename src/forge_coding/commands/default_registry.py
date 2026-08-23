"""Forge's built-in slash commands and the default command registry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from forge_coding.commands.registry import (
    CommandContext,
    CommandRegistry,
    CommandResult,
    CommandSession,
    SlashCommand,
)
from forge_coding.features.goals import GoalCommandAction
from forge_coding.providers.catalog import BUILTIN_PROVIDER_CATALOG, builtin_provider_entry
from forge_coding.providers.thinking import normalize_thinking_level
from forge_coding.resources import ResourceDiagnostic
from forge_coding.resources.subagent_profiles import format_profile_source
from forge_coding.sessions.manager import CodingSessionRecord
from forge_coding.sessions.reload import CodingReloadSummary, ReloadCategorySummary

BUILTIN_TUI_THEME_NAMES = ("forge-dark", "forge-light", "high-contrast")


def create_default_command_registry() -> CommandRegistry:
    """Create Forge's built-in slash command registry."""
    registry = CommandRegistry()
    registry.register(
        SlashCommand(
            name="quit",
            usage="/quit",
            description="Exit the current session.",
            handler=_exit_command,
            aliases=("exit",),
        )
    )
    registry.register(
        SlashCommand(
            name="new",
            usage="/new",
            description="Start a new session.",
            handler=_new_command,
            search_terms=("clear", "reset"),
        )
    )
    registry.register(
        SlashCommand(
            name="compact",
            usage="/compact [instructions]",
            description="Summarize and compact active context.",
            handler=_compact_command,
        )
    )
    registry.register(
        SlashCommand(
            name="todos",
            usage="/todos",
            description="Show the current Todo plan.",
            handler=_todos_command,
            search_terms=("plan", "tasks"),
        )
    )
    registry.register(
        SlashCommand(
            name="export",
            usage="/export [--format html|jsonl] [destination]",
            description="Export the current session.",
            handler=_export_command,
        )
    )
    registry.register(
        SlashCommand(
            name="goal",
            usage="/goal [objective|status|pause|resume|edit <objective>|clear]",
            description="Manage the current user goal.",
            handler=_goal_command,
            search_terms=("objective", "target", "task"),
        )
    )
    registry.register(
        SlashCommand(
            name="session",
            usage="/session",
            description="Show session info and stats.",
            handler=_status_command,
            search_terms=("info",),
        )
    )
    registry.register(
        SlashCommand(
            name="system",
            usage="/system",
            description="Show the active system prompt without saving it.",
            handler=_system_command,
            search_terms=("prompt", "instructions"),
        )
    )
    registry.register(
        SlashCommand(
            name="skill",
            usage="/skill:<name> [request]",
            description="Expand a loaded skill into your prompt.",
            handler=_skill_command,
            search_terms=("skills",),
        )
    )
    registry.register(
        SlashCommand(
            name="hotkeys",
            usage="/hotkeys",
            description="Show common keyboard shortcuts.",
            handler=_hotkeys_command,
            search_terms=("keys", "shortcuts", "bindings"),
        )
    )
    registry.register(
        SlashCommand(
            name="reload",
            usage="/reload",
            description="Reload local resources and project context.",
            handler=_reload_command,
        )
    )
    registry.register(
        SlashCommand(
            name="trust",
            usage="/trust [status|once|always|parent|deny]",
            description="Inspect or change project resource trust.",
            handler=_trust_command,
        )
    )
    registry.register(
        SlashCommand(
            name="agents",
            usage="/agents",
            description="List available coding subagents and profile diagnostics.",
            handler=_agents_command,
            search_terms=("subagents", "roles", "profiles"),
        )
    )
    registry.register(
        SlashCommand(
            name="resources",
            usage="/resources",
            description="List loaded coding resources and diagnostics.",
            handler=_resources_command,
        )
    )
    registry.register(
        SlashCommand(
            name="resume",
            usage="/resume [session-id]",
            description="Resume a previous session.",
            handler=_resume_command,
            search_terms=("history", "previous"),
        )
    )
    registry.register(
        SlashCommand(
            name="tree",
            usage="/tree",
            description="Branch from a previous session entry.",
            handler=_tree_command,
            search_terms=("branch", "history", "fork"),
        )
    )
    registry.register(
        SlashCommand(
            name="name",
            usage="/name <new name>",
            description="Rename the current session.",
            handler=_name_command,
            search_terms=("rename", "title"),
        )
    )
    registry.register(
        SlashCommand(
            name="model",
            usage="/model",
            description="Choose the active model.",
            handler=_model_command,
        )
    )
    registry.register(
        SlashCommand(
            name="scoped-models",
            usage="/scoped-models",
            description="Choose models available to quick-cycle with Ctrl+P.",
            handler=_scoped_models_command,
            search_terms=("scope", "quick", "cycle", "ctrl+p"),
        )
    )
    registry.register(
        SlashCommand(
            name="theme",
            usage="/theme [name]",
            description="Show or set the TUI theme.",
            handler=_theme_command,
            search_terms=("light", "dark", "contrast"),
        )
    )
    registry.register(
        SlashCommand(
            name="login",
            usage="/login [provider]",
            description="Save an API key for a built-in provider.",
            handler=_login_command,
        )
    )
    registry.register(
        SlashCommand(
            name="logout",
            usage="/logout [provider]",
            description="Remove saved credentials for a built-in provider.",
            handler=_logout_command,
        )
    )
    return registry


def _help_command(context: CommandContext) -> CommandResult:
    lines = ["Available commands:"]
    for command in context.registry.list_commands():
        lines.append(f"{command.usage}\t{command.description}")
    return CommandResult(handled=True, message="\n".join(lines))


def _exit_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, exit_requested=True, message="Exiting session.")


def _new_command(context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, new_session_requested=True)


def _compact_command(context: CommandContext) -> CommandResult:
    return CommandResult(
        handled=True,
        compact_summary=context.args.strip(),
    )


def _todos_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /todos")
    from forge_coding.features.planning import format_todos

    return CommandResult(handled=True, message=format_todos(context.session.todos))


def _export_command(context: CommandContext) -> CommandResult:
    try:
        export_format, destination = _parse_export_args(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))
    return CommandResult(
        handled=True,
        export_requested=True,
        export_destination=destination,
        export_format=export_format,
    )


_GOAL_USAGE = "/goal [objective|status|pause|resume|edit <objective>|clear]"
_GOAL_ACTIONS_WITHOUT_ARGUMENTS = {"status", "pause", "resume", "clear"}
_GOAL_MAX_OBJECTIVE_LENGTH = 4_000
_GoalActionName = Literal["start", "status", "pause", "resume", "edit", "clear"]


def _goal_command(context: CommandContext) -> CommandResult:
    """Parse a Goal command into an immutable intent.

    Applying an intent is deliberately owned by ``CodingSession``.  Slash
    command handlers stay synchronous so command parsing cannot update the
    session or append JSONL state before the async command consumer runs.
    """

    args = context.args.strip()
    if not args:
        return CommandResult(handled=True, goal_manager_requested=True)

    first, separator, remainder = args.partition(" ")
    keyword = first.casefold()
    remainder = remainder.strip() if separator else ""

    if keyword == "edit":
        if not remainder:
            return CommandResult(handled=True, message=f"Usage: {_GOAL_USAGE}")
        return _goal_action_result("edit", remainder)

    if keyword in _GOAL_ACTIONS_WITHOUT_ARGUMENTS:
        if remainder:
            return CommandResult(handled=True, message=f"Usage: {_GOAL_USAGE}")
        return _goal_action_result(keyword, None)

    # Any other non-empty text is the shorthand ``/goal <objective>``.
    # Reject option-looking input because Goal has no command-line options;
    # this avoids silently turning a typo such as ``--status`` into a goal.
    if args.startswith("-"):
        return CommandResult(handled=True, message=f"Usage: {_GOAL_USAGE}")
    return _goal_action_result("start", args)


def _goal_action_result(action: str, objective: str | None) -> CommandResult:
    if objective is not None:
        if not objective:
            return CommandResult(handled=True, message=f"Usage: {_GOAL_USAGE}")
        if len(objective) > _GOAL_MAX_OBJECTIVE_LENGTH:
            return CommandResult(
                handled=True,
                message=(
                    f"Goal objective must be {_GOAL_MAX_OBJECTIVE_LENGTH} characters or fewer."
                ),
            )
    try:
        intent = GoalCommandAction(
            action=cast(_GoalActionName, action),
            objective=objective,
        )
    except (TypeError, ValueError) as exc:
        # Keep parser failures user-facing and independent of the concrete
        # validation library used by the Goal core module.
        return CommandResult(handled=True, message=f"Invalid goal command: {exc}")
    return CommandResult(handled=True, goal_action=intent)


def _status_command(context: CommandContext) -> CommandResult:
    session = context.session
    context_usage = getattr(session, "context_usage", None)
    lines = [
        f"Model: {session.model}",
        f"CWD: {session.cwd}",
        f"Tools: {len(session.tools)}",
        f"Skills: {len(session.skills)}",
        f"Prompt templates: {len(session.prompt_templates)}",
        f"Context files: {len(session.context_files)}",
        f"Estimated context tokens: {session.context_token_estimate}",
        f"Context window: {session.context_window_tokens}",
    ]
    if context_usage is not None:
        lines.append(
            "Context token breakdown: "
            f"system={context_usage.system_tokens}, "
            f"messages={context_usage.message_tokens}, "
            f"tools={context_usage.tool_tokens}",
        )
    usage_totals = getattr(session, "usage_totals", None)
    if usage_totals is not None:
        lines.append(f"Model calls: {usage_totals.calls}")
        lines.append(
            "Usage totals: "
            f"input={usage_totals.input_tokens if usage_totals.input_tokens is not None else '?'} "
            "output="
            f"{usage_totals.output_tokens if usage_totals.output_tokens is not None else '?'} "
            f"total={usage_totals.total_tokens if usage_totals.total_tokens is not None else '?'}"
        )
        lines.append(
            "Usage cost: "
            f"{_usage_cost_text(usage_totals)}"
        )
        for purpose, purpose_totals in usage_totals.by_purpose.items():
            purpose_tokens = (
                purpose_totals.total_tokens
                if purpose_totals.total_tokens is not None
                else "?"
            )
            lines.append(
                f"Usage[{purpose}]: calls={purpose_totals.calls} "
                f"total={purpose_tokens}"
            )
        for provider_model, model_totals in usage_totals.by_provider_model.items():
            model_tokens = (
                model_totals.total_tokens if model_totals.total_tokens is not None else "?"
            )
            lines.append(
                f"Usage[{provider_model}]: calls={model_totals.calls} "
                f"total={model_tokens}"
            )
    lines.extend(_thinking_status_lines(session))
    lines.append(f"Resource diagnostics: {len(session.resource_diagnostics)}")
    if session.auto_compact_token_threshold is not None:
        lines.append(f"Auto compact threshold: {session.auto_compact_token_threshold}")
    if session.session_id is not None:
        lines.append(f"Session: {session.session_id}")
    if session.session_title:
        lines.append(f"Session name: {session.session_title}")
    return CommandResult(handled=True, message="\n".join(lines))


def _usage_cost_text(totals: object) -> str:
    """Format known/unknown pricing without presenting unknown as zero."""

    cost = getattr(totals, "cost", None)
    calls = getattr(totals, "calls", 0)
    known = getattr(totals, "known_cost_calls", 0)
    if cost is None:
        return "n/a"
    rendered = f"${cost:.6f}"
    if known != calls:
        rendered += f" ({known}/{calls} priced)"
    return rendered


def _system_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /system")
    return CommandResult(handled=True, message=context.session.system_prompt)


def _hotkeys_command(context: CommandContext) -> CommandResult:
    lines = [
        "Common keyboard shortcuts (defaults; the TUI /hotkeys shows your configured keys):",
        "- Enter: submit prompt",
        "- Shift+Enter: insert newline",
        "- Alt+Enter: queue follow-up while running",
        "- Esc: cancel active run (restores queued messages)",
        "- Alt+Up: restore queued messages to the editor",
        "- Ctrl+K: open slash-command completions",
        "- Ctrl+R: open session picker",
        "- Shift+Tab: cycle thinking mode",
        "- Ctrl+T: toggle thinking tokens",
        "- Ctrl+O: collapse or expand tool output",
        "- Ctrl+Shift+F: search the transcript",
        "- Ctrl+G: edit the prompt in an external editor",
        "- Ctrl+C: clear prompt input",
        "- Ctrl+D: quit",
    ]
    return CommandResult(handled=True, message="\n".join(lines))


def _skills_command(context: CommandContext) -> CommandResult:
    if not context.session.skills:
        lines = ["No skills loaded."]
        if context.session.resource_diagnostics:
            lines.append("")
            lines.extend(_format_diagnostics(context.session.resource_diagnostics, kind="skill"))
        return CommandResult(handled=True, message="\n".join(lines))

    lines = ["Available skills:"]
    for skill in sorted(context.session.skills, key=lambda item: item.name):
        description = skill.description or "No description"
        lines.append(f"- {skill.name}: {description}")
    lines.append("Use a skill with /skill:<name> [request].")
    if context.session.resource_diagnostics:
        lines.append("")
        lines.extend(_format_diagnostics(context.session.resource_diagnostics, kind="skill"))
    return CommandResult(handled=True, message="\n".join(lines))


def _resources_command(context: CommandContext) -> CommandResult:
    session = context.session
    profiles = tuple(getattr(session, "agents", ()))
    lines = [
        f"Skills: {len(session.skills)}",
        f"Prompt templates: {len(session.prompt_templates)}",
        f"Context files: {len(session.context_files)}",
        f"Subagents: {len(profiles)}",
    ]
    if session.resource_diagnostics:
        lines.append("")
        lines.extend(_format_diagnostics(session.resource_diagnostics))
    else:
        lines.append("Resource diagnostics: none")
    return CommandResult(handled=True, message="\n".join(lines))


def _agents_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /agents")
    session = context.session
    profiles = tuple(getattr(session, "agents", ()))
    if profiles:
        lines = ["Available subagents:"]
        for profile in profiles:
            if profile.tool_names is None:
                tools = "all configured tools"
            elif profile.tool_names:
                tools = ", ".join(profile.tool_names)
            else:
                tools = "none"
            lines.append(
                f"- {profile.name}: {profile.description} "
                f"(tools: {tools}; source: {format_profile_source(profile)})"
            )
    else:
        lines = ["No subagents available."]
    diagnostics = tuple(
        diagnostic for diagnostic in session.resource_diagnostics if diagnostic.kind == "subagent"
    )
    if diagnostics:
        lines.append("")
        lines.extend(_format_diagnostics(diagnostics, cwd=session.cwd))
    return CommandResult(handled=True, message="\n".join(lines))


def _reload_command(context: CommandContext) -> CommandResult:
    try:
        summary = context.session.reload()
    except (ValueError, RuntimeError) as exc:
        return CommandResult(handled=True, message=f"Could not reload: {exc}")

    return CommandResult(
        handled=True,
        message=_format_reload_summary(summary),
    )


def _trust_command(context: CommandContext) -> CommandResult:
    """Inspect or update trust without hot-swapping the active prompt."""
    if not context.args or context.args.casefold() == "status":
        status = getattr(context.session, "trust_status", None)
        if not callable(status):
            return CommandResult(handled=True, message="Project trust is unavailable.")
        return CommandResult(handled=True, message=status())

    decision = context.args.strip().casefold()
    if decision not in {"once", "always", "parent", "deny"}:
        return CommandResult(
            handled=True,
            message="Usage: /trust [status|once|always|parent|deny]",
        )
    setter = getattr(context.session, "set_trust_decision", None)
    if not callable(setter):
        return CommandResult(handled=True, message="Project trust is unavailable.")
    return CommandResult(handled=True, message=setter(decision))


def _context_command(context: CommandContext) -> CommandResult:
    session = context.session
    if not session.context_files:
        lines = ["No project context files loaded."]
        if session.resource_diagnostics:
            lines.append("")
            lines.extend(_format_diagnostics(session.resource_diagnostics, kind="context"))
        return CommandResult(handled=True, message="\n".join(lines))

    lines = ["Active project context files:"]
    lines.extend(f"- {context_file.path}" for context_file in session.context_files)
    if session.resource_diagnostics:
        lines.append("")
        lines.extend(_format_diagnostics(session.resource_diagnostics, kind="context"))
    return CommandResult(handled=True, message="\n".join(lines))


def _skill_command(context: CommandContext) -> CommandResult:
    return CommandResult(
        handled=True,
        message="Use /skill:<name> [request] to expand a loaded skill into your prompt.",
    )


def _resume_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True, resume_picker_requested=True)
    manager = context.session.session_manager
    if manager is None:
        return CommandResult(handled=True, message="Session manager is not available.")
    session_id = context.args.strip()
    if manager.get_session(session_id) is None:
        return CommandResult(handled=True, message=f"Unknown session: {session_id}")
    return CommandResult(
        handled=True,
        resume_session_id=session_id,
    )


def _tree_command(context: CommandContext) -> CommandResult:
    if context.args:
        return CommandResult(handled=True, message="Usage: /tree")
    return CommandResult(handled=True, tree_picker_requested=True)


def _name_command(context: CommandContext) -> CommandResult:
    manager = context.session.session_manager
    session_id = context.session.session_id
    if manager is None or session_id is None:
        return CommandResult(handled=True, message="Session manager is not available.")

    if not context.args:
        record = manager.get_session(session_id)
        title = (
            record.title if record is not None else context.session.session_title
        ) or "Untitled session"
        return CommandResult(
            handled=True,
            message=f"Current session name: {title}\nUsage: /name <new name>",
        )

    try:
        name = _validated_session_name(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))

    if manager.get_session(session_id) is None:
        context.session.ensure_session_indexed()

    updated = manager.touch_session(
        session_id,
        model=context.session.model,
        provider_name=context.session.provider_name,
        title=name,
    )
    if updated is None:
        return CommandResult(handled=True, message=f"Unknown current session: {session_id}")
    return CommandResult(handled=True, message=f"Session renamed: {updated.title}")


def _format_sessions(context: CommandContext) -> str:
    manager = context.session.session_manager
    if manager is None:
        return "Session manager is not available."

    records = manager.list_sessions(context.session.cwd)
    if not records:
        return "No sessions found."

    lines = ["Indexed sessions:"]
    for record in records:
        lines.append(_format_session_record(record))
    return "\n".join(lines)


def _model_command(context: CommandContext) -> CommandResult:
    refresh_error = _refresh_provider_settings(context.session)
    if refresh_error is not None:
        return refresh_error

    if context.args:
        model = context.args.strip()
        available_models = set(context.session.available_models)
        if available_models and model not in available_models:
            models = ", ".join(sorted(available_models))
            return CommandResult(
                handled=True,
                message=f"Unknown model for provider {context.session.provider_name}: {model}\n"
                f"Available models: {models}",
            )
        context.session.set_model(model)
        return CommandResult(handled=True, message=f"Current model: {model}")

    return CommandResult(handled=True, model_picker_requested=True)


def _scoped_models_command(context: CommandContext) -> CommandResult:
    refresh_error = _refresh_provider_settings(context.session)
    if refresh_error is not None:
        return refresh_error

    if context.args:
        return CommandResult(handled=True, message="Usage: /scoped-models")
    return CommandResult(handled=True, scoped_models_picker_requested=True)


def _thinking_command(context: CommandContext) -> CommandResult:
    session = context.session
    available = tuple(session.available_thinking_levels)
    if not context.args:
        lines = _thinking_status_lines(session)
        if available:
            lines.append(f"Available modes: {', '.join(available)}")
        else:
            lines.insert(1, f"Current model: {session.provider_name}:{session.model}")
        return CommandResult(handled=True, message="\n".join(lines))

    if not available:
        message = f"Thinking controls are unavailable for {session.provider_name}:{session.model}"
        reason = _thinking_unavailable_reason(session)
        if reason:
            message = f"{message}: {reason}"
        return CommandResult(
            handled=True,
            message=message,
        )
    try:
        level = normalize_thinking_level(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))
    if level not in available:
        modes = ", ".join(available)
        return CommandResult(
            handled=True,
            message=(
                f"Thinking mode {level} is not available for "
                f"{session.provider_name}:{session.model}\n"
                f"Available modes: {modes}"
            ),
        )
    return CommandResult(handled=True, thinking_level=level)


def _thinking_status_lines(session: CommandSession) -> list[str]:
    if tuple(session.available_thinking_levels):
        return [f"Thinking mode: {session.thinking_level}"]
    lines = ["Thinking mode: unavailable"]
    reason = _thinking_unavailable_reason(session)
    if reason:
        lines.append(f"Thinking unavailable: {reason}")
    return lines


def _thinking_unavailable_reason(session: CommandSession) -> str | None:
    reason = getattr(session, "thinking_unavailable_reason", None)
    return reason if isinstance(reason, str) and reason else None


def _theme_command(context: CommandContext) -> CommandResult:
    if not context.args:
        return CommandResult(handled=True, theme_picker_requested=True)

    theme_name = context.args.strip()
    if theme_name not in BUILTIN_TUI_THEME_NAMES:
        themes = ", ".join(BUILTIN_TUI_THEME_NAMES)
        return CommandResult(
            handled=True,
            message=f"Unknown theme: {theme_name}\nAvailable themes: {themes}",
        )
    return CommandResult(handled=True, theme=theme_name)


def _login_command(context: CommandContext) -> CommandResult:
    provider_name = context.args.strip()
    if provider_name in {"custom", "new", "add"}:
        return CommandResult(handled=True, custom_provider_login_requested=True)
    if provider_name:
        entry = builtin_provider_entry(provider_name)
        if entry is None:
            providers = ", ".join(entry.name for entry in BUILTIN_PROVIDER_CATALOG)
            return CommandResult(
                handled=True,
                message=(
                    f"Unknown login provider: {provider_name}\nAvailable providers: {providers}"
                ),
            )
        return CommandResult(handled=True, login_provider=entry.name)

    return CommandResult(handled=True, login_picker_requested=True)


def _logout_command(context: CommandContext) -> CommandResult:
    provider_name = context.args.strip()
    if provider_name:
        entry = builtin_provider_entry(provider_name)
        if entry is None:
            providers = ", ".join(entry.name for entry in BUILTIN_PROVIDER_CATALOG)
            return CommandResult(
                handled=True,
                message=(
                    f"Unknown logout provider: {provider_name}\nAvailable providers: {providers}"
                ),
            )
        return CommandResult(handled=True, logout_provider=entry.name)

    return CommandResult(handled=True, logout_picker_requested=True)


def _format_session_record(record: CodingSessionRecord) -> str:
    title = record.title or "Untitled"
    return f"- {record.id}: {title} ({record.model}) {record.cwd}"


def _format_diagnostics(
    diagnostics: Sequence[ResourceDiagnostic],
    *,
    kind: str | None = None,
    cwd: Path | None = None,
) -> list[str]:
    filtered = [diagnostic for diagnostic in diagnostics if kind is None or diagnostic.kind == kind]
    if not filtered:
        return ["Resource diagnostics: none"]
    lines = ["Resource diagnostics:"]
    for diagnostic in filtered:
        rendered = (
            diagnostic.format_safe(cwd=cwd)
            if diagnostic.kind == "subagent"
            else diagnostic.format()
        )
        lines.append(f"- {rendered}")
    return lines


def _refresh_provider_settings(session: CommandSession) -> CommandResult | None:
    try:
        session.reload_provider_settings()
    except ValueError as exc:
        return CommandResult(
            handled=True,
            message=f"Could not refresh provider settings: {exc}",
        )
    return None


def _format_reload_summary(summary: CodingReloadSummary) -> str:
    lines = [
        "Reloaded local coding resources and project context.",
        "Resources:",
        f"- Skills: {_format_reload_category(summary.skills)}",
        f"- Prompt templates: {_format_reload_category(summary.prompt_templates)}",
        *(
            [
                f"- Subagents: {_format_reload_category(summary.subagents)}",
            ]
            if summary.subagents is not None
            else []
        ),
        "Context:",
        f"- Project context files: {_format_reload_category(summary.context_files)}",
        "- Next-turn system prompt: "
        + ("rebuilt" if summary.system_prompt_rebuilt else "unchanged"),
        "Diagnostics:",
        f"- Resource diagnostics: {_format_reload_category(summary.diagnostics)}",
        "Provider config:",
        "- Not refreshed by /reload; use /login or /model for provider/model settings.",
    ]
    return "\n".join(lines)


def _format_reload_category(summary: ReloadCategorySummary) -> str:
    status = "changed" if summary.changed else "unchanged"
    delta = _format_count_delta(summary.delta)
    suffix = f", {delta}" if delta is not None else ""
    return f"{summary.after} total ({status}{suffix})"


def _format_count_delta(delta: int) -> str | None:
    if delta == 0:
        return None
    return f"{delta:+d}"


def _parse_export_args(args: str) -> tuple[str | None, Path | None]:
    parts = args.split()
    export_format: str | None = None
    destination: Path | None = None
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "--format":
            index += 1
            if index >= len(parts):
                raise ValueError("Usage: /export [--format html|jsonl] [destination]")
            export_format = parts[index]
        elif part.startswith("--format="):
            export_format = part.partition("=")[2]
        elif part.startswith("-"):
            raise ValueError(f"Unknown export option: {part}")
        elif destination is None:
            destination = Path(part).expanduser()
        else:
            raise ValueError("Usage: /export [--format html|jsonl] [destination]")
        index += 1
    return export_format, destination


def _validated_session_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise ValueError("Usage: /name <new name>")
    if any(char in name for char in "\r\n\t"):
        raise ValueError("Session name must be a single line.")
    return name
