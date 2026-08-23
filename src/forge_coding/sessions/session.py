"""Persistent coding-session wrapper built on AgentHarness."""

from __future__ import annotations

import asyncio
import string
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from forge_agent import (
    AgentEvent,
    AgentHarness,
    AgentHarnessConfig,
    ErrorEvent,
    GoalSnapshot,
    GoalUpdateEvent,
    HumanInputRequestedEvent,
    MessageEndEvent,
    QueuedMessages,
    QueueUpdateEvent,
    RetryEvent,
    SubagentRunner,
    SubagentRuntime,
    SubagentTrace,
    TodoItem,
    TodoUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from forge_agent.context import ForgeRuntimeContext
from forge_agent.message_codec import message_text
from forge_agent.retry import RetryPolicy, classify_model_error, redact_model_error
from forge_agent.session import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    JsonlSessionStorage,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    SessionState,
    SessionStorage,
    ThinkingLevelChangeEntry,
)
from forge_agent.session.entries import SessionEntry
from forge_agent.session.jsonl import entry_to_json_line
from forge_agent.session.storage import repair_torn_tail
from forge_agent.types import JSONValue
from forge_coding.commands import CommandRegistry, CommandResult, create_default_command_registry
from forge_coding.features.goals import (
    GOAL_MAX_AUTOMATIC_RUNS,
    GOAL_NAMESPACE,
    GoalCommandAction,
    GoalController,
    GoalMiddleware,
    goal_entry_data,
    goal_tombstone_data,
    latest_goal_snapshot,
)
from forge_coding.features.human_input import (
    create_ask_user_question_tool,
    create_human_input_middleware,
)
from forge_coding.features.planning import (
    TODO_NAMESPACE,
    create_todo_middleware,
    latest_todo_snapshot,
    todo_entry_data,
)
from forge_coding.features.subagents import (
    create_coding_subagent_specs,
    create_task_tool_definition,
    ensure_task_name_available,
)
from forge_coding.paths import ForgePaths
from forge_coding.providers.auth.credentials import FileCredentialStore, credentials_path
from forge_coding.providers.config import (
    ProviderConfig,
    ProviderConfigError,
    ProviderSettings,
    load_provider_settings,
    resolve_provider_selection,
    validate_provider_model,
)
from forge_coding.providers.runtime import aclose_model
from forge_coding.providers.thinking import (
    DEFAULT_THINKING_LEVEL,
    ThinkingLevel,
)
from forge_coding.resources import (
    ForgeResourcePaths,
    ResourceDiagnostic,
    ResourceError,
    TrustError,
    TrustResult,
    TrustStore,
    canonical_path,
    find_project_root,
    resolve_project_trust,
    resource_paths_with_cwd,
)
from forge_coding.resources.discovery import discover_project_context_with_diagnostics
from forge_coding.resources.prompt_templates import (
    PromptTemplate,
    expand_prompt_template_command,
    load_prompt_templates_with_diagnostics,
)
from forge_coding.resources.skills import Skill, expand_skill_command, load_skills_with_diagnostics
from forge_coding.resources.subagent_profiles import (
    CodingSubagentProfile,
    load_subagent_profiles,
)
from forge_coding.resources.system_prompt import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    build_system_prompt,
)
from forge_coding.sessions.branch_summary import summarize_branch_messages_with_model
from forge_coding.sessions.compaction import (
    CompactionPlan,
    _first_recent_context_index,
    _is_context_overflow_error,
    _last_compaction_details,
    _last_user_message_index,
    details_from_file_operations,
    extract_file_operations,
    file_operations_from_details,
    format_file_operations,
    merge_file_operations,
)
from forge_coding.sessions.context_usage import (
    DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_COMPACTION_RESERVE_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    SUMMARIZATION_SYSTEM_PROMPT,
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    build_compaction_summary_prompt,
    build_turn_prefix_summary_prompt,
    estimate_context_usage,
    summarize_messages_for_compaction,
    usage_aware_context_tokens,
)
from forge_coding.sessions.diagnostics import (
    AgentCallDiagnosticContext,
    AgentCallDiagnosticLogger,
    new_agent_call_run_id,
)
from forge_coding.sessions.export import (
    _resolve_export_destination,
    _session_export_title,
    _storage_path,
    export_session_artifact,
    normalize_export_format,
)
from forge_coding.sessions.manager import SessionManager
from forge_coding.sessions.model_selection import (
    ModelSelectionMixin,
    _coerced_thinking_level,
    _default_thinking_level_for_active_model,
    _initial_model_for_config,
    _initial_thinking_level_for_config,
    _runtime_model_for_state,
    _state_thinking_level,
)
from forge_coding.sessions.reload import CodingReloadSummary, ReloadCategorySummary
from forge_coding.sessions.terminal import (
    TerminalCommandResult,
    _terminal_command_context_message,
)
from forge_coding.sessions.tree import (
    SessionTreeBranchResult,
    SessionTreeChoice,
    _detach_missing_parents,
    _is_branchable_tree_entry,
    _is_tool_call_tree_entry,
    _last_parent_id_from_state,
    _latest_leaf_entry,
    _message_role,
    _messages_after_entry_on_active_path,
    _ordered_tree_entries,
    _tree_branch_indents,
    _tree_choice_label,
)
from forge_coding.sessions.usage import (
    USAGE_NAMESPACE,
    UsagePurpose,
    UsageRecord,
    UsageTotals,
    aggregate_usage_entries,
    merge_stream_metadata,
    usage_record_from_message,
)
from forge_coding.tools import ToolDefinition, ToolSet, create_bash_tool, create_coding_tool_set

StreamingBehavior = Literal["steer", "follow_up"]
SESSION_NAME_SYSTEM_PROMPT = (
    "You write concise coding-agent session names. Reply with only a short title, "
    "maximum four words, no quotes, no punctuation-only output."
)
TREE_RUNNING_MESSAGE = "Forge is still working. Press Escape to interrupt before using /tree."
SESSION_SWITCH_RUNNING_MESSAGE = (
    "Forge is still working. Press Escape to interrupt before switching sessions."
)
TURN_ERROR_NAMESPACE = "forge.turn_error.v1"


@dataclass(frozen=True, slots=True)
class SessionResources:
    """Forge-owned resources loaded around a coding session."""

    skills: tuple[Skill, ...]
    prompt_templates: tuple[PromptTemplate, ...]
    context_files: tuple[ProjectContextFile, ...]
    diagnostics: tuple[ResourceDiagnostic, ...]


@dataclass(slots=True)
class _GoalRunStats:
    """Bounded facts observed while consuming one settled Harness run."""

    persisted_count: int
    final_assistant_text: str = ""
    had_tool_calls: bool = False
    nonrecoverable_error: bool = False
    overflow_event: ErrorEvent | None = None
    auto_name_attempted: bool = False
    terminal_goal_stop: bool = False
    retry_attempts: list[dict[str, JSONValue]] = field(default_factory=list)
    final_error: ErrorEvent | None = None
    audit_persisted: bool = False


@dataclass(frozen=True, slots=True)
class _StreamedModelResult:
    """Text plus the allowlisted final response metadata of a helper call."""

    text: str
    message: object


@dataclass(frozen=True, slots=True)
class CodingSessionConfig:
    """Configuration for a persistent coding session."""

    provider: BaseChatModel
    model: str
    storage: SessionStorage
    cwd: Path
    system: str | None = None
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: tuple[ProjectContextFile, ...] = ()
    tools: Any = None
    resource_paths: ForgeResourcePaths | None = None
    session_id: str | None = None
    session_manager: SessionManager | None = None
    command_registry: CommandRegistry | None = None
    provider_name: str = "openai"
    provider_settings: ProviderSettings | None = None
    runtime_provider_config: ProviderConfig | None = None
    auto_compact_token_threshold: int | None = None
    auto_compact_enabled: bool = True
    thinking_level: ThinkingLevel = DEFAULT_THINKING_LEVEL
    index_on_first_persist: bool = False
    shell_command_prefix: str | None = None
    enable_subagents: bool = True
    interactive: bool = False
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    trust_result: TrustResult | None = None
    trust_override: str | None = None
    trust_store: TrustStore | None = None
    session_trust_decisions: dict[str, str] = field(default_factory=dict)


class CodingSession(ModelSelectionMixin):
    """Forge's coding-agent environment wrapper.

    `AgentHarness` owns the in-memory agent brain. `CodingSession` owns the
    coding-session environment around it: durable session entries, default coding
    tools, and a small command seam for later phases.
    """

    def __init__(
        self,
        config: CodingSessionConfig,
        *,
        state: SessionState,
        harness: AgentHarness,
        last_parent_id: str | None,
        skills: tuple[Skill, ...] = (),
        prompt_templates: tuple[PromptTemplate, ...] = (),
        context_files: tuple[ProjectContextFile, ...] = (),
        resource_diagnostics: tuple[ResourceDiagnostic, ...] = (),
        command_registry: CommandRegistry | None = None,
        pending_initial_entries: tuple[SessionEntry, ...] = (),
        tool_set: ToolSet | None = None,
        subagent_runner: SubagentRunner | None = None,
        subagent_profiles: tuple[CodingSubagentProfile, ...] = (),
        goal_controller: GoalController | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._harness = harness
        self._last_parent_id = last_parent_id
        self._pending_initial_entries = pending_initial_entries
        if tool_set is not None:
            self._tool_set = tool_set
        else:
            harness_config = getattr(harness, "config", None)
            harness_tools = getattr(harness_config, "tools", ())
            self._tool_set = ToolSet.from_tools(harness_tools)
        self._skills = skills
        self._prompt_templates = prompt_templates
        self._context_files = context_files
        self._resource_diagnostics = resource_diagnostics
        self._command_registry = command_registry or create_default_command_registry()
        self._provider_name = config.provider_name
        self._provider_settings = config.provider_settings
        self._runtime_provider_config = config.runtime_provider_config
        self._trust_result = config.trust_result
        self._trust_store = config.trust_store or TrustStore()
        self._session_trust_decisions = config.session_trust_decisions
        if (
            config.trust_result is not None
            and config.trust_result.source == "session"
            and config.trust_result.allowed
        ):
            self._session_trust_decisions[str(canonical_path(config.cwd))] = "allow"
        self._resource_paths = resource_paths_with_cwd(
            config.resource_paths,
            config.cwd,
            trust_result=config.trust_result,
        )
        self._auto_compact_token_threshold = config.auto_compact_token_threshold
        self._auto_compact_enabled = config.auto_compact_enabled
        self._thinking_level = _state_thinking_level(
            state,
            default=_default_thinking_level_for_active_model(self),
        )
        self._context_usage_cache: ContextUsageEstimate | None = None
        # Usage is rebuilt only when the durable active branch changes.  The
        # footer and /session command read this maintained projection rather
        # than scanning JSONL on every render.
        self._usage_totals_cache: UsageTotals = aggregate_usage_entries(
            getattr(state, "entries", ())
        )
        self._owned_providers: list[BaseChatModel] = []
        # Serializes session switches against complete CodingSession runs. The
        # lock protects ownership transitions; _run_active keeps that ownership
        # through compaction, overflow retry, persistence, and cancellation even
        # while the inner harness is briefly between model calls. Neither field
        # is transferred when a replacement session is adopted.
        self._switch_lock = asyncio.Lock()
        self._run_active = False
        self._run_task: asyncio.Task[Any] | None = None
        self._goal_replace_pending = False
        self._goal_replace_task: asyncio.Task[Any] | None = None
        self._diagnostic_logger = AgentCallDiagnosticLogger.from_paths(self._resource_paths.paths)
        self._credential_store = FileCredentialStore(
            credentials_path(self._resource_paths.paths) if self._resource_paths.paths else None
        )
        self._last_diagnostic_log_path: Path | None = None
        self._subagent_runner = subagent_runner
        self._subagent_profiles = subagent_profiles
        self._todos = latest_todo_snapshot(getattr(state, "custom_entries", ()))
        self._goal_controller = goal_controller or GoalController(
            latest_goal_snapshot(getattr(state, "custom_entries", ()))
        )
        self._persisted_goal_snapshot = self._goal_controller.snapshot
        self._goal_dirty = False

    @classmethod
    async def load(cls, config: CodingSessionConfig) -> CodingSession:
        """Load a coding session from append-only storage."""
        provided_trust_result = config.trust_result
        if provided_trust_result is not None and not _trust_result_matches_cwd(
            provided_trust_result,
            config.cwd,
        ):
            provided_trust_result = None
        trust_store = config.trust_store or TrustStore(
            provided_trust_result.store_path if provided_trust_result is not None else None
        )
        trust_result = provided_trust_result or resolve_project_trust(
            config.cwd,
            paths=config.resource_paths,
            store=trust_store,
            cli_override=config.trust_override,
            interactive=False,
        )
        effective_paths = resource_paths_with_cwd(
            config.resource_paths,
            config.cwd,
            trust_result=trust_result,
        )
        config = replace(
            config,
            resource_paths=effective_paths,
            trust_result=trust_result,
            trust_store=trust_store,
        )
        entries = await config.storage.read_all()
        pending_initial_entries: tuple[SessionEntry, ...] = ()
        if not entries:
            info = SessionInfoEntry(cwd=str(config.cwd))
            initial_model = _initial_model_for_config(config)
            model = ModelChangeEntry(
                parent_id=info.id,
                model=initial_model,
            )
            thinking = ThinkingLevelChangeEntry(
                parent_id=model.id,
                thinking_level=_initial_thinking_level_for_config(config, model=initial_model),
            )
            entries = [info, model, thinking]
            pending_initial_entries = (info, model, thinking)
        else:
            entries = _detach_missing_parents(entries)

        linear_state = SessionState.from_entries(entries)
        latest_leaf = _latest_leaf_entry(entries)
        state = (
            SessionState.from_entries(entries, leaf_id=latest_leaf.entry_id)
            if latest_leaf is not None
            else linear_state
        )
        base_tool_set = (
            config.tools
            if isinstance(config.tools, ToolSet)
            else (
                create_coding_tool_set(
                    cwd=config.cwd,
                    shell_command_prefix=config.shell_command_prefix,
                )
                if config.tools is None
                else ToolSet.from_tools(config.tools)
            )
        )
        resource_paths = resource_paths_with_cwd(config.resource_paths, config.cwd)
        resources = _load_session_resources(resource_paths, config.context_files)
        runtime_context = ForgeRuntimeContext(
            workspace_root=str(config.cwd),
            session_id=config.session_id,
            shell_command_prefix=config.shell_command_prefix,
        )
        ask_tool: BaseTool | None = None
        session_tool_set = base_tool_set
        if config.interactive:
            ask_tool = create_ask_user_question_tool()
            session_tool_set = session_tool_set.with_tools(ask_tool)
        goal_controller = GoalController(latest_goal_snapshot(state.custom_entries))
        harness_config = AgentHarnessConfig(
            provider=config.provider,
            model=_runtime_model_for_state(config, state),
            runtime_context=runtime_context,
            tools=list(session_tool_set.tools),
            middleware=(
                create_todo_middleware(include_system_prompt=config.system is None),
                GoalMiddleware(goal_controller),
            ),
            interactive=config.interactive,
            retry=config.retry,
        )
        subagent_runner: SubagentRunner | None = None
        subagent_profiles: tuple[CodingSubagentProfile, ...] = ()
        subagent_diagnostics: tuple[ResourceDiagnostic, ...] = ()
        if config.enable_subagents:
            ensure_task_name_available(base_tool_set.tools)
            loaded_subagents = load_subagent_profiles(
                resource_paths,
                available_tool_names=(tool.name for tool in base_tool_set.tools),
            )
            subagent_profiles = loaded_subagents.profiles
            subagent_diagnostics = loaded_subagents.diagnostics
            subagent_runner = SubagentRunner(
                runtime_reader=lambda: SubagentRuntime(
                    provider=harness_config.provider,
                    model=harness_config.model,
                    runtime_context=harness_config.runtime_context,
                ),
                specs=create_coding_subagent_specs(
                    cwd=config.cwd,
                    tools=base_tool_set,
                    skills=resources.skills,
                    context_files=resources.context_files,
                    system=config.system,
                    custom_system_prompt=config.custom_system_prompt,
                    append_system_prompt=config.append_system_prompt,
                    profiles=subagent_profiles,
                ),
            )
            session_tool_set = session_tool_set.with_tools(
                create_task_tool_definition(subagent_runner)
            )
            harness_config.tools = list(session_tool_set.tools)
        system = (
            config.system
            if config.system is not None
            else build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=config.cwd,
                    tools=session_tool_set,
                    skills=resources.skills,
                    custom_prompt=config.custom_system_prompt,
                    append_system_prompt=config.append_system_prompt,
                    context_files=resources.context_files,
                )
            )
        )
        harness_config.system = system
        if config.interactive:
            harness_config.middleware = (
                *harness_config.middleware,
                create_human_input_middleware(),
            )
        harness = AgentHarness(
            harness_config,
            messages=state.messages,
        )
        session = cls(
            config,
            state=state,
            harness=harness,
            last_parent_id=_last_parent_id_from_state(state),
            skills=resources.skills,
            prompt_templates=resources.prompt_templates,
            context_files=resources.context_files,
            resource_diagnostics=(*resources.diagnostics, *subagent_diagnostics),
            command_registry=config.command_registry,
            pending_initial_entries=pending_initial_entries,
            tool_set=session_tool_set,
            subagent_runner=subagent_runner,
            subagent_profiles=subagent_profiles,
            goal_controller=goal_controller,
        )
        await session._persist_loaded_interrupted_tool_repairs()
        if goal_controller.normalize_restored_active() is not None:
            session._goal_dirty = True
            await session._persist_goal_update()
        session._sync_thinking_level_to_active_model()
        session._refresh_runtime_provider()
        return session

    @property
    def cwd(self) -> Path:
        """Return the session working directory."""
        return self._config.cwd

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        """Return the tools available to the agent."""
        return self._tool_set.tools

    @property
    def tool_set(self) -> ToolSet:
        """Return the ordered product catalog for this session."""

        return self._tool_set

    @property
    def todos(self) -> tuple[TodoItem, ...]:
        """Return the latest durable Todo snapshot for the active branch."""

        return self._todos

    @property
    def goal(self) -> GoalSnapshot | None:
        """Return the current session Goal snapshot, if any."""

        return self._goal_controller.snapshot

    @property
    def is_waiting_for_input(self) -> bool:
        """Return whether the active graph is paused for questionnaire input."""

        return self._harness.is_waiting_for_input

    @property
    def pending_human_input(self) -> tuple[HumanInputRequestedEvent, ...]:
        """Return the current questionnaire request, if any."""

        return self._harness.pending_human_input

    @property
    def messages(self) -> tuple[Any, ...]:
        """Return the restored/current transcript."""
        return self._harness.messages

    @property
    def state(self) -> SessionState:
        """Return the last replayed durable session state."""
        return self._state

    async def tree_choices(self) -> tuple[SessionTreeChoice, ...]:
        """Return branchable session entries for a tree picker."""
        entries = await self._read_session_entries()
        branch_indents = _tree_branch_indents(entries)
        return tuple(
            SessionTreeChoice(
                entry_id=entry.id,
                label=_tree_choice_label(entry, branch_indent=branch_indents.get(entry.id, 0)),
                active=entry.id == self._state.active_leaf_id,
                is_tool_call=_is_tool_call_tree_entry(entry),
            )
            for entry in _ordered_tree_entries(entries)
            if _is_branchable_tree_entry(entry)
        )

    async def branch_to_entry(
        self,
        entry_id: str,
        *,
        summarize: bool = False,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
    ) -> SessionTreeBranchResult:
        """Move the active leaf to a previous entry, preserving existing history."""
        if self._goal_replace_pending:
            raise RuntimeError("Goal replacement is in progress")
        if self.is_running or self.is_waiting_for_input:
            raise RuntimeError(TREE_RUNNING_MESSAGE)
        entries = await self._read_session_entries()
        by_id = {entry.id: entry for entry in entries}
        if entry_id not in by_id:
            raise ValueError(f"Unknown session entry: {entry_id}")
        selected_entry = by_id[entry_id]
        if not _is_branchable_tree_entry(selected_entry):
            raise ValueError(f"Session entry cannot be branched from: {entry_id}")

        target_id: str | None = entry_id
        input_prefill: str | None = None
        summary_entry: BranchSummaryEntry | None = None
        summary_usage: list[object] = []
        if summarize:
            abandoned_messages = _messages_after_entry_on_active_path(
                entries,
                entry_id,
                self._last_parent_id,
            )
            if abandoned_messages:
                summary = await self._summarize_branch_messages(
                    abandoned_messages,
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                    usage_sink=summary_usage,
                )
                summary_entry = BranchSummaryEntry(
                    parent_id=entry_id,
                    branch_root_id=entry_id,
                    summary=summary,
                )
                await self._append_session_entry(summary_entry)
                target_id = summary_entry.id
                for usage_message in summary_usage:
                    usage_entry = await self._append_usage_message(
                        usage_message,
                        purpose="branch_summary",
                        parent_id=target_id,
                    )
                    target_id = usage_entry.id
        elif selected_entry.type == "message" and isinstance(selected_entry.message, HumanMessage):
            target_id = selected_entry.parent_id
            input_prefill = message_text(selected_entry.message)
        elif selected_entry.type in {"message", "compaction", "branch_summary"}:
            # Usage is an active-tree node immediately after each billable AI
            # message/helper entry.  Branching from the billable entry must
            # retain that child as the new parent so active totals stay intact.
            usage_child = _adjacent_usage_child(entries, selected_entry.id)
            if usage_child is not None:
                target_id = usage_child.id

        leaf = LeafEntry(parent_id=target_id, entry_id=target_id)
        await self._append_session_entry(leaf)
        self._last_parent_id = target_id

        await self._refresh_persisted_state(leaf_id=target_id)
        # A branch replays a historical snapshot without starting a managed
        # run.  Make a replayed active Goal explicitly resumable at this
        # lifecycle boundary; ordinary persistence must continue to preserve
        # active snapshots unchanged.
        if self._goal_controller.normalize_restored_active() is not None:
            self._goal_dirty = True
            await self._persist_goal_update()
        self._harness.replace_messages(self._state.messages)
        self._invalidate_context_usage_cache()
        self._thinking_level = _state_thinking_level(
            self._state,
            default=_default_thinking_level_for_active_model(self),
        )
        self._sync_thinking_level_to_active_model()
        self._refresh_runtime_provider()
        suffix = " with branch summary" if summary_entry is not None else ""
        if input_prefill is not None:
            return SessionTreeBranchResult(
                message=f"Branched session before {entry_id}.",
                input_prefill=input_prefill,
            )
        return SessionTreeBranchResult(message=f"Branched session at {target_id}{suffix}.")

    @property
    def storage(self) -> SessionStorage:
        """Return the backing session storage."""
        return self._config.storage

    async def export(
        self,
        destination: Path | None = None,
        *,
        format: str | None = None,
    ) -> Path:
        """Export the current session to a user-facing artifact."""
        entries = await self._read_session_entries()
        session_path = _storage_path(self._config.storage)
        export_format = normalize_export_format(
            format or (destination.suffix.removeprefix(".") if destination else "html")
        )
        output_path = _resolve_export_destination(
            destination,
            cwd=self.cwd,
            session_path=session_path,
            format=export_format,
        )
        return export_session_artifact(
            entries,
            output_path,
            title=_session_export_title(self),
            source=str(session_path) if session_path is not None else self.session_id,
            format=export_format,
        )

    @property
    def skills(self) -> tuple[Skill, ...]:
        """Return loaded skills."""
        return self._skills

    @property
    def prompt_templates(self) -> tuple[PromptTemplate, ...]:
        """Return loaded prompt templates."""
        return self._prompt_templates

    @property
    def context_files(self) -> tuple[ProjectContextFile, ...]:
        """Return active project context files."""
        return self._context_files

    @property
    def agents(self) -> tuple[CodingSubagentProfile, ...]:
        """Return the active declarative subagent profiles."""
        return self._subagent_profiles

    @property
    def context_token_estimate(self) -> int:
        """Return the active context size, preferring provider-reported usage.

        The newest assistant message's usage metadata measures the whole
        request context (system prompt, tools, messages) at that point;
        anything appended after it is estimated. Falls back to the
        deterministic heuristic when no fresh usage exists.
        """
        return usage_aware_context_tokens(
            system=self._harness.config.system,
            messages=self._harness.messages,
            tools=tuple(self._harness.config.tools),
            usage_cutoff_index=self._usage_cutoff_index(),
        )

    def _usage_cutoff_index(self) -> int | None:
        """Return the transcript index before which provider usage is stale.

        Messages kept by the latest compaction predate it and their usage
        metadata reflects a larger, pre-compaction context. Only messages
        appended after the latest compaction entry carry usage measured on
        the current context.
        """
        if not self._state.compaction_entries:
            return None
        last_compaction = self._state.compaction_entries[-1]
        entries_by_id = {entry.id: entry for entry in self._state.entries}
        for index, entry_id in enumerate(self._state.context_entry_ids):
            entry = entries_by_id.get(entry_id)
            if entry is not None and entry.timestamp > last_compaction.timestamp:
                return index
        return None

    @property
    def context_usage(self) -> ContextUsageEstimate:
        """Return structured context accounting for the active provider context."""
        if self._context_usage_cache is None:
            self._context_usage_cache = estimate_context_usage(
                system=self._harness.config.system,
                messages=self._harness.messages,
                tools=tuple(self._harness.config.tools),
            )
        return self._context_usage_cache

    @property
    def system_prompt(self) -> str:
        """Return the effective system prompt sent to the model."""
        return self._harness.config.system

    @property
    def auto_compact_token_threshold(self) -> int | None:
        """Return the effective automatic compaction threshold, if any."""
        if not self._auto_compact_enabled:
            return None
        if self._auto_compact_token_threshold is not None:
            return self._auto_compact_token_threshold
        return auto_compaction_threshold_for_context_window(self.context_window_tokens)

    @property
    def context_window_tokens(self) -> int:
        """Return the active model's configured context window, or Forge's fallback."""
        provider = self._active_provider_config()
        if provider is None:
            return DEFAULT_CONTEXT_WINDOW_TOKENS
        return provider.context_windows.get(self.model, DEFAULT_CONTEXT_WINDOW_TOKENS)

    @property
    def command_registry(self) -> CommandRegistry:
        """Return the slash-command registry used by this session."""
        return self._command_registry

    @property
    def resource_diagnostics(self) -> tuple[ResourceDiagnostic, ...]:
        """Return non-fatal resource discovery diagnostics."""
        return self._resource_diagnostics

    @property
    def trust_result(self) -> TrustResult | None:
        """Return the trust decision used to load current project resources."""
        return self._trust_result

    def trust_status(self) -> str:
        """Return a concise project-trust status for ``/trust``."""
        if self._trust_result is None:
            return "Project trust has not been evaluated."
        return self._trust_result.describe()

    def set_trust_decision(self, decision: str) -> str:
        """Change trust policy without reloading the active system prompt."""
        normalized = decision.strip().casefold()
        key = str(canonical_path(self.cwd))
        if normalized == "once":
            self._session_trust_decisions[key] = "allow"
            return "Trust once enabled for this session; run /reload to apply it."
        if normalized in {"always", "parent", "deny"}:
            try:
                self._trust_store.set(
                    self.cwd,
                    "deny" if normalized == "deny" else "allow",
                    scope="parent" if normalized == "parent" else "folder",
                    lock_timeout_seconds=0.0,
                )
            except TrustError as exc:
                return f"Could not save trust decision: {exc}"
            self._session_trust_decisions.pop(key, None)
            return "Trust decision saved; run /reload to apply it."
        return "Usage: /trust [status|once|always|parent|deny]"

    @property
    def subagent_traces(self) -> dict[str, dict[str, JSONValue]]:
        """Return validated display traces on the active session branch."""
        return _subagent_trace_index(self._state.custom_entries)

    @property
    def usage_totals(self) -> UsageTotals:
        """Return the active-branch usage/cost aggregate."""
        return self._usage_totals_cache

    @property
    def session_id(self) -> str | None:
        """Return this session's manager id, if indexed."""
        return self._config.session_id

    @property
    def session_title(self) -> str | None:
        """Return this session's indexed human-friendly title, if named."""
        if self._config.session_id is None or self._config.session_manager is None:
            return None
        record = self._config.session_manager.get_session(self._config.session_id)
        if record is None:
            return None
        return record.title

    @property
    def session_manager(self) -> SessionManager | None:
        """Return the session manager, if available."""
        return self._config.session_manager

    @property
    def is_running(self) -> bool:
        """Return whether this session currently has an active agent run."""
        return self._run_active or self._harness.is_running

    @property
    def queued_messages(self) -> QueuedMessages:
        """Return queued steering and follow-up messages."""
        return self._harness.queued_messages

    @property
    def queued_steering_messages(self) -> tuple[str, ...]:
        """Return queued steering message text for UI display."""
        return tuple(message_text(message) for message in self._harness.queued_messages.steering)

    @property
    def queued_follow_up_messages(self) -> tuple[str, ...]:
        """Return queued follow-up message text for UI display."""
        return tuple(message_text(message) for message in self._harness.queued_messages.follow_up)

    @property
    def last_diagnostic_log_path(self) -> Path | None:
        """Return the last diagnostic log path written by this session."""
        return self._last_diagnostic_log_path

    def cancel(self) -> None:
        """Cancel the currently running agent turn, if any."""
        self._harness.cancel()
        run_task = self._run_task
        if run_task is not None and run_task is not asyncio.current_task() and not run_task.done():
            run_task.cancel()
        replace_task = self._goal_replace_task
        if (
            replace_task is not None
            and replace_task is not asyncio.current_task()
            and not replace_task.done()
        ):
            replace_task.cancel()

    async def _wait_for_run_settled(self, run_task: asyncio.Task[Any] | None = None) -> None:
        """Wait for the session consumer and inner graph to finish unwinding."""

        task = self._run_task if run_task is None else run_task
        if task is not None and task is not asyncio.current_task():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    # The waiter itself was cancelled.  Do not turn that
                    # cancellation into a successful Goal replacement.
                    raise
                # Cancellation is the expected result of ``cancel()``; the
                # task is nevertheless fully settled once this await returns.
                pass
            except Exception:  # noqa: BLE001 - teardown must still close providers
                pass
            await self._harness.wait_until_idle()
            return
        await self._harness.wait_until_idle()

    def cancel_pending_input(self) -> int:
        """Pair and close a pending questionnaire during teardown."""

        return self._harness.cancel_pending_input()

    def queue_update_event(self) -> QueueUpdateEvent:
        """Return the current queue state as an agent event."""
        return self._harness.queue_update_event()

    def clear_queued_messages(self) -> QueuedMessages:
        """Clear queued steering and follow-up messages."""
        return self._harness.clear_queues()

    def pop_latest_follow_up_message(self) -> str | None:
        """Remove and return the most recently queued follow-up message."""
        message = self._harness.pop_latest_follow_up()
        return None if message is None else message_text(message)

    def pop_latest_steering_message(self) -> str | None:
        """Remove and return the most recently queued steering message."""
        message = self._harness.pop_latest_steering()
        return None if message is None else message_text(message)

    def reload(self) -> CodingReloadSummary:
        """Reload local coding resources and project context for future turns."""
        if self._run_active or self.is_waiting_for_input:
            raise RuntimeError("Cannot reload resources while Forge is running")

        trust_result = self._resolve_trust_for(
            self.cwd,
        )
        effective_paths = resource_paths_with_cwd(
            self._config.resource_paths,
            self.cwd,
            trust_result=trust_result,
        )

        before_skills = _skill_signatures(self._skills)
        before_prompt_templates = _prompt_template_signatures(self._prompt_templates)
        before_context_files = _context_file_signatures(self._context_files)
        before_diagnostics = _diagnostic_signatures(self._resource_diagnostics)
        before_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=self._skills,
            context_files=self._context_files,
        )

        resources = _load_session_resources(effective_paths, self._config.context_files)

        current_tool_set = self._tool_set
        base_tool_set = (
            ToolSet(
                tuple(definition for definition in current_tool_set if definition.name != "task")
            )
            if self._subagent_runner is not None
            else current_tool_set
        )
        loaded_subagents = None
        replacement_specs = None
        replacement_runner = None
        replacement_task_definition: ToolDefinition | None = None
        after_subagent_profiles: tuple[CodingSubagentProfile, ...] = ()
        if self._subagent_runner is not None:
            loaded_subagents = load_subagent_profiles(
                effective_paths,
                available_tool_names=(tool.name for tool in base_tool_set.tools),
            )
            after_subagent_profiles = loaded_subagents.profiles
            replacement_specs = create_coding_subagent_specs(
                cwd=self._config.cwd,
                tools=base_tool_set,
                skills=resources.skills,
                context_files=resources.context_files,
                system=self._config.system,
                custom_system_prompt=self._config.custom_system_prompt,
                append_system_prompt=self._config.append_system_prompt,
                profiles=after_subagent_profiles,
            )
            replacement_runner = SubagentRunner(
                runtime_reader=lambda: SubagentRuntime(
                    provider=self._harness.config.provider,
                    model=self._harness.config.model,
                    runtime_context=self._harness.config.runtime_context,
                ),
                specs=replacement_specs,
            )
            replacement_task_definition = create_task_tool_definition(replacement_runner)

        after_skills = _skill_signatures(resources.skills)
        after_prompt_templates = _prompt_template_signatures(resources.prompt_templates)
        after_context_files = _context_file_signatures(resources.context_files)
        combined_diagnostics = (
            (*resources.diagnostics, *loaded_subagents.diagnostics)
            if loaded_subagents is not None
            else resources.diagnostics
        )
        after_diagnostics = _diagnostic_signatures(combined_diagnostics)
        after_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=resources.skills,
            context_files=resources.context_files,
        )

        before_subagents = _subagent_profile_signatures(self._subagent_profiles)
        after_subagents = _subagent_profile_signatures(after_subagent_profiles)
        subagents_changed = before_subagents != after_subagents

        replacement_tool_set = (
            base_tool_set.with_tools(replacement_task_definition)
            if replacement_task_definition is not None
            else base_tool_set
        )

        rebuilt_system_prompt: str | None = None
        system_prompt_rebuilt = False
        if self._config.system is None and (
            before_system_prompt_inputs != after_system_prompt_inputs or subagents_changed
        ):
            rebuilt_system_prompt = build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=self._config.cwd,
                    tools=replacement_tool_set,
                    skills=resources.skills,
                    custom_prompt=self._config.custom_system_prompt,
                    append_system_prompt=self._config.append_system_prompt,
                    context_files=resources.context_files,
                )
            )
            system_prompt_rebuilt = True

        if replacement_runner is not None:
            self._subagent_runner = replacement_runner
            self._tool_set = replacement_tool_set
            self._harness.config.tools = list(replacement_tool_set.tools)
            self._subagent_profiles = after_subagent_profiles
            self._invalidate_context_usage_cache()

        self._skills = resources.skills
        self._prompt_templates = resources.prompt_templates
        self._context_files = resources.context_files
        self._resource_diagnostics = combined_diagnostics
        self._resource_paths = effective_paths
        self._trust_result = trust_result
        self._config = replace(
            self._config,
            resource_paths=effective_paths,
            trust_result=trust_result,
            trust_store=self._trust_store,
        )
        if rebuilt_system_prompt is not None:
            self._harness.config.system = rebuilt_system_prompt
            self._invalidate_context_usage_cache()

        return CodingReloadSummary(
            skills=_category_summary(before_skills, after_skills),
            prompt_templates=_category_summary(
                before_prompt_templates,
                after_prompt_templates,
            ),
            context_files=_category_summary(before_context_files, after_context_files),
            diagnostics=_category_summary(before_diagnostics, after_diagnostics),
            system_prompt_rebuilt=system_prompt_rebuilt,
            subagents=_category_summary(before_subagents, after_subagents),
        )

    def _resolve_trust_for(
        self,
        cwd: Path,
    ) -> TrustResult:
        """Re-run trust metadata discovery for a session target cwd."""
        return resolve_project_trust(
            cwd,
            paths=self._resource_paths,
            store=self._trust_store,
            cli_override=self._config.trust_override,
            session_decision=self._session_trust_decisions.get(str(canonical_path(cwd))),
            interactive=False,
        )

    def reload_provider_settings(self) -> None:
        """Reload provider settings for login and model-selection flows."""
        if self.is_waiting_for_input:
            raise RuntimeError("Cannot reload providers while Forge is waiting for human input")
        if self._provider_settings is None:
            return
        previous_settings = self._provider_settings
        previous_thinking_level = self._thinking_level
        self._provider_settings = load_provider_settings(self._resource_paths.paths)
        try:
            self._sync_thinking_level_to_active_model()
            self._refresh_runtime_provider()
        except ProviderConfigError:
            self._provider_settings = previous_settings
            self._thinking_level = previous_thinking_level
            raise

    async def resume(self, session_id: str) -> str:
        """Replace this session's active state with another indexed session.

        The whole check -> load -> adopt sequence runs under the session
        switch lock so a concurrent ``prompt()``/``continue_()`` (which takes
        the same lock to start a run) can never interleave into the swap.
        """
        async with self._switch_lock:
            return await self._resume_locked(session_id)

    async def _resume_locked(self, session_id: str) -> str:
        if self._goal_replace_pending:
            raise RuntimeError("Goal replacement is in progress")
        if self.is_running and not self.is_waiting_for_input:
            raise RuntimeError(SESSION_SWITCH_RUNNING_MESSAGE)
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")
        record = manager.get_session(session_id)
        if record is None:
            raise ValueError(f"Unknown session: {session_id}")
        provider_name = self._provider_name
        runtime_provider_config = self._runtime_provider_config
        model = self.model
        restore_record_model = False
        if record.provider_name:
            if self._provider_settings is None:
                raise ProviderConfigError(
                    "Cannot resume session provider without provider settings: "
                    f"{record.provider_name}"
                )
            try:
                runtime_provider_config = self._provider_settings.get_provider(record.provider_name)
            except ProviderConfigError as exc:
                raise ProviderConfigError(
                    f"Session provider is not configured: {record.provider_name}"
                ) from exc
            provider_name = runtime_provider_config.name
            model = record.model
            restore_record_model = True
            validate_provider_model(runtime_provider_config, model)

        same_cwd = canonical_path(record.cwd) == canonical_path(self.cwd)
        trust_result = self._resolve_trust_for(record.cwd)
        replacement = await type(self).load(
            CodingSessionConfig(
                provider=self._harness.config.provider,
                model=model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                system=self._config.system,
                custom_system_prompt=self._config.custom_system_prompt,
                append_system_prompt=self._config.append_system_prompt,
                context_files=self._config.context_files if same_cwd else (),
                tools=self._config.tools,
                resource_paths=self._config.resource_paths,
                session_id=record.id,
                session_manager=manager,
                command_registry=self._command_registry,
                provider_name=provider_name,
                provider_settings=self._provider_settings,
                runtime_provider_config=runtime_provider_config,
                auto_compact_token_threshold=self._auto_compact_token_threshold,
                auto_compact_enabled=self._auto_compact_enabled,
                thinking_level=self._thinking_level,
                shell_command_prefix=self._config.shell_command_prefix,
                enable_subagents=self._config.enable_subagents,
                interactive=self._config.interactive,
                trust_result=trust_result,
                trust_override=self._config.trust_override,
                trust_store=self._trust_store,
                session_trust_decisions=self._session_trust_decisions,
            )
        )
        if restore_record_model:
            if runtime_provider_config is None:
                raise ProviderConfigError(f"Session provider is not configured: {provider_name}")
            validate_provider_model(runtime_provider_config, replacement.model)
        else:
            # Records without a provider_name keep the current session's
            # model.  ``load()`` already refreshed with the *target state's*
            # model: reuse that provider when the models agree, otherwise
            # retire it before refreshing with the active model so no
            # client is constructed twice.
            keep_loaded_provider = replacement.model == self.model
            replacement._harness.config.model = self.model
            replacement._sync_thinking_level_to_active_model()
            if not keep_loaded_provider:
                for provider in replacement._owned_providers:
                    await aclose_model(provider)
                replacement._owned_providers.clear()
                replacement._refresh_runtime_provider()
        await self._pause_goal_for_session_transition_locked()
        await self._adopt_replacement(replacement)
        return f"Resumed session: {record.id}"

    async def new_session(self) -> str:
        """Replace this session's active state with a pending unindexed session.

        Same switch lock contract as :meth:`resume`.
        """
        async with self._switch_lock:
            return await self._new_session_locked()

    async def _new_session_locked(self) -> str:
        if self._goal_replace_pending:
            raise RuntimeError("Goal replacement is in progress")
        if self.is_running and not self.is_waiting_for_input:
            raise RuntimeError(SESSION_SWITCH_RUNNING_MESSAGE)
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")
        provider_name = self._provider_name
        model = self.model
        runtime_provider_config = self._runtime_provider_config
        thinking_level = self._thinking_level
        if self._provider_settings is not None:
            selection = resolve_provider_selection(self._provider_settings)
            provider_name = selection.provider.name
            model = selection.model
            runtime_provider_config = selection.provider
            thinking_level = _coerced_thinking_level(
                selection.provider,
                model=model,
                current=self._thinking_level,
            )

        trust_result = self._resolve_trust_for(self.cwd)
        record = manager.prepare_session(
            cwd=self.cwd,
            model=model,
            provider_name=provider_name,
        )
        replacement = await type(self).load(
            replace(
                self._config,
                provider=self._harness.config.provider,
                model=record.model or model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                session_id=record.id,
                provider_name=provider_name,
                provider_settings=self._provider_settings,
                runtime_provider_config=runtime_provider_config,
                thinking_level=thinking_level,
                index_on_first_persist=True,
                trust_result=trust_result,
                trust_override=self._config.trust_override,
                trust_store=self._trust_store,
                session_trust_decisions=self._session_trust_decisions,
            )
        )
        await self._pause_goal_for_session_transition_locked()
        await self._adopt_replacement(replacement)
        return f"Started new session: {record.id}"

    async def _adopt_replacement(self, replacement: CodingSession) -> None:
        """Atomically take over a fully-loaded replacement session's state.

        ``resume()`` and ``new_session()`` build and validate the replacement
        first, so any construction/validation failure leaves this session
        untouched.  This method transfers every runtime-owned field -- config,
        state, harness, last parent, pending initial entries, resources,
        command registry, provider settings, runtime provider config, resource
        paths, compaction/thinking state, credential store, diagnostics and
        owned providers -- then closes the providers retired by the swap.
        """
        replacement_owned = {id(provider) for provider in replacement._owned_providers}
        replacement_harness_provider = replacement._harness.config.provider
        retired = [
            provider
            for provider in self._owned_providers
            if id(provider) not in replacement_owned
            and provider is not replacement_harness_provider
        ]
        self._config = replacement._config
        self._state = replacement._state
        self._harness = replacement._harness
        self._tool_set = replacement._tool_set
        self._last_parent_id = replacement._last_parent_id
        self._pending_initial_entries = replacement._pending_initial_entries
        self._skills = replacement._skills
        self._prompt_templates = replacement._prompt_templates
        self._context_files = replacement._context_files
        self._resource_diagnostics = replacement._resource_diagnostics
        self._command_registry = replacement._command_registry
        self._provider_name = replacement._provider_name
        self._provider_settings = replacement._provider_settings
        self._runtime_provider_config = replacement._runtime_provider_config
        self._resource_paths = replacement._resource_paths
        self._trust_result = replacement._trust_result
        self._trust_store = replacement._trust_store
        self._session_trust_decisions = replacement._session_trust_decisions
        self._auto_compact_token_threshold = replacement._auto_compact_token_threshold
        self._auto_compact_enabled = replacement._auto_compact_enabled
        self._thinking_level = replacement._thinking_level
        self._context_usage_cache = replacement._context_usage_cache
        self._usage_totals_cache = replacement._usage_totals_cache
        self._owned_providers = replacement._owned_providers
        self._diagnostic_logger = replacement._diagnostic_logger
        self._credential_store = replacement._credential_store
        self._last_diagnostic_log_path = replacement._last_diagnostic_log_path
        self._subagent_runner = replacement._subagent_runner
        self._subagent_profiles = replacement._subagent_profiles
        self._todos = replacement._todos
        self._goal_controller = replacement._goal_controller
        self._persisted_goal_snapshot = replacement._persisted_goal_snapshot
        self._goal_dirty = replacement._goal_dirty
        self._run_active = False
        self._run_task = None
        self._goal_replace_pending = False
        self._goal_replace_task = None

        async def retire_providers() -> None:
            for provider in retired:
                try:
                    await aclose_model(provider)
                except Exception as exc:  # noqa: BLE001 - retirement must not fail the swap
                    self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                        context=self._diagnostic_context(),
                        phase="adopt_replacement",
                        exc=exc,
                    )

        if retired:
            retirement = asyncio.create_task(retire_providers())
            try:
                await asyncio.shield(retirement)
            except asyncio.CancelledError:
                # Adoption has already committed. Finish retiring providers so
                # cancellation cannot detach a live client from session ownership.
                await retirement
                raise

    async def compact(self, instructions: str | None = None) -> str:
        """Generate a manual compaction summary and rebuild active context.

        Like automatic compaction, manual compaction preserves the most
        recent ``DEFAULT_COMPACTION_KEEP_RECENT_TOKENS`` so the summary call
        stays bounded; a session without replaceable history reports a no-op.
        """
        if self.is_waiting_for_input:
            raise RuntimeError("Cannot compact while Forge is waiting for human input")
        plan = self._recent_preserving_compaction_plan()
        if plan is None:
            return "No context to compact."
        usage_sink: list[object] = []
        try:
            summary, details, summary_usage = await self._generate_compaction_summary(
                plan.messages_to_summarize,
                custom_instructions=instructions,
                turn_prefix_messages=plan.turn_prefix_messages,
                usage_sink=usage_sink,
            )
        except Exception:
            if usage_sink:
                await self._persist_helper_usage_messages(usage_sink, purpose="compaction")
            raise
        compaction = await self._append_compaction(
            summary,
            replace_entry_ids=plan.replace_entry_ids,
            details=details,
            tokens_before=self.context_token_estimate,
            usage_messages=summary_usage,
        )
        return f"Compacted {len(compaction.replaces_entry_ids)} context entries."

    async def aclose(self) -> None:
        """Close runtime providers created by this coding session.

        Every provider created by this session is closed exactly once; a
        failure closing one provider does not stop the remaining providers
        from closing.  Collected failures are raised together afterwards.
        """
        replacement_task = self._goal_replace_task
        if self.is_running and not self.is_waiting_for_input:
            run_task = self._run_task
            if self.goal is not None and self.goal.status == "active":
                self._goal_controller.pause(goal_id=self.goal.id, reason="cancelled")
                self._goal_dirty = True
            self.cancel()
            await self._wait_for_run_settled(run_task)
        if replacement_task is not None and replacement_task is not asyncio.current_task():
            replacement_task.cancel()
            await self._wait_for_run_settled(replacement_task)
        if self.is_waiting_for_input:
            await self._close_pending_human_input_locked()
            if self.goal is not None and self.goal.status == "active":
                self._goal_controller.pause(goal_id=self.goal.id, reason="cancelled")
                self._goal_dirty = True
        if self.goal is not None and self.goal.status == "active":
            self._goal_controller.pause(goal_id=self.goal.id, reason="cancelled")
            self._goal_dirty = True
        if self._goal_dirty:
            await self._persist_goal_update()
        errors: list[Exception] = []
        for provider in self._owned_providers:
            try:
                await aclose_model(provider)
            except Exception as exc:  # noqa: BLE001 - one bad provider must not leak others
                errors.append(exc)
        self._owned_providers.clear()
        if errors:
            raise ExceptionGroup("Failed to close session providers", errors)

    async def _close_pending_human_input_locked(self) -> None:
        """Persist a synthetic result before abandoning a paused HITL graph."""

        if not self.is_waiting_for_input:
            return
        before = len(self._harness.messages)
        self._harness.cancel_pending_input()
        if len(self._harness.messages) > before:
            await self._persist_messages_since(before)

    async def _pause_goal_for_session_transition_locked(self) -> None:
        """Stop an active Goal before replacing this session instance."""

        await self._close_pending_human_input_locked()
        if self.goal is not None and self.goal.status == "active":
            self._goal_controller.pause(goal_id=self.goal.id, reason="cancelled")
            self._goal_dirty = True
            await self._persist_goal_update()

    def handle_command(self, text: str) -> CommandResult:
        """Handle coding-session slash commands.

        Prompt-template slash commands are expansion directives, so they remain
        unhandled here and flow through `prompt()` for on-the-fly replacement.
        """
        if expand_prompt_template_command(text, self._prompt_templates) is not None:
            return CommandResult(handled=False)
        return self._command_registry.execute(self, text)

    def ensure_session_indexed(self) -> None:
        """Persist pending session metadata and add this session to the resume index."""
        if self._config.session_id is None or self._config.session_manager is None:
            return
        if self._config.session_manager.get_session(self._config.session_id) is None:
            self._config.session_manager.create_session(
                cwd=self.cwd,
                model=self.model,
                provider_name=self.provider_name,
                session_id=self._config.session_id,
            )
        self._config = replace(self._config, index_on_first_persist=False)
        self._ensure_session_file_initialized()

    def expand_prompt_text(self, text: str) -> str:
        """Expand prompt text using loaded markdown resources."""
        expanded_prompt = expand_prompt_template_command(text, self._prompt_templates)
        if expanded_prompt is not None:
            return expanded_prompt
        expanded_skill = expand_skill_command(text, self._skills)
        return expanded_skill if expanded_skill is not None else text

    async def run_terminal_command(
        self,
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        """Run a shell command in the session cwd, optionally adding output to context."""
        normalized_command = command.strip()
        if not normalized_command:
            raise ValueError("Terminal command cannot be empty")

        bash_tool = create_bash_tool(
            cwd=self.cwd,
            shell_command_prefix=self._config.shell_command_prefix,
        )
        result = await bash_tool.execute({"command": normalized_command})
        exit_code = None
        if result.data is not None:
            raw_exit_code = result.data.get("exit_code")
            exit_code = raw_exit_code if isinstance(raw_exit_code, int) else None

        if add_to_context:
            before_count = len(self._harness.messages)
            self._harness.append_message(
                HumanMessage(
                    content=_terminal_command_context_message(
                        normalized_command,
                        result.content,
                    )
                )
            )
            self._invalidate_context_usage_cache()
            await self._persist_messages_since(before_count)

        return TerminalCommandResult(
            command=normalized_command,
            output=result.content,
            exit_code=exit_code,
            ok=result.ok,
            added_to_context=add_to_context,
        )

    async def apply_goal_action(
        self,
        action: GoalCommandAction | object,
    ) -> AsyncIterator[AgentEvent]:
        """Apply one slash/TUI Goal intent without adding transcript messages."""

        raw_action = getattr(action, "action", getattr(action, "kind", None))
        objective = getattr(action, "objective", None)
        expected_goal_id = getattr(action, "goal_id", None)
        replace_requested = bool(getattr(action, "replace", False))
        if not isinstance(action, GoalCommandAction):
            if not isinstance(raw_action, str):
                raise ValueError("Goal action is missing its action name")
            action = GoalCommandAction(
                action=cast(
                    Literal["start", "status", "pause", "resume", "edit", "clear"],
                    raw_action,
                ),
                objective=objective,
                goal_id=expected_goal_id if isinstance(expected_goal_id, str) else None,
                replace=replace_requested,
            )

        intent = action
        assert isinstance(intent, GoalCommandAction)
        if intent.action == "status":
            yield GoalUpdateEvent(goal=self.goal)
            return
        if intent.replace:
            async for event in self._replace_goal_and_run(intent):
                yield event
            return

        running_stop = intent.action in {"pause", "clear"}
        reserved_run = False
        run_task_to_settle: asyncio.Task[Any] | None = None
        continuation_goal_id: str | None = None
        try:
            async with self._switch_lock:
                current_goal = self.goal
                if intent.goal_id is not None and (
                    current_goal is None or current_goal.id != intent.goal_id
                ):
                    raise RuntimeError("Goal changed while the manager was open; reopen /goal")
                if self._goal_replace_pending:
                    raise RuntimeError("Goal replacement is in progress")
                if self.is_waiting_for_input:
                    if not running_stop:
                        raise RuntimeError(
                            "Answer or cancel the questionnaire before changing this Goal"
                        )
                    await self._close_pending_human_input_locked()
                if self.is_running and not running_stop:
                    raise RuntimeError("Cannot change Goal while Forge is running")

                if intent.action == "start":
                    self._goal_controller.start(intent.objective or "")
                elif intent.action == "pause":
                    if self.goal is None:
                        raise RuntimeError("no Goal is active")
                    self._goal_controller.pause(goal_id=self.goal.id, reason="user")
                elif intent.action == "resume":
                    if self.goal is None:
                        raise RuntimeError("no Goal is active")
                    self._goal_controller.resume(goal_id=self.goal.id)
                elif intent.action == "edit":
                    if self.goal is None:
                        raise RuntimeError("no Goal is active")
                    self._goal_controller.edit(intent.objective or "", goal_id=self.goal.id)
                elif intent.action == "clear":
                    self._goal_controller.clear(
                        goal_id=self.goal.id if self.goal is not None else None
                    )
                else:  # pragma: no cover - GoalCommandAction validates the Literal
                    raise ValueError(f"Unsupported Goal action: {intent.action}")

                self._goal_dirty = True
                if running_stop:
                    run_task_to_settle = self._run_task
                    self._harness.cancel()
                persisted_event = await self._persist_goal_update()

                # Reserve the managed-run ownership before exposing the first
                # Goal event.  A concurrent prompt therefore queues/rejects
                # instead of racing the action's continuation.
                should_continue = intent.action in {"start", "resume"} or (
                    intent.action == "edit"
                    and self.goal is not None
                    and self.goal.status == "active"
                )
                if should_continue:
                    continuation_goal_id = self.goal.id if self.goal is not None else None
                    self._run_active = True
                    self._run_task = asyncio.current_task()
                    reserved_run = True

            if running_stop:
                await self._wait_for_run_settled(run_task_to_settle)

            if persisted_event is not None:
                yield persisted_event

            if should_continue:
                async with self._switch_lock:
                    current_goal = self.goal
                    can_continue = (
                        continuation_goal_id is not None
                        and current_goal is not None
                        and current_goal.id == continuation_goal_id
                        and current_goal.status == "active"
                        and not self.is_waiting_for_input
                        and not self._goal_replace_pending
                    )
                if can_continue:
                    async for run_event in self.continue_(
                        _run_already_owned=True,
                        _expected_goal_id=continuation_goal_id,
                    ):
                        yield run_event
        finally:
            if reserved_run and self._run_active:
                async with self._switch_lock:
                    self._run_active = False
                    if self._run_task is asyncio.current_task():
                        self._run_task = None

    async def _replace_goal_and_run(
        self,
        intent: GoalCommandAction,
    ) -> AsyncIterator[AgentEvent]:
        """Cancel, settle, clear, and start a replacement as one session intent."""

        run_task: asyncio.Task[Any] | None = None
        started_goal_id: str | None = None
        reserved_run = False
        try:
            async with self._switch_lock:
                current_goal = self.goal
                if intent.goal_id is not None and (
                    current_goal is None or current_goal.id != intent.goal_id
                ):
                    raise RuntimeError("Goal changed while the manager was open; reopen /goal")
                if self._goal_replace_pending:
                    raise RuntimeError("Goal replacement is already in progress")
                self._goal_replace_pending = True
                self._goal_replace_task = asyncio.current_task()
                if self.is_waiting_for_input:
                    await self._close_pending_human_input_locked()
                elif self.is_running:
                    run_task = self._run_task
                    self.cancel()

            await self._wait_for_run_settled(run_task)

            async with self._switch_lock:
                current_goal = self.goal
                if intent.goal_id is not None and (
                    current_goal is None or current_goal.id != intent.goal_id
                ):
                    raise RuntimeError("Goal changed while the manager was open; reopen /goal")
                self._goal_controller.clear(
                    goal_id=current_goal.id if current_goal is not None else None
                )
                self._goal_dirty = True
                cleared_event = await self._persist_goal_update()
                self._goal_controller.start(intent.objective or "")
                started_goal_id = self.goal.id if self.goal is not None else None
                self._goal_dirty = True
                started_event = await self._persist_goal_update()
                self._run_active = True
                self._run_task = asyncio.current_task()
                reserved_run = True

            if cleared_event is not None:
                yield cleared_event
            if started_event is not None:
                yield started_event
            async for event in self.continue_(
                _run_already_owned=True,
                _expected_goal_id=started_goal_id,
            ):
                yield event
        finally:
            async with self._switch_lock:
                self._goal_replace_pending = False
                if self._goal_replace_task is asyncio.current_task():
                    self._goal_replace_task = None
                if reserved_run and self._run_task is asyncio.current_task():
                    self._run_active = False
                    self._run_task = None

    async def _consume_harness_events(
        self,
        events: AsyncIterator[AgentEvent],
        *,
        stats: _GoalRunStats,
        trace_tool_call_ids: set[str],
        context: AgentCallDiagnosticContext,
        phase: str,
    ) -> AsyncIterator[AgentEvent]:
        """Consume one fully settled Harness stream and persist product state."""

        self._invalidate_context_usage_cache()
        async for event in events:
            if isinstance(event, RetryEvent):
                attempt: dict[str, JSONValue] = {
                    "attempt": event.attempt,
                    "max_attempts": event.max_attempts,
                    "delay_seconds": event.delay_seconds,
                    "message": redact_model_error(event.message),
                }
                if event.data is not None:
                    for key in ("kind", "status_code"):
                        value = event.data.get(key)
                        if isinstance(value, (str, int)) and not isinstance(value, bool):
                            attempt[key] = value
                if len(stats.retry_attempts) < 4:
                    stats.retry_attempts.append(attempt)
            if isinstance(event, ToolExecutionStartEvent):
                stats.had_tool_calls = True
            if isinstance(event, ToolExecutionUpdateEvent):
                persisted = await self._persist_subagent_trace_update(
                    event,
                    persisted_count=stats.persisted_count,
                    persisted_tool_call_ids=trace_tool_call_ids,
                )
                if persisted is None:
                    if _is_subagent_trace_update(event):
                        continue
                else:
                    stats.persisted_count = persisted
            if isinstance(event, MessageEndEvent):
                if isinstance(event.message, AIMessage):
                    stats.final_error = None
                    stats.nonrecoverable_error = False
                    stats.overflow_event = None
                if isinstance(event.message, AIMessage):
                    stats.final_assistant_text = message_text(event.message)
                stats.persisted_count = await self._persist_messages_since(stats.persisted_count)
                if not stats.auto_name_attempted and isinstance(event.message, HumanMessage):
                    stats.auto_name_attempted = True
                    await self._try_auto_name_session(message_text(event.message), context=context)
            if isinstance(event, TodoUpdateEvent):
                stats.persisted_count = await self._persist_todo_update(
                    event,
                    stats.persisted_count,
                )
            if isinstance(event, ToolExecutionEndEvent):
                self._invalidate_context_usage_cache()
                goal_event = await self._persist_goal_update()
                # A successful Goal terminal tool ends the managed run.  The
                # LangChain graph itself may otherwise ask the model again
                # after the tool result (a scripted or misbehaving model can
                # repeat the same terminal call until the graph turn limit).
                # Cancelling here keeps the single graph bounded while
                # preserving the terminal Goal snapshot and paired result.
                if self.goal is not None and self.goal.status in {"blocked", "complete"}:
                    stats.terminal_goal_stop = True
                    self._harness.request_cancel()
                if goal_event is not None:
                    # Persist first, then expose the product event.
                    yield goal_event
            if (
                isinstance(event, ErrorEvent)
                and event.recoverable
                and event.message == "Agent run cancelled"
                and stats.terminal_goal_stop
            ):
                # Terminal Goal tools cooperatively stop the graph so it cannot
                # start another model turn.  That internal cancellation is not
                # a user-visible interruption.
                continue
            if isinstance(event, ErrorEvent) and not event.recoverable:
                stats.nonrecoverable_error = True
                stats.final_error = event
                self._last_diagnostic_log_path = self._diagnostic_logger.log_error_event(
                    context=context,
                    phase=phase,
                    event=event,
                )
                if _is_context_overflow_error(event):
                    stats.overflow_event = event
            yield event
        stats.persisted_count = await self._persist_messages_since(stats.persisted_count)

    async def _persist_goal_transition_event(self) -> AsyncIterator[AgentEvent]:
        event = await self._persist_goal_update()
        if event is not None:
            yield event

    async def _persist_turn_error(self, stats: _GoalRunStats) -> None:
        """Persist one bounded model-failure audit row for a settled run."""

        if stats.audit_persisted or (not stats.retry_attempts and stats.final_error is None):
            return
        if stats.final_error is None:
            outcome = "recovered"
        else:
            classification = classify_model_error(RuntimeError(stats.final_error.message))
            kind = (
                stats.final_error.data.get("kind")
                if stats.final_error.data is not None
                else None
            )
            if kind == "overflow" or classification.kind == "overflow":
                outcome = "overflow"
            elif stats.retry_attempts and (kind == "transient" or classification.retryable):
                outcome = "exhausted"
            else:
                outcome = "non_retryable"
        data: dict[str, JSONValue] = {
            "version": 1,
            "outcome": outcome,
            "attempts": list(stats.retry_attempts),
        }
        if stats.final_error is not None:
            error_data: dict[str, JSONValue] = {
                "message": redact_model_error(stats.final_error.message),
                "recoverable": False,
            }
            data["error"] = error_data
            if stats.final_error.data is not None:
                kind = stats.final_error.data.get("kind")
                if isinstance(kind, str):
                    error_data["kind"] = kind
        entry = CustomEntry(
            parent_id=self._last_parent_id,
            namespace=TURN_ERROR_NAMESPACE,
            data=data,
        )
        await self._append_session_entry(entry)
        await self._append_session_entry(LeafEntry(parent_id=entry.id, entry_id=entry.id))
        self._last_parent_id = entry.id
        await self._refresh_persisted_state(leaf_id=entry.id)
        stats.audit_persisted = True

    async def _settle_goal_after_run(
        self,
        stats: _GoalRunStats,
        *,
        automatic: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Apply cancellation, errors, and progress safety checks after a run."""

        if self.goal is None:
            return
        if self._harness.was_last_run_interrupted:
            if self.goal.status == "active":
                self._goal_controller.pause(goal_id=self.goal.id, reason="cancelled")
                self._goal_dirty = True
            async for event in self._persist_goal_transition_event():
                yield event
            return
        if self.is_waiting_for_input:
            await self._persist_turn_error(stats)
            return
        if stats.nonrecoverable_error:
            if self.goal is not None and self.goal.status == "active":
                self._goal_controller.pause(goal_id=self.goal.id, reason="error")
                self._goal_dirty = True
            async for event in self._persist_goal_transition_event():
                yield event
            return
        if self.goal is None or self.goal.status != "active":
            return
        self._goal_controller.record_output(
            stats.final_assistant_text,
            goal_id=self.goal.id,
            had_tool_calls=stats.had_tool_calls,
        )
        self._goal_dirty = True
        if (
            automatic
            and self.goal is not None
            and self.goal.status == "active"
            and self.goal.automatic_runs >= GOAL_MAX_AUTOMATIC_RUNS
        ):
            self._goal_controller.pause(goal_id=self.goal.id, reason="automatic_limit")
            self._goal_dirty = True
        async for event in self._persist_goal_transition_event():
            yield event

    async def _run_goal_continuations(
        self,
        *,
        context: AgentCallDiagnosticContext,
        trace_tool_call_ids: set[str],
    ) -> AsyncIterator[AgentEvent]:
        """Continue an active Goal only after the previous run has settled."""

        while self.goal is not None and self.goal.status == "active":
            if self.is_waiting_for_input or self._harness.has_queued_messages():
                return
            if self._harness.was_last_run_interrupted:
                return
            snapshot = self.goal
            if snapshot is None:
                return
            if snapshot.automatic_runs >= GOAL_MAX_AUTOMATIC_RUNS:
                self._goal_controller.pause(goal_id=snapshot.id, reason="automatic_limit")
                self._goal_dirty = True
                async for event in self._persist_goal_transition_event():
                    yield event
                return

            # Count the coordinator-owned invocation, but keep the Goal active
            # for the 25th call so its dynamic prompt/tools remain available.
            self._goal_controller.record_automatic_run(
                goal_id=snapshot.id,
                pause_at_limit=False,
            )
            self._goal_dirty = True
            stats = _GoalRunStats(persisted_count=len(self._harness.messages))
            events = self._harness.continue_()
            async for event in self._consume_harness_events(
                events,
                stats=stats,
                trace_tool_call_ids=trace_tool_call_ids,
                context=context,
                phase="goal_agent_loop",
            ):
                yield event
            await self._persist_turn_error(stats)
            async for event in self._settle_goal_after_run(stats, automatic=True):
                yield event
            if self.goal is None or self.goal.status != "active":
                return
            if self.is_waiting_for_input or self._harness.has_queued_messages():
                return

    async def _run_prompt_turn(
        self,
        content: str,
        *,
        context: AgentCallDiagnosticContext,
        trace_tool_call_ids: set[str],
    ) -> AsyncIterator[AgentEvent]:
        """Run one user prompt, including overflow retry and Goal settling."""

        # A fresh user turn is an explicit progress signal.  Start a new
        # no-progress streak without discarding the lifetime automatic-run
        # counter.  Persist the reset before invoking the model so a failed or
        # cancelled prompt cannot resurrect the previous streak on replay.
        if self.goal is not None and self.goal.status == "active" and self.goal.no_progress_runs:
            self._goal_controller.reset_no_progress(goal_id=self.goal.id)
            self._goal_dirty = True
            async for event in self._persist_goal_transition_event():
                yield event

        stats = _GoalRunStats(persisted_count=len(self._harness.messages))
        events = self._harness.prompt(content)
        async for event in self._consume_harness_events(
            events,
            stats=stats,
            trace_tool_call_ids=trace_tool_call_ids,
            context=context,
            phase="agent_loop",
        ):
            yield event

        if self.is_waiting_for_input:
            await self._persist_turn_error(stats)
            return

        overflow_recovered = True
        if stats.overflow_event is not None:
            compacted = await self._try_overflow_compact(context=context)
            if compacted:
                stats.final_error = None
                stats.nonrecoverable_error = False
                stats.overflow_event = None
                async with self._switch_lock:
                    retry_events = self._harness.continue_()
                async for event in self._consume_harness_events(
                    retry_events,
                    stats=stats,
                    trace_tool_call_ids=trace_tool_call_ids,
                    context=context,
                    phase="agent_loop_retry",
                ):
                    yield event
                overflow_recovered = stats.overflow_event is None
            else:
                overflow_recovered = False

        await self._persist_turn_error(stats)

        async for event in self._settle_goal_after_run(stats, automatic=False):
            yield event

        if self.is_waiting_for_input or not overflow_recovered or stats.nonrecoverable_error:
            return
        await self._try_auto_compact(context=context, phase="auto_compact_after_prompt")
        if (
            self.goal is not None
            and self.goal.status == "active"
            and not stats.nonrecoverable_error
        ):
            async for event in self._run_goal_continuations(
                context=context,
                trace_tool_call_ids=trace_tool_call_ids,
            ):
                yield event

    async def prompt(
        self,
        content: str,
        *,
        streaming_behavior: StreamingBehavior | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Append a user prompt, run it, and continue an active Goal safely."""

        context = self._diagnostic_context()
        trace_tool_call_ids = set(self.subagent_traces)
        run_started = False
        harness_started = False
        try:
            async with self._switch_lock:
                context = self._diagnostic_context()
                if self.is_waiting_for_input:
                    raise RuntimeError(
                        "CodingSession is waiting for human input; "
                        "answer or cancel the questionnaire."
                    )
                if self._goal_replace_pending:
                    raise RuntimeError("Goal replacement is in progress")
                try:
                    expanded_content = self.expand_prompt_text(content)
                except ResourceError:
                    raise
                except Exception as exc:
                    self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                        context=context,
                        phase="expand_prompt",
                        exc=exc,
                    )
                    raise

                if self.is_running:
                    if streaming_behavior == "steer":
                        queued_event = self._harness.steer(expanded_content)
                    elif streaming_behavior == "follow_up":
                        queued_event = self._harness.follow_up(expanded_content)
                    else:
                        raise RuntimeError(
                            "CodingSession is already running; pass streaming_behavior "
                            "to queue a message."
                        )
                else:
                    queued_event = None
                    self._run_active = True
                    self._run_task = asyncio.current_task()
                    run_started = True
                    await self._try_auto_compact(
                        context=context,
                        phase="auto_compact_before_prompt",
                    )
                    harness_started = True

            if queued_event is not None:
                yield queued_event
                return

            async for event in self._run_prompt_turn(
                expanded_content,
                context=context,
                trace_tool_call_ids=trace_tool_call_ids,
            ):
                yield event
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="agent_loop",
                exc=exc,
            )
            raise
        finally:
            try:
                if harness_started and self._harness.was_last_run_interrupted:
                    await self._persist_messages_since(len(self._state.messages))
                    if self.goal is not None and self.goal.status == "active":
                        self._goal_controller.pause(goal_id=self.goal.id, reason="cancelled")
                        self._goal_dirty = True
                if self._goal_dirty:
                    await self._persist_goal_update()
            finally:
                if run_started:
                    async with self._switch_lock:
                        self._run_active = False
                        if self._run_task is asyncio.current_task():
                            self._run_task = None

    async def continue_(
        self,
        *,
        _run_already_owned: bool = False,
        _expected_goal_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Continue from the restored transcript without appending a user message."""

        context = self._diagnostic_context()
        trace_tool_call_ids = set(self.subagent_traces)
        persisted_count = len(self._harness.messages)
        run_started = False
        harness_started = False
        try:
            async with self._switch_lock:
                current_goal = self.goal
                if _expected_goal_id is not None and (
                    current_goal is None
                    or current_goal.id != _expected_goal_id
                    or current_goal.status != "active"
                ):
                    return
                context = self._diagnostic_context()
                if self.is_waiting_for_input:
                    raise RuntimeError(
                        "CodingSession is waiting for human input; "
                        "answer or cancel the questionnaire."
                    )
                if self._goal_replace_pending and not _run_already_owned:
                    raise RuntimeError("Goal replacement is in progress")
                if self.is_running and not _run_already_owned:
                    raise RuntimeError("CodingSession is already running")
                if not _run_already_owned:
                    self._run_active = True
                self._run_task = asyncio.current_task()
                run_started = True
                harness_started = True
                persisted_count = len(self._harness.messages)
                events = self._harness.continue_()

            stats = _GoalRunStats(persisted_count=persisted_count)
            async for event in self._consume_harness_events(
                events,
                stats=stats,
                trace_tool_call_ids=trace_tool_call_ids,
                context=context,
                phase="agent_loop",
            ):
                yield event
            await self._persist_turn_error(stats)
            async for event in self._settle_goal_after_run(stats, automatic=False):
                yield event
            if not self.is_waiting_for_input and not stats.nonrecoverable_error:
                await self._try_auto_compact(
                    context=context,
                    phase="auto_compact_after_continue",
                )
                if self.goal is not None and self.goal.status == "active":
                    async for event in self._run_goal_continuations(
                        context=context,
                        trace_tool_call_ids=trace_tool_call_ids,
                    ):
                        yield event
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="agent_loop",
                exc=exc,
            )
            raise
        finally:
            try:
                if harness_started and self._harness.was_last_run_interrupted:
                    await self._persist_messages_since(len(self._state.messages))
                    if self.goal is not None and self.goal.status == "active":
                        self._goal_controller.pause(goal_id=self.goal.id, reason="cancelled")
                        self._goal_dirty = True
                if self._goal_dirty:
                    await self._persist_goal_update()
            finally:
                if run_started:
                    async with self._switch_lock:
                        self._run_active = False
                        if self._run_task is asyncio.current_task():
                            self._run_task = None

    async def respond_to_human_input(
        self,
        response: str | Mapping[str, JSONValue] | Sequence[Mapping[str, JSONValue]],
    ) -> AsyncIterator[AgentEvent]:
        """Resume the paused graph, then continue an active Goal if safe."""

        context = self._diagnostic_context()
        trace_tool_call_ids = set(self.subagent_traces)
        persisted_count = len(self._harness.messages)
        run_started = False
        harness_started = False
        try:
            async with self._switch_lock:
                context = self._diagnostic_context()
                if self.is_running:
                    raise RuntimeError("CodingSession is already running")
                if self._goal_replace_pending:
                    raise RuntimeError("Goal replacement is in progress")
                if not self.is_waiting_for_input:
                    raise RuntimeError("CodingSession is not waiting for human input")
                self._run_active = True
                self._run_task = asyncio.current_task()
                run_started = True
                harness_started = True
                persisted_count = len(self._harness.messages)
                events = self._harness.respond_to_human_input(response)

            stats = _GoalRunStats(persisted_count=persisted_count)
            async for event in self._consume_harness_events(
                events,
                stats=stats,
                trace_tool_call_ids=trace_tool_call_ids,
                context=context,
                phase="human_input_resume",
            ):
                yield event
            await self._persist_turn_error(stats)
            async for event in self._settle_goal_after_run(stats, automatic=False):
                yield event
            if (
                not self.is_waiting_for_input
                and not stats.nonrecoverable_error
                and self.goal is not None
            ):
                async for event in self._run_goal_continuations(
                    context=context,
                    trace_tool_call_ids=trace_tool_call_ids,
                ):
                    yield event
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="human_input_resume",
                exc=exc,
            )
            raise
        finally:
            try:
                if harness_started and self._harness.was_last_run_interrupted:
                    await self._persist_messages_since(len(self._state.messages))
                    if self.goal is not None and self.goal.status == "active":
                        self._goal_controller.pause(goal_id=self.goal.id, reason="cancelled")
                        self._goal_dirty = True
                if self._goal_dirty:
                    await self._persist_goal_update()
            finally:
                if run_started:
                    async with self._switch_lock:
                        self._run_active = False
                        if self._run_task is asyncio.current_task():
                            self._run_task = None

    def _diagnostic_context(self) -> AgentCallDiagnosticContext:
        return AgentCallDiagnosticContext(
            provider_name=self._provider_name,
            model=self.model,
            cwd=self.cwd,
            session_id=self.session_id,
            run_id=new_agent_call_run_id(),
        )

    async def _persist_loaded_interrupted_tool_repairs(self) -> None:
        """Persist repairs for loaded sessions with dangling tool calls.

        Older Forge builds repaired interrupted tool-call transcripts only in the
        in-memory harness. If the app was later resumed from JSONL, the synthetic
        tool result was absent and providers rejected the whole transcript. Repair
        the active branch on load so resume/tree branches are durable and
        provider-safe.
        """
        repair = _interrupted_tool_repair_plan(
            self._state.messages,
            context_entry_ids=self._state.context_entry_ids,
        )
        if repair is None:
            return

        parent_id, suffix = repair
        for message in suffix:
            entry = MessageEntry(parent_id=parent_id, message=message)
            await self._append_session_entry(entry)
            parent_id = entry.id
        leaf = LeafEntry(parent_id=parent_id, entry_id=parent_id)
        await self._append_session_entry(leaf)
        self._last_parent_id = parent_id
        await self._refresh_persisted_state(leaf_id=parent_id)
        self._harness = AgentHarness(
            self._harness.config,
            messages=self._state.messages,
        )

    def _usage_provider_config(self) -> object | None:
        """Return the resolved catalog config used for the current call."""

        return self._active_provider_config() or self._runtime_provider_config

    async def _append_usage_record(
        self,
        record: UsageRecord,
        *,
        parent_id: str | None = None,
    ) -> CustomEntry:
        """Append one usage fact without adding a competing transcript node."""

        entry = CustomEntry(
            parent_id=self._last_parent_id if parent_id is None else parent_id,
            namespace=USAGE_NAMESPACE,
            data=record.to_data(),
        )
        await self._append_session_entry(entry)
        return entry

    async def _append_usage_message(
        self,
        message: object,
        *,
        purpose: UsagePurpose,
        parent_id: str | None = None,
    ) -> CustomEntry:
        record = usage_record_from_message(
            message,
            purpose=purpose,
            provider=self.provider_name,
            requested_model=self.model,
            provider_config=self._usage_provider_config(),
        )
        return await self._append_usage_record(record, parent_id=parent_id)

    async def _persist_helper_usage_messages(
        self,
        messages: Sequence[object],
        *,
        purpose: UsagePurpose,
        parent_id: str | None = None,
    ) -> None:
        """Persist completed helper calls even when their surrounding flow fails."""

        next_parent = self._last_parent_id if parent_id is None else parent_id
        for message in messages:
            usage_entry = await self._append_usage_message(
                message,
                purpose=purpose,
                parent_id=next_parent,
            )
            next_parent = usage_entry.id
        leaf = LeafEntry(parent_id=next_parent, entry_id=next_parent)
        await self._append_session_entry(leaf)
        self._last_parent_id = next_parent
        await self._refresh_persisted_state(leaf_id=next_parent)

    async def _persist_messages_since(self, persisted_count: int) -> int:
        """Persist completed harness messages after ``persisted_count``.

        Message lifecycle events are the durable-message boundary. Each persisted
        message advances the append-only tree and records a leaf pointer so tree
        navigation can observe the current branch while a run is still active.
        """
        new_messages = self._harness.messages[persisted_count:]
        if not new_messages:
            return persisted_count

        for message in new_messages:
            entry = MessageEntry(parent_id=self._last_parent_id, message=message)
            await self._append_session_entry(entry)
            self._last_parent_id = entry.id
            # Usage is the next active-tree node after its billable AI
            # MessageEntry.  LeafEntry remains only a storage pointer.
            if isinstance(message, AIMessage):
                usage_entry = await self._append_usage_message(
                    message,
                    purpose="agent",
                    parent_id=entry.id,
                )
                if usage_entry is not None:
                    self._last_parent_id = usage_entry.id
            leaf = LeafEntry(parent_id=self._last_parent_id, entry_id=self._last_parent_id)
            await self._append_session_entry(leaf)

        await self._refresh_persisted_state(leaf_id=self._last_parent_id)
        self._invalidate_context_usage_cache()
        return persisted_count + len(new_messages)

    async def _persist_todo_update(
        self,
        event: TodoUpdateEvent,
        persisted_count: int,
    ) -> int:
        """Persist the message/tool boundary before a versioned todo snapshot."""

        persisted_count = await self._persist_messages_since(persisted_count)
        if event.todos == self._todos:
            return persisted_count
        entry = CustomEntry(
            parent_id=self._last_parent_id,
            namespace=TODO_NAMESPACE,
            data=todo_entry_data(event.todos),
        )
        await self._append_session_entry(entry)
        leaf = LeafEntry(parent_id=entry.id, entry_id=entry.id)
        await self._append_session_entry(leaf)
        self._last_parent_id = entry.id
        self._todos = event.todos
        await self._refresh_persisted_state(leaf_id=entry.id)
        return persisted_count

    async def _persist_goal_update(self) -> GoalUpdateEvent | None:
        """Persist the current Goal snapshot before projecting its event."""

        snapshot = self.goal
        if snapshot == self._persisted_goal_snapshot and not self._goal_dirty:
            return None
        data = goal_tombstone_data() if snapshot is None else goal_entry_data(snapshot)
        entry = CustomEntry(
            parent_id=self._last_parent_id,
            namespace=GOAL_NAMESPACE,
            data=data,
        )
        await self._append_session_entry(entry)
        leaf = LeafEntry(parent_id=entry.id, entry_id=entry.id)
        await self._append_session_entry(leaf)
        self._last_parent_id = entry.id
        await self._refresh_persisted_state(leaf_id=entry.id)
        self._persisted_goal_snapshot = snapshot
        self._goal_dirty = False
        return GoalUpdateEvent(goal=self.goal)

    async def _persist_goal_and_yield(self) -> AsyncIterator[AgentEvent]:
        event = await self._persist_goal_update()
        if event is not None:
            yield event

    async def _persist_subagent_trace_update(
        self,
        event: ToolExecutionUpdateEvent,
        *,
        persisted_count: int,
        persisted_tool_call_ids: set[str],
    ) -> int | None:
        """Persist one valid trace between its parent task call and result."""
        decoded = _subagent_trace_event_data(event)
        if decoded is None:
            return None
        tool_call_id, trace_data = decoded
        if tool_call_id in persisted_tool_call_ids or tool_call_id in self.subagent_traces:
            return None
        if _has_root_tool_result(self._harness.messages, tool_call_id):
            self._log_subagent_trace_order_diagnostic(
                tool_call_id,
                reason="parent_result_already_exists",
            )
            return None
        if not _has_root_task_call(self._harness.messages, tool_call_id):
            self._log_subagent_trace_order_diagnostic(
                tool_call_id,
                reason="parent_task_call_missing",
            )
            return None

        persisted_count = await self._persist_messages_since(persisted_count)
        if not _has_root_task_call(self._state.messages, tool_call_id):
            self._log_subagent_trace_order_diagnostic(
                tool_call_id,
                reason="parent_task_call_missing_after_persist",
            )
            return None

        usage_facts = _subagent_usage_event_data(event)
        for fact in usage_facts:
            usage_entry = await self._append_usage_record(
                _usage_record_from_subagent_fact(
                    fact,
                    provider=self.provider_name,
                    requested_model=self.model,
                    provider_config=self._usage_provider_config(),
                ),
                parent_id=self._last_parent_id,
            )
            self._last_parent_id = usage_entry.id
        entry = CustomEntry(
            parent_id=self._last_parent_id,
            namespace="forge.subagent_trace",
            data=trace_data,
        )
        await self._append_session_entry(entry)
        leaf = LeafEntry(parent_id=entry.id, entry_id=entry.id)
        await self._append_session_entry(leaf)
        self._last_parent_id = entry.id
        await self._refresh_persisted_state(leaf_id=entry.id)
        persisted_tool_call_ids.add(tool_call_id)
        return persisted_count

    def _log_subagent_trace_order_diagnostic(self, tool_call_id: str, *, reason: str) -> None:
        """Record a bounded, non-fatal diagnostic for malformed trace ordering."""

        event = ErrorEvent(
            message="Dropped subagent trace due to invalid parent ordering",
            recoverable=True,
            data={
                "reason": reason,
                "tool_call_id": tool_call_id[:128],
            },
        )
        try:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_error_event(
                context=self._diagnostic_context(),
                phase="subagent_trace_order",
                event=event,
            )
        except Exception:  # noqa: BLE001 - diagnostics must not disrupt message pairing
            return

    def _invalidate_context_usage_cache(self) -> None:
        """Mark context accounting dirty after transcript/system/tool changes."""
        self._context_usage_cache = None

    async def _refresh_persisted_state(self, *, leaf_id: str | None) -> None:
        entries = await self._read_session_entries()
        self._state = SessionState.from_entries(entries, leaf_id=leaf_id)
        self._usage_totals_cache = aggregate_usage_entries(self._state.entries)
        self._todos = latest_todo_snapshot(self._state.custom_entries)
        durable_goal = latest_goal_snapshot(self._state.custom_entries)
        if not self._goal_dirty:
            self._goal_controller.restore(durable_goal)
            self._persisted_goal_snapshot = durable_goal
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=self.model,
                provider_name=self.provider_name,
            )

    async def _read_session_entries(self) -> list[SessionEntry]:
        """Read stored entries, detaching roots imported from external history."""
        return _detach_missing_parents(await self._config.storage.read_all())

    async def _append_session_entry(self, entry: SessionEntry) -> None:
        """Append one durable entry after flushing deferred session metadata."""
        await self._ensure_session_initialized()
        await self._config.storage.append(entry)

    async def _ensure_session_initialized(self) -> None:
        if not self._pending_initial_entries:
            return
        await self._write_pending_initial_entries()
        if self._config.index_on_first_persist:
            self._index_current_session()

    async def _write_pending_initial_entries(self) -> None:
        for entry in self._pending_initial_entries:
            await self._config.storage.append(entry)
        self._pending_initial_entries = ()

    def _ensure_session_file_initialized(self) -> None:
        if not self._pending_initial_entries:
            return
        for entry in self._pending_initial_entries:
            _append_session_entry_sync(self._config.storage, entry)
        self._pending_initial_entries = ()

    def _index_current_session(self) -> None:
        if self._config.session_id is None or self._config.session_manager is None:
            return
        existing = self._config.session_manager.get_session(self._config.session_id)
        if existing is not None:
            return
        self._config.session_manager.create_session(
            cwd=self.cwd,
            model=self.model,
            provider_name=self.provider_name,
            session_id=self._config.session_id,
        )

    async def _try_auto_compact(
        self,
        *,
        context: AgentCallDiagnosticContext,
        phase: str,
    ) -> bool:
        try:
            return await self._maybe_auto_compact()
        except Exception as exc:  # noqa: BLE001 - automatic compaction must not lose a turn
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase=phase,
                exc=exc,
            )
            return False

    async def _try_overflow_compact(
        self,
        *,
        context: AgentCallDiagnosticContext,
    ) -> bool:
        try:
            plan = self._recent_preserving_compaction_plan()
            if plan is None:
                return False
            usage_sink: list[object] = []
            try:
                summary, details, summary_usage = await self._generate_compaction_summary(
                    plan.messages_to_summarize,
                    turn_prefix_messages=plan.turn_prefix_messages,
                    usage_sink=usage_sink,
                )
            except Exception:
                if usage_sink:
                    await self._persist_helper_usage_messages(usage_sink, purpose="compaction")
                raise
            await self._append_compaction(
                summary,
                replace_entry_ids=plan.replace_entry_ids,
                details=details,
                tokens_before=self.context_token_estimate,
                usage_messages=summary_usage,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - the original overflow remains visible
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="overflow_compact",
                exc=exc,
            )
            return False

    async def _try_auto_name_session(
        self,
        first_message: str,
        *,
        context: AgentCallDiagnosticContext,
    ) -> None:
        if not self._should_auto_name_session():
            return
        usage_messages: list[object] = []
        try:
            title = await self._generate_session_name(first_message, usage_sink=usage_messages)
        except Exception as exc:  # noqa: BLE001 - naming must not interrupt the agent turn
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="auto_name_session",
                exc=exc,
            )
            title = _fallback_session_name(first_message)
        if usage_messages:
            await self._persist_helper_usage_messages(usage_messages, purpose="auto_name")
        if title is None:
            title = _fallback_session_name(first_message)
        if title is None:
            return
        self._set_auto_session_title(title)

    def _should_auto_name_session(self) -> bool:
        if self._config.session_id is None or self._config.session_manager is None:
            return False
        record = self._config.session_manager.get_session(self._config.session_id)
        if record is not None and record.title:
            return False
        return sum(isinstance(message, HumanMessage) for message in self._harness.messages) == 1

    async def _generate_session_name(
        self,
        first_message: str,
        *,
        usage_sink: list[object] | None = None,
    ) -> str | None:
        prompt = (
            "Create a concise session name for this first user message. "
            "Use at most four words.\n\n"
            f"User message:\n{first_message}"
        )
        provider = self._harness.config.provider
        if provider is None:
            raise RuntimeError("No active chat model is configured")
        result = await _stream_native_model_result(
            provider,
            system=SESSION_NAME_SYSTEM_PROMPT,
            messages=[HumanMessage(content=prompt)],
        )
        if usage_sink is not None:
            usage_sink.append(result.message)
        return _sanitize_session_name(result.text)

    def _set_auto_session_title(self, title: str) -> None:
        if self._config.session_id is None or self._config.session_manager is None:
            return
        existing = self._config.session_manager.get_session(self._config.session_id)
        if existing is not None and existing.title:
            return
        self._config.session_manager.touch_session(
            self._config.session_id,
            model=self.model,
            provider_name=self.provider_name,
            title=title,
        )

    async def _maybe_auto_compact(self) -> bool:
        threshold = self.auto_compact_token_threshold
        if threshold is None or threshold <= 0:
            return False
        if len(self._state.context_entry_ids) < 2:
            return False
        tokens_before = self.context_token_estimate
        if tokens_before <= threshold:
            return False
        plan = self._recent_preserving_compaction_plan()
        if plan is None:
            return False
        usage_sink: list[object] = []
        try:
            summary, details, summary_usage = await self._generate_compaction_summary(
                plan.messages_to_summarize,
                turn_prefix_messages=plan.turn_prefix_messages,
                usage_sink=usage_sink,
            )
        except Exception:
            if usage_sink:
                await self._persist_helper_usage_messages(usage_sink, purpose="compaction")
            raise
        await self._append_compaction(
            summary,
            replace_entry_ids=plan.replace_entry_ids,
            details=details,
            tokens_before=tokens_before,
            usage_messages=summary_usage,
        )
        return True

    async def _generate_compaction_summary(
        self,
        messages: tuple[Any, ...],
        *,
        custom_instructions: str | None = None,
        turn_prefix_messages: tuple[Any, ...] = (),
        usage_sink: list[object] | None = None,
    ) -> tuple[str, dict[str, list[str]], tuple[object, ...]]:
        """Summarize messages for compaction, appending file-operation context.

        Returns ``(summary_text, details, usage_messages)`` where details maps ``read_files``
        and ``modified_files`` to sorted path lists for durable, cumulative
        cross-compaction tracking. When the recent-keeping budget lands inside
        the newest turn, the turn prefix is summarized separately and merged
        (Pi's split-turn handling). Summarization calls are bounded by
        ``_summary_max_tokens`` and retried once on transient failures.
        """
        provider = self._harness.config.provider
        if provider is None:
            raise RuntimeError("No active chat model is configured")
        max_tokens = _summary_max_tokens(provider)
        system = SUMMARIZATION_SYSTEM_PROMPT

        usage_messages = usage_sink if usage_sink is not None else []
        summary = await _stream_summary_text(
            provider,
            system=system,
            messages=[
                HumanMessage(
                    content=build_compaction_summary_prompt(
                        messages,
                        custom_instructions=custom_instructions,
                    )
                )
            ],
            max_tokens=max_tokens,
            policy=self._config.retry,
            usage_sink=usage_messages,
        )
        summary = summary.strip()
        if turn_prefix_messages:
            prefix = await _stream_summary_text(
                provider,
                system=system,
                messages=[
                    HumanMessage(content=build_turn_prefix_summary_prompt(turn_prefix_messages))
                ],
                max_tokens=max_tokens,
                policy=self._config.retry,
                usage_sink=usage_messages,
            )
            prefix = prefix.strip()
            summary = (
                f"{summary}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}"
                if summary
                else prefix
            )
        if not summary:
            raise RuntimeError("Compaction summarization returned an empty summary")

        operations = merge_file_operations(
            extract_file_operations(messages),
            extract_file_operations(turn_prefix_messages),
            file_operations_from_details(_last_compaction_details(self._state)),
        )
        details = details_from_file_operations(operations)
        formatted = format_file_operations(details["read_files"], details["modified_files"])
        if formatted:
            summary = f"{summary}{formatted}"
        return summary, details, tuple(usage_messages)

    async def _summarize_branch_messages(
        self,
        messages: tuple[Any, ...],
        *,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
        usage_sink: list[object] | None = None,
    ) -> str:
        try:
            provider = self._harness.config.provider
            if provider is None:
                raise RuntimeError("No active chat model is configured")
            summary = await summarize_branch_messages_with_model(
                provider=provider,
                model=self.model,
                messages=messages,
                custom_instructions=custom_instructions,
                replace_instructions=replace_instructions,
                usage_sink=usage_sink,
            )
        except Exception:
            summary = None
        return summary or summarize_messages_for_compaction(messages)

    def _recent_preserving_compaction_plan(self) -> CompactionPlan | None:
        """Prepare a compaction that keeps the most recent context.

        Walks the active transcript with ``_first_recent_context_index`` and
        replaces everything older than the recent-keeping budget. When the
        budget lands inside the newest turn (a single turn larger than the
        budget), the turn's prefix is summarized separately as a split turn so
        the kept suffix keeps its context.
        """
        rows = self._active_context_rows()
        if len(rows) < 2:
            return None

        first_kept_index = _first_recent_context_index(
            rows,
            keep_recent_tokens=DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
        )
        if first_kept_index <= 0:
            return None

        replaced = rows[:first_kept_index]
        if not replaced:
            return None
        replace_entry_ids = tuple(entry_id for entry_id, _message in replaced)

        turn_start = _last_user_message_index(rows, end=first_kept_index)
        is_split_turn = (
            turn_start is not None and _message_role(rows[first_kept_index][1]) != "user"
        )
        if not is_split_turn:
            return CompactionPlan(
                replace_entry_ids=replace_entry_ids,
                messages_to_summarize=tuple(message for _entry_id, message in replaced),
            )
        return CompactionPlan(
            replace_entry_ids=replace_entry_ids,
            messages_to_summarize=tuple(message for _entry_id, message in rows[:turn_start]),
            turn_prefix_messages=tuple(
                message for _entry_id, message in rows[turn_start:first_kept_index]
            ),
        )

    def _active_context_rows(self) -> tuple[tuple[str, Any], ...]:
        return tuple(zip(self._state.context_entry_ids, self._state.messages, strict=True))

    async def _append_compaction(
        self,
        summary: str,
        *,
        replace_entry_ids: tuple[str, ...],
        details: dict[str, list[str]] | None = None,
        tokens_before: int | None = None,
        usage_messages: Sequence[object] = (),
    ) -> CompactionEntry:
        if not replace_entry_ids:
            raise ValueError("No active context messages to compact")

        compaction = CompactionEntry(
            parent_id=self._last_parent_id,
            summary=summary,
            replaces_entry_ids=list(replace_entry_ids),
            details=details,
            tokens_before=tokens_before,
        )
        await self._append_session_entry(compaction)
        self._last_parent_id = compaction.id
        for message in usage_messages:
            usage_entry = await self._append_usage_message(
                message,
                purpose="compaction",
                parent_id=self._last_parent_id,
            )
            if usage_entry is not None:
                self._last_parent_id = usage_entry.id
        leaf = LeafEntry(parent_id=self._last_parent_id, entry_id=self._last_parent_id)
        await self._append_session_entry(leaf)

        # Keep the product-facing plan visible after older message rows are
        # replaced by a compaction summary.  The snapshot is not model context.
        if self._todos:
            todo_entry = CustomEntry(
                parent_id=self._last_parent_id,
                namespace=TODO_NAMESPACE,
                data=todo_entry_data(self._todos),
            )
            await self._append_session_entry(todo_entry)
            todo_leaf = LeafEntry(parent_id=todo_entry.id, entry_id=todo_entry.id)
            await self._append_session_entry(todo_leaf)
            self._last_parent_id = todo_entry.id

        # Goal is product state, not model context.  Re-append the full
        # snapshot so a compacted branch can replay the same lifecycle without
        # introducing a LangGraph checkpoint or transcript message.
        if self.goal is not None:
            goal_entry = CustomEntry(
                parent_id=self._last_parent_id,
                namespace=GOAL_NAMESPACE,
                data=goal_entry_data(self.goal),
            )
            await self._append_session_entry(goal_entry)
            goal_leaf = LeafEntry(parent_id=goal_entry.id, entry_id=goal_entry.id)
            await self._append_session_entry(goal_leaf)
            self._last_parent_id = goal_entry.id

        await self._refresh_persisted_state(leaf_id=self._last_parent_id)
        self._harness.replace_messages(self._state.messages)
        self._invalidate_context_usage_cache()
        return compaction


def _adjacent_usage_child(entries: Sequence[SessionEntry], parent_id: str) -> CustomEntry | None:
    """Return the durable usage node immediately following ``parent_id``."""

    for index, entry in enumerate(entries[:-1]):
        if entry.id != parent_id:
            continue
        candidate = entries[index + 1]
        if (
            isinstance(candidate, CustomEntry)
            and candidate.namespace == USAGE_NAMESPACE
            and candidate.parent_id == parent_id
        ):
            return candidate
        return None
    return None


def _is_subagent_trace_update(event: ToolExecutionUpdateEvent) -> bool:
    data = event.data
    return isinstance(data, Mapping) and data.get("kind") == "subagent_trace"


def _subagent_trace_event_data(
    event: ToolExecutionUpdateEvent,
) -> tuple[str, dict[str, JSONValue]] | None:
    data = event.data
    if not isinstance(data, Mapping) or data.get("kind") != "subagent_trace":
        return None
    allowed = {
        "kind",
        "version",
        "agent",
        "items",
        "truncated",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "usage",
    }
    if (
        not {
            "kind",
            "version",
            "agent",
            "items",
            "truncated",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        }
        <= set(data)
        or set(data) - allowed
        or type(data.get("version")) is not int
        or data["version"] != 1
    ):
        return None
    tool_call_id = event.tool_call_id
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    core = {
        key: value
        for key, value in data.items()
        if key not in {"kind", "version", "usage"}
    }
    try:
        trace = SubagentTrace.from_dict(core)
    except (TypeError, ValueError):
        return None
    persisted: dict[str, JSONValue] = {
        "version": 1,
        "tool_call_id": tool_call_id,
        **trace.to_dict(),
    }
    return tool_call_id, persisted


def _subagent_trace_index(
    entries: tuple[CustomEntry, ...],
) -> dict[str, dict[str, JSONValue]]:
    traces: dict[str, dict[str, JSONValue]] = {}
    allowed = {
        "version",
        "tool_call_id",
        "agent",
        "items",
        "truncated",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
    for entry in entries:
        if entry.namespace != "forge.subagent_trace" or set(entry.data) != allowed:
            continue
        if type(entry.data.get("version")) is not int or entry.data["version"] != 1:
            continue
        tool_call_id = entry.data.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id or tool_call_id in traces:
            continue
        core = {
            key: value
            for key, value in entry.data.items()
            if key not in {"version", "tool_call_id"}
        }
        try:
            trace = SubagentTrace.from_dict(core)
        except (TypeError, ValueError):
            continue
        traces[tool_call_id] = {
            "version": 1,
            "tool_call_id": tool_call_id,
            **trace.to_dict(),
        }
    return traces


def _subagent_usage_event_data(
    event: ToolExecutionUpdateEvent,
) -> tuple[dict[str, JSONValue], ...]:
    """Validate private per-call usage facts carried with a trace update."""

    data = event.data
    if not isinstance(data, Mapping):
        return ()
    raw_usage = data.get("usage")
    if not isinstance(raw_usage, Sequence) or isinstance(raw_usage, (str, bytes, bytearray)):
        return ()
    facts: list[dict[str, JSONValue]] = []
    allowed = {
        "response_model",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
    }
    for raw in raw_usage[:8]:
        if not isinstance(raw, Mapping) or set(raw) != allowed:
            return ()
        response_model = raw.get("response_model")
        if response_model is not None and not isinstance(response_model, str):
            return ()
        fact: dict[str, JSONValue] = {"response_model": response_model}
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
        ):
            value = raw.get(key)
            if value is not None and (type(value) is not int or value < 0):
                return ()
            fact[key] = value
        facts.append(fact)
    return tuple(facts)


def _usage_record_from_subagent_fact(
    fact: Mapping[str, JSONValue],
    *,
    provider: str,
    requested_model: str,
    provider_config: object | None,
) -> UsageRecord:
    """Adapt a private child fact to the common pricing/normalization path."""

    usage: dict[str, object] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = fact.get(key)
        if value is not None:
            usage[key] = value
    details: dict[str, object] = {}
    if fact.get("cache_read_tokens") is not None:
        details["cache_read"] = fact["cache_read_tokens"]
    if fact.get("cache_write_tokens") is not None:
        details["cache_creation"] = fact["cache_write_tokens"]
    if details:
        usage["input_token_details"] = details
    response_model = fact.get("response_model")
    metadata = {"model_name": response_model} if isinstance(response_model, str) else {}
    message = SimpleNamespace(
        usage_metadata=usage or None,
        response_metadata=metadata,
    )
    return usage_record_from_message(
        cast(Any, message),
        purpose="subagent",
        provider=provider,
        requested_model=requested_model,
        provider_config=provider_config,
    )


def _has_root_task_call(messages: tuple[Any, ...] | list[Any], tool_call_id: str) -> bool:
    return any(
        isinstance(message, AIMessage)
        and any(
            call.get("id") == tool_call_id and call.get("name") == "task"
            for call in message.tool_calls
        )
        for message in messages
    )


def _has_root_tool_result(messages: tuple[Any, ...] | list[Any], tool_call_id: str) -> bool:
    return any(
        isinstance(message, ToolMessage) and str(message.tool_call_id) == tool_call_id
        for message in messages
    )


def _sanitize_session_name(text: str) -> str | None:
    cleaned = " ".join(text.split()).strip()
    cleaned = cleaned.strip("\"'`“”‘’")
    cleaned = cleaned.strip(string.punctuation + " ")
    words = [word.strip(string.punctuation + "\"'`“”‘’") for word in cleaned.split()]
    words = [word for word in words if word]
    if not words:
        return None
    return " ".join(words[:4])


def _fallback_session_name(first_message: str) -> str | None:
    return _sanitize_session_name(first_message)


def _category_summary(
    before: tuple[tuple[object, ...], ...],
    after: tuple[tuple[object, ...], ...],
) -> ReloadCategorySummary:
    return ReloadCategorySummary(
        before=len(before),
        after=len(after),
        changed=before != after,
    )


def _skill_signatures(skills: tuple[Skill, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (skill.name, str(skill.path), skill.description, skill.content) for skill in skills
    )


def _prompt_template_signatures(
    prompt_templates: tuple[PromptTemplate, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (template.name, str(template.path), template.description, template.content)
        for template in prompt_templates
    )


def _context_file_signatures(
    context_files: tuple[ProjectContextFile, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple((context_file.path, context_file.content) for context_file in context_files)


def _subagent_profile_signatures(
    profiles: tuple[CodingSubagentProfile, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            profile.name,
            profile.description,
            profile.prompt,
            profile.tool_names,
            profile.max_model_calls,
            profile.max_result_bytes,
            profile.source,
        )
        for profile in profiles
    )


def _diagnostic_signatures(
    diagnostics: tuple[ResourceDiagnostic, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            diagnostic.kind,
            diagnostic.message,
            str(diagnostic.path) if diagnostic.path is not None else None,
            diagnostic.name,
            diagnostic.severity,
        )
        for diagnostic in diagnostics
    )


def _system_prompt_resource_signatures(
    *,
    skills: tuple[Skill, ...],
    context_files: tuple[ProjectContextFile, ...],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    prompt_skills = tuple(
        (skill.name, str(skill.path), skill.description)
        for skill in sorted(skills, key=lambda item: item.name)
    )
    return (prompt_skills, _context_file_signatures(context_files))


def _trust_result_matches_cwd(result: TrustResult, cwd: Path) -> bool:
    """Ensure a preflight result is bound to the session's current project."""
    try:
        expected_cwd = canonical_path(cwd)
        expected_root = canonical_path(find_project_root(expected_cwd))
        return (
            canonical_path(result.cwd) == expected_cwd
            and canonical_path(result.project_root) == expected_root
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _load_session_resources(
    resource_paths: ForgeResourcePaths,
    explicit_context_files: tuple[ProjectContextFile, ...],
) -> SessionResources:
    loaded_skills, skill_diagnostics = load_skills_with_diagnostics(resource_paths)
    loaded_prompt_templates, prompt_diagnostics = load_prompt_templates_with_diagnostics(
        resource_paths
    )
    discovered_context, context_diagnostics = discover_project_context_with_diagnostics(
        resource_paths
    )
    return SessionResources(
        skills=tuple(loaded_skills),
        prompt_templates=tuple(loaded_prompt_templates),
        context_files=_merge_context_files(explicit_context_files, discovered_context),
        diagnostics=tuple([*skill_diagnostics, *prompt_diagnostics, *context_diagnostics]),
    )


def _merge_context_files(
    explicit: tuple[ProjectContextFile, ...],
    discovered: tuple[ProjectContextFile, ...],
) -> tuple[ProjectContextFile, ...]:
    merged: list[ProjectContextFile] = []
    seen: set[str] = set()
    for context_file in (*explicit, *discovered):
        if context_file.path in seen:
            continue
        seen.add(context_file.path)
        merged.append(context_file)
    return tuple(merged)


def _interrupted_tool_repair_plan(
    messages: tuple[Any, ...],
    *,
    context_entry_ids: tuple[str, ...],
) -> tuple[str, tuple[Any, ...]] | None:
    repaired: list[Any] = []
    returned_ids = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    for message in messages:
        repaired.append(message)
        if isinstance(message, AIMessage):
            calls = message.tool_calls
        else:
            continue
        for tool_call in calls:
            if isinstance(tool_call, Mapping):
                call_id = str(tool_call.get("id") or "")
                call_name = str(tool_call.get("name") or "tool")
            else:
                call_id = str(getattr(tool_call, "id", "") or "")
                call_name = str(getattr(tool_call, "name", "tool") or "tool")
            if call_id in returned_ids:
                continue
            returned_ids.add(call_id)
            content = "Tool call interrupted by user"
            repaired.append(
                ToolMessage(
                    tool_call_id=call_id,
                    name=call_name,
                    content=content,
                    status="error",
                )
            )

    if tuple(repaired) == messages:
        return None

    common_prefix_length = 0
    for old_message, repaired_message in zip(messages, repaired, strict=False):
        if old_message != repaired_message:
            break
        common_prefix_length += 1
    if common_prefix_length == 0:
        return None
    return context_entry_ids[common_prefix_length - 1], tuple(repaired[common_prefix_length:])


async def _stream_native_model_text(
    model: BaseChatModel,
    *,
    system: str,
    messages: list[Any],
    max_tokens: int | None = None,
) -> str:
    """Collect a text-only helper request through LangChain's native stream."""

    return (
        await _stream_native_model_result(
            model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
    ).text


async def _stream_native_model_result(
    model: BaseChatModel,
    *,
    system: str,
    messages: list[Any],
    max_tokens: int | None = None,
) -> _StreamedModelResult:
    """Collect helper text and final usage metadata without persisting text."""

    input_messages: list[Any] = [SystemMessage(content=system)]
    input_messages.extend(messages)
    text_parts: list[str] = []
    usage_metadata: Mapping[str, Any] | None = None
    response_metadata: Mapping[str, Any] | None = None
    stream = (
        model.astream(input_messages, max_tokens=max_tokens)
        if max_tokens is not None
        else model.astream(input_messages)
    )
    async for chunk in stream:
        text_parts.append(message_text(chunk))
        raw_usage = getattr(chunk, "usage_metadata", None)
        if isinstance(raw_usage, Mapping):
            usage_metadata = merge_stream_metadata(usage_metadata, raw_usage)
        raw_response = getattr(chunk, "response_metadata", None)
        if isinstance(raw_response, Mapping):
            response_metadata = merge_stream_metadata(response_metadata, raw_response)
    kwargs: dict[str, object] = {}
    if usage_metadata is not None:
        kwargs["usage_metadata"] = usage_metadata
    if response_metadata is not None:
        kwargs["response_metadata"] = response_metadata
    text = "".join(text_parts)
    if usage_metadata is not None and not all(
        type(usage_metadata.get(key)) is int
        for key in ("input_tokens", "output_tokens", "total_tokens")
    ):
        # Some adapters stream partial usage maps.  Avoid manufacturing zero
        # counts merely to satisfy AIMessage validation; the normalizer keeps
        # those fields as null.
        usage_message: object = SimpleNamespace(
            usage_metadata=usage_metadata,
            response_metadata=response_metadata or {},
        )
    else:
        try:
            usage_message = AIMessage(content=text, **kwargs)
        except (TypeError, ValueError):
            usage_message = SimpleNamespace(
                usage_metadata=usage_metadata,
                response_metadata=response_metadata or {},
            )
    return _StreamedModelResult(text=text, message=usage_message)


def _summary_max_tokens(provider: BaseChatModel) -> int:
    """Bound summarization output like Pi: 80% of the compaction reserve.

    Provider-level ``max_tokens`` caps the budget when configured.
    """
    budget = int(DEFAULT_COMPACTION_RESERVE_TOKENS * 0.8)
    model_cap = getattr(provider, "max_tokens", None)
    if isinstance(model_cap, int) and model_cap > 0:
        return min(budget, model_cap)
    return budget


async def _stream_summary_text(
    provider: BaseChatModel,
    *,
    system: str,
    messages: list[Any],
    max_tokens: int,
    policy: RetryPolicy | None = None,
    usage_sink: list[object] | None = None,
) -> str:
    """Run one summarization request with a single transient-failure retry.

    Compaction must not lose a turn to a dropped stream, so transient failures
    use the same bounded policy; cancellation is never swallowed.
    """
    policy = policy or RetryPolicy()
    max_retries = policy.max_retries if policy.enabled else 0
    for retry_index in range(max_retries + 1):
        try:
            result = await _stream_native_model_result(
                provider,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
            )
            if usage_sink is not None:
                usage_sink.append(result.message)
            return result.text
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            classification = classify_model_error(exc)
            if not classification.retryable or retry_index >= max_retries:
                raise
            delay = policy.delay(retry_index, classification.retry_after)
            if delay:
                await asyncio.sleep(delay)
    raise RuntimeError("unreachable summary retry loop")


def default_session_path(cwd: Path) -> Path:
    """Return Forge's default user-home session path for a project cwd."""
    return ForgePaths().default_session_path(cwd)


def jsonl_session_storage(path: str | Path) -> JsonlSessionStorage:
    """Convenience factory for local JSONL coding-session storage."""
    return JsonlSessionStorage(path)


def _append_session_entry_sync(storage: SessionStorage, entry: SessionEntry) -> None:
    """Append an entry synchronously for slash commands that cannot await storage."""
    if isinstance(storage, JsonlSessionStorage):
        storage.path.parent.mkdir(parents=True, exist_ok=True)
        repair_torn_tail(storage.path)
        with storage.path.open("ab") as file:
            file.write(entry_to_json_line(entry).encode("utf-8"))
        return
    raise RuntimeError("Session storage does not support synchronous initialization")
