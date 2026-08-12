"""Persistent coding-session wrapper built on AgentHarness."""

from __future__ import annotations

import asyncio
import string
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from forge_agent import (
    AgentEvent,
    AgentHarness,
    AgentHarnessConfig,
    ErrorEvent,
    MessageEndEvent,
    QueuedMessages,
    QueueUpdateEvent,
    SubagentRunner,
    SubagentRuntime,
    SubagentTrace,
    ToolExecutionEndEvent,
    ToolExecutionUpdateEvent,
)
from forge_agent.context import ForgeRuntimeContext
from forge_agent.message_codec import message_text
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
from forge_agent.session.tree import SessionTreeError, path_to_entry
from forge_agent.tools import ToolCall
from forge_agent.types import JSONValue
from forge_coding.branch_summary import summarize_branch_messages_with_model
from forge_coding.commands import CommandRegistry, CommandResult, create_default_command_registry
from forge_coding.context import discover_project_context_with_diagnostics
from forge_coding.context_window import (
    DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    SUMMARIZATION_SYSTEM_PROMPT,
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    build_compaction_summary_prompt,
    estimate_context_usage,
    estimate_message_tokens,
    summarize_messages_for_compaction,
)
from forge_coding.credentials import FileCredentialStore, credentials_path
from forge_coding.diagnostics import (
    AgentCallDiagnosticContext,
    AgentCallDiagnosticLogger,
    new_agent_call_run_id,
)
from forge_coding.paths import ForgePaths
from forge_coding.prompt_templates import (
    PromptTemplate,
    expand_prompt_template_command,
    load_prompt_templates_with_diagnostics,
)
from forge_coding.provider_config import (
    ProviderConfig,
    ProviderConfigError,
    ProviderSettings,
    load_provider_settings,
    provider_default_thinking_level,
    provider_has_usable_credentials,
    provider_thinking_levels,
    provider_thinking_unavailable_reason,
    resolve_provider_selection,
    save_default_provider_model,
    save_provider_thinking_level,
    toggle_saved_scoped_model,
    validate_provider_model,
)
from forge_coding.provider_runtime import aclose_model, create_model_provider
from forge_coding.reload import CodingReloadSummary, ReloadCategorySummary
from forge_coding.resources import (
    ForgeResourcePaths,
    ResourceDiagnostic,
    ResourceError,
    resource_paths_with_cwd,
)
from forge_coding.session_export import (
    default_session_export_artifact_path,
    export_session_artifact,
    normalize_export_format,
)
from forge_coding.session_manager import SessionManager
from forge_coding.skills import Skill, expand_skill_command, load_skills_with_diagnostics
from forge_coding.subagent_profiles import (
    CodingSubagentProfile,
    load_subagent_profiles,
)
from forge_coding.subagents import (
    create_coding_subagent_specs,
    create_task_tool,
    ensure_task_name_available,
)
from forge_coding.system_prompt import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    build_system_prompt,
)
from forge_coding.thinking import (
    DEFAULT_THINKING_LEVEL,
    THINKING_LEVELS,
    ThinkingLevel,
    next_thinking_level,
    normalize_thinking_level,
)
from forge_coding.tools import create_bash_tool, create_coding_tools

StreamingBehavior = Literal["steer", "follow_up"]
SESSION_NAME_SYSTEM_PROMPT = (
    "You write concise coding-agent session names. Reply with only a short title, "
    "maximum four words, no quotes, no punctuation-only output."
)
TREE_RUNNING_MESSAGE = "Forge is still working. Press Escape to interrupt before using /tree."
SESSION_SWITCH_RUNNING_MESSAGE = (
    "Forge is still working. Press Escape to interrupt before switching sessions."
)


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """A selectable model and the provider that serves it."""

    provider_name: str
    model: str


@dataclass(frozen=True, slots=True)
class TerminalCommandResult:
    """Result of an input-bar terminal command."""

    command: str
    output: str
    exit_code: int | None
    ok: bool
    added_to_context: bool


@dataclass(frozen=True, slots=True)
class SessionTreeChoice:
    """One branchable entry in the active session tree."""

    entry_id: str
    label: str
    active: bool = False
    is_tool_call: bool = False


@dataclass(frozen=True, slots=True)
class SessionTreeBranchResult:
    """Result of moving the active session tree leaf."""

    message: str
    input_prefill: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalCommandRequest:
    """Parsed input-bar terminal command request."""

    command: str
    add_to_context: bool


@dataclass(frozen=True, slots=True)
class SessionResources:
    """Forge-owned resources loaded around a coding session."""

    skills: tuple[Skill, ...]
    prompt_templates: tuple[PromptTemplate, ...]
    context_files: tuple[ProjectContextFile, ...]
    diagnostics: tuple[ResourceDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    """Prepared active-context entries for a compaction run."""

    replace_entry_ids: tuple[str, ...]
    messages_to_summarize: tuple[Any, ...]


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


class CodingSession:
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
        subagent_runner: SubagentRunner | None = None,
        subagent_profiles: tuple[CodingSubagentProfile, ...] = (),
    ) -> None:
        self._config = config
        self._state = state
        self._harness = harness
        self._last_parent_id = last_parent_id
        self._pending_initial_entries = pending_initial_entries
        self._skills = skills
        self._prompt_templates = prompt_templates
        self._context_files = context_files
        self._resource_diagnostics = resource_diagnostics
        self._command_registry = command_registry or create_default_command_registry()
        self._provider_name = config.provider_name
        self._provider_settings = config.provider_settings
        self._runtime_provider_config = config.runtime_provider_config
        self._resource_paths = resource_paths_with_cwd(config.resource_paths, config.cwd)
        self._auto_compact_token_threshold = config.auto_compact_token_threshold
        self._auto_compact_enabled = config.auto_compact_enabled
        self._thinking_level = _state_thinking_level(
            state,
            default=_default_thinking_level_for_active_model(self),
        )
        self._context_usage_cache: ContextUsageEstimate | None = None
        self._owned_providers: list[BaseChatModel] = []
        # Serializes session switches against complete CodingSession runs. The
        # lock protects ownership transitions; _run_active keeps that ownership
        # through compaction, overflow retry, persistence, and cancellation even
        # while the inner harness is briefly between model calls. Neither field
        # is transferred when a replacement session is adopted.
        self._switch_lock = asyncio.Lock()
        self._run_active = False
        self._diagnostic_logger = AgentCallDiagnosticLogger.from_paths(self._resource_paths.paths)
        self._credential_store = FileCredentialStore(
            credentials_path(self._resource_paths.paths) if self._resource_paths.paths else None
        )
        self._last_diagnostic_log_path: Path | None = None
        self._subagent_runner = subagent_runner
        self._subagent_profiles = subagent_profiles

    @classmethod
    async def load(cls, config: CodingSessionConfig) -> CodingSession:
        """Load a coding session from append-only storage."""
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
        base_tools = list(
            config.tools
            if config.tools is not None
            else create_coding_tools(
                cwd=config.cwd,
                shell_command_prefix=config.shell_command_prefix,
            )
        )
        resource_paths = resource_paths_with_cwd(config.resource_paths, config.cwd)
        resources = _load_session_resources(resource_paths, config.context_files)
        runtime_context = ForgeRuntimeContext(
            workspace_root=str(config.cwd),
            session_id=config.session_id,
            shell_command_prefix=config.shell_command_prefix,
        )
        harness_config = AgentHarnessConfig(
            provider=config.provider,
            model=_runtime_model_for_state(config, state),
            runtime_context=runtime_context,
            tools=base_tools,
        )
        subagent_runner: SubagentRunner | None = None
        subagent_profiles: tuple[CodingSubagentProfile, ...] = ()
        subagent_diagnostics: tuple[ResourceDiagnostic, ...] = ()
        if config.enable_subagents:
            ensure_task_name_available(base_tools)
            loaded_subagents = load_subagent_profiles(
                resource_paths,
                available_tool_names=(tool.name for tool in base_tools),
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
                    tools=base_tools,
                    skills=resources.skills,
                    context_files=resources.context_files,
                    system=config.system,
                    custom_system_prompt=config.custom_system_prompt,
                    append_system_prompt=config.append_system_prompt,
                    profiles=subagent_profiles,
                ),
            )
            harness_config.tools = [*base_tools, create_task_tool(subagent_runner)]
        system = (
            config.system
            if config.system is not None
            else build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=config.cwd,
                    tools=harness_config.tools,
                    skills=resources.skills,
                    custom_prompt=config.custom_system_prompt,
                    append_system_prompt=config.append_system_prompt,
                    context_files=resources.context_files,
                )
            )
        )
        harness_config.system = system
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
            subagent_runner=subagent_runner,
            subagent_profiles=subagent_profiles,
        )
        await session._persist_loaded_interrupted_tool_repairs()
        session._sync_thinking_level_to_active_model()
        session._refresh_runtime_provider()
        return session

    @property
    def cwd(self) -> Path:
        """Return the session working directory."""
        return self._config.cwd

    @property
    def model(self) -> str:
        """Return the active model for this session."""
        return self._harness.config.model

    @property
    def provider_name(self) -> str:
        """Return the active provider name."""
        return self._provider_name

    @property
    def available_providers(self) -> tuple[str, ...]:
        """Return provider names Forge can call with available credentials."""
        if self._provider_settings is None:
            return (self._provider_name,)
        return tuple(provider.name for provider in self._usable_provider_configs())

    @property
    def available_models(self) -> tuple[str, ...]:
        """Return model names for the active provider when it is usable."""
        if self._provider_settings is None:
            return (self.model,)
        try:
            provider = self._provider_settings.get_provider(self._provider_name)
        except ProviderConfigError:
            return (self.model,)
        if not self._provider_is_usable(provider):
            return ()
        return provider.models

    @property
    def available_model_choices(self) -> tuple[ModelChoice, ...]:
        """Return provider/model choices Forge can call with available credentials."""
        if self._provider_settings is None:
            return (ModelChoice(provider_name=self._provider_name, model=self.model),)
        return tuple(
            ModelChoice(provider_name=provider.name, model=model)
            for provider in self._usable_provider_configs()
            for model in provider.models
        )

    @property
    def scoped_model_choices(self) -> tuple[ModelChoice, ...]:
        """Return configured quick-switch model choices that are currently usable."""
        if self._provider_settings is None:
            return ()
        available = set(self.available_model_choices)
        return tuple(
            choice
            for choice in (
                ModelChoice(provider_name=item.provider, model=item.model)
                for item in self._provider_settings.scoped_models
            )
            if choice in available
        )

    @property
    def tools(self) -> tuple[BaseTool, ...]:
        """Return the tools available to the agent."""
        return tuple(self._harness.config.tools)

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
        if self.is_running:
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
                )
                summary_entry = BranchSummaryEntry(
                    parent_id=entry_id,
                    branch_root_id=entry_id,
                    summary=summary,
                )
                await self._append_session_entry(summary_entry)
                target_id = summary_entry.id
        elif selected_entry.type == "message" and isinstance(selected_entry.message, HumanMessage):
            target_id = selected_entry.parent_id
            input_prefill = message_text(selected_entry.message)

        leaf = LeafEntry(parent_id=target_id, entry_id=target_id)
        await self._append_session_entry(leaf)
        self._last_parent_id = target_id

        await self._refresh_persisted_state(leaf_id=target_id)
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
    def thinking_level(self) -> ThinkingLevel:
        """Return the active thinking mode for future turns."""
        return self._thinking_level

    @property
    def available_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        """Return thinking modes supported by the active provider/model."""
        if self._provider_settings is None:
            return THINKING_LEVELS
        provider = self._active_provider_config()
        if provider is None:
            return ()
        return provider_thinking_levels(provider, model=self.model)

    @property
    def thinking_unavailable_reason(self) -> str | None:
        """Return why thinking controls are unavailable for the active model."""
        if self.available_thinking_levels:
            return None
        provider = self._active_provider_config()
        if provider is None:
            return "Active provider settings are not available"
        return provider_thinking_unavailable_reason(provider, model=self.model)

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
        """Return a rough token estimate for the active provider context."""
        return self.context_usage.total_tokens

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
    def subagent_traces(self) -> dict[str, dict[str, JSONValue]]:
        """Return validated display traces on the active session branch."""
        return _subagent_trace_index(self._state.custom_entries)

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

    def set_model(self, model: str) -> None:
        """Switch the active model for future turns and make it the default."""
        provider = self._active_provider_config()
        if provider is not None:
            validate_provider_model(provider, model)
        self._harness.config.model = model
        self._sync_thinking_level_to_active_model()
        self._refresh_runtime_provider()
        self._persist_default_model_choice()
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=model,
                provider_name=self.provider_name,
            )

    def set_model_choice(self, choice: ModelChoice) -> None:
        """Switch provider/model as one operation."""
        if choice.provider_name == self.provider_name:
            self.set_model(choice.model)
            return
        self._set_provider_model(choice.provider_name, choice.model)

    def is_scoped_model(self, choice: ModelChoice) -> bool:
        """Return whether a provider/model pair is in the scoped model list."""
        return choice in self.scoped_model_choices

    def toggle_scoped_model(self, choice: ModelChoice) -> tuple[ModelChoice, ...]:
        """Add or remove a model from the persisted scoped model list."""
        if self._provider_settings is None:
            raise ProviderConfigError("Provider settings are not available for this session")
        available = set(self.available_model_choices)
        if choice not in available:
            raise ProviderConfigError(
                f"Model is not available: {choice.provider_name}:{choice.model}"
            )

        self._provider_settings = toggle_saved_scoped_model(
            provider_name=choice.provider_name,
            model=choice.model,
            paths=self._resource_paths.paths,
            fallback_settings=self._provider_settings,
        )
        self._sync_thinking_level_to_active_model()
        return self.scoped_model_choices

    def cycle_scoped_model(self, *, reverse: bool = False) -> ModelChoice:
        """Switch to the next configured scoped model."""
        scoped = self.scoped_model_choices
        if not scoped:
            raise ProviderConfigError("No scoped models configured.")
        current = ModelChoice(provider_name=self.provider_name, model=self.model)
        try:
            current_index = scoped.index(current)
        except ValueError:
            current_index = -1 if not reverse else 0
        delta = -1 if reverse else 1
        choice = scoped[(current_index + delta) % len(scoped)]
        self.set_model_choice(choice)
        return choice

    def set_provider(self, provider_name: str, *, persist_default: bool = True) -> None:
        """Switch the active provider and reset to that provider's default model."""
        if self._provider_settings is None:
            raise ProviderConfigError("Provider settings are not available for this session")
        provider_config = self._provider_settings.get_provider(provider_name)
        self._set_provider_model(
            provider_name,
            provider_config.default_model,
            persist_default=persist_default,
        )

    def _set_provider_model(
        self,
        provider_name: str,
        model: str,
        *,
        persist_default: bool = True,
    ) -> None:
        """Switch active provider/model without constructing an intermediate provider."""
        if self._provider_settings is None:
            raise ProviderConfigError("Provider settings are not available for this session")

        provider_config = self._provider_settings.get_provider(provider_name)
        if model not in provider_config.models:
            raise ProviderConfigError(f"Model is not configured: {provider_name}:{model}")
        thinking_level = _coerced_thinking_level(
            provider_config,
            model=model,
            current=self._thinking_level,
        )
        try:
            provider = create_model_provider(
                provider_config,
                credential_store=self._credential_store,
                model=model,
                thinking_level=thinking_level,
            )
        except RuntimeError as exc:
            raise ProviderConfigError(str(exc)) from exc
        self._owned_providers.append(provider)
        self._harness.config.provider = provider
        self._provider_name = provider_config.name
        self._runtime_provider_config = provider_config
        self._harness.config.model = model
        self._thinking_level = thinking_level
        if persist_default:
            self._persist_default_model_choice()
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=model,
                provider_name=self.provider_name,
            )

    async def set_thinking_level(self, level: str) -> str:
        """Persist and activate a thinking mode for future turns."""
        normalized = normalize_thinking_level(level)
        available = self.available_thinking_levels
        if not available:
            raise ValueError(_unavailable_thinking_message(self))
        if normalized not in available:
            modes = ", ".join(available)
            raise ValueError(
                f"Thinking mode {normalized} is not available for "
                f"{self._provider_name}:{self.model}. Available modes: {modes}"
            )
        if normalized == self._thinking_level:
            return f"Thinking mode: {normalized}"

        previous = self._thinking_level
        self._thinking_level = normalized
        try:
            self._refresh_runtime_provider()
        except ProviderConfigError:
            self._thinking_level = previous
            raise

        entry = ThinkingLevelChangeEntry(
            parent_id=self._last_parent_id,
            thinking_level=normalized,
        )
        await self._append_session_entry(entry)
        leaf = LeafEntry(parent_id=entry.id, entry_id=entry.id)
        await self._append_session_entry(leaf)
        self._last_parent_id = entry.id

        self._persist_thinking_level_choice()
        await self._refresh_persisted_state(leaf_id=entry.id)
        return f"Thinking mode: {normalized}"

    async def cycle_thinking_level(self) -> str:
        """Cycle to the next supported thinking mode and persist it."""
        return await self.set_thinking_level(
            next_thinking_level(
                self._thinking_level,
                available=self.available_thinking_levels,
            )
        )

    def _active_provider_config(self) -> ProviderConfig | None:
        if self._provider_settings is None:
            return None
        try:
            return self._provider_settings.get_provider(self._provider_name)
        except ProviderConfigError:
            return None

    def _sync_thinking_level_to_active_model(self) -> None:
        provider = self._active_provider_config()
        if provider is None:
            return
        self._thinking_level = _coerced_thinking_level(
            provider,
            model=self.model,
            current=self._thinking_level,
            preferred=provider.thinking_defaults.get(self.model),
        )

    def _persist_default_model_choice(self) -> None:
        if self._provider_settings is None:
            return
        self._provider_settings = save_default_provider_model(
            provider_name=self.provider_name,
            model=self.model,
            paths=self._resource_paths.paths,
            fallback_settings=self._provider_settings,
        )
        self._sync_thinking_level_to_active_model()

    def _persist_thinking_level_choice(self) -> None:
        if self._provider_settings is None:
            return
        provider = self._active_provider_config()
        if provider is None or self._thinking_level not in provider_thinking_levels(
            provider,
            model=self.model,
        ):
            return
        try:
            self._provider_settings = save_provider_thinking_level(
                provider_name=self.provider_name,
                model=self.model,
                thinking_level=self._thinking_level,
                paths=self._resource_paths.paths,
                fallback_settings=self._provider_settings,
            )
        except ProviderConfigError:
            return

    def _refresh_runtime_provider(self) -> None:
        if self._runtime_provider_config is None:
            return
        provider_config = self._active_provider_config() or self._runtime_provider_config
        validate_provider_model(provider_config, self.model)
        try:
            provider = create_model_provider(
                provider_config,
                credential_store=self._credential_store,
                model=self.model,
                thinking_level=self._thinking_level,
            )
        except RuntimeError as exc:
            raise ProviderConfigError(str(exc)) from exc
        self._owned_providers.append(provider)
        self._harness.config.provider = provider
        self._runtime_provider_config = provider_config

    def reload(self) -> CodingReloadSummary:
        """Reload local coding resources and project context for future turns."""
        if self._run_active:
            raise RuntimeError("Cannot reload resources while Forge is running")

        before_skills = _skill_signatures(self._skills)
        before_prompt_templates = _prompt_template_signatures(self._prompt_templates)
        before_context_files = _context_file_signatures(self._context_files)
        before_diagnostics = _diagnostic_signatures(self._resource_diagnostics)
        before_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=self._skills,
            context_files=self._context_files,
        )

        resources = _load_session_resources(self._resource_paths, self._config.context_files)

        current_tools = tuple(self._harness.config.tools)
        base_tools = (
            tuple(tool for tool in current_tools if tool.name != "task")
            if self._subagent_runner is not None
            else current_tools
        )
        loaded_subagents = None
        replacement_specs = None
        replacement_runner = None
        replacement_task_tool = None
        after_subagent_profiles: tuple[CodingSubagentProfile, ...] = ()
        if self._subagent_runner is not None:
            loaded_subagents = load_subagent_profiles(
                self._resource_paths,
                available_tool_names=(tool.name for tool in base_tools),
            )
            after_subagent_profiles = loaded_subagents.profiles
            replacement_specs = create_coding_subagent_specs(
                cwd=self._config.cwd,
                tools=base_tools,
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
            replacement_task_tool = create_task_tool(replacement_runner)

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

        replacement_tools = (
            [*base_tools, replacement_task_tool]
            if replacement_task_tool is not None
            else list(base_tools)
        )

        rebuilt_system_prompt: str | None = None
        system_prompt_rebuilt = False
        if self._config.system is None and (
            before_system_prompt_inputs != after_system_prompt_inputs or subagents_changed
        ):
            rebuilt_system_prompt = build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=self._config.cwd,
                    tools=replacement_tools,
                    skills=resources.skills,
                    custom_prompt=self._config.custom_system_prompt,
                    append_system_prompt=self._config.append_system_prompt,
                    context_files=resources.context_files,
                )
            )
            system_prompt_rebuilt = True

        if replacement_runner is not None:
            self._subagent_runner = replacement_runner
            self._harness.config.tools = replacement_tools
            self._subagent_profiles = after_subagent_profiles
            self._invalidate_context_usage_cache()

        self._skills = resources.skills
        self._prompt_templates = resources.prompt_templates
        self._context_files = resources.context_files
        self._resource_diagnostics = combined_diagnostics
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

    def reload_provider_settings(self) -> None:
        """Reload provider settings for login and model-selection flows."""
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
        if self.is_running:
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

        replacement = await type(self).load(
            CodingSessionConfig(
                provider=self._harness.config.provider,
                model=model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                system=self._config.system,
                custom_system_prompt=self._config.custom_system_prompt,
                append_system_prompt=self._config.append_system_prompt,
                context_files=self._config.context_files,
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
        await self._adopt_replacement(replacement)
        return f"Resumed session: {record.id}"

    async def new_session(self) -> str:
        """Replace this session's active state with a pending unindexed session.

        Same switch lock contract as :meth:`resume`.
        """
        async with self._switch_lock:
            return await self._new_session_locked()

    async def _new_session_locked(self) -> str:
        if self.is_running:
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
            )
        )
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
        self._auto_compact_token_threshold = replacement._auto_compact_token_threshold
        self._auto_compact_enabled = replacement._auto_compact_enabled
        self._thinking_level = replacement._thinking_level
        self._context_usage_cache = replacement._context_usage_cache
        self._owned_providers = replacement._owned_providers
        self._diagnostic_logger = replacement._diagnostic_logger
        self._credential_store = replacement._credential_store
        self._last_diagnostic_log_path = replacement._last_diagnostic_log_path
        self._subagent_runner = replacement._subagent_runner
        self._subagent_profiles = replacement._subagent_profiles

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
        """Generate a manual compaction summary and rebuild active context."""
        plan = self._manual_compaction_plan()
        summary = await self._generate_compaction_summary(
            plan.messages_to_summarize,
            custom_instructions=instructions,
        )
        compaction = await self._append_compaction(
            summary,
            replace_entry_ids=plan.replace_entry_ids,
        )
        return f"Compacted {len(compaction.replaces_entry_ids)} context entries."

    async def aclose(self) -> None:
        """Close runtime providers created by this coding session.

        Every provider created by this session is closed exactly once; a
        failure closing one provider does not stop the remaining providers
        from closing.  Collected failures are raised together afterwards.
        """
        errors: list[Exception] = []
        for provider in self._owned_providers:
            try:
                await aclose_model(provider)
            except Exception as exc:  # noqa: BLE001 - one bad provider must not leak others
                errors.append(exc)
        self._owned_providers.clear()
        if errors:
            raise ExceptionGroup("Failed to close session providers", errors)

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

    async def prompt(
        self,
        content: str,
        *,
        streaming_behavior: StreamingBehavior | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Append a user prompt, run the agent, and persist new messages."""
        context = self._diagnostic_context()
        persisted_count = 0
        persisted_trace_tool_call_ids = set(self.subagent_traces)
        auto_name_attempted = False
        overflow_event: ErrorEvent | None = None
        run_started = False
        harness_started = False
        try:
            async with self._switch_lock:
                context = self._diagnostic_context()
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
                    run_started = True
                    await self._try_auto_compact(
                        context=context,
                        phase="auto_compact_before_prompt",
                    )
                    persisted_count = len(self._harness.messages)
                    events = self._harness.prompt(expanded_content)
                    harness_started = True

            if queued_event is not None:
                yield queued_event
                return

            self._invalidate_context_usage_cache()
            async for event in events:
                if isinstance(event, ToolExecutionUpdateEvent):
                    persisted = await self._persist_subagent_trace_update(
                        event,
                        persisted_count=persisted_count,
                        persisted_tool_call_ids=persisted_trace_tool_call_ids,
                    )
                    if persisted is None:
                        if _is_subagent_trace_update(event):
                            continue
                    else:
                        persisted_count = persisted
                if isinstance(event, MessageEndEvent):
                    persisted_count = await self._persist_messages_since(persisted_count)
                    if not auto_name_attempted and isinstance(event.message, HumanMessage):
                        auto_name_attempted = True
                        await self._try_auto_name_session(
                            message_text(event.message), context=context
                        )
                if isinstance(event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                if isinstance(event, ErrorEvent) and not event.recoverable:
                    self._last_diagnostic_log_path = self._diagnostic_logger.log_error_event(
                        context=context,
                        phase="agent_loop",
                        event=event,
                    )
                    if _is_context_overflow_error(event):
                        overflow_event = event
                yield event
            persisted_count = await self._persist_messages_since(persisted_count)
            if overflow_event is not None:
                compacted = await self._try_overflow_compact(context=context)
                if compacted:
                    retry_persisted_count = len(self._harness.messages)
                    async with self._switch_lock:
                        retry_events = self._harness.continue_()
                    self._invalidate_context_usage_cache()
                    async for retry_event in retry_events:
                        if isinstance(retry_event, ToolExecutionUpdateEvent):
                            persisted = await self._persist_subagent_trace_update(
                                retry_event,
                                persisted_count=retry_persisted_count,
                                persisted_tool_call_ids=persisted_trace_tool_call_ids,
                            )
                            if persisted is None:
                                if _is_subagent_trace_update(retry_event):
                                    continue
                            else:
                                retry_persisted_count = persisted
                        if isinstance(retry_event, MessageEndEvent):
                            retry_persisted_count = await self._persist_messages_since(
                                retry_persisted_count
                            )
                        if isinstance(retry_event, ToolExecutionEndEvent):
                            self._invalidate_context_usage_cache()
                        if isinstance(retry_event, ErrorEvent) and not retry_event.recoverable:
                            self._last_diagnostic_log_path = (
                                self._diagnostic_logger.log_error_event(
                                    context=context,
                                    phase="agent_loop_retry",
                                    event=retry_event,
                                )
                            )
                        yield retry_event
                    await self._persist_messages_since(retry_persisted_count)
                return
            await self._try_auto_compact(context=context, phase="auto_compact_after_prompt")
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="agent_loop",
                exc=exc,
            )
            raise
        finally:
            # Persist whatever the completed (or cancelled) run left on the
            # harness transcript.  If the user cancelled mid-tool-execution the
            # harness appended a synthetic ToolMessage repairing the dangling
            # tool call in memory; without this final flush the JSONL would keep
            # an assistant tool call with no matching tool result, which some
            # providers reject on the next request after resume.  Only flush on
            # a genuine interrupt -- a consumer closing the stream early is not
            # one and must not index/materialise the session.
            try:
                if harness_started and self._harness.was_last_run_interrupted:
                    await self._persist_messages_since(persisted_count)
            finally:
                if run_started:
                    async with self._switch_lock:
                        self._run_active = False

    async def continue_(self) -> AsyncIterator[AgentEvent]:
        """Continue the agent from restored state and persist new messages."""
        context = self._diagnostic_context()
        persisted_count = 0
        persisted_trace_tool_call_ids = set(self.subagent_traces)
        run_started = False
        harness_started = False
        try:
            async with self._switch_lock:
                context = self._diagnostic_context()
                if self.is_running:
                    raise RuntimeError("CodingSession is already running")
                self._run_active = True
                run_started = True
                persisted_count = len(self._harness.messages)
                events = self._harness.continue_()
                harness_started = True
            self._invalidate_context_usage_cache()
            async for event in events:
                if isinstance(event, ToolExecutionUpdateEvent):
                    persisted = await self._persist_subagent_trace_update(
                        event,
                        persisted_count=persisted_count,
                        persisted_tool_call_ids=persisted_trace_tool_call_ids,
                    )
                    if persisted is None:
                        if _is_subagent_trace_update(event):
                            continue
                    else:
                        persisted_count = persisted
                if isinstance(event, MessageEndEvent):
                    persisted_count = await self._persist_messages_since(persisted_count)
                if isinstance(event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                if isinstance(event, ErrorEvent) and not event.recoverable:
                    self._last_diagnostic_log_path = self._diagnostic_logger.log_error_event(
                        context=context,
                        phase="agent_loop",
                        event=event,
                    )
                yield event
            await self._persist_messages_since(persisted_count)
            await self._try_auto_compact(context=context, phase="auto_compact_after_continue")
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="agent_loop",
                exc=exc,
            )
            raise
        finally:
            # Same contract as run: flush messages an interrupted (cancelled)
            # run appended, including synthetic interrupted-tool repairs, so a
            # resume from JSONL never submits a dangling assistant tool call.
            try:
                if harness_started and self._harness.was_last_run_interrupted:
                    await self._persist_messages_since(persisted_count)
            finally:
                if run_started:
                    async with self._switch_lock:
                        self._run_active = False

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
            leaf = LeafEntry(parent_id=entry.id, entry_id=entry.id)
            await self._append_session_entry(leaf)

        await self._refresh_persisted_state(leaf_id=self._last_parent_id)
        self._invalidate_context_usage_cache()
        return persisted_count + len(new_messages)

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

        previous_parent_id = self._last_parent_id
        entry = CustomEntry(
            parent_id=previous_parent_id,
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
            summary = await self._generate_compaction_summary(plan.messages_to_summarize)
            await self._append_compaction(summary, replace_entry_ids=plan.replace_entry_ids)
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
        try:
            title = await self._generate_session_name(first_message)
        except Exception as exc:  # noqa: BLE001 - naming must not interrupt the agent turn
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="auto_name_session",
                exc=exc,
            )
            title = _fallback_session_name(first_message)
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

    async def _generate_session_name(self, first_message: str) -> str | None:
        prompt = (
            "Create a concise session name for this first user message. "
            "Use at most four words.\n\n"
            f"User message:\n{first_message}"
        )
        provider = self._harness.config.provider
        if provider is None:
            raise RuntimeError("No active chat model is configured")
        return _sanitize_session_name(
            await _stream_native_model_text(
                provider,
                system=SESSION_NAME_SYSTEM_PROMPT,
                messages=[HumanMessage(content=prompt)],
            )
        )

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

    def _provider_is_usable(self, provider: ProviderConfig) -> bool:
        return provider_has_usable_credentials(
            provider,
            credential_reader=self._credential_store,
        )

    def _usable_provider_configs(self) -> tuple[ProviderConfig, ...]:
        if self._provider_settings is None:
            return ()
        return tuple(
            provider
            for provider in self._provider_settings.providers
            if self._provider_is_usable(provider)
        )

    async def _maybe_auto_compact(self) -> bool:
        threshold = self.auto_compact_token_threshold
        if threshold is None or threshold <= 0:
            return False
        if len(self._state.context_entry_ids) < 2:
            return False
        if self.context_token_estimate <= threshold:
            return False
        plan = self._recent_preserving_compaction_plan()
        if plan is None:
            return False
        summary = await self._generate_compaction_summary(plan.messages_to_summarize)
        await self._append_compaction(summary, replace_entry_ids=plan.replace_entry_ids)
        return True

    async def _generate_compaction_summary(
        self,
        messages: tuple[Any, ...],
        *,
        custom_instructions: str | None = None,
    ) -> str:
        prompt = build_compaction_summary_prompt(
            messages,
            custom_instructions=custom_instructions,
        )
        summary_messages: list[HumanMessage] = [HumanMessage(content=prompt)]
        provider = self._harness.config.provider
        if provider is None:
            raise RuntimeError("No active chat model is configured")
        summary = (
            await _stream_native_model_text(
                provider,
                system=SUMMARIZATION_SYSTEM_PROMPT,
                messages=summary_messages,
            )
        ).strip()
        if not summary:
            raise RuntimeError("Compaction summarization returned an empty summary")
        return summary

    async def _summarize_branch_messages(
        self,
        messages: tuple[Any, ...],
        *,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
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
            )
        except Exception:
            summary = None
        return summary or summarize_messages_for_compaction(messages)

    def _manual_compaction_plan(self) -> CompactionPlan:
        rows = self._active_context_rows()
        if not rows:
            raise ValueError("No active context messages to compact")
        return CompactionPlan(
            replace_entry_ids=tuple(entry_id for entry_id, _message in rows),
            messages_to_summarize=tuple(message for _entry_id, message in rows),
        )

    def _recent_preserving_compaction_plan(self) -> CompactionPlan | None:
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
        return CompactionPlan(
            replace_entry_ids=tuple(entry_id for entry_id, _message in replaced),
            messages_to_summarize=tuple(message for _entry_id, message in replaced),
        )

    def _active_context_rows(self) -> tuple[tuple[str, Any], ...]:
        return tuple(zip(self._state.context_entry_ids, self._state.messages, strict=True))

    async def _append_compaction(
        self,
        summary: str,
        *,
        replace_entry_ids: tuple[str, ...],
    ) -> CompactionEntry:
        if not replace_entry_ids:
            raise ValueError("No active context messages to compact")

        compaction = CompactionEntry(
            parent_id=self._last_parent_id,
            summary=summary,
            replaces_entry_ids=list(replace_entry_ids),
        )
        await self._append_session_entry(compaction)
        leaf = LeafEntry(parent_id=compaction.id, entry_id=compaction.id)
        await self._append_session_entry(leaf)
        self._last_parent_id = compaction.id

        await self._refresh_persisted_state(leaf_id=compaction.id)
        self._harness.replace_messages(self._state.messages)
        self._invalidate_context_usage_cache()
        return compaction


def _first_recent_context_index(
    rows: tuple[tuple[str, Any], ...],
    *,
    keep_recent_tokens: int,
) -> int:
    if keep_recent_tokens <= 0:
        return len(rows)

    accumulated_tokens = 0
    candidate_index: int | None = None
    for index in range(len(rows) - 1, -1, -1):
        _entry_id, message = rows[index]
        accumulated_tokens += estimate_message_tokens(message)
        if accumulated_tokens >= keep_recent_tokens:
            candidate_index = index
            break

    if candidate_index is None:
        return 0

    candidate_message = rows[candidate_index][1]
    if _message_role(candidate_message) == "user":
        if candidate_index > 0:
            return candidate_index
        next_user_index = _next_user_message_index(rows, start=1)
        return next_user_index if next_user_index is not None else 0

    next_user_index = _next_user_message_index(rows, start=candidate_index + 1)
    if next_user_index is not None:
        return next_user_index

    for index in range(candidate_index, len(rows)):
        if _message_role(rows[index][1]) != "tool":
            return index
    return len(rows)


def _next_user_message_index(
    rows: tuple[tuple[str, Any], ...],
    *,
    start: int,
) -> int | None:
    for index in range(start, len(rows)):
        if _message_role(rows[index][1]) == "user":
            return index
    return None


def _is_context_overflow_error(event: ErrorEvent) -> bool:
    text = event.message
    if event.data is not None:
        text = f"{text} {event.data}"
    normalized = text.lower()
    markers = (
        "context length",
        "context window",
        "context limit",
        "maximum context",
        "max context",
        "input is too long",
        "input length",
        "prompt is too long",
        "too many tokens",
        "token limit",
        "exceeds the limit",
        "exceeded the limit",
    )
    return any(marker in normalized for marker in markers)


def _detach_missing_parents(entries: list[SessionEntry]) -> list[SessionEntry]:
    """Return entries with dangling parent pointers detached from external history."""
    entry_ids = {entry.id for entry in entries}
    return [
        entry.model_copy(update={"parent_id": None})
        if entry.parent_id is not None and entry.parent_id not in entry_ids
        else entry
        for entry in entries
    ]


def _last_parent_id_from_state(state: SessionState) -> str | None:
    if state.active_leaf_id is not None:
        return state.active_leaf_id
    if state.entries:
        return state.entries[-1].id
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
    }
    if set(data) != allowed or type(data.get("version")) is not int or data["version"] != 1:
        return None
    tool_call_id = event.tool_call_id
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    core = {key: value for key, value in data.items() if key not in {"kind", "version"}}
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


def _latest_leaf_entry(entries: list[SessionEntry]) -> LeafEntry | None:
    for entry in reversed(entries):
        if isinstance(entry, LeafEntry):
            return entry
    return None


def _is_branchable_tree_entry(entry: SessionEntry) -> bool:
    if entry.type in {"compaction", "branch_summary"}:
        return True
    if entry.type != "message":
        return False
    return isinstance(
        entry.message,
        HumanMessage | AIMessage,
    )


def _tree_choice_label(entry: SessionEntry, *, branch_indent: int = 0) -> str:
    prefix = "  " * branch_indent
    return f"{prefix}{_tree_entry_title(entry)}"


def _tree_branch_indents(entries: list[SessionEntry]) -> dict[str, int]:
    children_by_parent: dict[str | None, list[str]] = {}
    for entry in entries:
        if entry.type != "leaf":
            children_by_parent.setdefault(entry.parent_id, []).append(entry.id)

    sibling_indexes = {
        child_id: index
        for children in children_by_parent.values()
        for index, child_id in enumerate(children)
    }
    indents: dict[str, int] = {}
    for entry in entries:
        if entry.type == "leaf":
            continue
        parent_indent = indents.get(entry.parent_id, 0) if entry.parent_id is not None else 0
        sibling_index = sibling_indexes.get(entry.id, 0)
        indents[entry.id] = parent_indent + (1 if sibling_index > 0 else 0)
    return indents


def _ordered_tree_entries(entries: list[SessionEntry]) -> tuple[SessionEntry, ...]:
    children_by_parent: dict[str | None, list[SessionEntry]] = {}
    for entry in entries:
        if entry.type != "leaf":
            children_by_parent.setdefault(entry.parent_id, []).append(entry)

    ordered: list[SessionEntry] = []
    seen: set[str] = set()
    expanded: set[str | None] = set()

    def append_descendants(root_parent_id: str | None) -> None:
        # Iterative depth-first walk rather than recursion so a long session (a
        # deep root-to-leaf entry chain) cannot exceed Python's recursion limit.
        # `expanded` also makes a malformed parent cycle terminate instead of
        # recursing forever. Emitting a node's direct children before descending,
        # and pushing them reversed so the first child is processed next,
        # preserves the original traversal order.
        stack: list[str | None] = [root_parent_id]
        while stack:
            parent_id = stack.pop()
            if parent_id in expanded:
                continue
            expanded.add(parent_id)
            children = children_by_parent.get(parent_id, [])
            for child in children:
                if child.id not in seen:
                    ordered.append(child)
                    seen.add(child.id)
            for child in reversed(children):
                stack.append(child.id)

    append_descendants(None)
    for entry in entries:
        if entry.type != "leaf" and entry.id not in seen:
            ordered.append(entry)
            seen.add(entry.id)
            append_descendants(entry.id)
    return tuple(ordered)


def _is_tool_call_tree_entry(entry: SessionEntry) -> bool:
    if entry.type != "message":
        return False
    message = entry.message
    if isinstance(message, AIMessage):
        return bool(message.tool_calls)
    return False


def _tree_entry_title(entry: SessionEntry) -> str:
    match entry.type:
        case "message":
            message = entry.message
            if isinstance(message, AIMessage) and message.tool_calls and not message_text(message):
                calls = message.tool_calls
                tool_names = ", ".join(
                    call.name if isinstance(call, ToolCall) else str(call.get("name", "tool"))
                    for call in calls
                )
                return f"tool call: {tool_names}"
            return f"{_message_role(message)}: {_message_text_preview(message)}"
        case "compaction":
            return f"compaction summary: {_short_preview(entry.summary)}"
        case "branch_summary":
            return f"branch summary: {_short_preview(entry.summary)}"
        case _:
            return entry.type


def _message_text_preview(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return _short_preview(content)
    return _short_preview(str(content))


def _message_role(message: Any) -> str:
    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(
        str(getattr(message, "type", "")), "message"
    )


def _short_preview(text: str, *, limit: int = 72) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized or "(empty)"
    return f"{normalized[: limit - 1]}..."


def _messages_after_entry_on_active_path(
    entries: list[SessionEntry],
    entry_id: str,
    active_leaf_id: str | None,
) -> tuple[Any, ...]:
    if active_leaf_id is None:
        return ()
    try:
        active_path = path_to_entry(entries, active_leaf_id)
    except SessionTreeError:
        return ()
    try:
        target_index = next(
            index for index, entry in enumerate(active_path) if entry.id == entry_id
        )
    except StopIteration:
        return ()
    return tuple(
        entry.message for entry in active_path[target_index + 1 :] if entry.type == "message"
    )


def _storage_path(storage: SessionStorage) -> Path | None:
    path = getattr(storage, "path", None)
    return path if isinstance(path, Path) else None


def _resolve_export_destination(
    destination: Path | None,
    *,
    cwd: Path,
    session_path: Path | None,
    format: str,
) -> Path:
    if destination is None:
        if session_path is not None:
            return default_session_export_artifact_path(
                session_path,
                destination_dir=cwd,
                format=format,
            )
        return cwd / f"forge-session.{format}"

    resolved = destination if destination.is_absolute() else cwd / destination
    if resolved.suffix:
        return resolved
    name = session_path.stem if session_path is not None else "forge-session"
    return default_session_export_artifact_path(
        Path(name),
        destination_dir=resolved,
        format=format,
    )


def _session_export_title(session: CodingSession) -> str:
    manager = session.session_manager
    session_id = session.session_id
    if manager is not None and session_id is not None:
        record = manager.get_session(session_id)
        if record is not None and record.title:
            return record.title
    return f"Forge session {session_id}" if session_id is not None else "Forge Session Export"


def _initial_model_for_config(config: CodingSessionConfig) -> str:
    if config.provider_settings is None or config.runtime_provider_config is None:
        return config.model
    provider = _provider_config_for_name(config, config.provider_name)
    if provider is None:
        return config.model
    try:
        validate_provider_model(provider, config.model)
    except ProviderConfigError:
        return provider.default_model
    return config.model


def _runtime_model_for_state(config: CodingSessionConfig, state: SessionState) -> str:
    state_model = state.model or config.model
    if config.provider_settings is None or config.runtime_provider_config is None:
        return state_model
    provider = _provider_config_for_name(config, config.provider_name)
    if provider is None:
        return state_model
    try:
        validate_provider_model(provider, state_model)
    except ProviderConfigError:
        return config.model if config.model in provider.models else provider.default_model
    return state_model


def _initial_thinking_level_for_config(
    config: CodingSessionConfig,
    *,
    model: str,
) -> ThinkingLevel:
    provider = _provider_config_for_name(config, config.provider_name)
    if provider is None:
        return config.thinking_level
    return _preferred_thinking_level_for_model(
        provider,
        model=model,
        fallback=config.thinking_level,
    )


def _provider_config_for_name(
    config: CodingSessionConfig,
    provider_name: str,
) -> ProviderConfig | None:
    if config.provider_settings is not None:
        try:
            return config.provider_settings.get_provider(provider_name)
        except ProviderConfigError:
            pass
    if config.runtime_provider_config is not None:
        return config.runtime_provider_config
    return None


def _state_thinking_level(
    state: SessionState,
    default: ThinkingLevel,
) -> ThinkingLevel:
    thinking_level = getattr(state, "thinking_level", None)
    if thinking_level is None:
        return default
    return normalize_thinking_level(thinking_level)


def _default_thinking_level_for_active_model(session: CodingSession) -> ThinkingLevel:
    provider = session._active_provider_config()
    if provider is None:
        return session._config.thinking_level
    return _preferred_thinking_level_for_model(
        provider,
        model=session.model,
        fallback=session._config.thinking_level,
    )


def _preferred_thinking_level_for_model(
    provider: ProviderConfig,
    *,
    model: str,
    fallback: ThinkingLevel,
) -> ThinkingLevel:
    levels = provider_thinking_levels(provider, model=model)
    preferred = provider.thinking_defaults.get(model)
    if preferred in levels:
        return preferred
    if fallback in levels or not levels:
        return fallback
    default = provider_default_thinking_level(provider, model=model)
    return default or levels[0]


def _coerced_thinking_level(
    provider: ProviderConfig,
    *,
    model: str,
    current: ThinkingLevel,
    preferred: ThinkingLevel | None = None,
) -> ThinkingLevel:
    levels = provider_thinking_levels(provider, model=model)
    if not levels or current in levels:
        return current
    if preferred in levels:
        return preferred
    default = provider_default_thinking_level(provider, model=model)
    return default or levels[0]


def _unavailable_thinking_message(session: CodingSession) -> str:
    message = f"Thinking controls are unavailable for {session.provider_name}:{session.model}"
    reason = session.thinking_unavailable_reason
    if reason:
        return f"{message}: {reason}"
    return message


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


def _terminal_command_context_message(command: str, output: str) -> str:
    return (
        "Terminal command executed by the user.\n\n"
        f"Command:\n```bash\n{command}\n```\n\n"
        f"Output:\n```text\n{output}\n```"
    )


def parse_terminal_command(text: str) -> TerminalCommandRequest | None:
    """Parse input-bar terminal command syntax."""
    stripped = text.strip()
    if stripped.startswith("!!"):
        command = stripped[2:].strip()
        if not command:
            return None
        return TerminalCommandRequest(command=command, add_to_context=False)
    if stripped.startswith("!"):
        command = stripped[1:].strip()
        if not command:
            return None
        return TerminalCommandRequest(command=command, add_to_context=True)
    return None


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
) -> str:
    """Collect a text-only helper request through LangChain's native stream."""

    input_messages: list[Any] = [SystemMessage(content=system)]
    input_messages.extend(messages)
    text_parts: list[str] = []
    async for chunk in model.astream(input_messages):
        text_parts.append(message_text(chunk))
    return "".join(text_parts)


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
