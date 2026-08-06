"""Runtime context injected into Forge LangChain tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ForgeRuntimeContext:
    """Non-model-visible execution context shared by Forge tools."""

    workspace_root: str
    session_id: str | None = None
    shell_command_prefix: str | None = None
    safety_policy: str = "workspace-bound"
    output_dir: str | None = None
