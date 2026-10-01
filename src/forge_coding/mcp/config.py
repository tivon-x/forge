"""Strict MCP configuration without resolving secrets or opening connections."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from forge_coding.paths import ForgePaths
from forge_coding.resources.trust import TrustResult, canonical_path


class MCPConfigError(ValueError):
    """A bounded diagnostic that never includes configuration values."""


class EnvironmentReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=256)


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    command: str | None = Field(default=None, min_length=1, max_length=4096)
    args: list[str] = Field(default_factory=list, max_length=256)
    cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    env: dict[str, EnvironmentReference] = Field(default_factory=dict, max_length=256)
    url: str | None = Field(default=None, max_length=4096)
    headers: dict[str, EnvironmentReference] = Field(default_factory=dict, max_length=256)
    enabled: bool = True
    exposure: Literal["direct", "codemode", "deferred", "hidden"] = "deferred"
    toolExposure: dict[str, Literal["direct", "codemode", "deferred", "hidden"]] = Field(
        default_factory=dict, max_length=2000
    )
    timeoutSeconds: float = Field(default=60.0, gt=0, le=3600)

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_timeout(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("timeoutSeconds"), bool):
            raise ValueError("Invalid timeout")
        return value

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if (self.command is None) == (self.url is None):
            raise ValueError("Exactly one transport is required")
        if self.command is not None and self.headers:
            raise ValueError("HTTP headers require HTTP transport")
        if self.url is not None:
            if self.args or self.cwd or self.env:
                raise ValueError("Process options require stdio transport")
            parsed = urlsplit(self.url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Invalid HTTP endpoint")
            if parsed.port == 0:
                raise ValueError("Invalid HTTP port")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in self.env):
            raise ValueError("Invalid environment name")
        if any(not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) for key in self.headers):
            raise ValueError("Invalid header name")
        return self


class _ConfigDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mcpServers: dict[str, MCPServerConfig] = Field(max_length=256)


@dataclass(frozen=True, slots=True)
class MCPServer:
    name: str
    config: MCPServerConfig
    source: Path
    workspace: Path
    fingerprint: str


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MCPConfigError("Duplicate MCP configuration key")
        result[key] = value
    return result


def load_mcp_servers(
    cwd: Path, *, paths: ForgePaths | None = None, trust: TrustResult | None = None
) -> dict[str, MCPServer]:
    """Load whole-server overrides; configuration never grants execution permission."""
    paths = paths or ForgePaths()
    workspace = canonical_path(cwd)
    sources = [paths.mcp_config_path]
    if trust is not None and trust.allowed and canonical_path(trust.cwd) == workspace:
        project = paths.project_mcp_config_path(workspace)
        if not project.resolve().is_relative_to(workspace):
            raise MCPConfigError("Project MCP configuration escapes workspace")
        sources.append(project)
    servers: dict[str, MCPServer] = {}
    for source in sources:
        try:
            if not source.exists():
                continue
            with source.open("rb") as stream:
                data = stream.read(256 * 1024 + 1)
            if len(data) > 256 * 1024:
                raise MCPConfigError("MCP configuration exceeds size limit")
            document = _ConfigDocument.model_validate(
                json.loads(data, object_pairs_hook=_unique_object)
            )
            for name, config in document.mcpServers.items():
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
                    raise MCPConfigError("Invalid MCP server name")
                directory = workspace if config.cwd is None else workspace / config.cwd
                if not canonical_path(directory).is_relative_to(workspace):
                    raise MCPConfigError("MCP process directory escapes workspace")
                normalized = {
                    "workspace": str(workspace),
                    "source": str(canonical_path(source)),
                    "config": config.model_dump(),
                    "cwd": str(canonical_path(directory)),
                }
                fingerprint = sha256(
                    json.dumps(normalized, sort_keys=True).encode("utf-8")
                ).hexdigest()
                servers[name] = MCPServer(name, config, source, workspace, fingerprint)
        except MCPConfigError:
            raise
        except (OSError, ValueError, RuntimeError, ValidationError):
            raise MCPConfigError("Invalid or unreadable MCP configuration") from None
    return servers
