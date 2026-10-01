"""Offline config acceptance before any MCP connection is implemented.

Failure modes: trust bypass, partial override, inline credentials, duplicate
keys, path escape, stale authorization and diagnostic leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_coding.mcp.config import MCPConfigError, load_mcp_servers
from forge_coding.paths import ForgePaths
from forge_coding.resources.trust import TrustResult


def test_mcp_config_trust_override_and_fingerprint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = ForgePaths(home=tmp_path / "user")
    paths.home.mkdir()
    paths.mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "command": "python",
                        "args": ["user.py"],
                        "env": {"TOKEN": {"env": "FIXTURE_TOKEN"}},
                    }
                }
            }
        )
    )
    project = paths.project_mcp_config_path(workspace)
    project.parent.mkdir()
    project.write_text(
        json.dumps({"mcpServers": {"local": {"url": "http://127.0.0.1:9999/mcp", "enabled": True}}})
    )
    trust = TrustResult(
        workspace, workspace, True, True, "allow", "fixture", tmp_path / "trust.json"
    )
    user = load_mcp_servers(workspace, paths=paths)
    selected = load_mcp_servers(workspace, paths=paths, trust=trust)
    assert user["local"].config.command == "python"
    assert selected["local"].config.command is None
    assert selected["local"].config.env == {}
    assert selected["local"].fingerprint != user["local"].fingerprint
    first = selected["local"].fingerprint
    project.write_text('{"mcpServers":{"local":{"url":"http://localhost:9999/mcp"}}}')
    assert load_mcp_servers(workspace, paths=paths, trust=trust)["local"].fingerprint != first
    project.write_text("INVALID AND MUST NOT BE READ WITHOUT TRUST")
    assert load_mcp_servers(workspace, paths=paths)["local"].config.command == "python"
    evidence = tmp_path / "mcp-config-evidence.json"
    evidence.write_text(
        json.dumps({"trust_gate": True, "whole_override": True, "changed_fingerprint": True})
    )
    print(f"MCP configuration evidence: {evidence}")


@pytest.mark.parametrize(
    "payload",
    [
        '{"mcpServers":{"x":{"command":"python","url":"http://localhost/mcp"}}}',
        '{"mcpServers":{"x":{"url":"https://localhost/mcp","headers":{"X-Key":"DUMMY"}}}}',
        '{"mcpServers":{"x":{"command":"python","env":{"TOKEN":"DUMMY"}}}}',
        '{"mcpServers":{"x":{"url":"https://user:DUMMY@localhost/mcp"}}}',
        '{"mcpServers":{"x":{"command":"python","timeoutSeconds":true}}}',
        '{"mcpServers":{"x":{"command":"python","unknown":"DUMMY"}}}',
        '{"mcpServers":{"x":{"command":"python"},"x":{"command":"other"}}}',
        '{"mcpServers":{"x":{"url":"http://localhost/mcp?token=DUMMY"}}}',
    ],
)
def test_mcp_config_fails_closed_without_value_leak(tmp_path: Path, payload: str) -> None:
    paths = ForgePaths(home=tmp_path)
    paths.mcp_config_path.write_text(payload)
    with pytest.raises(MCPConfigError) as captured:
        load_mcp_servers(tmp_path, paths=paths)
    assert "DUMMY" not in str(captured.value)
