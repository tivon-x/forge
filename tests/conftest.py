from pathlib import Path
from typing import Any

import pytest
from pydantic import Field, create_model

from forge_agent.tools import ToolExecutor

_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def make_native_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    executor: ToolExecutor,
):
    """Build a native Forge tool from a simple JSON schema (test helper)."""
    from forge_coding.tools.base import _create_native_tool

    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", ()))
    fields: dict[str, Any] = {}
    for prop_name, prop in properties.items():
        prop_type = _JSON_TYPES.get(prop.get("type"), Any)
        prop_description = prop.get("description")
        if prop_name in required:
            fields[prop_name] = (prop_type, Field(description=prop_description))
        else:
            fields[prop_name] = (
                prop_type | None,
                Field(default=None, description=prop_description),
            )
    args_schema = create_model(f"{name.title()}ToolInput", **fields)
    return _create_native_tool(
        name=name,
        description=description,
        args_schema=args_schema,
        executor=executor,
    )


def isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point home-directory lookups at the pytest temp directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Path.home() resolves via USERPROFILE on Windows, so HOME alone does not
    # isolate tests from the developer's real ~/.forge settings.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
