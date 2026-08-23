from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool

from forge_coding.tools import ToolSet, create_coding_tool_set


def test_coding_tool_set_preserves_order_and_native_identity() -> None:
    catalog = create_coding_tool_set()
    assert [definition.name for definition in catalog] == ["read", "write", "edit", "bash"]
    assert [tool.name for tool in catalog.tools] == ["read", "write", "edit", "bash"]
    assert all(
        definition.tool is tool for definition, tool in zip(catalog, catalog.tools, strict=True)
    )
    assert "runtime" not in catalog.by_name["read"].input_schema.get("properties", {})


def test_tool_set_selects_in_catalog_order_and_rejects_bad_requests() -> None:
    catalog = create_coding_tool_set()
    selected = catalog.select(("bash", "read"))
    assert [definition.name for definition in selected] == ["read", "bash"]
    with pytest.raises(ValueError, match="duplicate"):
        catalog.select(("read", "read"))
    with pytest.raises(ValueError, match="unknown"):
        catalog.select(("missing",))


def test_tool_set_with_tools_wraps_plain_native_tools_without_prompt_metadata() -> None:
    catalog = ToolSet()
    custom = StructuredTool.from_function(
        func=lambda value: value,
        name="custom",
        description="Return the value",
    )
    extended = catalog.with_tools(custom)
    assert extended.tools == (custom,)
    definition = extended.by_name["custom"]
    assert definition.label == "custom"
    assert definition.prompt_snippet is None
    assert definition.prompt_guidelines == ()


def test_definition_compatibility_properties_derive_from_native_tool() -> None:
    definition = create_coding_tool_set().by_name["read"]
    
    assert definition.name == definition.tool.name
    assert definition.description == definition.tool.description
    assert definition.args_schema is definition.tool.args_schema
