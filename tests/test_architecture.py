"""Architecture tests pinning Forge's three-package production surface."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from forge_agent import harness, langchain_runtime
from forge_coding import tools as coding_tools
from forge_coding.providers import runtime as provider_runtime
from forge_coding.sessions import session


def _src_root() -> Path:
    return Path(inspect.getsourcefile(harness)).parents[1]


def _production_files(package: str) -> tuple[Path, ...]:
    return tuple((_src_root() / package).rglob("*.py"))


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.append(node.module)
    return tuple(imported)


def _assert_package_does_not_import(package: str, forbidden_roots: tuple[str, ...]) -> None:
    for candidate in _production_files(package):
        for imported in _imports(candidate):
            assert not imported.startswith(forbidden_roots), (
                f"{candidate.relative_to(_src_root())} imports forbidden module {imported}"
            )


def test_three_package_dependency_direction() -> None:
    """The core cannot depend on coding or presentation; coding cannot depend on CLI."""
    _assert_package_does_not_import("forge_agent", ("forge_coding", "forge_cli"))
    _assert_package_does_not_import("forge_coding", ("forge_cli",))


def test_product_tool_catalog_stays_out_of_runtime_package() -> None:
    """ToolSet/ToolDefinition are coding product metadata, not agent facts."""
    for candidate in _production_files("forge_agent"):
        source = candidate.read_text(encoding="utf-8")
        assert "ToolDefinition" not in source
        assert "ToolSet" not in source


def test_forge_cli_root_stays_lightweight() -> None:
    """Importing the presentation package root must not eagerly load its frontends."""
    init_path = _src_root() / "forge_cli" / "__init__.py"
    assert _imports(init_path) == ("__future__",), (
        "forge_cli.__init__ must not import presentation submodules"
    )


def test_ui_libraries_are_owned_by_forge_cli() -> None:
    """Typer, Rich, and Textual stay in the presentation package."""
    ui_roots = ("textual", "rich", "typer")
    for package in ("forge_agent", "forge_coding"):
        for candidate in _production_files(package):
            imported = _imports(candidate)
            has_ui_import = any(
                name == root or name.startswith(f"{root}.")
                for name in imported
                for root in ui_roots
            )
            assert not has_ui_import, f"{candidate.relative_to(_src_root())} imports a UI library"


def test_provider_sdks_stay_out_of_forge_agent() -> None:
    provider_roots = (
        "anthropic",
        "google",
        "langchain_anthropic",
        "langchain_google_genai",
        "langchain_mistralai",
        "langchain_openai",
        "mistralai",
        "openai",
    )
    for candidate in _production_files("forge_agent"):
        for imported in _imports(candidate):
            assert not any(
                imported == root or imported.startswith(f"{root}.") for root in provider_roots
            ), f"{candidate.relative_to(_src_root())} imports provider SDK {imported}"


def test_migration_scaffolding_never_reappears() -> None:
    """Banned names and custom-loop scaffolding stay out of production source."""
    banned = (
        "chat_model",
        "ClosableModelProvider",
        "ClosableModel",
        "ForgeCodexCompatModel",
        "_CompatCodexCredentials",
        "is_langchain_message",
        "to_langchain_message",
        "ForgeProviderChatModel",
        "ForgeProviderRuntimeError",
        "_langchain_tool",
        "_from_langchain_messages",
        "_to_langchain_message",
        "native_transcript",
        "ModelProvider",
        "forge_ai",
        "legacy-style",
        "compatibility helper",
        "pre-migration",
        "provider-neutral",
        "StateGraph",
        "run_agent_loop",
    )
    for candidate in _src_root().rglob("*.py"):
        text = candidate.read_text(encoding="utf-8")
        for name in banned:
            if re.search(rf"\b{re.escape(name)}\b", text):
                raise AssertionError(f"{candidate} still contains {name}")


def test_legacy_protocol_files_stay_removed() -> None:
    src_root = _src_root()
    removed_paths = (
        src_root / "forge_ai",
        src_root / "forge_agent" / "messages.py",
        src_root / "forge_agent" / "provider.py",
        src_root / "forge_agent" / "loop.py",
        src_root / "forge_agent" / "compat.py",
        src_root / "forge_coding" / "compat.py",
        src_root / "forge_coding" / "cli.py",
        src_root / "forge_coding" / "rendering",
        src_root / "forge_coding" / "tui",
    )
    for path in removed_paths:
        assert not path.exists(), f"legacy or presentation path still present: {path}"


def test_production_loop_is_create_agent_only() -> None:
    runtime_source = inspect.getsource(langchain_runtime)
    assert "create_agent(" in runtime_source
    assert "astream_events" in runtime_source
    assert "StateGraph" not in runtime_source
    assert "run_agent_loop" not in runtime_source


def test_provider_ownership_uses_base_model() -> None:
    session_source = inspect.getsource(session)
    assert "list[BaseChatModel]" in session_source
    provider_source = inspect.getsource(provider_runtime)
    assert "ClosableModelProvider" not in provider_source


def test_tool_executors_receive_context_uniformly() -> None:
    tools_source = inspect.getsource(coding_tools)
    assert "inspect.signature" not in tools_source
    assert "signal=None, context=context" in tools_source
