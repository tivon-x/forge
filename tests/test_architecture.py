"""Architecture tests pinning the LangChain-native production surface.

These scans are the only migration-regression tests kept as a dedicated file:
they assert dependency direction and the removal surface for the whole
``src`` tree.  Everything else from the migration regression set now lives in
the module tests that own the code under test.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from forge_agent import harness, langchain_runtime, message_codec
from forge_coding import provider_runtime, session
from forge_coding import tools as coding_tools


def _src_root() -> Path:
    return Path(inspect.getsourcefile(harness)).parents[1]


def test_forge_agent_never_imports_forge_coding() -> None:
    for module in (harness, langchain_runtime, message_codec):
        source = inspect.getsource(module)
        assert "forge_coding" not in source, f"{module.__name__} must not import forge_coding"


def test_migration_scaffolding_never_reappears() -> None:
    """Banned names must stay out of production source.

    ``chat_model`` (the AgentHarnessConfig fallback), ``ClosableModel*``,
    identity message converters and the legacy protocol names were removed
    with the migration; matching them again means scaffolding crept back.
    """
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
    )
    for candidate in _src_root().rglob("*.py"):
        text = candidate.read_text(encoding="utf-8")
        for name in banned:
            # Word-boundary match so `langchain_openai.chat_models` does not
            # trip the `chat_model` token.
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
    )
    for path in removed_paths:
        assert not path.exists(), f"legacy protocol file still present: {path}"


def test_production_loop_is_create_agent_only() -> None:
    runtime_source = inspect.getsource(langchain_runtime)
    # The production agent loop is the official create_agent graph; there is
    # no second provider/tool loop anywhere.
    assert "create_agent(" in runtime_source
    assert "StateGraph" not in runtime_source
    assert "astream_events" in runtime_source


def test_provider_ownership_uses_base_model() -> None:
    session_source = inspect.getsource(session)
    assert "list[BaseChatModel]" in session_source
    provider_source = inspect.getsource(provider_runtime)
    assert "ClosableModelProvider" not in provider_source


def test_tool_executors_receive_context_uniformly() -> None:
    tools_source = inspect.getsource(coding_tools)
    assert "inspect.signature" not in tools_source
    assert "signal=None, context=context" in tools_source
