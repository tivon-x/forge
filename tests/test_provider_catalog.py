"""Tests for the TOML-backed provider catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_coding.paths import ForgePaths
from forge_coding.providers.catalog import (
    BUILTIN_PROVIDER_CATALOG,
    ProviderCatalogEntry,
    builtin_provider_entry,
)
from forge_coding.providers.catalog_loader import (
    CatalogError,
    builtin_catalog,
    builtin_catalog_resource_text,
    effective_catalog,
    user_catalog_path,
)
from forge_coding.providers.config import (
    ProviderConfigError,
    load_provider_settings,
    set_provider_thinking_level,
)

VALID_PROVIDER = """
[[providers]]
name = "nebius"
display_name = "Nebius AI Studio"
kind = "openai-compatible"
base_url = "https://api.studio.nebius.ai/v1"
api_key_env = "NEBIUS_API_KEY"
credential_name = "nebius"
models = ["deepseek-ai/DeepSeek-V4-Pro", "Qwen/Qwen3-Coder-480B-A35B-Instruct"]
default_model = "deepseek-ai/DeepSeek-V4-Pro"
docs_url = "https://studio.nebius.ai/docs"
thinking_levels = ["off", "low", "medium", "high"]
thinking_models = ["deepseek-ai/DeepSeek-V4-Pro"]
thinking_default = "medium"
thinking_parameter = "reasoning_effort"

[providers.context_windows]
"deepseek-ai/DeepSeek-V4-Pro" = 163840
"""


def _write_user_catalog(forge_home: Path, body: str) -> ForgePaths:
    paths = ForgePaths(home=forge_home)
    forge_home.mkdir(parents=True, exist_ok=True)
    user_catalog_path(paths).write_text(f"schema_version = 1\n{body}", encoding="utf-8")
    return paths


def test_builtin_catalog_matches_expected_providers() -> None:
    names = [entry.name for entry in BUILTIN_PROVIDER_CATALOG]
    assert names == [
        "openai",
        "openai-codex",
        "opencode",
        "opencode-go",
        "anthropic",
        "google",
        "deepseek",
        "xai",
        "groq",
        "cerebras",
        "nvidia",
        "openrouter",
        "zai",
        "mistral",
        "minimax",
        "minimax-cn",
        "moonshotai",
        "moonshotai-cn",
        "huggingface",
        "fireworks",
        "together",
        "vercel-ai-gateway",
        "xiaomi",
        "xiaomi-token-plan-cn",
        "xiaomi-token-plan-ams",
        "xiaomi-token-plan-sgp",
    ]


def test_builtin_catalog_golden_anthropic_entry() -> None:
    entry = builtin_provider_entry("anthropic")
    assert entry is not None
    assert entry.display_name == "Anthropic"
    assert entry.kind == "anthropic"
    assert entry.base_url == "https://api.anthropic.com"
    assert entry.api_key_env == "ANTHROPIC_API_KEY"
    assert entry.credential_name == "anthropic"
    assert entry.models == (
        "claude-fable-5",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5",
        "claude-opus-4-5-20251101",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
    )
    assert entry.default_model == "claude-opus-4-8"
    assert entry.docs_url == "https://docs.anthropic.com"
    assert entry.context_windows == {
        "claude-fable-5": 1_000_000,
        "claude-haiku-4-5": 200_000,
        "claude-haiku-4-5-20251001": 200_000,
        "claude-opus-4-5": 200_000,
        "claude-opus-4-5-20251101": 200_000,
        "claude-opus-4-6": 1_000_000,
        "claude-opus-4-7": 1_000_000,
        "claude-opus-4-8": 1_000_000,
        "claude-opus-5": 1_000_000,
        "claude-sonnet-4-5": 1_000_000,
        "claude-sonnet-4-5-20250929": 1_000_000,
        "claude-sonnet-4-6": 1_000_000,
        "claude-sonnet-5": 1_000_000,
    }
    assert entry.thinking_levels == ("off", "minimal", "low", "medium", "high", "xhigh", "max")
    assert entry.thinking_models == ()
    assert entry.thinking_default == "medium"
    assert entry.thinking_parameter == "anthropic.thinking"


def test_builtin_catalog_golden_nvidia_entry() -> None:
    entry = builtin_provider_entry("nvidia")
    assert entry is not None
    assert entry.display_name == "NVIDIA NIM"
    assert entry.kind == "openai-compatible"
    assert entry.base_url == "https://integrate.api.nvidia.com/v1"
    assert entry.api_key_env == "NVIDIA_API_KEY"
    assert entry.credential_name == "nvidia"
    assert entry.models == (
        "google/gemma-3-12b-it",
        "google/gemma-3-4b-it",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.2-11b-vision-instruct",
        "meta/llama-3.2-90b-vision-instruct",
        "meta/llama-3.3-70b-instruct",
        "minimaxai/minimax-m3",
        "mistralai/mistral-7b-instruct-v0.3",
        "moonshotai/kimi-k2.6",
        "nvidia/cosmos-reason2-8b",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.1-nemotron-nano-8b-v1",
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "nvidia/nemotron-3-nano-30b-a3b",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-nano-12b-v2-vl",
        "nvidia/nvidia-nemotron-nano-9b-v2",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "poolside/laguna-xs-2.1",
        "stepfun-ai/step-3.7-flash",
        "thinkingmachines/inkling",
        "z-ai/glm-5.2",
    )
    assert entry.default_model == "nvidia/nemotron-3-super-120b-a12b"
    assert entry.docs_url == "https://docs.api.nvidia.com/nim"
    assert entry.api == "openai-completions"
    assert entry.context_windows == {
        "google/gemma-3-12b-it": 131_072,
        "google/gemma-3-4b-it": 131_072,
        "meta/llama-3.1-70b-instruct": 128_000,
        "meta/llama-3.1-8b-instruct": 16_000,
        "meta/llama-3.2-11b-vision-instruct": 128_000,
        "meta/llama-3.2-90b-vision-instruct": 128_000,
        "meta/llama-3.3-70b-instruct": 128_000,
        "minimaxai/minimax-m3": 1_000_000,
        "mistralai/mistral-7b-instruct-v0.3": 65_536,
        "moonshotai/kimi-k2.6": 262_144,
        "nvidia/cosmos-reason2-8b": 131_072,
        "nvidia/llama-3.1-nemotron-70b-instruct": 128_000,
        "nvidia/llama-3.1-nemotron-nano-8b-v1": 131_072,
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1": 32_768,
        "nvidia/llama-3.1-nemotron-ultra-253b-v1": 128_000,
        "nvidia/llama-3.3-nemotron-super-49b-v1": 131_072,
        "nvidia/llama-3.3-nemotron-super-49b-v1.5": 131_072,
        "nvidia/nemotron-3-nano-30b-a3b": 131_072,
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": 256_000,
        "nvidia/nemotron-3-super-120b-a12b": 262_144,
        "nvidia/nemotron-3-ultra-550b-a55b": 1_000_000,
        "nvidia/nemotron-3.5-lightning-30b-a3b": 262_144,
        "nvidia/nemotron-nano-12b-v2-vl": 128_000,
        "nvidia/nvidia-nemotron-nano-9b-v2": 131_072,
        "openai/gpt-oss-120b": 128_000,
        "openai/gpt-oss-20b": 131_072,
        "poolside/laguna-xs-2.1": 262_144,
        "stepfun-ai/step-3.7-flash": 256_000,
        "thinkingmachines/inkling": 1_048_576,
        "z-ai/glm-5.2": 1_000_000,
    }
    assert entry.thinking_levels == ("off", "minimal", "low", "medium", "high")
    assert entry.thinking_models == ()
    assert entry.thinking_default == "medium"
    assert entry.thinking_parameter == "reasoning_effort"

    default_metadata = entry.model_metadata[entry.default_model]
    assert default_metadata.name == "Nemotron 3 Super"
    assert default_metadata.reasoning is True
    assert default_metadata.input == ("text",)
    assert default_metadata.context_window == 262_144
    assert default_metadata.max_tokens == 262_144
    assert default_metadata.cost == {
        "input": 0.2,
        "output": 0.8,
        "cacheRead": 0.0,
        "cacheWrite": 0.0,
    }

    gpt_oss_metadata = entry.model_metadata["openai/gpt-oss-120b"]
    assert gpt_oss_metadata.reasoning is True
    assert gpt_oss_metadata.context_window == 128_000
    assert gpt_oss_metadata.max_tokens == 8_192


def test_builtin_catalog_entries_are_internally_consistent() -> None:
    for entry in builtin_catalog():
        assert entry.default_model in entry.models
        assert set(entry.thinking_models) <= set(entry.models)
        assert set(entry.context_windows or {}) <= set(entry.models)
        if entry.thinking_default is not None:
            assert entry.thinking_levels is not None
            assert entry.thinking_default in entry.thinking_levels


def test_builtin_openai_carries_pi_cost_tiers(tmp_path: Path) -> None:
    """Pi's request-wide pricing tiers survive the catalog round trip."""
    entry = builtin_provider_entry("openai")
    assert entry is not None
    metadata = entry.model_metadata["gpt-5.5"]
    assert metadata.cost == {"input": 5.0, "output": 30.0, "cacheRead": 0.5, "cacheWrite": 0.0}
    assert len(metadata.cost_tiers) == 1
    tier = metadata.cost_tiers[0]
    assert tier.input_tokens_above == 272_000
    assert tier.input == 10.0
    assert tier.output == 45.0
    assert tier.cache_read == 1.0
    assert tier.cache_write == 0.0

    # The tier also survives the durable provider-config conversion.
    settings = load_provider_settings(ForgePaths(home=tmp_path / ".forge"))
    provider_metadata = settings.get_provider("openai").model_metadata["gpt-5.5"]
    assert provider_metadata.cost_tiers == metadata.cost_tiers
    assert provider_metadata.to_json()["cost_tiers"] == [
        {
            "input_tokens_above": 272_000,
            "input": 10.0,
            "output": 45.0,
            "cache_read": 1.0,
            "cache_write": 0.0,
        }
    ]


def test_stale_saved_thinking_defaults_are_dropped_on_load(tmp_path: Path) -> None:
    """Saved per-model thinking defaults that a catalog refresh invalidated
    must not fail the whole settings load; the save path still validates.
    """
    forge_home = tmp_path / ".forge"
    forge_home.mkdir()
    (forge_home / "providers.json").write_text(
        json.dumps(
            {
                "default_provider": "openai",
                "provider_preferences": {
                    # xhigh is not available for deepseek-v4-flash anymore.
                    "deepseek": {
                        "default_model": "deepseek-v4-flash",
                        "thinking_defaults": {"deepseek-v4-flash": "xhigh"},
                    },
                    "openai": {
                        "thinking_defaults": {"gpt-5.5": "high"},
                    },
                },
                "scoped_models": [],
            }
        ),
        encoding="utf-8",
    )
    settings = load_provider_settings(ForgePaths(home=forge_home))
    assert settings.get_provider("deepseek").thinking_defaults == {}
    assert settings.get_provider("openai").thinking_defaults == {"gpt-5.5": "high"}
    # The strict save path still rejects unavailable levels.
    with pytest.raises(ProviderConfigError, match="is not available"):
        set_provider_thinking_level(
            settings,
            provider_name="deepseek",
            model="deepseek-v4-flash",
            thinking_level="xhigh",
        )


def test_builtin_catalog_resource_is_packaged() -> None:
    assert "[[providers]]" in builtin_catalog_resource_text()


def test_effective_catalog_without_user_file_is_builtin(tmp_path: Path) -> None:
    paths = ForgePaths(home=tmp_path / ".forge")
    assert effective_catalog(paths) == builtin_catalog()


def test_user_catalog_adds_new_provider(tmp_path: Path) -> None:
    paths = _write_user_catalog(tmp_path / ".forge", VALID_PROVIDER)
    catalog = effective_catalog(paths)
    assert [entry.name for entry in catalog[:-1]] == [e.name for e in builtin_catalog()]
    entry = catalog[-1]
    assert entry.name == "nebius"
    assert entry.default_model == "deepseek-ai/DeepSeek-V4-Pro"
    assert entry.context_windows == {"deepseek-ai/DeepSeek-V4-Pro": 163_840}
    assert entry.thinking_levels == ("off", "low", "medium", "high")


def test_user_catalog_cost_tiers_round_trip(tmp_path: Path) -> None:
    """Cost tiers written through the user catalog survive a reload."""
    from forge_coding.providers.catalog import ModelCatalogMetadata, ModelCostTier
    from forge_coding.providers.catalog_loader import save_user_catalog_entries

    paths = ForgePaths(home=tmp_path / ".forge")
    tmp_path.mkdir(exist_ok=True)
    save_user_catalog_entries(
        [
            ProviderCatalogEntry(
                name="tiered",
                display_name="Tiered",
                kind="openai-compatible",
                base_url="https://example.com/v1",
                api_key_env="TIERED_API_KEY",
                credential_name="tiered",
                models=("tiered-1",),
                default_model="tiered-1",
                docs_url="https://example.com/docs",
                model_metadata={
                    "tiered-1": ModelCatalogMetadata(
                        name="Tiered 1",
                        reasoning=True,
                        cost={"input": 5.0, "output": 30.0, "cacheRead": 0.5, "cacheWrite": 0.0},
                        cost_tiers=(
                            ModelCostTier(
                                input_tokens_above=272_000,
                                input=10.0,
                                output=45.0,
                                cache_read=1.0,
                                cache_write=0.0,
                            ),
                        ),
                        context_window=272_000,
                        max_tokens=128_000,
                    ),
                },
            )
        ],
        paths=paths,
    )
    entry = next(e for e in effective_catalog(paths) if e.name == "tiered")
    metadata = entry.model_metadata["tiered-1"]
    assert metadata.cost_tiers == (
        ModelCostTier(
            input_tokens_above=272_000,
            input=10.0,
            output=45.0,
            cache_read=1.0,
            cache_write=0.0,
        ),
    )


def test_user_catalog_rejects_invalid_cost_tiers(tmp_path: Path) -> None:
    paths = _write_user_catalog(
        tmp_path / ".forge",
        """
[[providers]]
name = "bad-tiers"
display_name = "Bad"
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key_env = "BAD_API_KEY"
models = ["m1"]
default_model = "m1"
docs_url = "https://example.com/docs"

[providers.model_metadata.m1]
cost_tiers = [{ input_tokens_above = -1, input = 1, output = 2 }]
""",
    )
    with pytest.raises(CatalogError, match="input_tokens_above"):
        effective_catalog(paths)


def test_user_catalog_overlays_builtin_provider(tmp_path: Path) -> None:
    paths = _write_user_catalog(
        tmp_path / ".forge",
        """
[[providers]]
name = "anthropic"
models = ["claude-next-1"]
default_model = "claude-next-1"

[providers.context_windows]
"claude-next-1" = 500000
""",
    )
    entry = next(e for e in effective_catalog(paths) if e.name == "anthropic")
    assert entry.models[0] == "claude-next-1"
    assert "claude-sonnet-4-6" in entry.models
    assert entry.default_model == "claude-next-1"
    assert entry.context_windows is not None
    assert entry.context_windows["claude-next-1"] == 500_000
    assert entry.context_windows["claude-opus-4-7"] == 1_000_000
    # Untouched fields come from the builtin entry.
    assert entry.base_url == "https://api.anthropic.com"
    assert entry.thinking_parameter == "anthropic.thinking"


def test_user_catalog_thinking_fields_replace_as_group(tmp_path: Path) -> None:
    paths = _write_user_catalog(
        tmp_path / ".forge",
        """
[[providers]]
name = "anthropic"
thinking_levels = ["off", "high"]
thinking_default = "high"
""",
    )
    entry = next(e for e in effective_catalog(paths) if e.name == "anthropic")
    assert entry.thinking_levels == ("off", "high")
    assert entry.thinking_default == "high"
    assert entry.thinking_models == ()
    assert entry.thinking_parameter is None


def test_user_catalog_rejects_unknown_keys(tmp_path: Path) -> None:
    paths = _write_user_catalog(tmp_path / ".forge", VALID_PROVIDER.replace("docs_url", "docs_ur1"))
    with pytest.raises(CatalogError, match=r"providers\.nebius"):
        effective_catalog(paths)


def test_user_catalog_rejects_default_model_not_in_models(tmp_path: Path) -> None:
    paths = _write_user_catalog(
        tmp_path / ".forge",
        VALID_PROVIDER.replace(
            'default_model = "deepseek-ai/DeepSeek-V4-Pro"', 'default_model = "missing"'
        ),
    )
    with pytest.raises(CatalogError, match=r"providers\.nebius\.default_model"):
        effective_catalog(paths)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            VALID_PROVIDER.replace('display_name = "Nebius AI Studio"', 'display_name = ""'),
            r"providers\.nebius\.display_name",
        ),
        (
            VALID_PROVIDER.replace(
                'models = ["deepseek-ai/DeepSeek-V4-Pro", "Qwen/Qwen3-Coder-480B-A35B-Instruct"]',
                'models = [""]',
            ),
            r"providers\.nebius\.models",
        ),
        (
            VALID_PROVIDER.replace('"deepseek-ai/DeepSeek-V4-Pro" = 163840', '"" = 163840'),
            r"providers\.nebius\.context_windows",
        ),
        (
            VALID_PROVIDER.replace(
                '"deepseek-ai/DeepSeek-V4-Pro" = 163840',
                '"deepseek-ai/DeepSeek-V4-Pro" = 0',
            ),
            r"providers\.nebius\.context_windows",
        ),
        (
            VALID_PROVIDER.replace(
                '"deepseek-ai/DeepSeek-V4-Pro" = 163840',
                '"deepseek-ai/DeepSeek-V4-Pro" = -1',
            ),
            r"providers\.nebius\.context_windows",
        ),
        (
            VALID_PROVIDER.replace(
                '"deepseek-ai/DeepSeek-V4-Pro" = 163840',
                '"deepseek-ai/DeepSeek-V4-Pro" = true',
            ),
            r"providers\.nebius\.context_windows",
        ),
        (
            VALID_PROVIDER.replace(
                '"deepseek-ai/DeepSeek-V4-Pro" = 163840',
                '"deepseek-ai/DeepSeek-V4-Pro" = "163840"',
            ),
            r"providers\.nebius\.context_windows",
        ),
    ],
)
def test_user_catalog_rejects_empty_and_coerced_values(
    tmp_path: Path,
    body: str,
    match: str,
) -> None:
    paths = _write_user_catalog(tmp_path / ".forge", body)
    with pytest.raises(CatalogError, match=match):
        effective_catalog(paths)


def test_user_catalog_rejects_bad_kind(tmp_path: Path) -> None:
    paths = _write_user_catalog(
        tmp_path / ".forge", VALID_PROVIDER.replace("openai-compatible", "grpc")
    )
    with pytest.raises(CatalogError, match="kind"):
        effective_catalog(paths)


def test_user_catalog_rejects_malformed_toml(tmp_path: Path) -> None:
    paths = _write_user_catalog(tmp_path / ".forge", "[[providers]\nname =")
    with pytest.raises(CatalogError, match="invalid TOML"):
        effective_catalog(paths)


def test_user_catalog_provider_appears_in_settings(tmp_path: Path) -> None:
    paths = _write_user_catalog(tmp_path / ".forge", VALID_PROVIDER)
    settings = load_provider_settings(paths)
    provider = settings.get_provider("nebius")
    assert provider.base_url == "https://api.studio.nebius.ai/v1"
    assert provider.default_model == "deepseek-ai/DeepSeek-V4-Pro"


def test_user_catalog_provider_appears_with_existing_settings_file(tmp_path: Path) -> None:
    paths = _write_user_catalog(tmp_path / ".forge", VALID_PROVIDER)
    (tmp_path / ".forge" / "providers.json").write_text(
        '{"default_provider": "openai", "providers": [{"type": "openai-compatible", '
        '"name": "openai", "base_url": "https://api.openai.com/v1", '
        '"api_key_env": "OPENAI_API_KEY", "models": ["gpt-5.5"], '
        '"default_model": "gpt-5.5"}], "scoped_models": []}',
        encoding="utf-8",
    )
    settings = load_provider_settings(paths)
    assert settings.get_provider("nebius").models[0] == "deepseek-ai/DeepSeek-V4-Pro"
