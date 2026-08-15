"""Construct LangChain chat models from Forge's durable provider settings.

The provider catalog and credential store remain Forge-owned product concerns,
but the object crossing into the agent runtime is always a LangChain
``BaseChatModel``.  Integrations are imported lazily so a missing optional
provider produces an actionable error instead of an import-time crash.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from os import environ
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai.chatgpt_oauth import _ChatGPTToken

from forge_coding.credentials import FileCredentialStore, OAuthCredential
from forge_coding.oauth import (
    account_id_from_access_token,
    oauth_credential_is_expired,
    refresh_openai_codex_token,
)
from forge_coding.provider_config import (
    AnthropicProviderConfig,
    OpenAICodexProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderConfig,
    ProviderConfigError,
    _anthropic_thinking_budget_from_provider,
    _api_key_from_provider,
    _metadata_for_model,
    _model_base_url,
    _model_headers,
    _provider_api,
    _reasoning_effort_from_anthropic_provider,
    _reasoning_effort_from_provider,
    provider_thinking_levels,
    validate_provider_model,
)
from forge_coding.thinking import (
    ThinkingLevel,
    normalize_thinking_level,
    reasoning_effort_for_level,
)


class LoginRequiredChatModel(BaseChatModel):
    """Placeholder model that fails clearly before the user logs in.

    Lets the TUI open before a provider credential is configured; every model
    invocation raises the login message instead of talking to a model.  Native
    replacement for the historical ``LoginRequiredProvider`` placeholder.
    """

    _login_message: str

    def __init__(self, message: str) -> None:
        super().__init__()
        object.__setattr__(self, "_login_message", message)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError(self._login_message)

    @property
    def _llm_type(self) -> str:
        return "forge-login-required"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"llm_type": self._llm_type}


def create_model_provider(
    provider: ProviderConfig,
    *,
    credential_store: FileCredentialStore | None = None,
    model: str | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> BaseChatModel:
    """Create the official LangChain chat model for a Forge provider.

    ``BaseChatModel`` is the only runtime provider contract.  The optional
    integrations are loaded inside the branch that needs them and failures
    include the exact extra required to install.
    """

    if model is not None:
        validate_provider_model(provider, model)
    selected_model = model or provider.default_model
    credentials = credential_store or FileCredentialStore()

    if isinstance(provider, OpenAICodexProviderConfig):
        return _create_codex_model(
            provider,
            selected_model=selected_model,
            thinking_level=thinking_level,
            credential_store=credentials,
        )

    api_key = _api_key_from_provider(provider, credential_reader=credentials)
    if isinstance(provider, AnthropicProviderConfig):
        return _create_anthropic_model(
            provider,
            selected_model=selected_model,
            thinking_level=thinking_level,
            api_key=api_key,
        )

    if isinstance(provider, OpenAICompatibleProviderConfig):
        # Model-level api metadata wins over the provider default, so one
        # provider can mix APIs (Pi's opencode serves anthropic-messages,
        # google-generative-ai, openai-completions, and openai-responses
        # models behind a single gateway).
        model_api = _provider_api(provider, selected_model)
        if model_api == "anthropic-messages":
            return _create_anthropic_model(
                provider,
                selected_model=selected_model,
                thinking_level=thinking_level,
                api_key=api_key,
            )
        if model_api == "google-generative-ai":
            return _create_google_model(
                provider,
                selected_model=selected_model,
                thinking_level=thinking_level,
                api_key=api_key,
            )
        if model_api == "mistral-conversations":
            return _create_mistral_model(
                provider,
                selected_model=selected_model,
                thinking_level=thinking_level,
                api_key=api_key,
            )
        return _create_openai_model(
            provider,
            selected_model=selected_model,
            thinking_level=thinking_level,
            api_key=api_key,
        )

    raise ProviderConfigError(f"Unsupported provider config: {provider.name}")


def _create_openai_model(
    provider: OpenAICompatibleProviderConfig,
    *,
    selected_model: str,
    thinking_level: ThinkingLevel | None,
    api_key: str,
) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in clean installs
        raise ProviderConfigError(
            "Provider requires the LangChain OpenAI integration. "
            "Install it with: pip install 'forge-ai' (or 'forge-ai[providers]')."
        ) from exc

    metadata = _metadata_for_model(provider, selected_model)
    kwargs: dict[str, Any] = {
        "model": selected_model,
        "api_key": api_key,
        "base_url": _model_base_url(provider, selected_model),
        "default_headers": _model_headers(provider, selected_model) or None,
        "timeout": provider.timeout_seconds,
        "max_retries": provider.max_retries,
        "streaming": True,
    }
    effort = _reasoning_effort_from_provider(
        provider,
        model=selected_model,
        thinking_level=thinking_level,
    )
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    # Per-model api metadata wins over the provider default, so a provider can
    # mix APIs (Pi's xai serves most models on completions and grok-4.5 on
    # openai-responses).
    if _provider_api(provider, selected_model) == "openai-responses":
        kwargs["use_responses_api"] = True
    if metadata is not None and metadata.max_tokens is not None:
        kwargs["max_completion_tokens"] = metadata.max_tokens
    return ChatOpenAI(**kwargs)


def _create_anthropic_model(
    provider: ProviderConfig,
    *,
    selected_model: str,
    thinking_level: ThinkingLevel | None,
    api_key: str,
) -> BaseChatModel:
    """Create ChatAnthropic for an anthropic-messages model.

    Accepts any provider config whose model-level api is anthropic-messages,
    including mixed-API gateways like Pi's opencode.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in clean installs
        raise ProviderConfigError(
            "Provider requires the LangChain Anthropic integration. "
            "Install it with: pip install 'forge-ai[providers]'."
        ) from exc

    kwargs: dict[str, Any] = {
        "model_name": selected_model,
        "api_key": api_key,
        "base_url": _model_base_url(provider, selected_model),
        "default_headers": _model_headers(provider, selected_model) or None,
        "timeout": provider.timeout_seconds,
        "max_retries": provider.max_retries,
        "streaming": True,
    }
    budget = _anthropic_thinking_budget_from_provider(
        provider,
        model=selected_model,
        thinking_level=thinking_level,
    )
    effort = _reasoning_effort_from_anthropic_provider(
        provider,
        model=selected_model,
        thinking_level=thinking_level,
    )
    if budget is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if effort is not None:
        kwargs["effort"] = effort
    return ChatAnthropic(**kwargs)


def _create_google_model(
    provider: OpenAICompatibleProviderConfig,
    *,
    selected_model: str,
    thinking_level: ThinkingLevel | None,
    api_key: str,
) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in clean installs
        raise ProviderConfigError(
            "Provider requires the LangChain Google integration. "
            "Install it with: pip install 'forge-ai[providers]'."
        ) from exc

    metadata = _metadata_for_model(provider, selected_model)
    kwargs: dict[str, Any] = {
        "model": selected_model,
        "google_api_key": api_key,
        "additional_headers": _model_headers(provider, selected_model) or None,
        "request_timeout": provider.timeout_seconds,
        "retries": provider.max_retries,
        "streaming": True,
    }
    if metadata is not None and metadata.max_tokens is not None:
        kwargs["max_tokens"] = metadata.max_tokens
    # The gateway base URL already carries the version path (Pi convention),
    # so stop the google-genai SDK from appending another /v1beta.
    kwargs["base_url"] = _model_base_url(provider, selected_model)
    kwargs["api_version"] = ""
    normalized = _normalized_thinking_level(provider, selected_model, thinking_level)
    if normalized in {"minimal", "low", "medium", "high"}:
        kwargs["thinking_level"] = cast(Any, normalized)
    return ChatGoogleGenerativeAI(**kwargs)


def _create_mistral_model(
    provider: OpenAICompatibleProviderConfig,
    *,
    selected_model: str,
    thinking_level: ThinkingLevel | None,
    api_key: str,
) -> BaseChatModel:
    try:
        from langchain_mistralai import ChatMistralAI
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in clean installs
        raise ProviderConfigError(
            "Provider requires the LangChain Mistral integration. "
            "Install it with: pip install 'forge-ai[providers]'."
        ) from exc

    metadata = _metadata_for_model(provider, selected_model)
    kwargs: dict[str, Any] = {
        "model": selected_model,
        "api_key": api_key,
        "base_url": _model_base_url(provider, selected_model),
        "timeout": int(provider.timeout_seconds),
        "max_retries": provider.max_retries,
        "streaming": True,
    }
    if metadata is not None and metadata.max_tokens is not None:
        kwargs["max_tokens"] = metadata.max_tokens
    del thinking_level
    return ChatMistralAI(**kwargs)


def _create_codex_model(
    provider: OpenAICodexProviderConfig,
    *,
    selected_model: str,
    thinking_level: ThinkingLevel | None,
    credential_store: FileCredentialStore,
) -> BaseChatModel:
    """Create the isolated experimental LangChain Codex model.

    The private import is intentionally kept in this one factory.  The class
    pins its own official Codex endpoint and rejects caller-controlled base
    URLs, preventing OAuth tokens from being sent to a custom host.
    """

    try:
        from langchain_openai.chat_models.codex import _ChatOpenAICodex
        from langchain_openai.chatgpt_oauth import _ChatGPTOAuthTokenProvider
    except ModuleNotFoundError as exc:  # pragma: no cover - default install includes it
        raise ProviderConfigError(
            "Experimental Codex support requires langchain-openai==1.4.1."
        ) from exc

    resolver = OpenAICodexCredentialResolver(provider, credential_store=credential_store)

    token_provider = _ForgeCodexTokenProvider(resolver)
    if not isinstance(token_provider, _ChatGPTOAuthTokenProvider):
        raise TypeError("Forge Codex token provider does not satisfy LangChain's OAuth contract")

    class ForgeCodexChatModel(_ChatOpenAICodex):
        """Native Codex model.

        Lifecycle cleanup is handled by :func:`aclose_model`, which closes
        ``root_async_client`` (``AsyncOpenAI``) for this class.
        """

    reasoning_effort = _codex_reasoning_effort(
        provider,
        model=selected_model,
        thinking_level=thinking_level,
    )
    native_model = ForgeCodexChatModel(
        model=selected_model,
        token_provider=token_provider,
        timeout=provider.timeout_seconds,
        max_retries=provider.max_retries,
        reasoning_effort=reasoning_effort,
        instructions="You are Forge, a coding agent.",
        originator="forge",
    )
    return native_model


def _normalized_thinking_level(
    provider: OpenAICompatibleProviderConfig,
    model: str,
    thinking_level: ThinkingLevel | None,
) -> str | None:
    if thinking_level is None:
        return None
    levels = provider_thinking_levels(provider, model=model)
    normalized = normalize_thinking_level(thinking_level)
    if normalized not in levels:
        return None
    if normalized == "off":
        return None
    if normalized == "minimal":
        return "low"
    return normalized


def _codex_reasoning_effort(
    provider: OpenAICodexProviderConfig,
    *,
    model: str | None,
    thinking_level: ThinkingLevel | None,
) -> str | None:
    if thinking_level is None or provider.thinking_parameter != "reasoning.effort":
        return None
    levels = provider_thinking_levels(provider, model=model)
    if not levels:
        return None
    normalized = normalize_thinking_level(thinking_level)
    if normalized not in levels:
        selected_model = model or provider.default_model
        available = ", ".join(levels)
        raise ProviderConfigError(
            f"Thinking mode {normalized} is not available for "
            f"{provider.name}:{selected_model}. Available modes: {available}"
        )
    if normalized in {"off", "minimal"}:
        return None if normalized == "off" else "low"
    return reasoning_effort_for_level(normalized)


# Environment-variable Codex tokens carry no expiry; store a far-future
# datetime (year 3001, the ``datetime.fromtimestamp`` ceiling on Windows) in
# milliseconds.  The token provider divides by 1000 before
# ``datetime.fromtimestamp``, so the value must stay inside datetime's range
# (a 2**63-style sentinel would wrap to 1970 again).
_ENV_TOKEN_EXPIRY_MS = 32_536_850_399_000


class _ForgeCodexTokenProvider:
    """ChatGPT OAuth token provider backed by Forge's credential store.

    Module-private so the factory stays on the LangChain contract while the
    class remains directly testable.  Forge stores ``OAuthCredential.expires``
    in milliseconds; ``_ChatGPTToken.expires_at`` is a timezone-aware
    ``datetime``, so the conversion divides by 1000 exactly once here.
    """

    def __init__(self, resolver: OpenAICodexCredentialResolver) -> None:
        self._resolver = resolver

    def _token_from_credentials(self, credential: OAuthCredential) -> Any:
        return _ChatGPTToken(
            access_token=credential.access,
            refresh_token=credential.refresh or "forge-env-token",
            expires_at=datetime.fromtimestamp(credential.expires / 1000, tz=UTC),
            account_id=credential.account_id,
        )

    def get_token(self) -> Any:
        return self._token_from_credentials(self._resolver.sync_resolve())

    async def aget_token(self) -> Any:
        return self._token_from_credentials(await self._resolver.resolve())

    def get_access_token(self) -> str:
        return cast(str, self.get_token().access_token)

    async def aget_access_token(self) -> str:
        return cast(str, (await self.aget_token()).access_token)


class OpenAICodexCredentialResolver:
    """Resolve and refresh Forge Codex OAuth credentials without ``~/.codex``."""

    def __init__(
        self,
        provider: OpenAICodexProviderConfig,
        *,
        credential_store: FileCredentialStore,
    ) -> None:
        self._provider = provider
        self._credential_store = credential_store

    async def resolve(self) -> OAuthCredential:
        credential_name = self._provider.credential_name
        if credential_name:
            credential = self._credential_store.get_oauth(credential_name)
            if credential is not None:
                return await self._refresh_if_needed(credential_name, credential)

        access_token = environ.get(self._provider.api_key_env)
        if access_token:
            account_id = account_id_from_access_token(access_token)
            if account_id is None:
                raise RuntimeError(
                    f"{self._provider.api_key_env} must contain an OpenAI Codex access JWT"
                )
            return OAuthCredential(
                access=access_token,
                refresh="forge-env-token",
                expires=_ENV_TOKEN_EXPIRY_MS,
                account_id=account_id,
            )

        raise RuntimeError(
            f"Missing OpenAI Codex OAuth credentials. Run /login {self._provider.name}."
        )

    def sync_resolve(self) -> OAuthCredential:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.resolve())
        credential_name = self._provider.credential_name
        if credential_name:
            credential = self._credential_store.get_oauth(credential_name)
            if credential is not None and not oauth_credential_is_expired(credential):
                return credential
        access_token = environ.get(self._provider.api_key_env)
        if access_token:
            account_id = account_id_from_access_token(access_token)
            if account_id is not None:
                return OAuthCredential(
                    access=access_token,
                    refresh="forge-env-token",
                    expires=_ENV_TOKEN_EXPIRY_MS,
                    account_id=account_id,
                )
        raise RuntimeError("Codex credentials must be loaded before synchronous invocation")

    async def _refresh_if_needed(
        self,
        credential_name: str,
        credential: OAuthCredential,
    ) -> OAuthCredential:
        if not oauth_credential_is_expired(credential):
            return credential
        refreshed = await refresh_openai_codex_token(credential.refresh)
        self._credential_store.set_oauth(credential_name, refreshed)
        return refreshed


async def aclose_model(model: BaseChatModel) -> None:
    """Close a LangChain model's async client when it exposes one.

    Explicitly covers the currently supported official provider objects, each
    closed at most once:

    - ``BaseChatModel.aclose()`` when the integration defines it;
    - OpenAI ``root_async_client.close()`` (sync ``AsyncOpenAI`` close);
    - Anthropic ``_async_client.close()`` (sync ``AsyncAnthropic`` close);
    - Mistral ``async_client.aclose()`` (async ``httpx.AsyncClient``).

    Missing close interfaces are safe no-ops.  A failure closing one client
    does not stop the remaining clients from being closed; all collected
    errors are raised together afterwards.
    """

    seen: set[int] = set()
    errors: list[Exception] = []
    aclose = getattr(model, "aclose", None)
    if callable(aclose):
        try:
            result = aclose()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - one bad client must not leak others
            errors.append(exc)
    for client in (
        getattr(model, "async_client", None),
        getattr(model, "root_async_client", None),
        getattr(model, "_async_client", None),
    ):
        try:
            await _close_client(client, seen)
        except Exception as exc:  # noqa: BLE001 - one bad client must not leak others
            errors.append(exc)
    if errors:
        raise ExceptionGroup("Failed to close model client", errors)


async def _close_client(client: object, seen: set[int]) -> None:
    """Close one client object exactly once, preferring its async close."""
    if client is None or id(client) in seen:
        return
    seen.add(id(client))
    close = getattr(client, "aclose", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result
        return
    close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result
