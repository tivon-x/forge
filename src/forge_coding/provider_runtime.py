"""Construct LangChain chat models from Forge's durable provider settings.

The provider catalog and credential store remain Forge-owned product concerns,
but the object crossing into the agent runtime is always a LangChain
``BaseChatModel``.  Integrations are imported lazily so a missing optional
provider produces an actionable error instead of an import-time crash.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from datetime import UTC, datetime
from os import environ
from typing import Any, cast

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai.chatgpt_oauth import _ChatGPTToken

from forge_coding.credentials import FileCredentialStore, OAuthCredential
from forge_coding.http_proxy import create_async_client
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

        class _ForgeReasoningChatOpenAI(ChatOpenAI):
            """``ChatOpenAI`` that preserves third-party ``reasoning_content`` fields.

            langchain-openai's ``ChatOpenAI`` targets the official OpenAI API
            only and deliberately drops non-standard response fields
            (documented at the top of its module); DeepSeek/vLLM-style
            providers stream their thinking text in
            ``delta.reasoning_content``.  Forge reads
            ``additional_kwargs["reasoning_content"]`` to project
            ``ThinkingDeltaEvent`` rows, so this subclass restores the field
            onto emitted chunks and messages instead of losing it at the
            provider boundary.
            """

            def _convert_chunk_to_generation_chunk(
                self,
                chunk: dict[str, Any],
                default_chunk_class: type,
                base_generation_info: dict[str, Any] | None,
            ) -> ChatGenerationChunk | None:
                generation_chunk = super()._convert_chunk_to_generation_chunk(
                    chunk,
                    default_chunk_class,
                    base_generation_info,
                )
                if generation_chunk is not None and isinstance(
                    generation_chunk.message, AIMessageChunk
                ):
                    _preserve_chunk_reasoning_content(generation_chunk.message, chunk)
                return generation_chunk

            def _create_chat_result(
                self,
                response: dict[str, Any] | Any,
                generation_info: dict[str, Any] | None = None,
            ) -> ChatResult:
                result = super()._create_chat_result(response, generation_info)
                response_dict = (
                    response if isinstance(response, dict) else response.model_dump(warnings=False)
                )
                choices = response_dict.get("choices")
                if not isinstance(choices, list):
                    return result
                for choice, generation in zip(choices, result.generations, strict=False):
                    if not isinstance(choice, Mapping) or not isinstance(
                        generation.message, AIMessage
                    ):
                        continue
                    raw_message = choice.get("message")
                    if not isinstance(raw_message, Mapping):
                        continue
                    reasoning = raw_message.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        generation.message.additional_kwargs["reasoning_content"] = reasoning
                return result

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
    # One dedicated httpx client per provider instance.  langchain-openai
    # caches its default async client by (base_url, timeout), so two providers
    # on the same endpoint would share one client; Forge's ``aclose_model``
    # would then close it for the sibling provider too, and the wrapper's
    # ``__del__`` re-closes it when the cache evicts it.  A dedicated client
    # keeps provider teardown local to the provider being closed.
    kwargs["http_async_client"] = _dedicated_openai_async_client(provider, selected_model)
    # Per-model api metadata wins over the provider default, so a provider can
    # mix APIs (Pi's xai serves most models on completions and grok-4.5 on
    # openai-responses).
    if _provider_api(provider, selected_model) == "openai-responses":
        kwargs["use_responses_api"] = True
    if metadata is not None and metadata.max_tokens is not None:
        kwargs["max_completion_tokens"] = metadata.max_tokens
    return _ForgeReasoningChatOpenAI(**kwargs)


def _dedicated_openai_async_client(
    provider: OpenAICompatibleProviderConfig,
    model: str,
) -> httpx.AsyncClient:
    """Build the per-provider httpx client passed as ``http_async_client``."""
    return create_async_client(
        base_url=_model_base_url(provider, model),
        timeout=provider.timeout_seconds,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=5.0,
        ),
    )


def _preserve_chunk_reasoning_content(
    message: AIMessageChunk,
    chunk: Mapping[str, Any],
) -> None:
    """Fold one streamed ``reasoning_content`` delta onto its chunk message.

    Each chunk carries only its own fragment; langchain-core's chunk merge
    concatenates string ``additional_kwargs`` values, so the accumulated
    message ends up with the full reasoning text.
    """
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        return
    reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        message.additional_kwargs["reasoning_content"] = reasoning


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
    """Close one client object exactly once, preferring its async close.

    Clients that langchain caches per ``(base_url, timeout)`` are
    process-lifetime singletons shared by every provider on the same endpoint;
    closing one would break its sibling providers for the rest of the process
    (the next request raises ``APIConnectionError: Connection error.`` from a
    closed httpx client).  Those shared clients are left for process exit.
    """
    if client is None or id(client) in seen:
        return
    seen.add(id(client))
    if _is_langchain_cached_client(client):
        return
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


def _is_langchain_cached_client(client: object) -> bool:
    """Return whether ``client`` wraps a langchain-cached shared httpx client.

    langchain-openai and langchain-anthropic both cache their default async
    httpx client (``_AsyncHttpxClientWrapper``) with ``lru_cache`` keyed by
    base URL and timeout.  Walk the ``_client`` chain (provider proxy -> SDK
    client -> httpx client) and skip those shared wrappers; every other client
    (Forge's dedicated httpx clients, per-instance SDK clients) stays closable.
    """
    wrapper_types: list[type] = []
    for module, attribute in (
        ("langchain_openai.chat_models._client_utils", "_AsyncHttpxClientWrapper"),
        ("langchain_anthropic._client_utils", "_AsyncHttpxClientWrapper"),
    ):
        try:
            wrapper = getattr(__import__(module, fromlist=[attribute]), attribute)
        except (ImportError, AttributeError):
            continue
        if isinstance(wrapper, type):
            wrapper_types.append(wrapper)
    if not wrapper_types:
        return False
    underlying: object = client
    for _ in range(4):
        if isinstance(underlying, tuple(wrapper_types)):
            return True
        next_client = getattr(underlying, "_client", None)
        if next_client is None:
            return False
        underlying = next_client
    return False
