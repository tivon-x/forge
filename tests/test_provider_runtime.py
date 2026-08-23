import json
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatResult

from forge_coding.providers import runtime as provider_runtime
from forge_coding.providers.auth.credentials import FileCredentialStore, OAuthCredential
from forge_coding.providers.config import (
    AnthropicProviderConfig,
    OpenAICodexProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderConfigError,
    ProviderModelMetadata,
)
from forge_coding.providers.runtime import (
    OpenAICodexCredentialResolver,
    aclose_model,
    create_model_provider,
)


@pytest.fixture()
def credential_store(tmp_path) -> FileCredentialStore:
    return FileCredentialStore(tmp_path / "credentials.json")


def test_create_model_provider_returns_openai_codex_provider(tmp_path) -> None:
    store = FileCredentialStore(tmp_path / "credentials.json")

    provider = create_model_provider(
        OpenAICodexProviderConfig(),
        credential_store=store,
    )

    assert isinstance(provider, BaseChatModel)
    assert provider.model_name == "gpt-5.5"


def test_create_model_provider_rejects_model_not_declared_for_provider(tmp_path) -> None:
    store = FileCredentialStore(tmp_path / "credentials.json")
    provider_config = OpenAICompatibleProviderConfig(
        name="local",
        models=("qwen",),
        default_model="qwen",
    )

    with pytest.raises(
        ProviderConfigError,
        match="Model is not configured for provider local: llama",
    ):
        create_model_provider(provider_config, credential_store=store, model="llama")


def test_create_model_provider_maps_codex_reasoning_effort_like_pi(tmp_path) -> None:
    store = FileCredentialStore(tmp_path / "credentials.json")
    provider_config = OpenAICodexProviderConfig(
        thinking_levels=("off", "minimal", "low", "medium", "high", "xhigh"),
        thinking_models=("gpt-5.5",),
        thinking_parameter="reasoning.effort",
    )

    off_provider = create_model_provider(
        provider_config,
        credential_store=store,
        model="gpt-5.5",
        thinking_level="off",
    )
    minimal_provider = create_model_provider(
        provider_config,
        credential_store=store,
        model="gpt-5.5",
        thinking_level="minimal",
    )
    xhigh_provider = create_model_provider(
        provider_config,
        credential_store=store,
        model="gpt-5.5",
        thinking_level="xhigh",
    )

    assert isinstance(off_provider, BaseChatModel)
    assert isinstance(minimal_provider, BaseChatModel)
    assert isinstance(xhigh_provider, BaseChatModel)
    assert off_provider.reasoning_effort is None
    assert minimal_provider.reasoning_effort == "low"
    assert xhigh_provider.reasoning_effort == "xhigh"


@pytest.mark.anyio
async def test_openai_codex_credential_resolver_refreshes_expired_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "openai-codex",
        OAuthCredential(
            access="old-access",
            refresh="old-refresh",
            expires=1,
            account_id="old-account",
        ),
    )

    async def fake_refresh(refresh_token: str) -> OAuthCredential:
        assert refresh_token == "old-refresh"
        return OAuthCredential(
            access="new-access",
            refresh="new-refresh",
            expires=9999999999999,
            account_id="new-account",
        )

    monkeypatch.setattr(provider_runtime, "refresh_openai_codex_token", fake_refresh)

    resolver = OpenAICodexCredentialResolver(
        OpenAICodexProviderConfig(),
        credential_store=store,
    )

    credentials = await resolver.resolve()

    assert credentials.access == "new-access"
    assert credentials.account_id == "new-account"
    assert store.get_oauth("openai-codex") == OAuthCredential(
        access="new-access",
        refresh="new-refresh",
        expires=9999999999999,
        account_id="new-account",
    )


# --------------------------------------------------------------------------- #
# G5: official LangChain model construction for the production provider extras
# --------------------------------------------------------------------------- #
def test_create_openai_model_returns_chat_openai(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = create_model_provider(
        OpenAICompatibleProviderConfig(
            name="openai",
            models=("gpt-4o",),
            default_model="gpt-4o",
        ),
        credential_store=credential_store,
    )

    from langchain_openai import ChatOpenAI

    assert isinstance(provider, ChatOpenAI)
    assert isinstance(provider, BaseChatModel)
    assert provider.model_name == "gpt-4o"


def test_create_anthropic_model_returns_chat_anthropic(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    provider = create_model_provider(
        AnthropicProviderConfig(),
        credential_store=credential_store,
    )

    from langchain_anthropic import ChatAnthropic

    assert isinstance(provider, ChatAnthropic)
    assert isinstance(provider, BaseChatModel)


def test_create_google_model_returns_chat_google_generative_ai(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    provider = create_model_provider(
        OpenAICompatibleProviderConfig(
            name="google",
            api="google-generative-ai",
            api_key_env="GOOGLE_API_KEY",
            models=("gemini-2.5-pro",),
            default_model="gemini-2.5-pro",
        ),
        credential_store=credential_store,
    )

    from langchain_google_genai import ChatGoogleGenerativeAI

    assert isinstance(provider, ChatGoogleGenerativeAI)
    assert isinstance(provider, BaseChatModel)


def test_create_model_provider_dispatches_mixed_api_provider_per_model(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    """Pi's opencode gateway mixes APIs behind one provider; each model routes
    to the integration matching its model-level api metadata.
    """
    pytest.importorskip("langchain_openai")
    pytest.importorskip("langchain_anthropic")
    pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")

    provider = OpenAICompatibleProviderConfig(
        name="opencode",
        api="openai-completions",
        api_key_env="OPENCODE_API_KEY",
        models=("claude-haiku-4-5", "gemini-3-flash", "gpt-5.5", "kimi-k2.6"),
        default_model="kimi-k2.6",
        model_metadata={
            "claude-haiku-4-5": ProviderModelMetadata(
                api="anthropic-messages",
                base_url="https://opencode.ai/zen",
                reasoning=True,
            ),
            "gemini-3-flash": ProviderModelMetadata(
                api="google-generative-ai",
                base_url="https://opencode.ai/zen/v1",
                reasoning=True,
            ),
            "gpt-5.5": ProviderModelMetadata(
                api="openai-responses",
                base_url="https://opencode.ai/zen/v1",
                reasoning=True,
            ),
            "kimi-k2.6": ProviderModelMetadata(
                api="openai-completions",
                base_url="https://opencode.ai/zen/v1",
                reasoning=True,
            ),
        },
    )

    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI

    anthropic_model = create_model_provider(
        provider, credential_store=credential_store, model="claude-haiku-4-5"
    )
    google_model = create_model_provider(
        provider, credential_store=credential_store, model="gemini-3-flash"
    )
    responses_model = create_model_provider(
        provider, credential_store=credential_store, model="gpt-5.5"
    )
    completions_model = create_model_provider(
        provider, credential_store=credential_store, model="kimi-k2.6"
    )

    assert isinstance(anthropic_model, ChatAnthropic)
    assert anthropic_model.anthropic_api_url == "https://opencode.ai/zen"
    assert isinstance(google_model, ChatGoogleGenerativeAI)
    assert google_model.base_url == "https://opencode.ai/zen/v1"
    assert isinstance(responses_model, ChatOpenAI)
    assert responses_model.use_responses_api is True
    assert responses_model.openai_api_base == "https://opencode.ai/zen/v1"
    assert isinstance(completions_model, ChatOpenAI)
    assert completions_model.use_responses_api is not True


def test_create_mistral_model_returns_chat_mistral_ai(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    pytest.importorskip("langchain_mistralai")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    provider = create_model_provider(
        OpenAICompatibleProviderConfig(
            name="mistral",
            api="mistral-conversations",
            api_key_env="MISTRAL_API_KEY",
            models=("mistral-large",),
            default_model="mistral-large",
        ),
        credential_store=credential_store,
    )

    from langchain_mistralai import ChatMistralAI

    assert isinstance(provider, ChatMistralAI)
    assert isinstance(provider, BaseChatModel)


def test_create_provider_reports_missing_integration_with_install_hint(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "langchain_openai":
            raise ModuleNotFoundError("No module named 'langchain_openai'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(ProviderConfigError, match="OpenAI integration"):
        create_model_provider(
            OpenAICompatibleProviderConfig(name="openai"),
            credential_store=credential_store,
        )


# --------------------------------------------------------------------------- #
# G5: mock-transport request-parameter coverage for the production extras
# --------------------------------------------------------------------------- #


def _capture_handler(requests: list[tuple[str, str, dict[str, str], bytes]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url), dict(request.headers), request.content))
        body = request.content.decode(errors="replace")
        if '"stream":true' in body:
            chunks = [
                'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
                '"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},'
                '"finish_reason":null}]}',
                'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            return httpx.Response(
                200,
                content=("\n\n".join(chunks) + "\n\n").encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return handler


def _anthropic_handler(requests: list[tuple[str, str, dict[str, str], bytes]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url), dict(request.headers), request.content))
        events = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
            '"type":"message","role":"assistant","content":[],"model":"m","stop_reason":null}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"hi"}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            'event: message_delta\ndata: {"type":"message_delta",'
            '"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]
        return httpx.Response(
            200,
            content=("\n\n".join(events) + "\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )

    return handler


def _google_handler(requests: list[tuple[str, str, dict[str, str], bytes]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url), dict(request.headers), request.content))
        # The genai SDK accepts a plain JSON body even for the streaming URL.
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}}]},
        )

    return handler


@pytest.mark.anyio
async def test_openai_model_mock_transport_request_parameters(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = OpenAICompatibleProviderConfig(
        name="openai",
        models=("gpt-4o",),
        default_model="gpt-4o",
    )
    provider = create_model_provider(config, credential_store=credential_store)

    from openai import AsyncOpenAI

    captured: list[tuple[str, str, dict[str, str], bytes]] = []
    transport = httpx.MockTransport(_capture_handler(captured))
    base_url = provider.openai_api_base
    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url=base_url,
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    provider.root_async_client = sdk
    provider.async_client = sdk.chat.completions

    result = await provider.ainvoke([HumanMessage(content="hello")])
    assert message_text(result) == "hi"
    assert len(captured) == 1
    method, url, headers, body = captured[0]
    assert method == "POST"
    assert url.startswith(base_url) and url.endswith("/chat/completions")
    assert headers.get("authorization") == "Bearer test-key"
    payload = json.loads(body)
    assert payload["model"] == "gpt-4o"
    assert payload["stream"] is True
    assert payload["messages"][0]["content"] == "hello"


@pytest.mark.anyio
async def test_openai_model_preserves_deepseek_style_reasoning_content(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    """DeepSeek/vLLM ``reasoning_content`` survives the provider boundary.

    langchain-openai's plain ``ChatOpenAI`` drops non-standard response
    fields, so Forge's reasoning-aware subclass must restore them onto chunk
    and aggregated messages or thinking tokens never reach the TUI.
    """
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = OpenAICompatibleProviderConfig(
        name="deepseek",
        models=("deepseek-v4-flash",),
        default_model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
    )
    provider = create_model_provider(config, credential_store=credential_store)

    from openai import AsyncOpenAI

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode(errors="replace")
        assert '"stream":true' in body
        chunks = [
            'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":null,'
            '"reasoning_content":"Let me reason"}}]}',
            'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
            '"choices":[{"index":0,"delta":{"reasoning_content":" carefully."}}]}',
            'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
            '"choices":[{"index":0,"delta":{"content":"Final answer"}}]}',
            'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        return httpx.Response(
            200,
            content=("\n\n".join(chunks) + "\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )

    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url=provider.openai_api_base,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    provider.root_async_client = sdk
    provider.async_client = sdk.chat.completions

    thinking_deltas: list[str] = []
    text_deltas: list[str] = []
    async for chunk in provider.astream([HumanMessage(content="hello")]):
        message = getattr(chunk, "message", chunk)
        reasoning = message.additional_kwargs.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            thinking_deltas.append(reasoning)
        if message.content:
            text_deltas.append(message.content)

    assert thinking_deltas == ["Let me reason", " carefully."]
    assert text_deltas == ["Final answer"]

    aggregated = await provider.ainvoke([HumanMessage(content="hello")])
    assert aggregated.additional_kwargs.get("reasoning_content") == "Let me reason carefully."
    assert aggregated.content == "Final answer"


@pytest.mark.anyio
async def test_aclose_model_never_breaks_sibling_openai_provider(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    """Closing one provider must not close another provider on the same endpoint.

    langchain-openai caches its default async httpx client by (base_url,
    timeout), so without Forge's dedicated per-provider clients, two deepseek
    providers would share one httpx client; closing one (session resume/new)
    then makes the sibling fail with ``APIConnectionError: Connection error.``
    ("client has been closed") for the rest of the process.
    """
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = OpenAICompatibleProviderConfig(
        name="deepseek",
        models=("deepseek-v4-flash",),
        default_model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
    )
    first = create_model_provider(config, credential_store=credential_store)
    second = create_model_provider(config, credential_store=credential_store)

    first_client = first.root_async_client._client
    second_client = second.root_async_client._client
    assert first_client is not second_client

    from openai import AsyncOpenAI

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode(errors="replace")
        if '"stream":true' in body:
            chunks = [
                'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
                '"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}',
                'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            return httpx.Response(
                200,
                content=("\n\n".join(chunks) + "\n\n").encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}}
                ],
            },
        )

    def install_mock_sdk(provider: Any) -> Any:
        sdk = AsyncOpenAI(
            api_key="test-key",
            base_url=provider.openai_api_base,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )
        provider.root_async_client = sdk
        provider.async_client = sdk.chat.completions
        return sdk

    first_sdk = install_mock_sdk(first)
    second_sdk = install_mock_sdk(second)

    await aclose_model(first)
    assert first_sdk._client.is_closed
    assert not second_sdk._client.is_closed

    streamed = ""
    async for chunk in second.astream([HumanMessage(content="hello")]):
        streamed += getattr(chunk, "content", "") or ""
    assert "hi" in streamed


@pytest.mark.anyio
async def test_aclose_model_skips_langchain_cached_anthropic_client(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    """The shared langchain-cached httpx client is left for process exit.

    ChatAnthropic builds its client from langchain-anthropic's lru_cache-ed
    default, so two anthropic providers share one httpx client.  Closing one
    provider must not close the shared client or the sibling breaks.
    """
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = AnthropicProviderConfig()
    first = create_model_provider(config, credential_store=credential_store)
    second = create_model_provider(config, credential_store=credential_store)

    first_client = first._async_client._client
    second_client = second._async_client._client
    assert first_client is second_client

    await aclose_model(first)
    assert not second_client.is_closed


@pytest.mark.anyio
async def test_anthropic_model_mock_transport_request_parameters(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = AnthropicProviderConfig()
    provider = create_model_provider(config, credential_store=credential_store)

    import langchain_anthropic.chat_models as anthropic_chat_models

    captured: list[tuple[str, str, dict[str, str], bytes]] = []

    def fake_async_client(**kwargs: Any) -> Any:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(_anthropic_handler(captured)),
            base_url=kwargs.get("base_url"),
            timeout=kwargs.get("timeout", 10),
        )

    monkeypatch.setattr(anthropic_chat_models, "_get_default_async_httpx_client", fake_async_client)

    result = await provider.ainvoke([HumanMessage(content="hello")])
    assert message_text(result) == "hi"
    assert len(captured) == 1
    method, url, headers, body = captured[0]
    assert method == "POST"
    assert url.endswith("/v1/messages")
    assert headers.get("x-api-key") == "test-key"
    payload = json.loads(body)
    assert payload["model"] == config.default_model
    assert payload["stream"] is True
    assert payload["messages"][0]["content"] == "hello"


@pytest.mark.anyio
async def test_mistral_model_mock_transport_request_parameters(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    pytest.importorskip("langchain_mistralai")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    config = OpenAICompatibleProviderConfig(
        name="mistral",
        api="mistral-conversations",
        api_key_env="MISTRAL_API_KEY",
        models=("mistral-large",),
        default_model="mistral-large",
    )
    provider = create_model_provider(config, credential_store=credential_store)

    captured: list[tuple[str, str, dict[str, str], bytes]] = []
    api_key = provider.mistral_api_key
    api_key_text = (
        api_key.get_secret_value() if hasattr(api_key, "get_secret_value") else str(api_key)
    )
    provider.async_client = httpx.AsyncClient(
        base_url=getattr(provider, "endpoint", None) or "https://api.mistral.ai",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key_text}",
        },
        transport=httpx.MockTransport(_capture_handler(captured)),
    )

    result = await provider.ainvoke([HumanMessage(content="hello")])
    assert message_text(result) == "hi"
    assert len(captured) == 1
    method, url, headers, body = captured[0]
    assert method == "POST"
    assert url.endswith("/chat/completions")
    assert headers.get("authorization") == "Bearer test-key"
    payload = json.loads(body)
    assert payload["model"] == "mistral-large"
    assert payload["stream"] is True
    assert payload["messages"][0]["content"] == "hello"


def test_google_model_mock_transport_request_parameters(
    monkeypatch: pytest.MonkeyPatch, credential_store: FileCredentialStore
) -> None:
    pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    config = OpenAICompatibleProviderConfig(
        name="google",
        api="google-generative-ai",
        api_key_env="GOOGLE_API_KEY",
        models=("gemini-2.5-pro",),
        default_model="gemini-2.5-pro",
    )
    provider = create_model_provider(config, credential_store=credential_store)

    from google.genai import Client, types

    captured: list[tuple[str, str, dict[str, str], bytes]] = []
    mock = httpx.MockTransport(_google_handler(captured))
    api_key = provider.google_api_key
    api_key_text = (
        api_key.get_secret_value() if hasattr(api_key, "get_secret_value") else str(api_key)
    )
    provider.client = Client(
        api_key=api_key_text,
        http_options=types.HttpOptions(
            base_url="https://generativelanguage.googleapis.com",
            httpx_client=httpx.Client(transport=mock),
            httpx_async_client=httpx.AsyncClient(transport=mock),
        ),
    )

    result = provider.invoke([HumanMessage(content="hello")])
    assert message_text(result) == "hi"
    assert len(captured) == 1
    method, url, headers, body = captured[0]
    assert method == "POST"
    assert url.startswith("https://generativelanguage.googleapis.com")
    assert "streamGenerateContent" in url
    assert headers.get("x-goog-api-key") == "test-key"
    payload = json.loads(body)
    assert payload["contents"][0]["parts"][0]["text"] == "hello"


def message_text(message: object) -> str:
    """Extract plain text from a chat result for offline assertions."""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Codex token provider: OAuthCredential.expires is milliseconds; the LangChain
# token dataclass expects a timezone-aware datetime.  The conversion divides by
# 1000 exactly once, so a one-hour-ahead millisecond expiry yields a datetime
# within 1 ms of the stored value and expired credentials still refresh.
# --------------------------------------------------------------------------- #
def test_codex_token_provider_sync_and_async_accept_future_millisecond_expiry(
    tmp_path,
) -> None:
    from time import time

    store = FileCredentialStore(tmp_path / "credentials.json")
    expires_ms = int(time() * 1000) + 3_600_000
    store.set_oauth(
        "openai-codex",
        OAuthCredential(
            access="future-access",
            refresh="future-refresh",
            expires=expires_ms,
            account_id="future-account",
        ),
    )
    resolver = OpenAICodexCredentialResolver(
        OpenAICodexProviderConfig(),
        credential_store=store,
    )
    token_provider = provider_runtime._ForgeCodexTokenProvider(resolver)

    sync_token = token_provider.get_token()
    async_token = asyncio_run(token_provider.aget_token())

    assert sync_token.access_token == "future-access"
    assert async_token.access_token == "future-access"
    assert sync_token.account_id == "future-account"
    assert async_token.account_id == "future-account"
    assert sync_token.expires_at.tzinfo is not None
    assert async_token.expires_at.tzinfo is not None
    # The stored value is millisecond precision; the round trip through the
    # datetime conversion must not drift by more than one millisecond.
    assert abs(sync_token.expires_at.timestamp() * 1000 - expires_ms) <= 1.0
    assert abs(async_token.expires_at.timestamp() * 1000 - expires_ms) <= 1.0


def test_codex_token_provider_datetime_matches_milliseconds_exactly(tmp_path) -> None:
    from datetime import UTC, datetime
    from time import time

    store = FileCredentialStore(tmp_path / "credentials.json")
    expires_ms = int(time() * 1000) + 3_600_000
    store.set_oauth(
        "openai-codex",
        OAuthCredential(
            access="access",
            refresh="refresh",
            expires=expires_ms,
            account_id="account",
        ),
    )
    resolver = OpenAICodexCredentialResolver(
        OpenAICodexProviderConfig(),
        credential_store=store,
    )
    token = provider_runtime._ForgeCodexTokenProvider(resolver).get_token()

    expected = datetime.fromtimestamp(expires_ms / 1000, tz=UTC)
    assert token.expires_at == expected


@pytest.mark.anyio
async def test_codex_env_token_expiry_is_far_future(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Environment-variable Codex tokens must not wrap to 1970 after /1000."""
    from datetime import UTC, datetime, timedelta

    from forge_coding.providers.auth.oauth import oauth_credential_is_expired

    monkeypatch.setenv("OPENAI_CODEX_ACCESS_TOKEN", "env-jwt-token")
    monkeypatch.setattr(
        provider_runtime,
        "account_id_from_access_token",
        lambda _token: "env-account",
    )
    store = FileCredentialStore(tmp_path / "credentials.json")
    resolver = OpenAICodexCredentialResolver(
        OpenAICodexProviderConfig(),
        credential_store=store,
    )

    token = await provider_runtime._ForgeCodexTokenProvider(resolver).aget_token()
    credential = await resolver.resolve()

    assert token.expires_at > datetime.now(UTC) + timedelta(days=365 * 10)
    assert not oauth_credential_is_expired(credential)
    assert token.expires_at.year >= 3000


@pytest.mark.anyio
async def test_codex_token_provider_refreshes_expired_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "openai-codex",
        OAuthCredential(
            access="expired-access",
            refresh="expired-refresh",
            expires=1,
            account_id="expired-account",
        ),
    )

    async def fake_refresh(refresh_token: str) -> OAuthCredential:
        assert refresh_token == "expired-refresh"
        return OAuthCredential(
            access="fresh-access",
            refresh="fresh-refresh",
            expires=1_784_882_400_000,
            account_id="fresh-account",
        )

    monkeypatch.setattr(provider_runtime, "refresh_openai_codex_token", fake_refresh)

    resolver = OpenAICodexCredentialResolver(
        OpenAICodexProviderConfig(),
        credential_store=store,
    )
    token = await provider_runtime._ForgeCodexTokenProvider(resolver).aget_token()

    assert token.access_token == "fresh-access"
    assert token.refresh_token == "fresh-refresh"
    assert token.account_id == "fresh-account"


def asyncio_run(awaitable: Any) -> Any:
    import asyncio

    return asyncio.run(awaitable)


# --------------------------------------------------------------------------- #
# aclose_model: official provider client surfaces close exactly once; missing
# interfaces are no-ops; one failing client does not leak the others.
# --------------------------------------------------------------------------- #
class _FakeClient:
    """Fake provider client with optional sync/async close and failure."""

    def __init__(self, name: str, *, async_close: bool = False, fail: bool = False) -> None:
        self.name = name
        self.async_close = async_close
        self.fail = fail
        self.closed = 0

    def close(self) -> None:
        if self.fail:
            raise RuntimeError(f"{self.name} sync close failed")
        self.closed += 1

    async def aclose(self) -> None:
        if self.fail:
            raise RuntimeError(f"{self.name} async close failed")
        self.closed += 1


class _NoCloseClient:
    """Client-like object without any close interface."""


class _FakeOpenAIModel(BaseChatModel):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "root_async_client", _FakeClient("openai-root"))
        object.__setattr__(self, "async_client", _NoCloseClient())

    @property
    def _llm_type(self) -> str:
        return "fake-openai"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[])


class _FakeAnthropicModel(BaseChatModel):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_async_client", _FakeClient("anthropic"))

    @property
    def _llm_type(self) -> str:
        return "fake-anthropic"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[])


class _FakeMistralModel(BaseChatModel):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "async_client", _FakeClient("mistral", async_close=True))

    @property
    def _llm_type(self) -> str:
        return "fake-mistral"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[])


class _FakeGenericModel(BaseChatModel):
    closed: int = 0

    async def aclose(self) -> None:
        self.closed += 1

    @property
    def _llm_type(self) -> str:
        return "fake-generic"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[])


@pytest.mark.anyio
async def test_aclose_model_closes_openai_root_async_client() -> None:
    model = _FakeOpenAIModel()

    await provider_runtime.aclose_model(model)

    assert model.root_async_client.closed == 1


@pytest.mark.anyio
async def test_aclose_model_closes_anthropic_async_client() -> None:
    model = _FakeAnthropicModel()

    await provider_runtime.aclose_model(model)

    assert model._async_client.closed == 1


@pytest.mark.anyio
async def test_aclose_model_closes_mistral_async_client() -> None:
    model = _FakeMistralModel()

    await provider_runtime.aclose_model(model)

    assert model.async_client.closed == 1


@pytest.mark.anyio
async def test_aclose_model_uses_generic_aclose() -> None:
    model = _FakeGenericModel()

    await provider_runtime.aclose_model(model)

    assert model.closed == 1


@pytest.mark.anyio
async def test_aclose_model_never_closes_same_client_twice() -> None:
    model = _FakeMistralModel()
    shared = _FakeClient("shared", async_close=True)
    object.__setattr__(model, "async_client", shared)
    object.__setattr__(model, "root_async_client", shared)

    await provider_runtime.aclose_model(model)

    assert shared.closed == 1


@pytest.mark.anyio
async def test_aclose_model_no_op_for_model_without_clients() -> None:
    class Bare(BaseChatModel):
        def __init__(self) -> None:
            super().__init__()
            object.__setattr__(self, "closed_count", 0)

        @property
        def _llm_type(self) -> str:
            return "bare"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
            del messages, stop, run_manager, kwargs
            return ChatResult(generations=[])

    await provider_runtime.aclose_model(Bare())  # must not raise


@pytest.mark.anyio
async def test_aclose_model_continues_after_one_client_fails() -> None:
    model = _FakeOpenAIModel()
    object.__setattr__(model, "root_async_client", _FakeClient("openai-root", fail=True))
    object.__setattr__(model, "async_client", _FakeClient("openai-completions", async_close=True))

    with pytest.raises(ExceptionGroup):
        await provider_runtime.aclose_model(model)

    assert model.async_client.closed == 1
