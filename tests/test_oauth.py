import base64
from json import dumps
from time import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from forge_coding.providers.auth.oauth import (
    OPENAI_CODEX_ACCOUNT_CLAIM,
    OPENAI_CODEX_CLIENT_ID,
    account_id_from_access_token,
    create_openai_codex_authorization_flow,
    parse_authorization_input,
    refresh_openai_codex_token,
)


def test_create_openai_codex_authorization_flow_includes_pkce_and_codex_params() -> None:
    flow = create_openai_codex_authorization_flow(originator="forge-test")

    url = urlparse(flow.url)
    params = parse_qs(url.query)

    assert url.geturl().startswith("https://auth.openai.com/oauth/authorize?")
    assert params["response_type"] == ["code"]
    assert params["client_id"] == [OPENAI_CODEX_CLIENT_ID]
    assert params["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert params["scope"] == ["openid profile email offline_access"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["codex_cli_simplified_flow"] == ["true"]
    assert params["originator"] == ["forge-test"]
    assert params["state"] == [flow.state]
    assert params["code_challenge"][0]
    assert flow.verifier


def test_parse_authorization_input_accepts_redirect_url_query_and_raw_code() -> None:
    assert (
        parse_authorization_input("http://localhost:1455/auth/callback?code=abc&state=state-1").code
        == "abc"
    )
    assert parse_authorization_input("code=abc&state=state-1").state == "state-1"
    assert parse_authorization_input("abc#state-1").state == "state-1"
    assert parse_authorization_input("abc").code == "abc"


def test_account_id_from_access_token_reads_openai_auth_claim() -> None:
    assert account_id_from_access_token(_jwt("account-1")) == "account-1"
    assert account_id_from_access_token("not-a-jwt") is None


@pytest.mark.anyio
async def test_refresh_openai_codex_token_returns_oauth_credential() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        assert "grant_type=refresh_token" in body
        assert "client_id=" in body
        return httpx.Response(
            200,
            json={
                "access_token": _jwt("account-2"),
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        credential = await refresh_openai_codex_token("old-refresh", client=client)

    assert credential.access == _jwt("account-2")
    assert credential.refresh == "new-refresh"
    assert credential.account_id == "account-2"
    assert credential.expires > 0


@pytest.mark.anyio
async def test_refresh_openai_codex_token_preserves_refresh_and_reads_jwt_expiry() -> None:
    expires = int(time()) + 3600
    access_token = _jwt("account-3", expires=expires)

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        assert "grant_type=refresh_token" in body
        return httpx.Response(200, json={"access_token": access_token})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        credential = await refresh_openai_codex_token("old-refresh", client=client)

    assert credential.access == access_token
    assert credential.refresh == "old-refresh"
    assert credential.account_id == "account-3"
    assert credential.expires == expires * 1000


def _jwt(account_id: str, *, expires: int | None = None) -> str:
    payload = {OPENAI_CODEX_ACCOUNT_CLAIM: {"chatgpt_account_id": account_id}}
    if expires is not None:
        payload["exp"] = expires
    return ".".join(
        [
            _base64url(dumps({"alg": "none"}).encode()),
            _base64url(dumps(payload).encode()),
            "signature",
        ]
    )


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


# --------------------------------------------------------------------------- #
# OAuth errors must never embed token values: they surface in the TUI and in
# diagnostic logs.
# --------------------------------------------------------------------------- #
def test_oauth_errors_never_contain_token_values() -> None:
    from forge_coding.providers.auth.oauth import (
        OAuthError,
        _required_token_field,
        _token_expiry,
    )

    fake_access = "fake-access-token-123"
    fake_refresh = "fake-refresh-token-456"
    missing_field = {"access_token": fake_access, "refresh_token": fake_refresh}
    try:
        _required_token_field(missing_field, "expires_in", action="exchange")
    except OAuthError as exc:
        assert fake_access not in str(exc)
        assert fake_refresh not in str(exc)
        assert "expires_in" in str(exc)

    invalid_expiry = {
        "access_token": fake_access,
        "refresh_token": fake_refresh,
        "expires_in": "not-a-number",
    }
    try:
        _token_expiry(invalid_expiry, fake_access, action="refresh")
    except OAuthError as exc:
        assert fake_access not in str(exc)
        assert fake_refresh not in str(exc)

    missing_expiry = {"access_token": fake_access, "refresh_token": fake_refresh}
    try:
        _token_expiry(missing_expiry, fake_access, action="refresh")
    except OAuthError as exc:
        assert fake_access not in str(exc)
        assert fake_refresh not in str(exc)


@pytest.mark.anyio
async def test_oauth_error_response_body_never_leaks_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge_coding.providers.auth.oauth import (
        OAuthError,
        exchange_openai_codex_authorization_code,
    )

    fake_access = "echoed-access-token"
    fake_refresh = "echoed-refresh-token"

    class FakeResponse:
        status_code = 401
        text = f'{{"access_token": "{fake_access}", "refresh_token": "{fake_refresh}"}}'

    class FakeClient:
        async def post(self, *args, **kwargs) -> FakeResponse:
            del args, kwargs
            return FakeResponse()

    with pytest.raises(OAuthError) as excinfo:
        await exchange_openai_codex_authorization_code("code", "verifier", client=FakeClient())  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert fake_access not in message
    assert fake_refresh not in message
    assert "401" in message
