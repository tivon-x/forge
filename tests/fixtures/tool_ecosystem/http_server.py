"""Local HTTP MCP + minimal OAuth issuer for offline SDK acceptance."""

import base64
import hashlib
import sys
import time
from urllib.parse import parse_qs, urlencode

import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Mount, Route

port = int(sys.argv[1])
origin = f"http://127.0.0.1:{port}"
server = FastMCP("HTTP fixture", list_page_size=1)


class FixtureOAuthState:
    def __init__(self) -> None:
        self.wrong_state = False
        self.refresh_failure = False
        self.authorization_expires_in = 3600
        self.refresh_expires_in = 3600
        self.authorization_requests = 0
        self.token_requests = 0
        self.authorization_code_grants = 0
        self.refresh_requests = 0
        self.pkce_valid_grants = 0
        self.pkce_rejections = 0
        self.missing_pkce_rejections = 0
        self.refresh_rotation_count = 0
        self.refresh_reuse_rejections = 0
        self.codes: dict[str, str] = {}
        self.valid_refresh_tokens = {"fixture-refresh"}
        self.used_refresh_tokens: set[str] = set()
        self.valid_access_tokens = {"fixture-oauth-token"}
        self.access_expires_at = {"fixture-oauth-token": float("inf")}
        self.next_access_token = 0
        self.next_refresh_token = 1

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "wrong_state": self.wrong_state,
            "refresh_failure": self.refresh_failure,
            "authorization_requests": self.authorization_requests,
            "token_requests": self.token_requests,
            "authorization_code_grants": self.authorization_code_grants,
            "refresh_requests": self.refresh_requests,
            "pkce_valid_grants": self.pkce_valid_grants,
            "pkce_rejections": self.pkce_rejections,
            "missing_pkce_rejections": self.missing_pkce_rejections,
            "refresh_rotation_count": self.refresh_rotation_count,
            "refresh_reuse_rejections": self.refresh_reuse_rejections,
        }

    def issue_tokens(self, *, expires_in: int, rotate_refresh: bool) -> dict[str, str | int]:
        self.next_access_token += 1
        access_token = (
            "fixture-oauth-token"
            if self.next_access_token == 1
            else f"fixture-oauth-token-{self.next_access_token}"
        )
        self.valid_access_tokens = {access_token}
        self.access_expires_at = {access_token: time.monotonic() + expires_in}
        refresh_token = next(iter(self.valid_refresh_tokens))
        if rotate_refresh:
            self.used_refresh_tokens.update(self.valid_refresh_tokens)
            self.next_refresh_token += 1
            refresh_token = f"fixture-refresh-{self.next_refresh_token}"
            self.valid_refresh_tokens = {refresh_token}
            self.refresh_rotation_count += 1
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "refresh_token": refresh_token,
            "expires_in": expires_in,
        }

    def expire_access(self) -> None:
        expired_at = time.monotonic() - 1
        for token in self.valid_access_tokens:
            self.access_expires_at[token] = expired_at


oauth_state = FixtureOAuthState()


@server.tool
def echo(value: str) -> dict[str, str]:
    """Echo a structured HTTP result."""
    return {"value": value}


@server.resource("fixture://note")
def note() -> str:
    return "fixture resource"


@server.resource("fixture://second")
def second_note() -> str:
    return "second resource"


@server.resource("fixture://note/{name}")
def named_note(name: str) -> str:
    return f"fixture {name}"


async def protected(request: Request):
    return JSONResponse({"resource": origin + "/mcp", "authorization_servers": [origin]})


async def metadata(request: Request):
    return JSONResponse(
        {
            "issuer": origin,
            "authorization_endpoint": origin + "/authorize",
            "token_endpoint": origin + "/token",
            "registration_endpoint": origin + "/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["none"],
            "code_challenge_methods_supported": ["S256"],
        }
    )


async def register(request: Request):
    return JSONResponse({**await request.json(), "client_id": "fixture-client"})


async def authorize(request: Request):
    values = request.query_params
    challenge = values.get("code_challenge")
    if not challenge or values.get("code_challenge_method") != "S256":
        oauth_state.missing_pkce_rejections += 1
        return JSONResponse(
            {"error": "invalid_request", "error_description": "PKCE is required"}, status_code=400
        )
    oauth_state.authorization_requests += 1
    code = f"fixture-code-{oauth_state.authorization_requests}"
    oauth_state.codes[code] = challenge
    callback_state = "fixture-wrong-state" if oauth_state.wrong_state else values["state"]
    return RedirectResponse(
        values["redirect_uri"] + "?" + urlencode({"code": code, "state": callback_state})
    )


async def token(request: Request):
    oauth_state.token_requests += 1
    values = parse_qs((await request.body()).decode())
    grant_type = values.get("grant_type", [None])[0]
    if grant_type == "authorization_code":
        code = values.get("code", [None])[0]
        challenge = oauth_state.codes.get(code or "")
        verifier = values.get("code_verifier", [None])[0]
        if not verifier:
            oauth_state.missing_pkce_rejections += 1
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        if not challenge:
            oauth_state.pkce_rejections += 1
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        actual = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        if code is None or actual != challenge:
            oauth_state.pkce_rejections += 1
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        oauth_state.codes.pop(code, None)
        oauth_state.authorization_code_grants += 1
        oauth_state.pkce_valid_grants += 1
        return JSONResponse(
            oauth_state.issue_tokens(
                expires_in=oauth_state.authorization_expires_in,
                rotate_refresh=False,
            )
        )
    if grant_type == "refresh_token":
        oauth_state.refresh_requests += 1
        refresh_token = values.get("refresh_token", [None])[0]
        if oauth_state.refresh_failure or refresh_token not in oauth_state.valid_refresh_tokens:
            if refresh_token in oauth_state.used_refresh_tokens:
                oauth_state.refresh_reuse_rejections += 1
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        return JSONResponse(
            oauth_state.issue_tokens(
                expires_in=oauth_state.refresh_expires_in,
                rotate_refresh=True,
            )
        )
    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


async def control(request: Request):
    values = await request.json()
    if "wrong_state" in values:
        oauth_state.wrong_state = bool(values["wrong_state"])
    if "refresh_failure" in values:
        oauth_state.refresh_failure = bool(values["refresh_failure"])
    if "authorization_expires_in" in values:
        oauth_state.authorization_expires_in = int(values["authorization_expires_in"])
    if "refresh_expires_in" in values:
        oauth_state.refresh_expires_in = int(values["refresh_expires_in"])
    if values.get("expire_access"):
        oauth_state.expire_access()
    return JSONResponse(oauth_state.snapshot())


class AuthGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            headers = dict(scope["headers"])
            authorization = headers.get(b"authorization", b"").decode()
            token = authorization[7:] if authorization.startswith("Bearer ") else None
            if (
                token is None
                or token not in oauth_state.valid_access_tokens
                or time.monotonic() >= oauth_state.access_expires_at.get(token, 0)
            ):
                response = JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": (
                            f'Bearer resource_metadata="{origin}'
                            '/.well-known/oauth-protected-resource"'
                        )
                    },
                )
                return await response(scope, receive, send)
        await self.app(scope, receive, send)


mcp_app = server.http_app(path="/mcp")
app = Starlette(
    routes=[
        Route("/.well-known/oauth-protected-resource", protected),
        Route("/.well-known/oauth-protected-resource/mcp", protected),
        Route("/.well-known/oauth-authorization-server", metadata),
        Route("/register", register, methods=["POST"]),
        Route("/authorize", authorize),
        Route("/token", token, methods=["POST"]),
        Route("/control", control, methods=["POST"]),
        Mount("/", app=mcp_app),
    ],
    lifespan=mcp_app.lifespan,
)
if __name__ == "__main__":
    uvicorn.run(AuthGate(app), host="127.0.0.1", port=port, log_level="critical")
