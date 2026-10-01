"""HTTP/OAuth offline acceptance before the auth and resource implementation.

Failure modes: login granting execution, hidden browser prompts, cross-origin
headers, leaked tokens in tool output, resources bypassing authorization,
expiry and refresh rotation, state mismatch, and PKCE validation.
"""

import asyncio
import base64
import hashlib
import json
import socket
import sys
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest

from forge_coding.mcp.runtime import MCPError, MCPRuntime
from forge_coding.paths import ForgePaths


@asynccontextmanager
async def _fixture_server(tmp_path: Path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).parent / "fixtures/tool_ecosystem/http_server.py"),
        str(port),
        cwd=tmp_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with httpx2.AsyncClient(trust_env=False) as client:
            async with asyncio.timeout(20):
                while True:
                    try:
                        await client.get(origin + "/.well-known/oauth-protected-resource")
                        break
                    except httpx2.ConnectError:
                        await asyncio.sleep(0.05)
        yield origin
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


async def _fixture_control(origin: str, **values: object) -> dict[str, object]:
    async with httpx2.AsyncClient(trust_env=False) as client:
        response = await client.post(origin + "/control", json=values)
        response.raise_for_status()
        return response.json()


def _redirect_handler(visits: list[asyncio.Task[None]]):
    async def redirect(self, url: str) -> None:
        async def visit() -> None:
            async with httpx2.AsyncClient(follow_redirects=True, trust_env=False) as client:
                for _ in range(100):
                    try:
                        await client.get(url)
                        return
                    except httpx2.ConnectError:
                        await asyncio.sleep(0.05)
                raise AssertionError("OAuth callback never started")

        visits.append(asyncio.create_task(visit()))

    return redirect


def _short_oauth_callback(monkeypatch) -> None:
    from fastmcp.client.auth import OAuth

    original_init = OAuth.__init__

    def init(self, *args: object, **kwargs: object) -> None:
        kwargs["callback_timeout"] = 5.0
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(OAuth, "__init__", init)


async def _authorize_code(
    origin: str, *, verifier: str, client: httpx2.AsyncClient, state: str
) -> str:
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    response = await client.get(
        origin + "/authorize",
        params={
            "response_type": "code",
            "client_id": "fixture-client",
            "redirect_uri": "http://127.0.0.1:1/callback",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )
    assert response.status_code == 307
    return parse_qs(urlsplit(response.headers["location"]).query)["code"][0]


@pytest.mark.anyio
async def test_sdk_oauth_and_resources_end_to_end(tmp_path: Path, monkeypatch) -> None:
    from fastmcp.client.auth import OAuth

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).parent / "fixtures/tool_ecosystem/http_server.py"),
        str(port),
        cwd=tmp_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    visits = []

    async def redirect(self, url):
        async def visit():
            async with httpx2.AsyncClient(follow_redirects=True, trust_env=False) as client:
                for _ in range(100):
                    try:
                        await client.get(url)
                        return
                    except httpx2.ConnectError:
                        await asyncio.sleep(0.05)
                raise AssertionError("OAuth callback never started")

        visits.append(asyncio.create_task(visit()))

    monkeypatch.setattr(OAuth, "redirect_handler", redirect)
    paths = ForgePaths(home=tmp_path / "user")
    paths.home.mkdir()
    paths.mcp_config_path.write_text(
        json.dumps({"mcpServers": {"fixture": {"url": origin + "/mcp"}}})
    )
    runtime = MCPRuntime(tmp_path, paths=paths)
    try:
        async with httpx2.AsyncClient(trust_env=False) as client:
            async with asyncio.timeout(20):
                while True:
                    try:
                        await client.get(origin + "/.well-known/oauth-protected-resource")
                        break
                    except httpx2.ConnectError:
                        await asyncio.sleep(0.05)
        with pytest.raises(MCPError):
            await runtime.enable("fixture")
        assert not visits
        try:
            await runtime.login("fixture")
        except MCPError:
            print(runtime.describe("logs", "fixture"))
            raise
        assert not runtime.tools
        await runtime.enable("fixture")
        listing = next(t for t in runtime.resource_tools if t.name == "list_mcp_resources")
        resources = await listing.ainvoke({"server": "fixture"})
        assert "fixture://note" in str(resources) and "fixture://second" in str(resources)
        templates = next(
            t for t in runtime.resource_tools if t.name == "list_mcp_resource_templates"
        )
        assert "{name}" in str(await templates.ainvoke({"server": "fixture"}))
        reader = next(t for t in runtime.resource_tools if t.name == "read_mcp_resource")
        assert "fixture resource" in str(
            await reader.ainvoke({"server": "fixture", "uri": "fixture://note"})
        )
        with pytest.raises(MCPError):
            await reader.ainvoke({"server": "fixture", "uri": "fixture://unknown"})
        assert "fixture bob" in str(
            await reader.ainvoke({"server": "fixture", "uri": "fixture://note/bob"})
        )
        echo = next(t for t in runtime.tools if t.name == "fixture_echo")
        result = await echo.ainvoke({"value": "fixture-oauth-token"})
        assert "fixture-oauth-token" not in str(result)
        await runtime.logout("fixture")
        assert not runtime.tools
        with pytest.raises(MCPError):
            await reader.ainvoke({"server": "fixture", "uri": "fixture://note"})
        (tmp_path / "oauth-evidence.json").write_text(
            json.dumps({"login_separate": True, "resources": True, "logout_revoked": True})
        )
        print(f"OAuth evidence: {tmp_path / 'oauth-evidence.json'}")
    finally:
        await runtime.aclose()
        for visit in visits:
            if not visit.done():
                visit.cancel()
        await asyncio.gather(*visits, return_exceptions=True)
        if process.returncode is None:
            process.kill()
        await process.wait()


@pytest.mark.anyio
async def test_http_redirect_never_forwards_custom_credentials(tmp_path: Path, monkeypatch) -> None:
    forwarded = []

    class Target(BaseHTTPRequestHandler):
        def do_POST(self):
            forwarded.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

        do_GET = do_POST

        def log_message(self, *args):
            pass

    destination = ThreadingHTTPServer(("127.0.0.1", 0), Target)

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{destination.server_port}/mcp")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST

        def log_message(self, *args):
            pass

    source = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    for server in (source, destination):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    paths = ForgePaths(home=tmp_path)
    monkeypatch.setenv("FORGE_FIXTURE_HEADER", "fixture-header-secret")
    paths.mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fixture": {
                        "url": f"http://127.0.0.1:{source.server_port}/mcp",
                        "headers": {"X-Private-Key": {"env": "FORGE_FIXTURE_HEADER"}},
                    }
                }
            }
        )
    )
    runtime = MCPRuntime(tmp_path, paths=paths)
    try:
        with pytest.raises(MCPError):
            await runtime.enable("fixture")
        assert not forwarded and not runtime.tools
        assert "fixture-header-secret" not in runtime.describe("logs", "fixture")
    finally:
        await runtime.aclose()
        for server in (source, destination):
            await asyncio.to_thread(server.shutdown)
            server.server_close()


@pytest.mark.anyio
async def test_local_oauth_issuer_enforces_pkce(tmp_path: Path) -> None:
    async with _fixture_server(tmp_path) as origin:
        async with httpx2.AsyncClient(trust_env=False) as client:
            wrong_verifier = "B" * 64
            valid_verifier = "A" * 64
            missing_verifier = "C" * 64
            wrong_code = await _authorize_code(
                origin, verifier=wrong_verifier, client=client, state="wrong"
            )
            wrong = await client.post(
                origin + "/token",
                data={
                    "grant_type": "authorization_code",
                    "code": wrong_code,
                    "client_id": "fixture-client",
                    "redirect_uri": "http://127.0.0.1:1/callback",
                    "code_verifier": "D" * 64,
                },
            )
            assert wrong.status_code == 400

            missing_code = await _authorize_code(
                origin, verifier=missing_verifier, client=client, state="missing"
            )
            missing = await client.post(
                origin + "/token",
                data={
                    "grant_type": "authorization_code",
                    "code": missing_code,
                    "client_id": "fixture-client",
                    "redirect_uri": "http://127.0.0.1:1/callback",
                },
            )
            assert missing.status_code == 400

            valid_code = await _authorize_code(
                origin, verifier=valid_verifier, client=client, state="valid"
            )
            valid = await client.post(
                origin + "/token",
                data={
                    "grant_type": "authorization_code",
                    "code": valid_code,
                    "client_id": "fixture-client",
                    "redirect_uri": "http://127.0.0.1:1/callback",
                    "code_verifier": valid_verifier,
                },
            )
            assert valid.status_code == 200

        snapshot = await _fixture_control(origin)
        assert snapshot["pkce_rejections"] == 1
        assert snapshot["missing_pkce_rejections"] == 1
        assert snapshot["pkce_valid_grants"] == 1
        (tmp_path / "oauth-pkce-evidence.json").write_text(
            json.dumps(
                {
                    "valid_s256": True,
                    "wrong_verifier_rejected": snapshot["pkce_rejections"] == 1,
                    "missing_verifier_rejected": snapshot["missing_pkce_rejections"] == 1,
                    "token_requests": snapshot["token_requests"],
                }
            )
        )


@pytest.mark.anyio
async def test_sdk_oauth_rejects_state_mismatch_before_token_exchange(
    tmp_path: Path, monkeypatch
) -> None:
    from fastmcp.client.auth import OAuth

    _short_oauth_callback(monkeypatch)
    visits: list[asyncio.Task[None]] = []
    monkeypatch.setattr(OAuth, "redirect_handler", _redirect_handler(visits))
    async with _fixture_server(tmp_path) as origin:
        await _fixture_control(origin, wrong_state=True)
        paths = ForgePaths(home=tmp_path / "user")
        paths.home.mkdir()
        paths.mcp_config_path.write_text(
            json.dumps({"mcpServers": {"fixture": {"url": origin + "/mcp"}}})
        )
        runtime = MCPRuntime(tmp_path, paths=paths)
        try:
            with pytest.raises(MCPError):
                await runtime.login("fixture")
            await asyncio.gather(*visits)
            snapshot = await _fixture_control(origin)
            assert snapshot["authorization_requests"] == 1
            assert snapshot["token_requests"] == 0
            assert snapshot["authorization_code_grants"] == 0
            assert not runtime.tools
            (tmp_path / "oauth-state-evidence.json").write_text(
                json.dumps(
                    {
                        "state_mismatch_rejected": True,
                        "token_requests": snapshot["token_requests"],
                        "authorization_code_grants": snapshot["authorization_code_grants"],
                        "browser_visits": len(visits),
                    }
                )
            )
        finally:
            await runtime.aclose()
            for visit in visits:
                if not visit.done():
                    visit.cancel()
            await asyncio.gather(*visits, return_exceptions=True)


@pytest.mark.anyio
async def test_server_expiry_returns_401_without_sdk_refresh(tmp_path: Path, monkeypatch) -> None:
    from fastmcp.client.auth import OAuth

    _short_oauth_callback(monkeypatch)
    visits: list[asyncio.Task[None]] = []
    monkeypatch.setattr(OAuth, "redirect_handler", _redirect_handler(visits))
    async with _fixture_server(tmp_path) as origin:
        paths = ForgePaths(home=tmp_path / "user")
        paths.home.mkdir()
        paths.mcp_config_path.write_text(
            json.dumps({"mcpServers": {"fixture": {"url": origin + "/mcp"}}})
        )
        runtime = MCPRuntime(tmp_path, paths=paths)
        try:
            await runtime.login("fixture")
            await asyncio.gather(*visits)
            await _fixture_control(origin, expire_access=True)
            async with httpx2.AsyncClient(trust_env=False) as client:
                expired = await client.get(
                    origin + "/mcp",
                    headers={"Authorization": "Bearer fixture-oauth-token"},
                )
                missing_scheme = await client.get(
                    origin + "/mcp",
                    headers={"Authorization": "fixture-oauth-token"},
                )
            assert expired.status_code == 401
            assert missing_scheme.status_code == 401
            with pytest.raises(MCPError):
                await runtime.enable("fixture")
            snapshot = await _fixture_control(origin)
            assert snapshot["refresh_requests"] == 0
            assert snapshot["authorization_requests"] == 1
            assert len(visits) == 1
            (tmp_path / "oauth-server-expiry-evidence.json").write_text(
                json.dumps(
                    {
                        "old_access_rejected": expired.status_code == 401,
                        "missing_bearer_rejected": missing_scheme.status_code == 401,
                        "sdk_401_refresh_supported": False,
                        "refresh_requests": snapshot["refresh_requests"],
                        "browser_visits": len(visits),
                    }
                )
            )
        finally:
            await runtime.logout("fixture")
            await runtime.aclose()
            for visit in visits:
                if not visit.done():
                    visit.cancel()
            await asyncio.gather(*visits, return_exceptions=True)


@pytest.mark.anyio
async def test_sdk_oauth_refreshes_expired_token_and_rotates_refresh_token(
    tmp_path: Path, monkeypatch
) -> None:
    from fastmcp.client.auth import OAuth

    _short_oauth_callback(monkeypatch)
    visits: list[asyncio.Task[None]] = []
    monkeypatch.setattr(OAuth, "redirect_handler", _redirect_handler(visits))
    async with _fixture_server(tmp_path) as origin:
        await _fixture_control(origin, authorization_expires_in=3)
        paths = ForgePaths(home=tmp_path / "user")
        paths.home.mkdir()
        paths.mcp_config_path.write_text(
            json.dumps({"mcpServers": {"fixture": {"url": origin + "/mcp"}}})
        )
        runtime = MCPRuntime(tmp_path, paths=paths)
        try:
            await runtime.login("fixture")
            await asyncio.gather(*visits)
            await asyncio.sleep(3.1)
            before = await _fixture_control(origin)
            assert before["refresh_requests"] == 0
            await runtime.enable("fixture")
            after = await _fixture_control(origin)
            assert after["refresh_requests"] == before["refresh_requests"] + 1
            assert after["refresh_rotation_count"] == 1
            assert after["authorization_code_grants"] == 1
            assert len(visits) == 1
            async with httpx2.AsyncClient(trust_env=False) as client:
                replay = await client.post(
                    origin + "/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": "fixture-refresh",
                        "client_id": "fixture-client",
                    },
                )
            assert replay.status_code == 400
            after_replay = await _fixture_control(origin)
            assert after_replay["refresh_reuse_rejections"] == 1
            assert runtime.tools
            (tmp_path / "oauth-refresh-evidence.json").write_text(
                json.dumps(
                    {
                        "expired_token_refreshed": True,
                        "refresh_grant_requests": after["refresh_requests"],
                        "refresh_rotated": after["refresh_rotation_count"] == 1,
                        "old_refresh_rejected": after_replay["refresh_reuse_rejections"] == 1,
                        "browser_visits": len(visits),
                    }
                )
            )
        finally:
            await runtime.logout("fixture")
            await runtime.aclose()
            for visit in visits:
                if not visit.done():
                    visit.cancel()
            await asyncio.gather(*visits, return_exceptions=True)


@pytest.mark.anyio
async def test_sdk_oauth_refresh_failure_stays_noninteractive(tmp_path: Path, monkeypatch) -> None:
    from fastmcp.client.auth import OAuth

    _short_oauth_callback(monkeypatch)
    visits: list[asyncio.Task[None]] = []
    monkeypatch.setattr(OAuth, "redirect_handler", _redirect_handler(visits))
    async with _fixture_server(tmp_path) as origin:
        await _fixture_control(origin, authorization_expires_in=3, refresh_failure=True)
        paths = ForgePaths(home=tmp_path / "user")
        paths.home.mkdir()
        paths.mcp_config_path.write_text(
            json.dumps({"mcpServers": {"fixture": {"url": origin + "/mcp"}}})
        )
        runtime = MCPRuntime(tmp_path, paths=paths)
        try:
            await runtime.login("fixture")
            await asyncio.gather(*visits)
            await asyncio.sleep(3.1)
            before = await _fixture_control(origin)
            assert before["refresh_requests"] == 0
            with pytest.raises(MCPError):
                await runtime.enable("fixture")
            snapshot = await _fixture_control(origin)
            assert snapshot["refresh_requests"] == 1
            assert snapshot["authorization_requests"] == 1
            assert snapshot["token_requests"] == 2
            assert len(visits) == 1
            assert not runtime.tools
            (tmp_path / "oauth-refresh-failure-evidence.json").write_text(
                json.dumps(
                    {
                        "refresh_failed": True,
                        "noninteractive_rejected": True,
                        "refresh_requests": snapshot["refresh_requests"],
                        "browser_visits": len(visits),
                    }
                )
            )
        finally:
            await runtime.logout("fixture")
            await runtime.aclose()
            for visit in visits:
                if not visit.done():
                    visit.cancel()
            await asyncio.gather(*visits, return_exceptions=True)
