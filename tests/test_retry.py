from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import Field

from fake_models import ScriptedChatModel, tool_call_ai
from forge_agent import AgentHarnessConfig, ErrorEvent, HumanInputRequestedEvent, RetryEvent
from forge_agent.langchain_runtime import run_langchain_agent
from forge_agent.retry import (
    ForgeModelRetryMiddleware,
    RetryPolicy,
    classify_model_error,
    redact_model_error,
)
from forge_agent.session import CustomEntry, JsonlSessionStorage
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.sessions.session import _stream_summary_text


class StatusError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


class RetryChatModel(ScriptedChatModel):
    failures: list[BaseException] = Field(default_factory=list)

    def __init__(self, failures: list[BaseException], response: AIMessage) -> None:
        super().__init__([response])
        object.__setattr__(self, "failures", list(failures))

    def _next_response(self) -> AIMessage:
        failures = self.failures
        if failures:
            raise failures.pop(0)
        return super()._next_response()


def _policy(*, enabled: bool = True, max_retries: int = 3) -> RetryPolicy:
    return RetryPolicy(
        enabled=enabled,
        max_retries=max_retries,
        initial_delay=0,
        max_delay=0,
    )


def test_classify_model_error_prefers_quota_over_429() -> None:
    error = StatusError("insufficient_quota", 429)
    result = classify_model_error(error)

    assert result.retryable is False
    assert result.kind == "quota"


def test_classify_model_error_marks_overflow_non_retryable() -> None:
    result = classify_model_error(StatusError("context_length_exceeded", 500))

    assert result.retryable is False
    assert result.kind == "overflow"


@pytest.mark.parametrize(
    "message",
    [
        "RateLimitError",
        "Provider returned error",
        "getaddrinfo ENOTFOUND api.example.test",
        "stream ended before message_stop",
        "ResourceExhausted",
    ],
)
def test_classify_model_error_matches_pi_transient_text(message: str) -> None:
    result = classify_model_error(RuntimeError(message))

    assert result.retryable is True
    assert result.kind == "transient"


@pytest.mark.parametrize("message", ["GoUsageLimitError", "FreeUsageLimitError"])
def test_classify_model_error_matches_pi_non_retryable_limits(message: str) -> None:
    result = classify_model_error(StatusError(message, 429))

    assert result.retryable is False
    assert result.kind == "quota"


def test_classify_connection_abort_is_transient_and_retry_after_is_bounded() -> None:
    error = StatusError("connection aborted", 503, {"Retry-After": "20"})
    result = classify_model_error(error)

    assert result.retryable is True
    assert result.kind == "transient"
    assert _policy().delay(0, result.retry_after) == 0
    assert RetryPolicy().delay(0, result.retry_after) == 8


@pytest.mark.parametrize("status", [408, 409])
def test_classify_model_error_retries_pi_request_statuses(status: int) -> None:
    assert classify_model_error(StatusError("request failed", status)).retryable is True


def test_retry_after_ms_takes_precedence() -> None:
    result = classify_model_error(
        StatusError("temporary outage", 503, {"Retry-After-Ms": "1500", "Retry-After": "7"})
    )

    assert result.retry_after == 1.5


def test_redact_model_error_removes_root_level_absolute_path() -> None:
    assert "/tmp" not in redact_model_error("failed at /tmp")


def test_redact_model_error_removes_bearer_and_bounds_utf8_bytes() -> None:
    redacted = redact_model_error("Authorization: Bearer secret-token " + "中" * 2_048)

    assert "secret-token" not in redacted
    assert len(redacted.encode("utf-8")) <= 2_048


@pytest.mark.anyio
async def test_runtime_retries_transient_model_error_and_projects_retry_event() -> None:
    provider = RetryChatModel([StatusError("temporary outage", 429)], AIMessage(content="ok"))
    events = [
        event
        async for event in run_langchain_agent(
            provider=provider,
            model="fake",
            system="You are Forge.",
            messages=[],
            retry_policy=_policy(),
        )
    ]

    retries = [event for event in events if isinstance(event, RetryEvent)]
    assert len(provider.calls) == 2
    assert len(retries) == 1
    assert retries[0].attempt == 1
    assert not [event for event in events if getattr(event, "type", None) == "error"]


@pytest.mark.anyio
async def test_retry_policy_disabled_preserves_single_model_call() -> None:
    provider = RetryChatModel([StatusError("temporary outage", 429)], AIMessage(content="ok"))
    events = [
        event
        async for event in run_langchain_agent(
            provider=provider,
            model="fake",
            system="You are Forge.",
            messages=[],
            retry_policy=_policy(enabled=False),
        )
    ]

    assert len(provider.calls) == 1
    assert not [event for event in events if isinstance(event, RetryEvent)]
    assert any(getattr(event, "type", None) == "error" for event in events)


@pytest.mark.anyio
async def test_runtime_redacts_final_model_error() -> None:
    provider = RetryChatModel(
        [
            RuntimeError(
                "Authorization: Bearer secret-token "
                "https://api.example.test/chat?api_key=secret C:\\repo\\file.py"
            )
        ],
        AIMessage(content="unreachable"),
    )
    events = [
        event
        async for event in run_langchain_agent(
            provider=provider,
            model="fake",
            system="You are Forge.",
            messages=[],
            retry_policy=_policy(enabled=False),
        )
    ]

    error = next(event for event in events if isinstance(event, ErrorEvent))
    assert "secret" not in error.message
    assert "?api_key" not in error.message
    assert "C:\\repo" not in error.message


@pytest.mark.anyio
async def test_runtime_exhausts_three_transient_retries() -> None:
    provider = RetryChatModel(
        [
            StatusError("temporary outage", 503),
            StatusError("temporary outage", 503),
            StatusError("temporary outage", 503),
            StatusError("temporary outage", 503),
        ],
        AIMessage(content="unreachable"),
    )
    events = [
        event
        async for event in run_langchain_agent(
            provider=provider,
            model="fake",
            system="You are Forge.",
            messages=[],
            retry_policy=_policy(),
        )
    ]

    retries = [event for event in events if isinstance(event, RetryEvent)]
    assert len(provider.calls) == 4
    assert [event.attempt for event in retries] == [1, 2, 3]
    assert any(getattr(event, "data", {}).get("kind") == "transient" for event in events)


@pytest.mark.anyio
async def test_runtime_does_not_retry_quota_error() -> None:
    provider = RetryChatModel(
        [StatusError("insufficient_quota", 429)],
        AIMessage(content="unreachable"),
    )
    events = [
        event
        async for event in run_langchain_agent(
            provider=provider,
            model="fake",
            system="You are Forge.",
            messages=[],
            retry_policy=_policy(),
        )
    ]

    assert len(provider.calls) == 1
    assert not [event for event in events if isinstance(event, RetryEvent)]
    assert any(getattr(event, "data", {}).get("kind") == "quota" for event in events)


@pytest.mark.anyio
async def test_retry_backoff_cancellation_has_no_followup_call() -> None:
    calls = 0
    middleware = ForgeModelRetryMiddleware(RetryPolicy(initial_delay=2, max_delay=2))

    async def handler(_request: object) -> object:
        nonlocal calls
        calls += 1
        raise StatusError("temporary outage", 503)

    request = SimpleNamespace(runtime=Runtime(stream_writer=lambda _payload: None))
    task = asyncio.create_task(middleware.awrap_model_call(request, handler))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == 1


@pytest.mark.anyio
async def test_compaction_respects_disabled_retry_policy() -> None:
    provider = RetryChatModel(
        [StatusError("temporary outage", 503)],
        AIMessage(content="unreachable"),
    )

    with pytest.raises(StatusError):
        await _stream_summary_text(
            provider,
            system="Summarize.",
            messages=[],
            max_tokens=100,
            policy=_policy(enabled=False),
        )

    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_compaction_does_not_retry_quota_error() -> None:
    provider = RetryChatModel(
        [StatusError("insufficient_quota", 429)],
        AIMessage(content="unreachable"),
    )

    with pytest.raises(StatusError):
        await _stream_summary_text(
            provider,
            system="Summarize.",
            messages=[],
            max_tokens=100,
            policy=_policy(),
        )

    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_session_persists_one_redacted_recovered_retry_audit(tmp_path: Path) -> None:
    provider = RetryChatModel(
        [
            StatusError(
                "GET https://api.example.test/v1/chat?api_key=secret C:\\repo\\file.py",
                503,
            )
        ],
        AIMessage(content="ok"),
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            retry=_policy(),
            enable_subagents=False,
        )
    )

    _ = [event async for event in session.prompt("hello")]
    entries = await storage.read_all()
    audits = [entry for entry in entries if isinstance(entry, CustomEntry)]

    assert len(audits) == 1
    assert audits[0].namespace == "forge.turn_error.v1"
    assert audits[0].data["outcome"] == "recovered"
    attempt_text = str(audits[0].data["attempts"])
    assert "secret" not in attempt_text
    assert "?api_key" not in attempt_text
    assert "C:\\repo" not in attempt_text


@pytest.mark.anyio
async def test_session_persists_recovered_retry_before_human_input(tmp_path: Path) -> None:
    question = {
        "header": "Mode",
        "question": "Which mode should Forge use?",
        "options": [
            {"label": "Fast", "description": "Prioritize speed."},
            {"label": "Careful", "description": "Prioritize review."},
        ],
        "multi_select": False,
    }
    provider = RetryChatModel(
        [StatusError("temporary outage", 503)],
        tool_call_ai("ask-call", "ask_user_question", {"questions": [question]}),
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            retry=_policy(),
            enable_subagents=False,
            interactive=True,
        )
    )

    events = [event async for event in session.prompt("ask me")]
    entries = await storage.read_all()
    audits = [entry for entry in entries if isinstance(entry, CustomEntry)]

    assert any(isinstance(event, HumanInputRequestedEvent) for event in events)
    assert len(audits) == 1
    assert audits[0].data["outcome"] == "recovered"


@pytest.mark.anyio
async def test_session_persists_one_exhausted_retry_audit(tmp_path: Path) -> None:
    provider = RetryChatModel(
        [StatusError("temporary outage", 503)] * 4,
        AIMessage(content="unreachable"),
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Forge.",
            storage=storage,
            cwd=tmp_path,
            retry=_policy(),
            enable_subagents=False,
        )
    )

    _ = [event async for event in session.prompt("hello")]
    entries = await storage.read_all()
    audits = [entry for entry in entries if isinstance(entry, CustomEntry)]

    assert len(audits) == 1
    assert audits[0].data["outcome"] == "exhausted"
    attempts = audits[0].data["attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 3


def test_harness_config_has_enabled_retry_by_default() -> None:
    assert AgentHarnessConfig.__dataclass_fields__["retry"].default_factory() == RetryPolicy()
