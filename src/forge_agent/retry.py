"""Transient model-call retry policy shared by the agent and sessions."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal, cast

from langchain.agents.middleware import ModelRetryMiddleware

from forge_agent.types import JSONValue

ErrorKind = Literal[
    "transient",
    "abort",
    "overflow",
    "auth",
    "quota",
    "invalid_request",
    "unknown",
]

_QUOTA_RE = re.compile(
    r"(?:GoUsageLimitError|FreeUsageLimitError|insufficient[_ -]?quota|"
    r"quota(?:[ _-]+)?exceeded|billing|out of budget|available balance|"
    r"monthly usage|usage limit|free usage limit|go usage limit)",
    re.IGNORECASE,
)
_OVERFLOW_RE = re.compile(
    r"(?:context[_ -]?length[_ -]?exceeded|context window|maximum context|"
    r"prompt(?: input)?(?: is)? too long|input too long|too many tokens|token limit)",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"(?:unauthorized|authentication|invalid api key|permission denied|forbidden|"
    r"access denied|invalid credential)",
    re.IGNORECASE,
)
_INVALID_RE = re.compile(
    r"(?:invalid request|invalid parameter|bad request|malformed|unsupported parameter)",
    re.IGNORECASE,
)
_ABORT_RE = re.compile(
    r"(?:user (?:abort|interrupt)|(?:request|operation|run|model) "
    r"(?:abort(?:ed|ing)?|cancel(?:led|ed|ation)?))",
    re.IGNORECASE,
)
_TRANSIENT_RE = re.compile(
    r"(?:overloaded|rate.?limit|too many requests|(?:^|\D)429(?:\D|$)|"
    r"(?:^|\D)(?:500|502|503|504|524)(?:\D|$)|service.?unavailable|"
    r"server.?error|internal.?error|provider.?returned.?error|"
    r"exceeded request buffer limit while retrying upstream|network.?error|"
    r"connection(?: reset| refused| aborted| closed| error| lost)?|"
    r"other side closed|fetch failed|getaddrinfo|ENOTFOUND|EAI_AGAIN|"
    r"upstream.?connect|reset before headers|socket hang up|"
    r"socket connection was closed|socket(?: reset| closed| error)|"
    r"timed? ?out|timeout|terminated|websocket.?closed|websocket.?error|"
    r"ended without|stream ended before message_stop|"
    r"stream ended before a terminal response event|"
    r"http2 request did not get a response|retry delay|"
    r"you can retry your request|try your request again|"
    r"please retry your request|ResourceExhausted|broken pipe|"
    r"temporarily unavailable|stream (?:ended|interrupted|closed)|"
    r"incomplete read|eof|try again)",
    re.IGNORECASE,
)
_URL_QUERY_RE = re.compile(r"(https?://[^\s<>\"']+?)(?:\?[^\s<>\"']*)", re.IGNORECASE)
_AUTH_VALUE_RE = re.compile(
    r"((?:api[_ -]?key|access[_ -]?token|token)\s*[=:]\s*)"
    r"[^\s,;]+",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(authorization\s*:\s*bearer\s+)[^\s,;]+", re.IGNORECASE)
_KEY_RE = re.compile(r"\b(?:sk|rk|xoxb|ghp|glpat)-[A-Za-z0-9_-]{8,}\b")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\s<>\"']+")
_UNIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9:/])/(?:[^\s<>\"']+/)*[^\s<>\"']+")


@dataclass(frozen=True, slots=True)
class ModelErrorClassification:
    """Fail-closed classification of one provider/model exception."""

    kind: ErrorKind
    retryable: bool
    status_code: int | None = None
    retry_after: float | None = None


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential retry policy for one logical model call."""

    max_retries: int = 3
    initial_delay: float = 2.0
    backoff_factor: float = 2.0
    max_delay: float = 8.0
    jitter: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.initial_delay < 0 or self.backoff_factor < 0 or self.max_delay < 0:
            raise ValueError("retry delays must be non-negative")

    def delay(self, retry_index: int, retry_after: float | None = None) -> float:
        """Return the bounded delay before retry number ``retry_index``."""

        if retry_after is not None:
            return min(max(retry_after, 0.0), self.max_delay)
        return min(
            self.max_delay,
            self.initial_delay * (self.backoff_factor**retry_index),
        )


def _status_code(exc: BaseException) -> int | None:
    """Read structured status fields before inspecting exception text."""

    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        for name in ("status_code", "status", "http_status"):
            value = getattr(source, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        if isinstance(source, dict):
            for name in ("status_code", "status", "http_status"):
                value = source.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
    return None


def _headers(exc: BaseException) -> object | None:
    for source in (exc, getattr(exc, "response", None)):
        headers = getattr(source, "headers", None) if source is not None else None
        if headers is not None:
            return cast(object, headers)
    return None


def _retry_after(exc: BaseException) -> float | None:
    headers = _headers(exc)
    if headers is None:
        return None

    def header(name: str) -> object | None:
        if isinstance(headers, dict):
            return next(
                (candidate for key, candidate in headers.items() if str(key).lower() == name),
                None,
            )
        getter = getattr(headers, "get", None)
        return getter(name) if callable(getter) else None

    retry_after_ms = header("retry-after-ms")
    if isinstance(retry_after_ms, (int, float)) and not isinstance(retry_after_ms, bool):
        return max(float(retry_after_ms) / 1_000, 0.0)
    if isinstance(retry_after_ms, str):
        try:
            return max(float(retry_after_ms.strip()) / 1_000, 0.0)
        except ValueError:
            pass

    value = header("retry-after")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(float(value), 0.0)
    if not isinstance(value, str):
        return None
    try:
        return max(float(value.strip()), 0.0)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max((retry_at - datetime.now(UTC)).total_seconds(), 0.0)
        except (TypeError, ValueError, OverflowError):
            return None


def _error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        for name in ("code", "error_code", "type", "message", "body"):
            value = getattr(source, name, None)
            if value is not None and value is not exc:
                parts.append(str(value))
    return " ".join(parts)


def _is_abort_error(exc: BaseException, text: str) -> bool:
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return True
    for name in ("is_cancelled", "cancelled", "aborted"):
        if getattr(exc, name, False) is True:
            return True
    normalized = text.strip().lower()
    return normalized in {"abort", "aborted", "cancelled", "canceled"} or bool(
        _ABORT_RE.search(text)
    )


def classify_model_error(exc: BaseException) -> ModelErrorClassification:
    """Classify an exception using structured fields first and text as fallback."""

    text = _error_text(exc)
    status = _status_code(exc)
    if _is_abort_error(exc, text):
        return ModelErrorClassification("abort", False, status, _retry_after(exc))
    if _OVERFLOW_RE.search(text):
        return ModelErrorClassification("overflow", False, status, _retry_after(exc))
    if _QUOTA_RE.search(text):
        return ModelErrorClassification("quota", False, status, _retry_after(exc))
    if status in {401, 403} or _AUTH_RE.search(text):
        return ModelErrorClassification("auth", False, status, _retry_after(exc))
    if status is not None and 400 <= status < 500 and status not in {408, 409, 429}:
        return ModelErrorClassification("invalid_request", False, status, _retry_after(exc))
    if status in {408, 409, 429} or status is not None and 500 <= status < 600:
        return ModelErrorClassification("transient", True, status, _retry_after(exc))
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)) or _TRANSIENT_RE.search(text):
        return ModelErrorClassification("transient", True, status, _retry_after(exc))
    if _INVALID_RE.search(text):
        return ModelErrorClassification("invalid_request", False, status, _retry_after(exc))
    return ModelErrorClassification("unknown", False, status, _retry_after(exc))


def redact_model_error(exc: BaseException | str, *, limit: int = 2_048) -> str:
    """Bound provider text before it reaches UI events or durable audit data."""

    text = str(exc)
    text = _BEARER_RE.sub(r"\1<redacted>", text)
    text = _AUTH_VALUE_RE.sub(r"\1<redacted>", text)
    text = _KEY_RE.sub("<redacted>", text)
    text = _URL_QUERY_RE.sub(r"\1?<redacted>", text)
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = _UNIX_PATH_RE.sub("<path>", text)
    return text.encode("utf-8")[: max(0, limit)].decode("utf-8", errors="ignore")


async def retry_model_call[RetryResult](
    operation: Callable[[], Awaitable[RetryResult]],
    *,
    policy: RetryPolicy | None = None,
) -> RetryResult:
    """Run one model helper call with the shared bounded retry policy."""

    policy = policy or RetryPolicy()
    max_retries = policy.max_retries if policy.enabled else 0
    for retry_index in range(max_retries + 1):
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            classification = classify_model_error(exc)
            if not classification.retryable or retry_index >= max_retries:
                raise
            delay = policy.delay(retry_index, classification.retry_after)
            if delay:
                await asyncio.sleep(delay)
    raise RuntimeError("unreachable model helper retry loop")


class ForgeModelRetryMiddleware(ModelRetryMiddleware):
    """LangChain's model middleware with Forge classification and events."""

    def __init__(self, policy: RetryPolicy | None = None, **kwargs: object) -> None:
        if policy is None:
            raw_max_retries = kwargs.pop("max_retries", 3)
            raw_initial_delay = kwargs.pop("initial_delay", 2.0)
            raw_backoff_factor = kwargs.pop("backoff_factor", 2.0)
            raw_max_delay = kwargs.pop("max_delay", 8.0)
            raw_jitter = kwargs.pop("jitter", False)
            policy = RetryPolicy(
                max_retries=raw_max_retries if isinstance(raw_max_retries, int) else 3,
                initial_delay=(
                    float(raw_initial_delay)
                    if isinstance(raw_initial_delay, (int, float))
                    and not isinstance(raw_initial_delay, bool)
                    else 2.0
                ),
                backoff_factor=(
                    float(raw_backoff_factor)
                    if isinstance(raw_backoff_factor, (int, float))
                    and not isinstance(raw_backoff_factor, bool)
                    else 2.0
                ),
                max_delay=(
                    float(raw_max_delay)
                    if isinstance(raw_max_delay, (int, float))
                    and not isinstance(raw_max_delay, bool)
                    else 8.0
                ),
                jitter=raw_jitter if isinstance(raw_jitter, bool) else False,
            )
        elif kwargs:
            raise TypeError(f"unexpected retry options: {', '.join(sorted(kwargs))}")
        super().__init__(
            max_retries=policy.max_retries,
            retry_on=(Exception,),
            on_failure="error",
            backoff_factor=policy.backoff_factor,
            initial_delay=policy.initial_delay,
            max_delay=policy.max_delay,
            jitter=policy.jitter,
        )
        self.policy = policy

    async def awrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.policy.enabled:
            return await handler(request)
        for retry_index in range(self.policy.max_retries + 1):
            try:
                return await handler(request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                classification = classify_model_error(exc)
                if not classification.retryable or retry_index >= self.policy.max_retries:
                    raise
                delay = self.policy.delay(retry_index, classification.retry_after)
                payload: dict[str, JSONValue] = {
                    "type": "forge.model_retry.v1",
                    "attempt": retry_index + 1,
                    "max_attempts": self.policy.max_retries,
                    "delay_seconds": delay,
                    "message": redact_model_error(exc),
                    "kind": classification.kind,
                }
                if classification.status_code is not None:
                    payload["status_code"] = classification.status_code
                with suppress(Exception):
                    request.runtime.stream_writer(payload)
                if delay:
                    await asyncio.sleep(delay)
        raise RuntimeError("unreachable retry loop")


__all__ = [
    "ErrorKind",
    "ForgeModelRetryMiddleware",
    "ModelErrorClassification",
    "RetryPolicy",
    "classify_model_error",
    "redact_model_error",
    "retry_model_call",
]
