"""Small, provider-aware guidance for final model errors."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from forge_agent import ErrorEvent
from forge_agent.retry import classify_model_error, redact_model_error
from forge_coding.providers.catalog import builtin_provider_entry
from forge_coding.providers.config import ProviderConfig

GuidanceKind = Literal["auth", "missing_api_key", "unknown_model", "unknown_provider"]
StructuredGuidanceKind = GuidanceKind | Literal["suppress"]

_ENV_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}\Z")
_QUOTED_LABEL_PATTERN = r"[\"'`][^\"'`\r\n]{1,96}[\"'`]"
_PLAIN_LABEL_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}"
_MISSING_API_KEY_RE = re.compile(
    r"(?:missing|no|without|unset|not[ -]+(?:configured|set|provided)).{0,48}"
    r"(?:api[ _-]*key|credential|access[ _-]*token)"
    r"|(?:api[ _-]*key|credential|access[ _-]*token).{0,48}"
    r"(?:missing|not[ -]+(?:configured|set|provided)|required)",
    re.IGNORECASE,
)
_UNKNOWN_MODEL_RE = re.compile(
    rf"(?:unknown|invalid|unsupported)\s+model\s*[:=]?\s*"
    rf"(?:{_QUOTED_LABEL_PATTERN}|{_PLAIN_LABEL_PATTERN})(?=\s|$|[.,;])"
    r"|\bmodel\s+(?:[\"'`][^\"'`\r\n]{1,96}[\"'`]"
    r"|(?!response\b|output\b|result\b|request\b|cache\b)"
    r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,95})\s+(?:was\s+|is\s+)?"
    r"(?:not found|does not exist|not configured)\b",
    re.IGNORECASE,
)
_UNKNOWN_PROVIDER_RE = re.compile(
    rf"(?:unknown|invalid|unsupported)\s+provider\s*[:=]?\s*"
    rf"(?:{_QUOTED_LABEL_PATTERN}|{_PLAIN_LABEL_PATTERN})(?=\s|$|[.,;])"
    r"|\bprovider\s+(?:[\"'`][^\"'`\r\n]{1,96}[\"'`]"
    r"|(?!response\b|returned\b|request\b)"
    r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,95})\s+(?:was\s+|is\s+)?"
    r"(?:not found|does not exist|not configured)\b",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"\b(?:401|403)\b|unauthori[sz]ed|forbidden|invalid\s+api[ _-]*key|"
    r"authentication\s+(?:failed|error)|invalid\s+credentials",
    re.IGNORECASE,
)


def project_provider_error(
    event: ErrorEvent,
    *,
    provider_name: str,
    model: str,
    provider_config: ProviderConfig | None = None,
) -> ErrorEvent:
    """Return a bounded final error with actionable provider guidance.

    The message is redacted before classification or interpolation. Structured
    fields are preferred; text matching is deliberately narrow and only fills
    in a category when the structured data has no actionable category.
    """

    redacted_message = redact_model_error(event.message)
    if event.recoverable:
        return event.model_copy(update={"message": redacted_message})

    kind = _structured_kind(event.data)
    if kind is None:
        kind = _message_kind(redacted_message)
    if kind == "suppress":
        return event.model_copy(update={"message": redacted_message})
    guidance = _guidance(
        kind,
        provider_name=provider_name,
        model=model,
        provider_config=provider_config,
    )
    if guidance is None:
        return event.model_copy(update={"message": redacted_message})
    message = f"{redacted_message}\n\n{guidance}" if redacted_message else guidance
    return event.model_copy(update={"message": message})


def _structured_kind(data: Mapping[str, object] | None) -> StructuredGuidanceKind | None:
    if not isinstance(data, Mapping):
        return None

    for key in ("missing_api_key", "api_key_missing", "missing_key", "missing_credential"):
        if data.get(key) is True:
            return "missing_api_key"

    for key in ("code", "error_code", "error_type", "type", "reason"):
        kind = _token_kind(data.get(key))
        if kind is not None:
            return kind

    kind = _token_kind(data.get("kind"))
    if kind is not None:
        return kind
    status = _status_code(data)
    if status in {401, 403}:
        return "auth"

    resource = _normalized_token(data.get("resource"))
    code = _normalized_token(data.get("code"))
    if code in {"not_found", "notfound", "unavailable"}:
        if resource == "model":
            return "unknown_model"
        if resource == "provider":
            return "unknown_provider"
    return None


def _status_code(data: Mapping[str, object]) -> int | None:
    for key in ("status_code", "status", "http_status"):
        value = data.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            try:
                return int(value.strip())
            except ValueError:
                continue
    return None


def _token_kind(value: object) -> StructuredGuidanceKind | None:
    token = _normalized_token(value)
    if token is None:
        return None
    if token in {
        "401",
        "403",
        "auth",
        "authentication",
        "authentication_error",
        "unauthorized",
        "forbidden",
        "invalid_api_key",
        "invalid_credentials",
        "permission_denied",
    }:
        return "auth"
    if token in {
        "missing_api_key",
        "api_key_missing",
        "missing_key",
        "missing_credential",
        "credentials_missing",
    }:
        return "missing_api_key"
    if token in {
        "unknown_model",
        "model_not_found",
        "model_not_configured",
        "model_unavailable",
        "invalid_model",
    }:
        return "unknown_model"
    if token in {
        "unknown_provider",
        "provider_not_found",
        "provider_not_configured",
        "invalid_provider",
    }:
        return "unknown_provider"
    if token in {"abort", "overflow", "quota", "transient"}:
        return "suppress"
    return None


def _message_kind(message: str) -> StructuredGuidanceKind | None:
    if classify_model_error(RuntimeError(message)).kind in {
        "abort",
        "overflow",
        "quota",
        "transient",
    }:
        return "suppress"
    if _MISSING_API_KEY_RE.search(message):
        return "missing_api_key"
    if _UNKNOWN_MODEL_RE.search(message):
        return "unknown_model"
    if _UNKNOWN_PROVIDER_RE.search(message):
        return "unknown_provider"
    if _AUTH_RE.search(message):
        return "auth"
    return None


def _guidance(
    kind: GuidanceKind | None,
    *,
    provider_name: str,
    model: str,
    provider_config: ProviderConfig | None,
) -> str | None:
    if kind is None:
        return None

    provider = _safe_label(provider_name) or _safe_label(
        getattr(provider_config, "name", None)
    )
    provider_display = provider or "the current provider"
    model_display = _safe_label(model) or "the selected model"
    api_key_env = _safe_env(getattr(provider_config, "api_key_env", None))

    if kind == "missing_api_key":
        return (
            f"Missing API key for provider {provider_display}. "
            f"{_credential_guidance(provider, api_key_env)}"
        )
    if kind == "auth":
        return (
            f"Authentication failed for provider {provider_display}. "
            f"{_credential_guidance(provider, api_key_env)}"
        )
    if kind == "unknown_model":
        return (
            f"Model {model_display} is not configured for {provider_display}. "
            "Run forge models or /model to choose a configured model."
        )
    return (
        f"Provider {provider_display} is not configured. "
        "Run forge models or /login to choose a provider."
    )


def _credential_guidance(provider: str | None, api_key_env: str | None) -> str:
    if provider is not None and builtin_provider_entry(provider) is not None:
        command = f"Run /login {provider}"
        return f"{command} or set {api_key_env}." if api_key_env else f"{command}."
    if api_key_env:
        return f"Set {api_key_env} or run /login custom to update provider configuration."
    return "Run /login to configure credentials."


def _normalized_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().casefold().replace("-", "_").replace(" ", "_")
    return token or None


def _safe_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    label = redact_model_error(value, limit=96).strip()
    return label if _LABEL_RE.fullmatch(label) else None


def _safe_env(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    env = value.strip()
    return env if _ENV_RE.fullmatch(env) else None


__all__ = ["GuidanceKind", "project_provider_error"]
