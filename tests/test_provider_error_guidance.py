from forge_agent import ErrorEvent
from forge_coding.providers.config import OpenAICompatibleProviderConfig
from forge_coding.providers.error_guidance import project_provider_error


def _provider() -> OpenAICompatibleProviderConfig:
    return OpenAICompatibleProviderConfig(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        models=("gpt-test",),
        default_model="gpt-test",
    )


def _project(message: str, data: dict[str, object] | None = None) -> ErrorEvent:
    return project_provider_error(
        ErrorEvent(message=message, recoverable=False, data=data),
        provider_name="openai",
        model="gpt-test",
        provider_config=_provider(),
    )


def test_structured_auth_error_uses_provider_login_and_redacts_message() -> None:
    event = _project(
        "HTTP 401 api_key=sk-secret-value at C:\\Users\\me\\forge.py",
        {"status_code": 401, "kind": "auth"},
    )

    assert "<redacted>" in event.message
    assert "C:\\Users\\me" not in event.message
    assert "Authentication failed for provider openai" in event.message
    assert "/login openai" in event.message
    assert "OPENAI_API_KEY" in event.message


def test_structured_missing_key_error_is_more_specific_than_generic_auth() -> None:
    event = _project("request rejected", {"code": "missing_api_key"})

    assert "Missing API key for provider openai" in event.message
    assert "/login openai" in event.message
    assert "OPENAI_API_KEY" in event.message


def test_structured_status_wins_over_conflicting_message_fallback() -> None:
    event = _project("unknown model gpt-test", {"status_code": 403})

    assert "Authentication failed" in event.message
    assert "/model" not in event.message


def test_explicit_non_actionable_kind_suppresses_conflicting_status() -> None:
    event = _project("quota exceeded", {"kind": "quota", "status_code": 403})
    code_event = _project("request rejected", {"code": "quota", "status_code": 403})

    assert event.message == "quota exceeded"
    assert "/login" not in event.message
    assert code_event.message == "request rejected"
    assert "/login" not in code_event.message


def test_structured_model_and_provider_errors_point_to_existing_commands() -> None:
    model_event = _project("request rejected", {"code": "model_not_found"})
    provider_event = _project("request rejected", {"kind": "unknown_provider"})

    assert "forge models" in model_event.message
    assert "/model" in model_event.message
    assert "forge models" in provider_event.message
    assert "/login" in provider_event.message


def test_string_fallback_is_conservative_and_redacted() -> None:
    auth = _project("provider returned 403: invalid api key sk-live-secret")
    model = _project("model 'acme-large' was not found")
    named_model = _project("unknown model acme-large")
    ordinary = _project("provider returned an unexpected response")

    assert "/login openai" in auth.message
    assert "<redacted>" in auth.message
    assert "/model" in model.message
    assert "forge models" in model.message
    assert "/model" in named_model.message
    assert "/login" not in ordinary.message
    assert "/model" not in ordinary.message
    assert "forge models" not in ordinary.message

    transient = _project("provider is temporarily unavailable", {"kind": "transient"})
    quota_status = _project("provider returned 403: quota exceeded")
    assert "forge models" not in transient.message
    assert "/login" not in transient.message
    assert "/login" not in quota_status.message

    model_cache = _project("model response was not found in cache")
    provider_resource = _project("provider returned resource not found")
    unspecified = _project("unknown model")
    assert "/model" not in model_cache.message
    assert "forge models" not in provider_resource.message
    assert "/model" not in unspecified.message


def test_recoverable_error_does_not_receive_provider_guidance() -> None:
    event = project_provider_error(
        ErrorEvent(message="HTTP 401", recoverable=True),
        provider_name="acme",
        model="acme-small",
        provider_config=_provider(),
    )

    assert event.message == "HTTP 401"


def test_custom_provider_guidance_uses_existing_configuration_flow() -> None:
    event = project_provider_error(
        ErrorEvent(message="HTTP 401", recoverable=False, data={"status_code": 401}),
        provider_name="acme",
        model="acme-small",
        provider_config=OpenAICompatibleProviderConfig(
            name="acme",
            api_key_env="ACME_API_KEY",
            models=("acme-small",),
            default_model="acme-small",
        ),
    )

    assert "/login acme" not in event.message
    assert "ACME_API_KEY" in event.message
    assert "/login custom" in event.message


def test_guidance_rejects_control_characters_in_provider_and_model_labels() -> None:
    event = project_provider_error(
        ErrorEvent(message="model 'requested-model' was not found", recoverable=False),
        provider_name="evil\x1b[31m",
        model="bad\nmodel",
    )

    assert "\x1b" not in event.message
    assert "bad\nmodel" not in event.message
    assert "the current provider" in event.message
    assert "the selected model" in event.message
