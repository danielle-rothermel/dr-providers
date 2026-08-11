from dr_providers import (
    GenerationControls,
    ProviderStopReason,
    ProviderTransportFailure,
    ProviderTransportResponse,
    anthropic_messages_config,
    parse_anthropic_messages_body,
)
from dr_providers.lifecycle import (
    AcceptAllSemanticResponseClassifier,
    ProviderInvocationOutcome,
    classify_provider_invocation,
)
from dr_providers.outcomes.evidence import ProviderInvocationEvidence
from dr_providers.outcomes.models import is_failure, is_response
from dr_providers.translation.common import (
    PROVIDER_ERROR_ENVELOPE_CODE,
    RESPONSE_INCOMPLETE_NO_TEXT_CODE,
    RESPONSE_NO_TEXT_CODE,
)


def test_anthropic_body_parses_parts() -> None:
    body = {
        "id": "msg-1",
        "model": "claude",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }

    response = parse_anthropic_messages_body(
        body,
        config=anthropic_messages_config(
            model="claude", controls=GenerationControls(token_limit=64)
        ),
    )

    assert isinstance(response, ProviderTransportResponse)
    assert response.text == "hello"
    assert response.stop_reason is ProviderStopReason.STOP
    assert response.usage is not None
    assert response.usage.total_tokens == 7


def _config():
    return anthropic_messages_config(
        model="claude", controls=GenerationControls(token_limit=64)
    )


def _classify(body: dict) -> ProviderInvocationOutcome:
    """Classify a wire body through the real parse and classify path."""
    outcome = parse_anthropic_messages_body(body, config=_config())
    evidence = ProviderInvocationEvidence(
        request_identity_hash="a" * 64,
        response=outcome if is_response(outcome) else None,
        failure=outcome if is_failure(outcome) else None,
    )
    return classify_provider_invocation(
        evidence, AcceptAllSemanticResponseClassifier()
    )


# A thinking-capable model whose thinking consumes the whole token limit
# returns stop_reason "max_tokens" with thinking-only content and no text.
THINKING_ONLY_BODY = {
    "id": "msg-truncated",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "stop_reason": "max_tokens",
    "content": [{"type": "thinking", "thinking": "Let me work through this"}],
    "usage": {"input_tokens": 108, "output_tokens": 2049},
}

EMPTY_GENERATION_BODY = {
    "id": "msg-empty",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": ""}],
    "usage": {"input_tokens": 10, "output_tokens": 0},
}

NO_STOP_REASON_BODY = {
    "id": "msg-unknown",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "   "}],
}


def test_thinking_only_max_tokens_body_is_truncated_no_text() -> None:
    outcome = parse_anthropic_messages_body(
        THINKING_ONLY_BODY, config=_config()
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == RESPONSE_INCOMPLETE_NO_TEXT_CODE
    assert (
        _classify(THINKING_ONLY_BODY)
        is ProviderInvocationOutcome.TRUNCATED_NO_TEXT
    )


def test_blank_text_with_end_turn_is_empty_generation() -> None:
    outcome = parse_anthropic_messages_body(
        EMPTY_GENERATION_BODY, config=_config()
    )

    assert isinstance(outcome, ProviderTransportResponse)
    assert outcome.text == ""
    assert outcome.stop_reason is ProviderStopReason.STOP
    assert (
        _classify(EMPTY_GENERATION_BODY)
        is ProviderInvocationOutcome.EMPTY_GENERATION
    )


def test_blank_text_without_stop_reason_stays_missing_generation_text() -> (
    None
):
    outcome = parse_anthropic_messages_body(
        NO_STOP_REASON_BODY, config=_config()
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == RESPONSE_NO_TEXT_CODE
    assert (
        _classify(NO_STOP_REASON_BODY)
        is ProviderInvocationOutcome.MISSING_GENERATION_TEXT
    )


# Captured live from api.anthropic.com: an error body carries type "error"
# and an error object in place of message content.
ERROR_ENVELOPE_BODY = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "Your credit balance is too low to access the API.",
    },
    "request_id": "req-1",
}


def test_error_envelope_without_content_is_provider_rejection() -> None:
    outcome = parse_anthropic_messages_body(
        ERROR_ENVELOPE_BODY, config=_config()
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == PROVIDER_ERROR_ENVELOPE_CODE
    assert outcome.response_body == ERROR_ENVELOPE_BODY
    assert (
        _classify(ERROR_ENVELOPE_BODY)
        is ProviderInvocationOutcome.PROVIDER_REJECTION
    )


def test_error_envelope_beside_text_content_stays_success() -> None:
    body = {
        "id": "msg-partial",
        "type": "error",
        "model": "claude",
        "stop_reason": "end_turn",
        "error": {"type": "overloaded_error", "message": "a later leg failed"},
        "content": [{"type": "text", "text": "usable text"}],
    }

    outcome = parse_anthropic_messages_body(body, config=_config())

    assert isinstance(outcome, ProviderTransportResponse)
    assert outcome.text == "usable text"
    assert _classify(body) is ProviderInvocationOutcome.SUCCESS
