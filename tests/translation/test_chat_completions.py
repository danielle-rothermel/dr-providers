import pytest

from dr_providers import (
    ProviderStopReason,
    ProviderTransportFailure,
    ProviderTransportResponse,
    RecoverabilityClass,
    openai_chat_config,
    parse_chat_completions_body,
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


def test_chat_body_parses_parts() -> None:
    body = {
        "id": "chatcmpl-1",
        "model": "m-actual",
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "cost": 0.001,
        },
    }

    response = parse_chat_completions_body(
        body, config=openai_chat_config(model="m")
    )

    assert isinstance(response, ProviderTransportResponse)
    assert response.text == "hi"
    assert response.usage is not None
    assert response.usage.reasoning_tokens == 2
    assert response.cost is not None
    assert response.cost.total_cost == 0.001
    assert response.model == "m-actual"
    assert response.stop_reason is ProviderStopReason.STOP
    assert response.response_body == body


@pytest.mark.parametrize(
    "body",
    [{}, {"choices": []}, {"choices": [{"message": {}}]}],
    ids=["missing", "empty", "no_text"],
)
def test_chat_parse_failures_are_typed(body: dict) -> None:
    outcome = parse_chat_completions_body(
        body, config=openai_chat_config(model="m")
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.recoverability is RecoverabilityClass.PERMANENT


def _classify(body: dict) -> ProviderInvocationOutcome:
    """Classify a wire body through the real parse and classify path."""
    outcome = parse_chat_completions_body(
        body, config=openai_chat_config(model="m")
    )
    evidence = ProviderInvocationEvidence(
        request_identity_hash="c" * 64,
        response=outcome if is_response(outcome) else None,
        failure=outcome if is_failure(outcome) else None,
    )
    return classify_provider_invocation(
        evidence, AcceptAllSemanticResponseClassifier()
    )


# Captured live from OpenRouter: deepseek/deepseek-r1 at max_tokens=1
# returns HTTP 200 with a null content and finish_reason "length".
TRUNCATED_NO_TEXT_BODY = {
    "id": "gen-truncated",
    "model": "deepseek/deepseek-r1",
    "choices": [
        {
            "index": 0,
            "finish_reason": "length",
            "native_finish_reason": "length",
            "message": {"role": "assistant", "content": None, "refusal": None},
        }
    ],
    "usage": {"prompt_tokens": 14, "completion_tokens": 1, "total_tokens": 15},
}

EMPTY_GENERATION_BODY = {
    "id": "gen-empty",
    "model": "m-actual",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": ""},
        }
    ],
    "usage": {"prompt_tokens": 17, "completion_tokens": 0, "total_tokens": 17},
}

NO_STOP_REASON_BODY = {
    "id": "gen-unknown",
    "model": "m-actual",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
}


def test_blank_text_with_length_stop_is_truncated_no_text() -> None:
    outcome = parse_chat_completions_body(
        TRUNCATED_NO_TEXT_BODY, config=openai_chat_config(model="m")
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == RESPONSE_INCOMPLETE_NO_TEXT_CODE
    assert (
        _classify(TRUNCATED_NO_TEXT_BODY)
        is ProviderInvocationOutcome.TRUNCATED_NO_TEXT
    )


def test_blank_text_with_genuine_stop_is_empty_generation() -> None:
    outcome = parse_chat_completions_body(
        EMPTY_GENERATION_BODY, config=openai_chat_config(model="m")
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
    outcome = parse_chat_completions_body(
        NO_STOP_REASON_BODY, config=openai_chat_config(model="m")
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == RESPONSE_NO_TEXT_CODE
    assert (
        _classify(NO_STOP_REASON_BODY)
        is ProviderInvocationOutcome.MISSING_GENERATION_TEXT
    )


# OpenRouter delivers upstream provider errors inside HTTP 200 as an error
# envelope with no choices.
ERROR_ENVELOPE_BODY = {
    "error": {
        "message": "Provider returned error",
        "code": 429,
        "metadata": {"provider_name": "Upstream"},
    },
    "user_id": "user-1",
}


def test_error_envelope_without_choices_is_provider_rejection() -> None:
    outcome = parse_chat_completions_body(
        ERROR_ENVELOPE_BODY, config=openai_chat_config(model="m")
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == PROVIDER_ERROR_ENVELOPE_CODE
    assert outcome.response_body == ERROR_ENVELOPE_BODY
    assert (
        _classify(ERROR_ENVELOPE_BODY)
        is ProviderInvocationOutcome.PROVIDER_REJECTION
    )


def test_error_envelope_with_empty_choices_is_provider_rejection() -> None:
    """An empty choices list carries no more text than an absent one."""
    body = {**ERROR_ENVELOPE_BODY, "choices": []}

    outcome = parse_chat_completions_body(
        body, config=openai_chat_config(model="m")
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == PROVIDER_ERROR_ENVELOPE_CODE
    assert _classify(body) is ProviderInvocationOutcome.PROVIDER_REJECTION


@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_error_envelope_with_blank_text_outranks_the_stop_reason(
    finish_reason: str | None,
) -> None:
    body = {
        **ERROR_ENVELOPE_BODY,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": "   "},
            }
        ],
    }

    outcome = parse_chat_completions_body(
        body, config=openai_chat_config(model="m")
    )

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == PROVIDER_ERROR_ENVELOPE_CODE
    assert _classify(body) is ProviderInvocationOutcome.PROVIDER_REJECTION


def test_error_envelope_beside_text_bearing_choice_stays_success() -> None:
    body = {
        "id": "gen-partial",
        "model": "m-actual",
        "error": {"message": "a later upstream leg failed", "code": 500},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "usable text"},
            }
        ],
    }

    outcome = parse_chat_completions_body(
        body, config=openai_chat_config(model="m")
    )

    assert isinstance(outcome, ProviderTransportResponse)
    assert outcome.text == "usable text"
    assert _classify(body) is ProviderInvocationOutcome.SUCCESS
