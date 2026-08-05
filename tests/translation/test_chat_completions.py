import pytest

from dr_providers import (
    FailureClass,
    ProviderTransportFailure,
    ProviderTransportResponse,
    openai_chat_config,
    parse_chat_completions_body,
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
    assert response.finish_reason == "stop"
    assert response.raw_body == body


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
    assert outcome.failure_class is FailureClass.PERMANENT
