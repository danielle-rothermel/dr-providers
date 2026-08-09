from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import httpx
import pytest

from dr_providers import (
    FailureClass,
    HttpProvider,
    MessageRole,
    PromptMessage,
    Provider,
    ProviderCallRequest,
    ProviderKind,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    ScriptedOutcome,
    ScriptedProvider,
    Transcript,
    openai_chat_config,
)

OutcomeKind = Literal["success", "failure"]
ProviderFactory = Callable[[OutcomeKind], Provider]

REQUEST = ProviderCallRequest(
    config=openai_chat_config(model="m"),
    transcript=Transcript(
        messages=(PromptMessage(role=MessageRole.USER, content="say hello"),)
    ),
)


def _scripted_provider(outcome_kind: OutcomeKind) -> Provider:
    if outcome_kind == "success":
        outcome = ScriptedOutcome(text="hello")
    else:
        outcome = ScriptedOutcome(
            failure=ProviderTransportFailure(
                failure_class=FailureClass.RATE_LIMITED,
                code="scripted_rate_limit",
                message="slow down",
            )
        )
    return ScriptedProvider([outcome])


def _http_provider(outcome_kind: OutcomeKind) -> Provider:
    def handler(_request: httpx.Request) -> httpx.Response:
        if outcome_kind == "failure":
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return HttpProvider(
        policy=ProviderTransportPolicy(
            provider_kind=ProviderKind.OPENAI,
            api_key_env="TEST_API_KEY",
            base_url="https://example.test/v1",
        ),
        _client_factory=lambda **_kwargs: httpx.Client(
            transport=httpx.MockTransport(handler)
        ),
        api_key="test-key",
    )


@pytest.mark.parametrize(
    "provider_factory",
    [_scripted_provider, _http_provider],
    ids=("scripted", "http"),
)
def test_provider_returns_common_typed_success(
    provider_factory: ProviderFactory,
) -> None:
    evidence = provider_factory("success").invoke(REQUEST)
    outcome = evidence.outcome

    assert isinstance(outcome, ProviderTransportResponse)
    assert outcome.text == "hello"
    assert evidence.request_identity_hash == REQUEST.identity_hash


@pytest.mark.parametrize(
    "provider_factory",
    [_scripted_provider, _http_provider],
    ids=("scripted", "http"),
)
def test_provider_returns_common_typed_failure(
    provider_factory: ProviderFactory,
) -> None:
    outcome = provider_factory("failure").invoke(REQUEST).outcome

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.failure_class is FailureClass.RATE_LIMITED
