from __future__ import annotations

from typing import Any

import httpx

from dr_providers import (
    ApiKeyEnv,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderBaseUrl,
    ProviderCallConfig,
    ProviderCallRequest,
    ProviderKind,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    ReasoningEffort,
    Transcript,
    openai_chat_config,
)
from dr_providers.outcomes.conformance import (
    MODEL_SUBSTITUTION_CODE,
    REASONING_NOT_OBSERVED_CODE,
)
from dr_providers.transport.http import HttpProvider

MESSAGES = (PromptMessage(role=MessageRole.USER, content="write add"),)
CHAT_BODY_OK: dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "m",
    "choices": [
        {
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}
OPENAI_POLICY = ProviderTransportPolicy(
    provider_kind=ProviderKind.OPENAI,
    api_key_env=str(ApiKeyEnv.OPENAI),
    base_url=str(ProviderBaseUrl.OPENAI),
)


def request_for(
    config: ProviderCallConfig, messages=MESSAGES
) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config, transcript=Transcript(messages=messages)
    )


def openai_request(**control_overrides: Any) -> ProviderCallRequest:
    controls = GenerationControls(**control_overrides)
    return request_for(openai_chat_config(model="m", controls=controls))


def mock_provider(
    handler: Any,
    *,
    policy: ProviderTransportPolicy = OPENAI_POLICY,
) -> HttpProvider:
    return HttpProvider(
        policy=policy,
        _client_factory=lambda **_kwargs: httpx.Client(
            transport=httpx.MockTransport(handler)
        ),
        api_key="test-key",
    )


class TestConformance:
    def test_reasoning_not_observed_warning(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        outcome = provider.invoke(
            openai_request(reasoning=ReasoningEffort.LOW)
        ).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        codes = [w.code for w in outcome.warnings]
        assert REASONING_NOT_OBSERVED_CODE in codes

    def test_model_substitution_warning(self) -> None:
        body = dict(CHAT_BODY_OK)
        body["model"] = "m-other"
        provider = mock_provider(lambda _req: httpx.Response(200, json=body))
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        codes = [w.code for w in outcome.warnings]
        assert MODEL_SUBSTITUTION_CODE in codes

    def test_clean_response_has_no_warnings(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.warnings == ()
