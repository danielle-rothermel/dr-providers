from __future__ import annotations

from dr_providers import (
    CostInfo,
    FailureClass,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportFailure,
    ProviderTransportResponse,
    ProviderTransportWarning,
    ScriptedOutcome,
    ScriptedProvider,
    TokenUsage,
    Transcript,
    openai_chat_config,
)

MESSAGES = (
    PromptMessage(role=MessageRole.SYSTEM, content="be brief"),
    PromptMessage(role=MessageRole.USER, content="write add"),
)


def request_for(config, messages=MESSAGES) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config, transcript=Transcript(messages=messages)
    )


class TestScriptedProvider:
    def test_scripted_response_preserves_every_outcome_field(self) -> None:
        request = request_for(openai_chat_config(model="full-model"))
        usage = TokenUsage(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            reasoning_tokens=3,
        )
        cost = CostInfo(total_cost=0.012, currency="USD")
        warning = ProviderTransportWarning(
            code="scripted_notice",
            message="scripted warning",
            metadata={"source": "fixture"},
        )
        provider = ScriptedProvider(
            [
                ScriptedOutcome(
                    text="scripted full response",
                    raw_body={"id": "raw-scripted", "nested": {"ok": True}},
                    usage=usage,
                    cost=cost,
                    warnings=(warning,),
                    finish_reason="length",
                )
            ]
        )

        outcome = provider.complete(request)

        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome == ProviderTransportResponse(
            text="scripted full response",
            raw_body={"id": "raw-scripted", "nested": {"ok": True}},
            usage=usage,
            cost=cost,
            warnings=(warning,),
            finish_reason="length",
            response_id="scripted-response-1",
            model="full-model",
        )
        assert provider.requests == [request]
        assert provider.payloads == [
            {
                "model": "full-model",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "write add"},
                ],
            }
        ]

    def test_scripted_failure_returns_typed_outcome(self) -> None:
        failure = ProviderTransportFailure(
            failure_class=FailureClass.RATE_LIMITED,
            message="scripted 429",
            retryable=True,
        )
        provider = ScriptedProvider([ScriptedOutcome(failure=failure)])
        outcome = provider.complete(request_for(openai_chat_config(model="m")))
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.retryable is True
        assert outcome.raw_request["model"] == "m"

    def test_outcomes_are_consumed_in_order_then_last_repeats(self) -> None:
        provider = ScriptedProvider(
            [ScriptedOutcome(text="first"), ScriptedOutcome(text="second")]
        )
        requests = [
            request_for(openai_chat_config(model="model-1")),
            request_for(openai_chat_config(model="model-2")),
            request_for(openai_chat_config(model="model-3")),
        ]

        first = provider.complete(requests[0])
        second = provider.complete(requests[1])
        repeated = provider.complete(requests[2])

        assert isinstance(first, ProviderTransportResponse)
        assert isinstance(second, ProviderTransportResponse)
        assert isinstance(repeated, ProviderTransportResponse)
        assert (first.text, second.text, repeated.text) == (
            "first",
            "second",
            "second",
        )
        assert (
            first.response_id,
            second.response_id,
            repeated.response_id,
        ) == (
            "scripted-response-1",
            "scripted-response-2",
            "scripted-response-3",
        )
        assert provider.requests == requests
        assert [payload["model"] for payload in provider.payloads] == [
            "model-1",
            "model-2",
            "model-3",
        ]

    def test_response_carries_conformance_warnings(self) -> None:
        provider = ScriptedProvider(
            [
                ScriptedOutcome(
                    text="over budget",
                    usage=TokenUsage(completion_tokens=99),
                )
            ]
        )
        request = request_for(
            openai_chat_config(
                model="m", controls=GenerationControls(token_limit=10)
            )
        )
        outcome = provider.complete(request)
        assert isinstance(outcome, ProviderTransportResponse)
        codes = [warning.code for warning in outcome.warnings]
        assert "token_limit_exceeded" in codes
