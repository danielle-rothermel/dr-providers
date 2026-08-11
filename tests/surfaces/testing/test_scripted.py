from __future__ import annotations

from dr_providers import (
    CostInfo,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderStopReason,
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
                    response_body={
                        "id": "response-scripted",
                        "nested": {"ok": True},
                    },
                    usage=usage,
                    cost=cost,
                    warnings=(warning,),
                    stop_reason=ProviderStopReason.LENGTH,
                )
            ]
        )

        evidence = provider.invoke(request)
        outcome = evidence.outcome

        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome == ProviderTransportResponse(
            text="scripted full response",
            response_body={
                "id": "response-scripted",
                "nested": {"ok": True},
            },
            usage=usage,
            cost=cost,
            warnings=(warning,),
            stop_reason=ProviderStopReason.LENGTH,
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
        assert evidence.http_request is None
        assert evidence.policy_identity is None
        assert evidence.request_identity_hash == request.identity_hash

    def test_outcomes_are_consumed_in_order_then_last_repeats(self) -> None:
        provider = ScriptedProvider(
            [ScriptedOutcome(text="first"), ScriptedOutcome(text="second")]
        )
        requests = [
            request_for(openai_chat_config(model="model-1")),
            request_for(openai_chat_config(model="model-2")),
            request_for(openai_chat_config(model="model-3")),
        ]

        first = provider.invoke(requests[0]).outcome
        second = provider.invoke(requests[1]).outcome
        repeated = provider.invoke(requests[2]).outcome

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

    def test_high_usage_without_wire_length_has_no_truncation_signal(
        self,
    ) -> None:
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
        outcome = provider.invoke(request).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.stop_reason is ProviderStopReason.STOP
        assert outcome.warnings == ()
