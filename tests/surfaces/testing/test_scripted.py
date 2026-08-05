"""Deterministic scripted-provider surface tests."""

from __future__ import annotations

from hashlib import sha256

from dr_providers import (
    FailureClass,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportFailure,
    ProviderTransportResponse,
    ScriptedOutcome,
    ScriptedProvider,
    TokenUsage,
    Transcript,
    openai_chat_config,
    openai_responses_config,
    parse_responses_body,
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
    def test_scripted_text_response(self) -> None:
        provider = ScriptedProvider(
            [ScriptedOutcome(text="scripted", finish_reason="stop")]
        )
        outcome = provider.complete(request_for(openai_chat_config(model="m")))
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.text == "scripted"
        assert provider.payloads[0]["model"] == "m"

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

    def test_last_outcome_repeats(self) -> None:
        provider = ScriptedProvider([ScriptedOutcome(text="only")])
        request = request_for(openai_chat_config(model="m"))
        first = provider.complete(request)
        second = provider.complete(request)
        assert isinstance(first, ProviderTransportResponse)
        assert isinstance(second, ProviderTransportResponse)
        assert first.text == "only"
        assert second.text == "only"
        assert len(provider.requests) == 2

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

    def test_diagnostics_response_id_hash_stays_diagnostic(self) -> None:
        body = {
            "id": "resp-diag",
            "status": "failed",
            "output": [],
        }
        outcome = parse_responses_body(
            body, config=openai_responses_config(model="m")
        )
        assert isinstance(outcome, ProviderTransportFailure)
        expected = sha256(b"resp-diag").hexdigest()[:16]
        assert outcome.metadata["diagnostics"]["response_id_hash"] == expected
