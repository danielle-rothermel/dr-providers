"""Contract tests for the v0.2 kernel (config, payload, parse, failures)."""

from __future__ import annotations

import pytest

from dr_providers.kernel import (
    EndpointKind,
    FailureClass,
    FixtureOutcome,
    FixtureProvider,
    LlmRequest,
    MessageRole,
    PromptMessage,
    ProviderKind,
    RateLimitedProviderError,
    RequestControl,
    UnsupportedControlError,
    build_payload,
    classify_status_code,
    endpoint_path,
    failure_record,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    parse_chat_completions_body,
    parse_response,
    parse_responses_body,
    sanitize_kwargs,
    token_usage_from_body,
)

MESSAGES = (
    PromptMessage(role=MessageRole.SYSTEM, content="be brief"),
    PromptMessage(role=MessageRole.USER, content="write add"),
)


class TestConfigPresets:
    def test_openrouter_chat(self) -> None:
        config = openrouter_chat_config(model="m")
        assert config.provider_kind is ProviderKind.OPENROUTER
        assert config.endpoint_kind is EndpointKind.CHAT_COMPLETIONS
        assert config.api_key_env == "OPENROUTER_API_KEY"
        assert config.throttle_identity == "openrouter:chat_completions:m"

    def test_openai_responses(self) -> None:
        config = openai_responses_config(model="m")
        assert config.endpoint_kind is EndpointKind.RESPONSES
        assert config.token_limit_parameter.value == "max_output_tokens"

    def test_gemini_compat_preset(self) -> None:
        config = gemini_chat_config(model="gemini-2.5-flash")
        assert config.provider_kind is ProviderKind.GEMINI
        assert config.api_key_env == "GEMINI_API_KEY"
        assert config.base_url is not None
        assert "generativelanguage.googleapis.com" in config.base_url
        assert config.endpoint_kind is EndpointKind.CHAT_COMPLETIONS

    def test_explicit_throttle_key_wins(self) -> None:
        config = openai_chat_config(model="m").model_copy(
            update={"throttle_key": "custom"}
        )
        assert config.throttle_identity == "custom"


class TestBuildPayload:
    def test_chat_payload_shape(self) -> None:
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
            temperature=0.2,
            token_limit=64,
        )
        payload = build_payload(request)
        assert payload == {
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "write add"},
            ],
            "temperature": 0.2,
            "max_completion_tokens": 64,
        }

    def test_responses_payload_lifts_system_to_instructions(self) -> None:
        request = LlmRequest(
            provider_config=openai_responses_config(model="m"),
            messages=MESSAGES,
            token_limit=64,
        )
        payload = build_payload(request)
        assert payload["instructions"] == "be brief"
        assert payload["input"] == [{"role": "user", "content": "write add"}]
        assert payload["max_output_tokens"] == 64
        assert "messages" not in payload

    def test_reasoning_top_level_for_openai(self) -> None:
        request = LlmRequest(
            provider_config=openai_responses_config(model="m"),
            messages=MESSAGES,
            reasoning={"effort": "low"},
        )
        assert build_payload(request)["reasoning"] == {"effort": "low"}

    def test_reasoning_inline_body_for_openrouter(self) -> None:
        request = LlmRequest(
            provider_config=openrouter_chat_config(model="m"),
            messages=MESSAGES,
            reasoning={"effort": "low"},
        )
        payload = build_payload(request)
        assert payload["reasoning"] == {"effort": "low"}

    def test_unsupported_control_raises_loudly(self) -> None:
        config = openai_chat_config(model="m").model_copy(
            update={
                "supported_controls": frozenset(
                    {RequestControl.TOKEN_LIMIT, RequestControl.REASONING}
                )
            }
        )
        request = LlmRequest(
            provider_config=config,
            messages=MESSAGES,
            temperature=0.5,
        )
        with pytest.raises(UnsupportedControlError) as exc_info:
            build_payload(request)
        assert exc_info.value.failure.metadata["control"] == "temperature"

    def test_unsupported_control_drop_opt_in(self) -> None:
        config = openai_chat_config(model="m").model_copy(
            update={
                "supported_controls": frozenset(
                    {RequestControl.TOKEN_LIMIT, RequestControl.REASONING}
                ),
                "allow_unsupported_control_drop": True,
            }
        )
        request = LlmRequest(
            provider_config=config,
            messages=MESSAGES,
            temperature=0.5,
        )
        assert "temperature" not in build_payload(request)

    def test_endpoint_paths(self) -> None:
        assert endpoint_path(openai_chat_config(model="m")) == (
            "/chat/completions"
        )
        assert endpoint_path(openai_responses_config(model="m")) == (
            "/responses"
        )


class TestParseResponses:
    def test_chat_body_parses_parts(self) -> None:
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
            body,
            config=openai_chat_config(model="m"),
            payload={"model": "m"},
        )
        assert response.text == "hi"
        assert response.usage is not None
        assert response.usage.reasoning_tokens == 2
        assert response.cost is not None
        assert response.cost.total_cost == 0.001
        assert response.model == "m-actual"
        assert response.finish_reason == "stop"
        assert response.payload == {"model": "m"}
        assert response.continuation_handle is None

    def test_responses_body_sets_continuation_handle(self) -> None:
        body = {
            "id": "resp-1",
            "status": "completed",
            "output_text": "hi",
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        response = parse_responses_body(
            body, config=openai_responses_config(model="m")
        )
        assert response.continuation_handle == "resp-1"
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 3

    def test_responses_output_parts_fallback(self) -> None:
        body = {
            "id": "resp-2",
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "part one "},
                        {"type": "output_text", "text": "part two"},
                    ]
                }
            ],
        }
        response = parse_responses_body(
            body, config=openai_responses_config(model="m")
        )
        assert response.text == "part one part two"

    def test_parse_dispatches_by_endpoint_kind(self) -> None:
        chat_body = {
            "choices": [{"message": {"content": "x"}}],
        }
        response = parse_response(
            chat_body, config=openrouter_chat_config(model="m")
        )
        assert response.text == "x"

    @pytest.mark.parametrize(
        "body",
        [{}, {"choices": []}, {"choices": [{"message": {}}]}],
        ids=["missing", "empty", "no_text"],
    )
    def test_chat_parse_failures_are_permanent(self, body: dict) -> None:
        from dr_providers.kernel import PermanentProviderError

        with pytest.raises(PermanentProviderError):
            parse_chat_completions_body(
                body, config=openai_chat_config(model="m")
            )


class TestFailures:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (429, FailureClass.RATE_LIMITED),
            (500, FailureClass.TRANSIENT),
            (503, FailureClass.TRANSIENT),
            (408, FailureClass.TRANSIENT),
            (409, FailureClass.TRANSIENT),
            (425, FailureClass.TRANSIENT),
            (400, FailureClass.PERMANENT),
            (401, FailureClass.PERMANENT),
            (404, FailureClass.PERMANENT),
        ],
    )
    def test_classify_status_code(
        self, status: int, expected: FailureClass
    ) -> None:
        assert classify_status_code(status) is expected

    def test_failure_record_retryable_follows_class(self) -> None:
        rate_limited = failure_record(
            failure_class=FailureClass.RATE_LIMITED, message="slow down"
        )
        assert rate_limited.retryable is True
        permanent = failure_record(
            failure_class=FailureClass.PERMANENT, message="bad key"
        )
        assert permanent.retryable is False

    def test_sanitize_kwargs_redacts_credentials(self) -> None:
        assert sanitize_kwargs({"api_key": "secret", "temperature": 0.7}) == {
            "api_key": "<redacted>",
            "temperature": 0.7,
        }

    def test_token_usage_absent_when_no_fields(self) -> None:
        assert token_usage_from_body({"usage": {}}) is None
        assert token_usage_from_body({}) is None


class TestFixtureProvider:
    def test_scripted_text_response(self) -> None:
        provider = FixtureProvider(
            [FixtureOutcome(text="scripted", finish_reason="stop")]
        )
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
            idempotency_key="attempt-1",
        )
        response = provider.complete(request)
        assert response.text == "scripted"
        assert response.payload["model"] == "m"
        assert provider.requests[0].idempotency_key == "attempt-1"

    def test_scripted_failure_raises_carrier(self) -> None:
        failure = failure_record(
            failure_class=FailureClass.RATE_LIMITED,
            message="scripted 429",
        )
        provider = FixtureProvider([FixtureOutcome(failure=failure)])
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
        )
        with pytest.raises(RateLimitedProviderError) as exc_info:
            provider.complete(request)
        assert exc_info.value.failure.retryable is True

    def test_last_outcome_repeats(self) -> None:
        provider = FixtureProvider([FixtureOutcome(text="only")])
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
        )
        assert provider.complete(request).text == "only"
        assert provider.complete(request).text == "only"
        assert len(provider.requests) == 2
