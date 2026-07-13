"""Contract tests for the v0.2 kernel (config, payload, parse, failures)."""

from __future__ import annotations

import json

import pytest

from dr_providers import (
    EndpointKind,
    FailureClass,
    LlmRequest,
    MessageRole,
    PermanentProviderError,
    PromptMessage,
    ProviderKind,
    RateLimitedProviderError,
    ReasoningEffort,
    RequestControl,
    ScriptedOutcome,
    ScriptedProvider,
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

    def test_reasoning_object_for_openai_responses(self) -> None:
        request = LlmRequest(
            provider_config=openai_responses_config(model="m"),
            messages=MESSAGES,
            reasoning=ReasoningEffort.LOW,
        )
        payload = build_payload(request)
        assert payload["reasoning"] == {"effort": "low"}
        assert "reasoning_effort" not in payload

    def test_reasoning_object_for_openrouter(self) -> None:
        request = LlmRequest(
            provider_config=openrouter_chat_config(model="m"),
            messages=MESSAGES,
            reasoning=ReasoningEffort.HIGH,
        )
        payload = build_payload(request)
        assert payload["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in payload

    def test_reasoning_effort_field_for_openai_chat(self) -> None:
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
            reasoning=ReasoningEffort.MEDIUM,
        )
        payload = build_payload(request)
        assert payload["reasoning_effort"] == "medium"
        assert "reasoning" not in payload

    def test_reasoning_effort_field_for_gemini(self) -> None:
        request = LlmRequest(
            provider_config=gemini_chat_config(model="m"),
            messages=MESSAGES,
            reasoning=ReasoningEffort.MINIMAL,
        )
        payload = build_payload(request)
        assert payload["reasoning_effort"] == "minimal"
        assert "reasoning" not in payload

    def test_top_p_transported(self) -> None:
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
            top_p=0.9,
        )
        assert build_payload(request)["top_p"] == 0.9

    def test_top_p_unsupported_raises(self) -> None:
        config = openai_chat_config(model="m").model_copy(
            update={
                "supported_controls": frozenset(
                    {RequestControl.TEMPERATURE, RequestControl.TOKEN_LIMIT}
                )
            }
        )
        request = LlmRequest(
            provider_config=config,
            messages=MESSAGES,
            top_p=0.9,
        )
        with pytest.raises(UnsupportedControlError) as exc_info:
            build_payload(request)
        assert exc_info.value.failure.metadata["control"] == "top_p"

    def test_top_p_unsupported_drop_opt_in(self) -> None:
        config = openai_chat_config(model="m").model_copy(
            update={
                "supported_controls": frozenset(
                    {RequestControl.TEMPERATURE, RequestControl.TOKEN_LIMIT}
                ),
                "allow_unsupported_control_drop": True,
            }
        )
        request = LlmRequest(
            provider_config=config,
            messages=MESSAGES,
            top_p=0.9,
        )
        assert "top_p" not in build_payload(request)

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
        )
        assert response.text == "hi"
        assert response.usage is not None
        assert response.usage.reasoning_tokens == 2
        assert response.cost is not None
        assert response.cost.total_cost == 0.001
        assert response.model == "m-actual"
        assert response.finish_reason == "stop"

    def test_responses_body_reports_response_id(self) -> None:
        body = {
            "id": "resp-1",
            "status": "completed",
            "output_text": "SDK convenience must not win",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        response = parse_responses_body(
            body, config=openai_responses_config(model="m")
        )
        assert response.response_id == "resp-1"
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 3
        assert response.text == "hi"
        assert response.diagnostics is not None
        assert response.diagnostics.output_text_len == 2

    def test_responses_output_parts_fallback(self) -> None:
        body = {
            "id": "resp-2",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "part one "},
                        {"type": "output_text", "text": "part two"},
                    ],
                }
            ],
        }
        response = parse_responses_body(
            body, config=openai_responses_config(model="m")
        )
        assert response.text == "part one part two"

    @pytest.mark.parametrize(
        ("body", "expected_code"),
        [
            (
                {
                    "id": "resp-private-refusal",
                    "status": "completed",
                    "prompt": "PRIVATE_PROMPT",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "refusal",
                                    "refusal": "PRIVATE_REFUSAL",
                                }
                            ],
                        }
                    ],
                },
                "response_refusal",
            ),
            (
                {
                    "id": "resp-private-output",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "arguments": "PRIVATE_OUTPUT",
                        }
                    ],
                },
                "response_no_text",
            ),
            (
                {
                    "id": "resp-private-malformed",
                    "prompt": "PRIVATE_PROMPT",
                    "output": {"text": "PRIVATE_OUTPUT"},
                },
                "response_parse_error",
            ),
        ],
        ids=["refusal", "tool_only", "malformed"],
    )
    def test_responses_failure_metadata_is_content_free(
        self, body: dict, expected_code: str
    ) -> None:
        with pytest.raises(PermanentProviderError) as exc_info:
            parse_responses_body(
                body, config=openai_responses_config(model="m")
            )

        failure = exc_info.value.failure
        assert failure.code == expected_code
        serialized_failure = json.dumps(failure.model_dump())
        for private_value in (
            "PRIVATE_PROMPT",
            "PRIVATE_OUTPUT",
            "PRIVATE_REFUSAL",
            body["id"],
        ):
            assert private_value not in serialized_failure
        assert "response_preview" not in failure.metadata
        assert len(failure.metadata["diagnostics"]["response_id_hash"]) == 16

    def test_failed_response_failure_is_entirely_content_free(self) -> None:
        body = {
            "id": "PRIVATE_RESPONSE_ID",
            "status": "failed",
            "prompt": "PRIVATE_PROMPT",
            "error": {
                "code": "safe_provider_code",
                "message": "PRIVATE_ERROR_MESSAGE",
            },
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "PRIVATE_OUTPUT"},
                        {"type": "refusal", "refusal": "PRIVATE_REFUSAL"},
                    ],
                },
                {
                    "type": "function_call",
                    "name": "PRIVATE_TOOL_NAME",
                    "arguments": "PRIVATE_TOOL_ARGUMENTS",
                },
            ],
        }

        with pytest.raises(PermanentProviderError) as exc_info:
            parse_responses_body(
                body, config=openai_responses_config(model="m")
            )

        failure = exc_info.value.failure
        assert failure.code == "response_failed"
        assert failure.message == "provider response failed"
        assert failure.metadata["diagnostics"]["provider_error_code"] == (
            "safe_provider_code"
        )
        serialized_failure = json.dumps(failure.model_dump())
        for private_value in (
            "PRIVATE_RESPONSE_ID",
            "PRIVATE_PROMPT",
            "PRIVATE_ERROR_MESSAGE",
            "PRIVATE_OUTPUT",
            "PRIVATE_REFUSAL",
            "PRIVATE_TOOL_NAME",
            "PRIVATE_TOOL_ARGUMENTS",
        ):
            assert private_value not in serialized_failure

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


class TestScriptedProvider:
    def test_scripted_text_response(self) -> None:
        provider = ScriptedProvider(
            [ScriptedOutcome(text="scripted", finish_reason="stop")]
        )
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
            idempotency_key="attempt-1",
        )
        response = provider.complete(request)
        assert response.text == "scripted"
        assert provider.payloads[0]["model"] == "m"
        assert provider.requests[0].idempotency_key == "attempt-1"

    def test_scripted_failure_raises_carrier(self) -> None:
        failure = failure_record(
            failure_class=FailureClass.RATE_LIMITED,
            message="scripted 429",
        )
        provider = ScriptedProvider([ScriptedOutcome(failure=failure)])
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
        )
        with pytest.raises(RateLimitedProviderError) as exc_info:
            provider.complete(request)
        assert exc_info.value.failure.retryable is True

    def test_last_outcome_repeats(self) -> None:
        provider = ScriptedProvider([ScriptedOutcome(text="only")])
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
        )
        assert provider.complete(request).text == "only"
        assert provider.complete(request).text == "only"
        assert len(provider.requests) == 2

    def test_response_carries_conformance_warnings(self) -> None:
        from dr_providers import TokenUsage

        provider = ScriptedProvider(
            [
                ScriptedOutcome(
                    text="over budget",
                    usage=TokenUsage(completion_tokens=99),
                )
            ]
        )
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
            token_limit=10,
        )
        response = provider.complete(request)
        codes = [warning.code for warning in response.warnings]
        assert "token_limit_exceeded" in codes

    def test_scripted_warnings_appended_once(self) -> None:
        from dr_providers import LlmWarning, TokenUsage

        scripted = LlmWarning(code="scripted", message="script")
        provider = ScriptedProvider(
            [
                ScriptedOutcome(
                    text="over budget",
                    usage=TokenUsage(completion_tokens=99),
                    warnings=(scripted,),
                )
            ]
        )
        request = LlmRequest(
            provider_config=openai_chat_config(model="m"),
            messages=MESSAGES,
            token_limit=10,
        )
        response = provider.complete(request)
        codes = [warning.code for warning in response.warnings]
        assert codes.count("scripted") == 1
        assert codes.count("token_limit_exceeded") == 1
