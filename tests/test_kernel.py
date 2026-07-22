"""Contract tests for the kernel: config, payload, parse, failures.

Covers Provider Call Config presets, ``build_payload`` for every
protocol, the least-processed parsers (no-throw, content-free), the
failure taxonomy, and ScriptedProvider.
"""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from dr_providers import (
    ControlConstraints,
    FailureClass,
    GenerationControls,
    MessageRole,
    PromptMessage,
    Protocol,
    ProviderBodyExtensions,
    ProviderCallDefinition,
    ProviderCallRequest,
    ProviderKind,
    ProviderTransportFailure,
    ProviderTransportResponse,
    ReasoningEffort,
    RequestControl,
    ScriptedOutcome,
    ScriptedProvider,
    TokenLimitParameter,
    TokenUsage,
    Transcript,
    UnsupportedControlError,
    anthropic_messages_config,
    build_payload,
    classify_status_code,
    failure_record,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    parse_chat_completions_body,
    parse_response,
    parse_responses_body,
    protocol_path,
    sanitize_kwargs,
    token_usage_from_body,
)
from dr_providers.route import ModelRoute

MESSAGES = (
    PromptMessage(role=MessageRole.SYSTEM, content="be brief"),
    PromptMessage(role=MessageRole.USER, content="write add"),
)


def request_for(config, messages=MESSAGES) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config, transcript=Transcript(messages=messages)
    )


class TestConfigPresets:
    def test_openrouter_chat(self) -> None:
        config = openrouter_chat_config(model="m")
        assert config.route.provider is ProviderKind.OPENROUTER
        assert config.route.protocol is Protocol.CHAT_COMPLETIONS
        env = config.definition.constraints.token_limit_parameter
        assert env is TokenLimitParameter.MAX_COMPLETION_TOKENS
        assert config.quota_identity.model_dump() == {
            "provider": "openrouter",
            "protocol": "chat_completions",
            "model": "m",
        }

    def test_openai_responses(self) -> None:
        config = openai_responses_config(model="m")
        assert config.route.protocol is Protocol.RESPONSES
        param = config.definition.constraints.token_limit_parameter
        assert param.value == "max_output_tokens"

    def test_gemini_compat_preset(self) -> None:
        config = gemini_chat_config(model="gemini-2.5-flash")
        assert config.route.provider is ProviderKind.GEMINI
        assert config.route.protocol is Protocol.CHAT_COMPLETIONS

    def test_anthropic_messages_preset(self) -> None:
        config = anthropic_messages_config(model="claude")
        assert config.route.provider is ProviderKind.ANTHROPIC
        assert config.route.protocol is Protocol.ANTHROPIC_MESSAGES
        param = config.definition.constraints.token_limit_parameter
        assert param is TokenLimitParameter.MAX_TOKENS


class TestBuildPayload:
    def test_chat_payload_shape(self) -> None:
        request = request_for(
            openai_chat_config(
                model="m",
                controls=GenerationControls(temperature=0.2, token_limit=64),
            )
        )
        assert build_payload(request) == {
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "write add"},
            ],
            "temperature": 0.2,
            "max_completion_tokens": 64,
        }

    def test_responses_payload_lifts_system_to_instructions(self) -> None:
        request = request_for(
            openai_responses_config(
                model="m", controls=GenerationControls(token_limit=64)
            )
        )
        payload = build_payload(request)
        assert payload["instructions"] == "be brief"
        assert payload["input"] == [{"role": "user", "content": "write add"}]
        assert payload["max_output_tokens"] == 64
        assert "messages" not in payload

    def test_anthropic_payload_lifts_system_and_sets_max_tokens(self) -> None:
        request = request_for(
            anthropic_messages_config(
                model="claude", controls=GenerationControls(token_limit=64)
            )
        )
        payload = build_payload(request)
        assert payload["system"] == "be brief"
        assert payload["messages"] == [
            {"role": "user", "content": "write add"}
        ]
        assert payload["max_tokens"] == 64
        assert "input" not in payload

    def test_reasoning_object_for_openai_responses(self) -> None:
        request = request_for(
            openai_responses_config(
                model="m",
                controls=GenerationControls(reasoning=ReasoningEffort.LOW),
            )
        )
        payload = build_payload(request)
        assert payload["reasoning"] == {"effort": "low"}
        assert "reasoning_effort" not in payload

    def test_reasoning_object_for_openrouter(self) -> None:
        request = request_for(
            openrouter_chat_config(
                model="m",
                controls=GenerationControls(reasoning=ReasoningEffort.HIGH),
            )
        )
        payload = build_payload(request)
        assert payload["reasoning"] == {"effort": "high"}

    def test_reasoning_effort_field_for_openai_chat(self) -> None:
        request = request_for(
            openai_chat_config(
                model="m",
                controls=GenerationControls(reasoning=ReasoningEffort.MEDIUM),
            )
        )
        payload = build_payload(request)
        assert payload["reasoning_effort"] == "medium"
        assert "reasoning" not in payload

    def test_reasoning_effort_field_for_gemini(self) -> None:
        request = request_for(
            gemini_chat_config(
                model="m",
                controls=GenerationControls(reasoning=ReasoningEffort.MINIMAL),
            )
        )
        assert build_payload(request)["reasoning_effort"] == "minimal"

    def test_top_p_transported(self) -> None:
        request = request_for(
            openai_chat_config(
                model="m", controls=GenerationControls(top_p=0.9)
            )
        )
        assert build_payload(request)["top_p"] == 0.9

    def test_extra_body_merged_into_payload(self) -> None:
        request = request_for(
            openai_chat_config(
                model="m",
                extensions=ProviderBodyExtensions(extra_body={"seed": 7}),
            )
        )
        assert build_payload(request)["seed"] == 7

    def test_protocol_paths(self) -> None:
        assert protocol_path(openai_chat_config(model="m")) == (
            "/chat/completions"
        )
        assert protocol_path(openai_responses_config(model="m")) == (
            "/responses"
        )
        assert protocol_path(anthropic_messages_config(model="m")) == (
            "/messages"
        )


class TestDefinitionValidation:
    def _constrained_definition(
        self,
        supported: frozenset[RequestControl],
        *,
        allow_drop: bool = False,
        required: frozenset[RequestControl] = frozenset(),
    ) -> ProviderCallDefinition:
        return ProviderCallDefinition(
            definition_id="test.chat",
            route=ModelRoute(
                provider=ProviderKind.OPENAI,
                protocol=Protocol.CHAT_COMPLETIONS,
                model="m",
            ),
            constraints=ControlConstraints(
                supported_controls=supported,
                token_limit_parameter=(
                    TokenLimitParameter.MAX_COMPLETION_TOKENS
                ),
                allow_unsupported_control_drop=allow_drop,
            ),
            required_controls=required,
        )

    def test_unsupported_control_rejected_at_materialize(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT})
        )
        with pytest.raises(UnsupportedControlError) as exc_info:
            definition.materialize(
                controls=GenerationControls(temperature=0.5)
            )
        assert exc_info.value.failure.metadata["control"] == "temperature"

    def test_unsupported_control_drop_opt_in(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT}), allow_drop=True
        )
        config = definition.materialize(
            controls=GenerationControls(temperature=0.5)
        )
        request = request_for(config)
        assert "temperature" not in build_payload(request)

    def test_required_control_must_be_assigned(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT}),
            required=frozenset({RequestControl.TOKEN_LIMIT}),
        )
        with pytest.raises(UnsupportedControlError) as exc_info:
            definition.materialize(controls=GenerationControls())
        assert exc_info.value.failure.code == "missing_required_control"

    def test_undeclared_extension_rejected(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT})
        )
        with pytest.raises(UnsupportedControlError) as exc_info:
            definition.materialize(
                extensions=ProviderBodyExtensions(extra_body={"nope": 1})
            )
        assert exc_info.value.failure.code == "undeclared_extension"


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

    def test_anthropic_body_parses_parts(self) -> None:
        body = {
            "id": "msg-1",
            "model": "claude",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
        response = parse_response(
            body, config=anthropic_messages_config(model="claude")
        )
        assert isinstance(response, ProviderTransportResponse)
        assert response.text == "hello"
        assert response.finish_reason == "end_turn"
        assert response.usage is not None
        assert response.usage.total_tokens == 7

    def test_responses_body_reports_response_id(self) -> None:
        body = {
            "id": "resp-1",
            "status": "completed",
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
        assert isinstance(response, ProviderTransportResponse)
        assert response.response_id == "resp-1"
        assert response.finish_reason == "stop"
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
        assert isinstance(response, ProviderTransportResponse)
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
        failure = parse_responses_body(
            body, config=openai_responses_config(model="m")
        )
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.code == expected_code
        # metadata must be content-free; the raw body is retained
        # separately in raw_response_body (least-processed evidence).
        serialized_metadata = json.dumps(failure.metadata)
        for private_value in (
            "PRIVATE_PROMPT",
            "PRIVATE_OUTPUT",
            "PRIVATE_REFUSAL",
            body["id"],
        ):
            assert private_value not in serialized_metadata
        assert len(failure.metadata["diagnostics"]["response_id_hash"]) == 16
        assert failure.raw_response_body == body

    def test_parse_dispatches_by_protocol(self) -> None:
        chat_body = {"choices": [{"message": {"content": "x"}}]}
        response = parse_response(
            chat_body, config=openrouter_chat_config(model="m")
        )
        assert isinstance(response, ProviderTransportResponse)
        assert response.text == "x"

    @pytest.mark.parametrize(
        "body",
        [{}, {"choices": []}, {"choices": [{"message": {}}]}],
        ids=["missing", "empty", "no_text"],
    )
    def test_chat_parse_failures_are_typed(self, body: dict) -> None:
        outcome = parse_chat_completions_body(
            body, config=openai_chat_config(model="m")
        )
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.failure_class is FailureClass.PERMANENT


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
