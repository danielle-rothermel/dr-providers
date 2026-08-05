"""Provider response wire-translation tests."""

from __future__ import annotations

import json

import pytest

from dr_providers import (
    FailureClass,
    GenerationControls,
    ProviderTransportFailure,
    ProviderTransportResponse,
    anthropic_messages_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    parse_chat_completions_body,
    parse_response,
    parse_responses_body,
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
            body,
            config=anthropic_messages_config(
                model="claude", controls=GenerationControls(token_limit=64)
            ),
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
