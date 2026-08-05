"""Provider request wire-translation tests."""

from __future__ import annotations

import json

from dr_providers import (
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderBodyExtensions,
    ProviderCallRequest,
    ReasoningEffort,
    Transcript,
    anthropic_messages_config,
    build_payload,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    protocol_path,
)

MESSAGES = (
    PromptMessage(role=MessageRole.SYSTEM, content="be brief"),
    PromptMessage(role=MessageRole.USER, content="write add"),
)


def request_for(config, messages=MESSAGES) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config, transcript=Transcript(messages=messages)
    )


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

    def test_reasoning_output_config_for_anthropic(self) -> None:
        request = request_for(
            anthropic_messages_config(
                model="claude",
                controls=GenerationControls(
                    token_limit=64, reasoning=ReasoningEffort.MEDIUM
                ),
            )
        )
        payload = build_payload(request)
        # Anthropic Messages takes {"output_config": {"effort": ...}}, not a
        # top-level {"reasoning": ...}.
        assert payload["output_config"] == {"effort": "medium"}
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

    def test_nested_extra_body_payload_is_json_serializable(self) -> None:
        # Frozen extension values must be thawed on the way into the wire
        # payload, or httpx's json= encoding would reject the request.
        request = request_for(
            openai_chat_config(
                model="m",
                extensions=ProviderBodyExtensions(
                    extra_body={"provider": {"order": ["a", "b"]}}
                ),
            )
        )
        payload = build_payload(request)
        assert json.loads(json.dumps(payload))["provider"] == {
            "order": ["a", "b"]
        }

    def test_protocol_paths(self) -> None:
        assert protocol_path(openai_chat_config(model="m")) == (
            "/chat/completions"
        )
        assert protocol_path(openai_responses_config(model="m")) == (
            "/responses"
        )
        assert protocol_path(
            anthropic_messages_config(
                model="m", controls=GenerationControls(token_limit=64)
            )
        ) == ("/messages")
