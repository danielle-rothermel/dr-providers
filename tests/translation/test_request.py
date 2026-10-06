from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from dr_providers import (
    ControlConstraints,
    ControlValidationError,
    GenerationControls,
    MessageRole,
    ModelRoute,
    PromptMessage,
    PromptRendering,
    Protocol,
    ProviderBodyExtensions,
    ProviderCallDefinition,
    ProviderCallRequest,
    ProviderKind,
    ReasoningEffort,
    RequestControl,
    TokenLimitParameter,
    Transcript,
    Verbosity,
    anthropic_messages_config,
    build_payload,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    protocol_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dr_providers import ProviderCallConfig

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

    def test_unsupported_control_refuses_construction(self) -> None:
        definition = ProviderCallDefinition(
            definition_id="test.chat",
            route=ModelRoute(
                provider=ProviderKind.OPENAI,
                protocol=Protocol.CHAT_COMPLETIONS,
                model="m",
            ),
            constraints=ControlConstraints(
                supported_controls=frozenset({RequestControl.TOKEN_LIMIT}),
                token_limit_parameter=(
                    TokenLimitParameter.MAX_COMPLETION_TOKENS
                ),
            ),
        )
        with pytest.raises(ControlValidationError) as exc_info:
            definition.materialize(
                controls=GenerationControls(temperature=0.5)
            )
        assert exc_info.value.failure.code == "unsupported_control"

    def test_seed_transported(self) -> None:
        request = request_for(
            openai_chat_config(
                model="m",
                controls=GenerationControls(seed=7),
            )
        )
        assert build_payload(request)["seed"] == 7

    @pytest.mark.parametrize(
        "config_factory", [openai_chat_config, openrouter_chat_config]
    )
    def test_verbosity_transported(
        self, config_factory: Callable[..., ProviderCallConfig]
    ) -> None:
        request = request_for(
            config_factory(
                model="m",
                controls=GenerationControls(verbosity=Verbosity.LOW),
            )
        )
        assert build_payload(request)["verbosity"] == "low"

    def test_unset_verbosity_absent_from_payload(self) -> None:
        request = request_for(openai_chat_config(model="m"))
        assert "verbosity" not in build_payload(request)

    def test_extra_body_merged_into_payload(self) -> None:
        request = request_for(
            openai_chat_config(
                model="m",
                extensions=ProviderBodyExtensions(extra_body={"user": "eval"}),
            )
        )
        assert build_payload(request)["user"] == "eval"

    def test_nested_extra_body_payload_is_json_serializable(self) -> None:
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


@pytest.mark.parametrize(
    ("config_factory", "expected_body"),
    [
        (
            openai_chat_config,
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "systemuserassistant"}
                ],
                "max_completion_tokens": 64,
                "reasoning_effort": "low",
                "temperature": 0.2,
                "top_p": 0.9,
                "user": "eval",
            },
        ),
        (
            openai_responses_config,
            {
                "model": "m",
                "input": [{"role": "user", "content": "systemuserassistant"}],
                "max_output_tokens": 64,
                "reasoning": {"effort": "low"},
                "temperature": 0.2,
                "top_p": 0.9,
                "user": "eval",
            },
        ),
        (
            anthropic_messages_config,
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "systemuserassistant"}
                ],
                "max_tokens": 64,
                "output_config": {"effort": "low"},
                "temperature": 0.2,
                "top_p": 0.9,
                "user": "eval",
            },
        ),
    ],
    ids=["chat-completions", "responses", "anthropic-messages"],
)
def test_flat_text_body_preserves_controls_and_extensions(
    config_factory: Callable[..., ProviderCallConfig],
    expected_body: dict[str, object],
) -> None:
    request = request_for(
        config_factory(
            model="m",
            prompt_rendering=PromptRendering.FLAT_TEXT,
            controls=GenerationControls(
                token_limit=64,
                reasoning=ReasoningEffort.LOW,
                temperature=0.2,
                top_p=0.9,
            ),
            extensions=ProviderBodyExtensions(extra_body={"user": "eval"}),
        ),
        messages=(
            PromptMessage(role=MessageRole.SYSTEM, content="system"),
            PromptMessage(role=MessageRole.USER, content="user"),
            PromptMessage(role=MessageRole.ASSISTANT, content="assistant"),
        ),
    )
    payload = build_payload(request)

    assert payload == expected_body
    assert "instructions" not in payload
    assert "system" not in payload


def test_flat_text_single_message_preserves_content_bytes() -> None:
    content = " \tPréfixe\r\n🙂\n\nSuffixe  "
    request = request_for(
        openai_chat_config(
            model="m", prompt_rendering=PromptRendering.FLAT_TEXT
        ),
        messages=(PromptMessage(role=MessageRole.SYSTEM, content=content),),
    )

    payload = build_payload(request)

    assert payload == {
        "model": "m",
        "messages": [{"role": "user", "content": content}],
    }
    assert payload["messages"][0]["content"].encode("utf-8") == content.encode(
        "utf-8"
    )


def test_flat_text_inserts_no_separator_between_any_message_roles() -> None:
    request = request_for(
        openai_chat_config(
            model="m", prompt_rendering=PromptRendering.FLAT_TEXT
        ),
        messages=(
            PromptMessage(role=MessageRole.SYSTEM, content="a"),
            PromptMessage(role=MessageRole.USER, content="b"),
            PromptMessage(role=MessageRole.TOOL, content="c"),
            PromptMessage(role=MessageRole.ASSISTANT, content="d"),
        ),
    )

    assert build_payload(request) == {
        "model": "m",
        "messages": [{"role": "user", "content": "abcd"}],
    }


@pytest.mark.parametrize(
    ("config_factory", "expected_body"),
    [
        (
            openai_chat_config,
            {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "assistant"},
                ],
                "max_completion_tokens": 64,
            },
        ),
        (
            openai_responses_config,
            {
                "model": "m",
                "instructions": "system",
                "input": [
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "assistant"},
                ],
                "max_output_tokens": 64,
            },
        ),
        (
            anthropic_messages_config,
            {
                "model": "m",
                "system": "system",
                "messages": [
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "assistant"},
                ],
                "max_tokens": 64,
            },
        ),
    ],
    ids=["chat-completions", "responses", "anthropic-messages"],
)
def test_role_messages_preserves_assistant_ending_transcript(
    config_factory: Callable[..., ProviderCallConfig],
    expected_body: dict[str, object],
) -> None:
    request = request_for(
        config_factory(model="m", controls=GenerationControls(token_limit=64)),
        messages=(
            PromptMessage(role=MessageRole.SYSTEM, content="system"),
            PromptMessage(role=MessageRole.USER, content="user"),
            PromptMessage(role=MessageRole.ASSISTANT, content="assistant"),
        ),
    )

    assert build_payload(request) == expected_body


@pytest.mark.parametrize(
    ("config_factory", "message_key", "expected_controls"),
    [
        (openai_chat_config, "messages", {"max_completion_tokens": 64}),
        (openai_responses_config, "input", {"max_output_tokens": 64}),
        (anthropic_messages_config, "messages", {"max_tokens": 64}),
    ],
    ids=["chat-completions", "responses", "anthropic-messages"],
)
@pytest.mark.parametrize(
    ("rendering", "expected_messages"),
    [
        (PromptRendering.ROLE_MESSAGES, []),
        (PromptRendering.FLAT_TEXT, [{"role": "user", "content": ""}]),
    ],
    ids=["role-messages", "flat-text"],
)
def test_empty_transcript_rendering_is_explicit(
    config_factory: Callable[..., ProviderCallConfig],
    message_key: str,
    expected_controls: dict[str, int],
    rendering: PromptRendering,
    expected_messages: list[dict[str, str]],
) -> None:
    request = request_for(
        config_factory(
            model="m",
            controls=GenerationControls(token_limit=64),
            prompt_rendering=rendering,
        ),
        messages=(),
    )

    payload = build_payload(request)

    assert payload == {
        "model": "m",
        message_key: expected_messages,
        **expected_controls,
    }
    assert "instructions" not in payload
    assert "system" not in payload
    assert request.transcript.messages == ()
    assert request.identity_payload()["transcript"] == []
