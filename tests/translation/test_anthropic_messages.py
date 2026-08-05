"""Anthropic Messages response-translation tests."""

from dr_providers import (
    GenerationControls,
    ProviderTransportResponse,
    anthropic_messages_config,
    parse_anthropic_messages_body,
)


def test_anthropic_body_parses_parts() -> None:
    body = {
        "id": "msg-1",
        "model": "claude",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }

    response = parse_anthropic_messages_body(
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
