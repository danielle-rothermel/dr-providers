"""Response parser dispatch tests."""

from collections.abc import Mapping
from typing import Any

import pytest

from dr_providers import (
    GenerationControls,
    ProviderCallConfig,
    ProviderTransportResponse,
    anthropic_messages_config,
    openai_chat_config,
    openai_responses_config,
)
from dr_providers.translation import response as response_translation


@pytest.mark.parametrize(
    ("parser_name", "config"),
    [
        ("parse_chat_completions_body", openai_chat_config(model="m")),
        ("parse_responses_body", openai_responses_config(model="m")),
        (
            "parse_anthropic_messages_body",
            anthropic_messages_config(
                model="m", controls=GenerationControls(token_limit=1)
            ),
        ),
    ],
    ids=["chat_completions", "responses", "anthropic_messages"],
)
def test_parse_response_dispatches_by_protocol(
    monkeypatch: pytest.MonkeyPatch,
    parser_name: str,
    config: ProviderCallConfig,
) -> None:
    body = {"sentinel": "body"}
    expected = ProviderTransportResponse(text=parser_name)
    calls: list[tuple[Mapping[str, Any], ProviderCallConfig]] = []

    def parser(
        received_body: Mapping[str, Any], *, config: ProviderCallConfig
    ) -> ProviderTransportResponse:
        calls.append((received_body, config))
        return expected

    monkeypatch.setattr(response_translation, parser_name, parser)

    assert response_translation.parse_response(body, config=config) is expected
    assert calls == [(body, config)]
