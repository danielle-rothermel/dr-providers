from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_providers import ModelRoute, Protocol, ProviderKind


@pytest.mark.parametrize(
    ("provider", "protocol"),
    [
        (ProviderKind.OPENROUTER, Protocol.CHAT_COMPLETIONS),
        (ProviderKind.OPENAI, Protocol.CHAT_COMPLETIONS),
        (ProviderKind.OPENAI, Protocol.RESPONSES),
        (ProviderKind.GEMINI, Protocol.CHAT_COMPLETIONS),
        (ProviderKind.ANTHROPIC, Protocol.ANTHROPIC_MESSAGES),
    ],
)
def test_supported_provider_protocol_pairs_are_accepted(
    provider: ProviderKind,
    protocol: Protocol,
) -> None:
    route = ModelRoute(provider=provider, protocol=protocol, model="m")

    assert route.provider is provider
    assert route.protocol is protocol


@pytest.mark.parametrize(
    ("provider", "protocol"),
    [
        (ProviderKind.OPENROUTER, Protocol.RESPONSES),
        (ProviderKind.OPENROUTER, Protocol.ANTHROPIC_MESSAGES),
        (ProviderKind.OPENAI, Protocol.ANTHROPIC_MESSAGES),
        (ProviderKind.GEMINI, Protocol.RESPONSES),
        (ProviderKind.GEMINI, Protocol.ANTHROPIC_MESSAGES),
        (ProviderKind.ANTHROPIC, Protocol.CHAT_COMPLETIONS),
        (ProviderKind.ANTHROPIC, Protocol.RESPONSES),
    ],
)
def test_unsupported_provider_protocol_pairs_are_rejected(
    provider: ProviderKind,
    protocol: Protocol,
) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "unsupported provider/protocol combination: "
            f"{provider.value}\\+{protocol.value}"
        ),
    ):
        ModelRoute(provider=provider, protocol=protocol, model="m")
