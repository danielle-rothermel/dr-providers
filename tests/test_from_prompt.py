from unittest.mock import Mock

from dr_providers import LlmResponse, ProviderName
from dr_providers.query.from_prompt import query_from_prompt


def test_query_from_prompt_builds_system_and_user_messages() -> None:
    provider = Mock()
    provider.generate.return_value = LlmResponse(
        provider=str(ProviderName.OPENROUTER),
        model="test-model",
        latency_ms=10,
        text="ok",
    )

    response = query_from_prompt(
        provider,
        ProviderName.OPENROUTER,
        model="test-model",
        prompt="user message",
        system="system message",
    )

    provider.generate.assert_called_once()
    request = provider.generate.call_args.args[0]
    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[0].content == "system message"
    assert request.messages[1].role == "user"
    assert request.messages[1].content == "user message"
    assert response.text == "ok"
