from unittest.mock import Mock

import httpx
import pytest

from dr_providers import LlmRequest, Message, MessageRole, ProviderName
from dr_providers.config import ReasoningSpec
from dr_providers.names import EffortLevel
from dr_providers.query.transport import ApiProvider
from dr_providers.query.transport_config import ProviderConfig


@pytest.fixture
def config() -> ProviderConfig:
    return ProviderConfig(
        name="openrouter",
        base_url="https://example.com/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        api_key="test-key",
        chat_path="/chat/completions",
    )


def _request_with_reasoning(
    reasoning: ReasoningSpec | None,
) -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENROUTER,
        model="test-model",
        messages=[Message(role=MessageRole.USER, content="hi")],
        reasoning=reasoning,
    )


def _success_response() -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {"content": "hello"},
                    "finish_reason": "stop",
                }
            ]
        },
    )


def _posted_payload(
    *,
    config: ProviderConfig,
    reasoning: ReasoningSpec | None,
) -> dict[str, object]:
    client = Mock(spec=httpx.Client)
    client.post.return_value = _success_response()
    provider = ApiProvider(config=config, client=client)

    provider.generate(_request_with_reasoning(reasoning))

    return client.post.call_args.kwargs["json"]


def test_json_payload_omits_reasoning_when_unspecified(
    config: ProviderConfig,
) -> None:
    payload = _posted_payload(config=config, reasoning=None)

    assert "reasoning" not in payload


def test_json_payload_includes_reasoning_enabled_flag(
    config: ProviderConfig,
) -> None:
    payload = _posted_payload(
        config=config,
        reasoning=ReasoningSpec(enabled=True),
    )

    assert payload["reasoning"] == {"enabled": True}


def test_json_payload_includes_reasoning_effort(
    config: ProviderConfig,
) -> None:
    payload = _posted_payload(
        config=config,
        reasoning=ReasoningSpec(effort=EffortLevel.HIGH),
    )

    assert payload["reasoning"] == {"effort": EffortLevel.HIGH}


def test_json_payload_includes_default_reasoning_spec(
    config: ProviderConfig,
) -> None:
    payload = _posted_payload(config=config, reasoning=ReasoningSpec())

    assert payload["reasoning"] == {"reasoning": True}
