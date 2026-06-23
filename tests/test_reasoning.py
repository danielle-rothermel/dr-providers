import pytest

from dr_providers import LlmRequest, Message, MessageRole, ProviderName
from dr_providers.config import ReasoningSpec
from dr_providers.names import EffortLevel
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


def test_json_payload_omits_reasoning_when_unspecified(
    config: ProviderConfig,
) -> None:
    payload = _request_with_reasoning(None).prepare(config).json_payload()

    assert "reasoning" not in payload


def test_json_payload_includes_reasoning_enabled_flag(
    config: ProviderConfig,
) -> None:
    payload = (
        _request_with_reasoning(ReasoningSpec(enabled=True))
        .prepare(config)
        .json_payload()
    )

    assert payload["reasoning"] == {"enabled": True}


def test_json_payload_includes_reasoning_effort(
    config: ProviderConfig,
) -> None:
    payload = (
        _request_with_reasoning(ReasoningSpec(effort=EffortLevel.HIGH))
        .prepare(config)
        .json_payload()
    )

    assert payload["reasoning"] == {"effort": EffortLevel.HIGH}


def test_json_payload_includes_default_reasoning_spec(
    config: ProviderConfig,
) -> None:
    payload = (
        _request_with_reasoning(ReasoningSpec()).prepare(config).json_payload()
    )

    assert payload["reasoning"] == {"reasoning": True}
