import pytest

from dr_providers import (
    LlmRequest,
    Message,
    MessageRole,
    ProviderName,
    ReasoningSpec,
    SamplingControls,
)
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


@pytest.fixture
def llm_request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENROUTER,
        model="test-model",
        messages=[Message(role=MessageRole.USER, content="hi")],
        max_tokens=100,
        sampling=SamplingControls(temperature=0.5, top_p=0.9),
        reasoning=ReasoningSpec(effort=EffortLevel.LOW),
    )


def test_prepare_populates_transport_fields(
    llm_request: LlmRequest, config: ProviderConfig
) -> None:
    prepared = llm_request.prepare(config)

    assert prepared.api_key == "test-key"
    assert prepared.base_url == config.base_url
    assert prepared.idempotency_key
    assert prepared.extra_body == {"reasoning": {"effort": EffortLevel.LOW}}
    assert llm_request.api_key is None


def test_idempotency_key_from_metadata(config: ProviderConfig) -> None:
    llm_request = LlmRequest(
        provider=ProviderName.OPENROUTER,
        model="test-model",
        messages=[Message(role=MessageRole.USER, content="hi")],
        metadata={"idempotency_key": "fixed-key"},
    )
    prepared = llm_request.prepare(config)
    assert prepared.idempotency_key == "fixed-key"


def test_unprepared_endpoint_raises(llm_request: LlmRequest) -> None:
    with pytest.raises(ValueError, match="not prepared"):
        llm_request.endpoint()


def test_prepared_endpoint_and_headers(
    llm_request: LlmRequest, config: ProviderConfig
) -> None:
    prepared = llm_request.prepare(config)
    assert prepared.endpoint() == "https://example.com/api/v1/chat/completions"
    headers = prepared.headers()
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["Idempotency-Key"]


def test_json_payload_includes_sampling_and_tokens(
    llm_request: LlmRequest, config: ProviderConfig
) -> None:
    prepared = llm_request.prepare(config)
    payload = prepared.json_payload()

    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 100
    assert payload["max_completion_tokens"] == 100
    assert "reasoning" in payload


def test_json_payload_extra_body_conflict(config: ProviderConfig) -> None:
    llm_request = LlmRequest(
        provider=ProviderName.OPENROUTER,
        model="test-model",
        messages=[Message(role=MessageRole.USER, content="hi")],
    )
    prepared = llm_request.prepare(config).model_copy(
        update={"extra_body": {"model": "override"}}
    )
    with pytest.raises(ValueError, match="extra_body conflicts"):
        prepared.json_payload()
