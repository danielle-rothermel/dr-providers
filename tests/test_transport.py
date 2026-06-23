from unittest.mock import Mock

import httpx
import pytest

from dr_providers import (
    LlmRequest,
    Message,
    MessageRole,
    ProviderName,
    ProviderTransportError,
)
from dr_providers.query import transport as transport_module
from dr_providers.query.transport import ApiProvider
from dr_providers.query.transport_config import ProviderConfig


@pytest.fixture
def config() -> ProviderConfig:
    return ProviderConfig(
        name="test",
        base_url="https://example.com/api/v1",
        api_key_env="TEST_API_KEY",
        api_key="test-key",
        chat_path="/chat/completions",
    )


@pytest.fixture
def llm_request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENROUTER,
        model="test-model",
        messages=[Message(role=MessageRole.USER, content="hi")],
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


def test_generate_retries_transport_errors_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    config: ProviderConfig,
    llm_request: LlmRequest,
) -> None:
    client = Mock(spec=httpx.Client)
    client.post.side_effect = [
        httpx.TransportError("first"),
        httpx.TimeoutException("second"),
        _success_response(),
    ]
    sleep = Mock()
    monkeypatch.setattr(transport_module.time, "sleep", sleep)
    monkeypatch.setattr(
        transport_module,
        "_retry_delay_seconds",
        Mock(return_value=0.01),
    )

    provider = ApiProvider(config=config, client=client)
    response = provider.generate(llm_request)

    assert response.text == "hello"
    assert client.post.call_count == 3
    assert sleep.call_count == 2


def test_generate_stops_retrying_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    config: ProviderConfig,
    llm_request: LlmRequest,
) -> None:
    final_error = httpx.TransportError("third")
    client = Mock(spec=httpx.Client)
    client.post.side_effect = [
        httpx.TransportError("first"),
        httpx.TransportError("second"),
        final_error,
    ]
    monkeypatch.setattr(transport_module.time, "sleep", Mock())
    monkeypatch.setattr(
        transport_module,
        "_retry_delay_seconds",
        Mock(return_value=0.01),
    )

    provider = ApiProvider(config=config, client=client)
    with pytest.raises(
        ProviderTransportError, match="HTTP request failed"
    ) as exc:
        provider.generate(llm_request)

    assert exc.value.__cause__ is final_error
    assert client.post.call_count == 3


def test_generate_does_not_retry_http_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    config: ProviderConfig,
    llm_request: LlmRequest,
) -> None:
    client = Mock(spec=httpx.Client)
    client.post.return_value = httpx.Response(
        status_code=503,
        json={},
        text="unavailable",
    )
    sleep = Mock()
    monkeypatch.setattr(transport_module.time, "sleep", sleep)

    provider = ApiProvider(config=config, client=client)
    with pytest.raises(ProviderTransportError, match="transient error"):
        provider.generate(llm_request)

    client.post.assert_called_once()
    sleep.assert_not_called()
