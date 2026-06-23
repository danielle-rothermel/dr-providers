import json
from unittest.mock import Mock

import pytest

from dr_providers import (
    LlmRequest,
    Message,
    MessageRole,
    ProviderName,
    ProviderSemanticError,
    ProviderTransportError,
)
from dr_providers.query.response import llm_response_from_http


@pytest.fixture
def llm_request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENROUTER,
        model="test-model",
        messages=[Message(role=MessageRole.USER, content="hi")],
    )


def _mock_response(
    *,
    status_code: int = 200,
    json_body: object | None = None,
    text: str = "",
) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.text = text
    if json_body is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = json_body
    return response


def _parse(
    llm_request: LlmRequest,
    response: Mock,
    *,
    latency_ms: int,
) -> None:
    llm_response_from_http(
        response,
        llm_request,
        latency_ms=latency_ms,
    )


def test_llm_response_from_http_success(llm_request: LlmRequest) -> None:
    response = _mock_response(
        json_body={
            "choices": [
                {
                    "message": {"content": "hello"},
                    "finish_reason": "stop",
                }
            ]
        },
        text='{"choices":[{"message":{"content":"hello"}}]}',
    )
    result = llm_response_from_http(response, llm_request, latency_ms=42)
    assert result.text == "hello"
    assert result.finish_reason == "stop"
    assert result.latency_ms == 42
    assert result.provider == str(ProviderName.OPENROUTER)
    assert result.model == "test-model"


def test_llm_response_from_http_transient_error(
    llm_request: LlmRequest,
) -> None:
    response = _mock_response(status_code=503, json_body={}, text="error")
    with pytest.raises(ProviderTransportError, match="transient error"):
        _parse(llm_request, response, latency_ms=1)


def test_llm_response_from_http_semantic_error(
    llm_request: LlmRequest,
) -> None:
    response = _mock_response(status_code=400, json_body={}, text="bad")
    with pytest.raises(ProviderSemanticError, match="rejected request"):
        _parse(llm_request, response, latency_ms=1)


def test_llm_response_from_http_invalid_json(
    llm_request: LlmRequest,
) -> None:
    response = _mock_response(status_code=200, text="not json")
    response.json.side_effect = json.JSONDecodeError("bad", "doc", 0)
    with pytest.raises(ProviderTransportError, match="invalid JSON"):
        _parse(llm_request, response, latency_ms=1)


def test_llm_response_from_http_missing_choices(
    llm_request: LlmRequest,
) -> None:
    response = _mock_response(
        status_code=200,
        json_body={"choices": []},
        text='{"choices":[]}',
    )
    with pytest.raises(ProviderSemanticError, match="missing choices"):
        _parse(llm_request, response, latency_ms=1)
