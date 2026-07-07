"""Contract tests for the raw-httpx transport and conformance checks."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from dr_providers.kernel import (
    FailureClass,
    LlmRequest,
    MessageRole,
    PermanentProviderError,
    PromptMessage,
    ProviderFailureError,
    RateLimitedProviderError,
    TransientProviderError,
    openai_chat_config,
)
from dr_providers.kernel.conformance import (
    MODEL_SUBSTITUTION_CODE,
    REASONING_NOT_OBSERVED_CODE,
    TOKEN_LIMIT_EXCEEDED_CODE,
)
from dr_providers.kernel.transport import (
    HttpProvider,
    TransportPolicy,
)

MESSAGES = (PromptMessage(role=MessageRole.USER, content="write add"),)
CHAT_BODY_OK: dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "m",
    "choices": [
        {
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


def request(**overrides: Any) -> LlmRequest:
    fields: dict[str, Any] = {
        "provider_config": openai_chat_config(model="m"),
        "messages": MESSAGES,
    }
    fields.update(overrides)
    return LlmRequest(**fields)


def mock_provider(
    handler: Any,
    *,
    policy: TransportPolicy | None = None,
) -> tuple[HttpProvider, list[float]]:
    delays: list[float] = []
    provider = HttpProvider(
        policy=policy,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        api_key="test-key",
        sleep=delays.append,
        rng=lambda: 0.5,
    )
    return provider, delays


class TestHttpProvider:
    def test_success_posts_payload_and_parses(self) -> None:
        seen: dict[str, Any] = {}

        def handler(http_request: httpx.Request) -> httpx.Response:
            seen["url"] = str(http_request.url)
            seen["auth"] = http_request.headers.get("authorization")
            seen["idempotency"] = http_request.headers.get("idempotency-key")
            return httpx.Response(200, json=CHAT_BODY_OK)

        provider, _ = mock_provider(handler)
        response = provider.complete(request(idempotency_key="attempt-9"))
        assert response.text == "hello"
        assert seen["url"] == "https://api.openai.com/v1/chat/completions"
        assert seen["auth"] == "Bearer test-key"
        assert seen["idempotency"] == "attempt-9"
        assert response.payload["model"] == "m"

    @pytest.mark.parametrize(
        ("status", "error_type"),
        [
            (429, RateLimitedProviderError),
            (500, TransientProviderError),
            (400, PermanentProviderError),
        ],
    )
    def test_http_status_classification(
        self,
        status: int,
        error_type: type[ProviderFailureError],
    ) -> None:
        provider, _ = mock_provider(
            lambda _req: httpx.Response(status, text="nope")
        )
        with pytest.raises(error_type) as exc_info:
            provider.complete(request())
        assert exc_info.value.failure.code == f"http_status_{status}"

    def test_transport_error_is_transient(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("boom")

        provider, _ = mock_provider(handler)
        with pytest.raises(TransientProviderError) as exc_info:
            provider.complete(request())
        assert exc_info.value.failure.failure_class is (FailureClass.TRANSIENT)

    def test_invalid_json_is_permanent(self) -> None:
        provider, _ = mock_provider(
            lambda _req: httpx.Response(200, text="<html>")
        )
        with pytest.raises(PermanentProviderError) as exc_info:
            provider.complete(request())
        assert exc_info.value.failure.code == "invalid_response_json"

    def test_missing_api_key_is_permanent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = HttpProvider(
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
                )
            )
        )
        with pytest.raises(PermanentProviderError) as exc_info:
            provider.complete(request())
        assert exc_info.value.failure.code == "missing_api_key"

    def test_no_retry_by_default(self) -> None:
        calls: list[int] = []

        def handler(_req: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(500)

        provider, delays = mock_provider(handler)
        with pytest.raises(TransientProviderError):
            provider.complete(request())
        assert len(calls) == 1
        assert delays == []

    def test_opt_in_retry_is_bounded_with_backoff(self) -> None:
        calls: list[int] = []

        def handler(_req: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(500)

        provider, delays = mock_provider(
            handler,
            policy=TransportPolicy(max_retries=2, backoff_base_seconds=1.0),
        )
        with pytest.raises(TransientProviderError):
            provider.complete(request())
        assert len(calls) == 3
        # rng=0.5 → jitter multiplier 0.75; exponential base 1.0, 2.0.
        assert delays == [0.75, 1.5]

    def test_retry_recovers_on_success(self) -> None:
        responses = [
            httpx.Response(500),
            httpx.Response(200, json=CHAT_BODY_OK),
        ]

        def handler(_req: httpx.Request) -> httpx.Response:
            return responses.pop(0)

        provider, delays = mock_provider(
            handler, policy=TransportPolicy(max_retries=1)
        )
        assert provider.complete(request()).text == "hello"
        assert len(delays) == 1

    def test_permanent_failure_never_retries(self) -> None:
        calls: list[int] = []

        def handler(_req: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(401)

        provider, _ = mock_provider(
            handler, policy=TransportPolicy(max_retries=3)
        )
        with pytest.raises(PermanentProviderError):
            provider.complete(request())
        assert len(calls) == 1


class TestConformance:
    def test_reasoning_not_observed_warning(self) -> None:
        provider, _ = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        response = provider.complete(request(reasoning={"effort": "low"}))
        codes = [w.code for w in response.warnings]
        assert REASONING_NOT_OBSERVED_CODE in codes

    def test_token_limit_exceeded_warning(self) -> None:
        body = dict(CHAT_BODY_OK)
        body["usage"] = {"prompt_tokens": 1, "completion_tokens": 99}
        provider, _ = mock_provider(
            lambda _req: httpx.Response(200, json=body)
        )
        response = provider.complete(request(token_limit=10))
        codes = [w.code for w in response.warnings]
        assert TOKEN_LIMIT_EXCEEDED_CODE in codes

    def test_model_substitution_warning(self) -> None:
        body = dict(CHAT_BODY_OK)
        body["model"] = "m-other"
        provider, _ = mock_provider(
            lambda _req: httpx.Response(200, json=body)
        )
        response = provider.complete(request())
        codes = [w.code for w in response.warnings]
        assert MODEL_SUBSTITUTION_CODE in codes

    def test_clean_response_has_no_warnings(self) -> None:
        provider, _ = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        assert provider.complete(request()).warnings == ()
