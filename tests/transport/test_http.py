from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from dr_providers import (
    ApiKeyEnv,
    FailureClass,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderBaseUrl,
    ProviderCallConfig,
    ProviderCallRequest,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    Transcript,
    anthropic_messages_config,
    openai_chat_config,
)
from dr_providers.transport.http import HttpProvider

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
ANTHROPIC_BODY_OK: dict[str, Any] = {
    "id": "msg-1",
    "model": "claude",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "hello"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

OPENAI_POLICY = ProviderTransportPolicy(
    api_key_env=str(ApiKeyEnv.OPENAI),
    base_url=str(ProviderBaseUrl.OPENAI),
)
ANTHROPIC_POLICY = ProviderTransportPolicy(
    api_key_env=str(ApiKeyEnv.ANTHROPIC),
    base_url=str(ProviderBaseUrl.ANTHROPIC),
)


def request_for(
    config: ProviderCallConfig, messages=MESSAGES
) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config, transcript=Transcript(messages=messages)
    )


def openai_request(**control_overrides: Any) -> ProviderCallRequest:
    controls = GenerationControls(**control_overrides)
    return request_for(openai_chat_config(model="m", controls=controls))


def mock_provider(
    handler: Any,
    *,
    policy: ProviderTransportPolicy = OPENAI_POLICY,
) -> HttpProvider:
    return HttpProvider(
        policy=policy,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        api_key="test-key",
    )


class TestHttpProvider:
    def test_success_posts_payload_and_parses(self) -> None:
        seen: dict[str, Any] = {}

        def handler(http_request: httpx.Request) -> httpx.Response:
            seen["url"] = str(http_request.url)
            seen["auth"] = http_request.headers.get("authorization")
            return httpx.Response(200, json=CHAT_BODY_OK)

        provider = mock_provider(handler)
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.text == "hello"
        assert seen["url"] == "https://api.openai.com/v1/chat/completions"
        assert seen["auth"] == "Bearer test-key"
        assert outcome.model == "m"

    @pytest.mark.parametrize("status", [200, 202, 299])
    def test_every_2xx_status_dispatches_provider_body(
        self, status: int
    ) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(status, json=CHAT_BODY_OK)
        )

        outcome = provider.complete(openai_request())

        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.text == "hello"

    @pytest.mark.parametrize("status", [199, 300, 399, 400])
    def test_every_non_2xx_status_retains_http_failure_evidence(
        self, status: int
    ) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(status, json=CHAT_BODY_OK)
        )

        outcome = provider.complete(openai_request())

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == f"http_status_{status}"
        assert outcome.status_code == status
        assert outcome.response_body == CHAT_BODY_OK
        assert outcome.request_body

    def test_redirect_is_not_followed_even_when_client_default_follows(
        self,
    ) -> None:
        calls: list[str] = []

        def handler(http_request: httpx.Request) -> httpx.Response:
            calls.append(str(http_request.url))
            if len(calls) == 1:
                return httpx.Response(
                    302,
                    headers={"location": "https://example.test/redirected"},
                    json=CHAT_BODY_OK,
                )
            return httpx.Response(200, json=CHAT_BODY_OK)

        client = httpx.Client(
            transport=httpx.MockTransport(handler), follow_redirects=True
        )
        provider = HttpProvider(
            policy=OPENAI_POLICY, client=client, api_key="test-key"
        )

        outcome = provider.complete(openai_request())

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "http_status_302"
        assert calls == ["https://api.openai.com/v1/chat/completions"]

    @pytest.mark.parametrize(
        "body",
        [None, False, 0, "provider body", ["provider", "body"]],
        ids=["null", "boolean", "number", "string", "list"],
    )
    def test_valid_json_non_mapping_is_typed_parse_failure(
        self, body: Any
    ) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, text=json.dumps(body))
        )

        outcome = provider.complete(openai_request())

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.failure_class is FailureClass.PERMANENT
        assert outcome.code == "response_parse_error"
        assert outcome.retryable is False
        assert outcome.response_body == body
        assert outcome.status_code == 200
        assert outcome.request_body

    def test_anthropic_uses_x_api_key_and_version_headers(self) -> None:
        seen: dict[str, Any] = {}

        def handler(http_request: httpx.Request) -> httpx.Response:
            seen["url"] = str(http_request.url)
            seen["x_api_key"] = http_request.headers.get("x-api-key")
            seen["version"] = http_request.headers.get("anthropic-version")
            seen["auth"] = http_request.headers.get("authorization")
            return httpx.Response(200, json=ANTHROPIC_BODY_OK)

        config = anthropic_messages_config(
            model="claude", controls=GenerationControls(token_limit=16)
        )
        provider = mock_provider(handler, policy=ANTHROPIC_POLICY)
        outcome = provider.complete(request_for(config))
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.text == "hello"
        assert seen["url"] == "https://api.anthropic.com/v1/messages"
        assert seen["x_api_key"] == "test-key"
        assert seen["version"]
        assert seen["auth"] is None

    @pytest.mark.parametrize(
        ("status", "failure_class"),
        [
            (429, FailureClass.RATE_LIMITED),
            (500, FailureClass.TRANSIENT),
            (400, FailureClass.PERMANENT),
        ],
    )
    def test_http_status_classification_no_throw(
        self, status: int, failure_class: FailureClass
    ) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(status, text="nope")
        )
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.failure_class is failure_class
        assert outcome.code == f"http_status_{status}"
        assert outcome.status_code == status

    def test_transport_error_is_transient_no_throw(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        provider = mock_provider(handler)
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.failure_class is FailureClass.TRANSIENT
        assert outcome.code == "transport_error"
        assert outcome.retryable is True

    @pytest.mark.parametrize(
        ("error", "expected_code"),
        [
            (httpx.ConnectError("down"), "transport_error"),
            (httpx.ConnectTimeout("slow connect"), "timeout"),
            (httpx.ReadTimeout("idle stall"), "stalled_response"),
        ],
    )
    def test_httpx_error_classification(
        self, error: httpx.HTTPError, expected_code: str
    ) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise error

        provider = mock_provider(handler)
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == expected_code
        assert outcome.failure_class is FailureClass.TRANSIENT
        assert outcome.retryable is True

    def test_invalid_json_is_permanent_no_throw(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, text="<html>")
        )
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "invalid_response_json"

    def test_missing_api_key_is_permanent_no_throw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = HttpProvider(
            policy=OPENAI_POLICY,
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
                )
            ),
        )
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "missing_api_key"

    def test_missing_base_url_is_permanent_no_throw(self) -> None:
        policy = ProviderTransportPolicy(
            api_key_env=str(ApiKeyEnv.OPENAI), base_url=None
        )
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK),
            policy=policy,
        )
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "missing_base_url"


class TestNativeRetry:
    def test_no_retry_by_default(self) -> None:
        calls: list[int] = []

        def handler(_req: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(500)

        provider = mock_provider(handler)
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportFailure)
        assert len(calls) == 1

    def test_native_retry_count_repeats_retryable(self) -> None:
        calls: list[int] = []

        def handler(_req: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(500)

        policy = ProviderTransportPolicy(
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=str(ProviderBaseUrl.OPENAI),
            native_retry_count=2,
        )
        provider = mock_provider(handler, policy=policy)
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportFailure)
        assert len(calls) == 3

    def test_native_retry_recovers_on_success(self) -> None:
        responses = [
            httpx.Response(500),
            httpx.Response(200, json=CHAT_BODY_OK),
        ]

        def handler(_req: httpx.Request) -> httpx.Response:
            return responses.pop(0)

        policy = ProviderTransportPolicy(
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=str(ProviderBaseUrl.OPENAI),
            native_retry_count=1,
        )
        provider = mock_provider(handler, policy=policy)
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.text == "hello"

    def test_permanent_failure_never_retries(self) -> None:
        calls: list[int] = []

        def handler(_req: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(401)

        policy = ProviderTransportPolicy(
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=str(ProviderBaseUrl.OPENAI),
            native_retry_count=3,
        )
        provider = mock_provider(handler, policy=policy)
        provider.complete(openai_request())
        assert len(calls) == 1


class TestHttpProviderLifecycle:
    def test_owned_per_call_client_closed_after_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_client = httpx.Client
        created: list[httpx.Client] = []

        def tracking_client(*_args: object, **_kwargs: object) -> httpx.Client:
            client = real_client(
                transport=httpx.MockTransport(
                    lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
                )
            )
            created.append(client)
            return client

        monkeypatch.setattr(httpx, "Client", tracking_client)
        with HttpProvider(policy=OPENAI_POLICY, api_key="k") as provider:
            outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportResponse)
        assert created
        assert all(client.is_closed for client in created)

    def test_injected_client_left_open(self) -> None:
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
            )
        )
        with HttpProvider(
            policy=OPENAI_POLICY, client=client, api_key="k"
        ) as provider:
            outcome = provider.complete(openai_request())
            assert isinstance(outcome, ProviderTransportResponse)
        assert not client.is_closed
        client.close()
