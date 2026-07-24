"""Contract tests for the raw-httpx transport, outcomes, and evidence.

Covers the no-throw Provider Transport Outcome, native retry-zero,
protocol-specific headers (Bearer vs Anthropic x-api-key), conformance
warnings, and stable Provider Invocation Evidence (complete raw
request/response, credential redaction, no silent truncation).
"""

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
    ReasoningEffort,
    Transcript,
    anthropic_messages_config,
    openai_chat_config,
)
from dr_providers.conformance import (
    MODEL_SUBSTITUTION_CODE,
    REASONING_NOT_OBSERVED_CODE,
    TOKEN_LIMIT_EXCEEDED_CODE,
)
from dr_providers.transport import HttpProvider

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
        # A non-timeout httpx.HTTPError (here ConnectError) exercises the
        # generic transport_error branch, distinct from the timeout branch.
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
        # Unit-level classification: a ConnectError is a generic transport
        # error; a ConnectTimeout is a plain timeout; a ReadTimeout is an idle
        # stall (no bytes within the idle window) -> stalled_response.
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
        assert len(calls) == 3  # one initial + two native retries

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


class TestInvocationEvidence:
    def test_evidence_binds_request_policy_and_success_body(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        request = openai_request(token_limit=64)
        evidence = provider.invoke(request)

        # identity payloads are deeply frozen (nested lists become tuples of
        # frozen maps), so compare via the JSON-serialized (thawed) form.
        stable = evidence.stable_payload()
        assert stable["request_identity"] == request.identity_payload()
        assert stable["policy_identity"] == OPENAI_POLICY.identity_payload()
        assert isinstance(evidence.outcome, ProviderTransportResponse)
        assert evidence.response is not None
        # complete least-processed raw success body, no truncation.
        assert evidence.response.raw_body == CHAT_BODY_OK
        assert evidence.raw_request.body["model"] == "m"

    def test_evidence_retains_complete_failure_body(self) -> None:
        long_message = "x" * 5000
        big_body = {"error": {"message": long_message}}
        provider = mock_provider(
            lambda _req: httpx.Response(400, json=big_body)
        )
        evidence = provider.invoke(openai_request())
        failure = evidence.failure
        assert failure is not None
        # no silent preview truncation of the failure evidence: the
        # complete raw body is retained verbatim.
        assert failure.raw_response_body == big_body
        assert long_message in json.dumps(failure.raw_response_body)

    def test_evidence_never_persists_authorization_header(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        evidence = provider.invoke(openai_request())
        headers = evidence.raw_request.headers
        assert headers.get("Authorization") == "<redacted>"
        serialized = evidence.to_stable_dict()
        assert "test-key" not in str(serialized)
        assert "Bearer test-key" not in str(serialized)

    def test_anthropic_evidence_redacts_x_api_key(self) -> None:
        config = anthropic_messages_config(
            model="claude", controls=GenerationControls(token_limit=8)
        )
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=ANTHROPIC_BODY_OK),
            policy=ANTHROPIC_POLICY,
        )
        evidence = provider.invoke(request_for(config))
        assert "test-key" not in str(evidence.to_stable_dict())

    def test_evidence_serialization_is_stable(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        request = openai_request(token_limit=64)
        first = provider.invoke(request).to_stable_dict()
        second = provider.invoke(request).to_stable_dict()
        assert first == second


class TestConformance:
    def test_reasoning_not_observed_warning(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        outcome = provider.complete(
            openai_request(reasoning=ReasoningEffort.LOW)
        )
        assert isinstance(outcome, ProviderTransportResponse)
        codes = [w.code for w in outcome.warnings]
        assert REASONING_NOT_OBSERVED_CODE in codes

    def test_token_limit_exceeded_warning(self) -> None:
        body = dict(CHAT_BODY_OK)
        body["usage"] = {"prompt_tokens": 1, "completion_tokens": 99}
        provider = mock_provider(lambda _req: httpx.Response(200, json=body))
        outcome = provider.complete(openai_request(token_limit=10))
        assert isinstance(outcome, ProviderTransportResponse)
        codes = [w.code for w in outcome.warnings]
        assert TOKEN_LIMIT_EXCEEDED_CODE in codes

    def test_model_substitution_warning(self) -> None:
        body = dict(CHAT_BODY_OK)
        body["model"] = "m-other"
        provider = mock_provider(lambda _req: httpx.Response(200, json=body))
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportResponse)
        codes = [w.code for w in outcome.warnings]
        assert MODEL_SUBSTITUTION_CODE in codes

    def test_clean_response_has_no_warnings(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        outcome = provider.complete(openai_request())
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.warnings == ()


class TestHttpProviderLifecycle:
    def test_owned_per_call_client_closed_after_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Owned wire calls build a per-call client; each must be closed as
        # its call completes, leaving no shared client for close() to hold.
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

    def test_close_is_idempotent(self) -> None:
        provider = HttpProvider(policy=OPENAI_POLICY, api_key="k")
        provider.close()
        provider.close()


class TestOutcomeGuards:
    def test_is_response_and_is_failure_narrow(self) -> None:
        from dr_providers import is_failure, is_response

        response = ProviderTransportResponse(text="hi")
        failure = ProviderTransportFailure(
            failure_class=FailureClass.PERMANENT,
            code="x",
            message="m",
            retryable=False,
        )
        assert is_response(response) is True
        assert is_failure(response) is False
        assert is_failure(failure) is True
        assert is_response(failure) is False
