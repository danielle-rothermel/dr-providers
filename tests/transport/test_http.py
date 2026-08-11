from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from _policy import make_transport_policy

from dr_providers import (
    ApiKeyEnv,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
    ProviderKind,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RecoverabilityClass,
    Transcript,
    anthropic_messages_config,
    openai_chat_config,
)
from dr_providers.lifecycle import (
    AcceptAllSemanticResponseClassifier,
    ProviderInvocationOutcome,
    classify_provider_invocation,
)
from dr_providers.transport import http as transport_http
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

OPENAI_POLICY = make_transport_policy(provider_kind=ProviderKind.OPENAI)
ANTHROPIC_POLICY = make_transport_policy(provider_kind=ProviderKind.ANTHROPIC)


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
        _client_factory=lambda **_kwargs: httpx.Client(
            transport=httpx.MockTransport(handler)
        ),
        api_key="test-key",
    )


class TestHttpProvider:
    @pytest.mark.parametrize(
        ("provider_request", "policy_kind", "secret_env", "endpoint"),
        [
            pytest.param(
                openai_request(),
                ProviderKind.ANTHROPIC,
                "ANTHROPIC_MISMATCH_SECRET",
                "https://anthropic-mismatch.example/v1",
                id="openai-request-anthropic-policy",
            ),
            pytest.param(
                request_for(
                    anthropic_messages_config(
                        model="claude",
                        controls=GenerationControls(token_limit=8),
                    )
                ),
                ProviderKind.OPENAI,
                "OPENAI_MISMATCH_SECRET",
                "https://openai-mismatch.example/v1",
                id="anthropic-request-openai-policy",
            ),
        ],
    )
    def test_route_policy_mismatch_precedes_request_and_credential_work(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_request: ProviderCallRequest,
        policy_kind: ProviderKind,
        secret_env: str,
        endpoint: str,
    ) -> None:
        payload_builds: list[ProviderCallRequest] = []
        credential_resolutions: list[ProviderCallConfig] = []
        wire_requests: list[httpx.Request] = []
        monkeypatch.setenv(secret_env, f"secret-for-{secret_env}")

        def record_payload_build(
            attempted_request: ProviderCallRequest,
        ) -> dict[str, Any]:
            payload_builds.append(attempted_request)
            return {}

        def record_credential_resolution(
            _provider: HttpProvider,
            config: ProviderCallConfig,
        ) -> dict[str, str]:
            credential_resolutions.append(config)
            return {}

        def handler(http_request: httpx.Request) -> httpx.Response:
            wire_requests.append(http_request)
            return httpx.Response(200, json=CHAT_BODY_OK)

        monkeypatch.setattr(
            transport_http,
            "build_payload",
            record_payload_build,
        )
        monkeypatch.setattr(
            HttpProvider,
            "_headers",
            record_credential_resolution,
        )
        policy = make_transport_policy(
            provider_kind=policy_kind,
            api_key_env=secret_env,
            base_url=endpoint,
        )
        provider = HttpProvider(
            policy=policy,
            _client_factory=lambda **_kwargs: httpx.Client(
                transport=httpx.MockTransport(handler)
            ),
        )

        with (
            provider,
            pytest.raises(
                ValueError,
                match="request route provider does not match transport policy",
            ),
        ):
            provider.invoke(provider_request)

        assert payload_builds == []
        assert credential_resolutions == []
        assert wire_requests == []

    def test_success_posts_payload_and_parses(self) -> None:
        seen: dict[str, Any] = {}

        def handler(http_request: httpx.Request) -> httpx.Response:
            seen["url"] = str(http_request.url)
            seen["auth"] = http_request.headers.get("authorization")
            return httpx.Response(200, json=CHAT_BODY_OK)

        provider = mock_provider(handler)
        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.text == "hello"
        assert seen["url"] == "https://api.openai.com/v1/chat/completions"
        assert seen["auth"] == "Bearer test-key"
        assert outcome.model == "m"
        assert evidence.http_request is not None

    @pytest.mark.parametrize("status", [200, 202, 299])
    def test_every_2xx_status_dispatches_provider_body(
        self, status: int
    ) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(status, json=CHAT_BODY_OK)
        )

        outcome = provider.invoke(openai_request()).outcome

        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.text == "hello"

    @pytest.mark.parametrize("status", [199, 400])
    def test_every_non_2xx_status_retains_http_failure_evidence(
        self, status: int
    ) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(status, json=CHAT_BODY_OK)
        )

        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == f"http_status_{status}"
        assert outcome.status_code == status
        assert outcome.response_body == CHAT_BODY_OK
        assert evidence.http_request is not None

    def test_payload_too_large_status_is_resource_exhaustion(self) -> None:
        """A 413 is the server detecting the local request-size condition."""
        provider = mock_provider(
            lambda _req: httpx.Response(413, json={"error": "too large"})
        )

        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "http_status_413"
        assert (
            outcome.recoverability is RecoverabilityClass.RESOURCE_EXHAUSTION
        )
        assert (
            classify_provider_invocation(
                evidence, AcceptAllSemanticResponseClassifier()
            )
            is ProviderInvocationOutcome.RESOURCE_EXHAUSTION
        )

    @pytest.mark.parametrize("status", [300, 302, 399])
    def test_redirect_status_is_distinguishable_from_a_rejection(
        self, status: int
    ) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(status, json=CHAT_BODY_OK)
        )

        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == f"http_redirect_{status}"
        assert outcome.recoverability is RecoverabilityClass.PERMANENT
        assert outcome.status_code == status
        assert outcome.response_body == CHAT_BODY_OK
        assert evidence.http_request is not None
        assert classify_provider_invocation(
            evidence, AcceptAllSemanticResponseClassifier()
        ) is (
            ProviderInvocationOutcome.PERMANENT_PROVIDER_OR_TRANSPORT_FAILURE
        )

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
            policy=OPENAI_POLICY,
            api_key="test-key",
            _client_factory=lambda **_kwargs: client,
        )

        outcome = provider.invoke(openai_request()).outcome

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "http_redirect_302"
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

        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.recoverability is RecoverabilityClass.PERMANENT
        assert outcome.code == "response_parse_error"
        assert outcome.response_body == body
        assert outcome.status_code == 200
        assert evidence.http_request is not None

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
        outcome = provider.invoke(request_for(config)).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.text == "hello"
        assert seen["url"] == "https://api.anthropic.com/v1/messages"
        assert seen["x_api_key"] == "test-key"
        assert seen["version"]
        assert seen["auth"] is None

    def test_http_status_failure_is_typed_no_throw(self) -> None:
        provider = mock_provider(lambda _req: httpx.Response(500, text="nope"))
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.recoverability is RecoverabilityClass.TRANSIENT
        assert outcome.code == "http_status_500"
        assert outcome.status_code == 500
        assert outcome.traceback is None

    def test_transport_error_is_transient_no_throw(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        provider = mock_provider(handler)
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.recoverability is RecoverabilityClass.TRANSIENT
        assert outcome.code == "transport_error"
        assert outcome.message == "provider transport error"
        assert outcome.metadata["exception_type"] == "ConnectError"
        assert outcome.traceback is not None
        assert "ConnectError" in outcome.traceback
        assert "boom" not in outcome.message

    @pytest.mark.parametrize(
        ("error", "expected_code", "expected_recoverability"),
        [
            (
                httpx.ConnectError("down"),
                "transport_error",
                RecoverabilityClass.TRANSIENT,
            ),
            (
                httpx.ConnectTimeout("slow connect"),
                "timeout",
                RecoverabilityClass.TRANSIENT,
            ),
            (
                httpx.ReadTimeout("idle stall"),
                "stalled_response",
                RecoverabilityClass.TRANSIENT,
            ),
            (
                httpx.LocalProtocolError("bad framing"),
                "transport_protocol_error",
                RecoverabilityClass.PERMANENT,
            ),
            (
                httpx.RemoteProtocolError("server disconnected"),
                "transport_remote_protocol_error",
                RecoverabilityClass.TRANSIENT,
            ),
            (
                httpx.PoolTimeout("pool starved"),
                "pool_timeout",
                RecoverabilityClass.TRANSIENT,
            ),
        ],
    )
    def test_httpx_error_classification(
        self,
        error: httpx.HTTPError,
        expected_code: str,
        expected_recoverability: RecoverabilityClass,
    ) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise error

        provider = mock_provider(handler)
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == expected_code
        assert outcome.recoverability is expected_recoverability
        assert outcome.message == (
            "provider transport timeout"
            if isinstance(error, httpx.TimeoutException)
            else "provider transport error"
        )
        assert outcome.metadata["exception_type"] == type(error).__name__
        assert outcome.traceback is not None
        assert type(error).__name__ in outcome.traceback
        assert str(error) not in outcome.message

    def test_http_status_failure_has_no_traceback(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(500, text="server error detail")
        )
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.traceback is None

    def test_invalid_json_is_permanent_no_throw(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, text="<html>")
        )
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "invalid_response_json"

    def test_missing_api_key_is_permanent_no_throw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = HttpProvider(
            policy=OPENAI_POLICY,
            _client_factory=lambda **_kwargs: httpx.Client(
                transport=httpx.MockTransport(
                    lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
                )
            ),
        )
        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "missing_api_key"
        assert evidence.http_request is None

    def test_missing_base_url_is_permanent_no_throw(self) -> None:
        policy = ProviderTransportPolicy(
            provider_kind=ProviderKind.OPENAI,
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=None,
            timeout_seconds=120.0,
            connect_timeout_seconds=30.0,
            idle_timeout_seconds=90.0,
            max_connections=10,
            max_keepalive_connections=5,
            max_request_bytes=1024 * 1024,
            max_response_bytes=8 * 1024 * 1024,
        )
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK),
            policy=policy,
        )
        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "missing_base_url"
        assert evidence.http_request is None
        assert (
            classify_provider_invocation(
                evidence, AcceptAllSemanticResponseClassifier()
            )
            is ProviderInvocationOutcome.MISSING_TRANSPORT_CONFIG
        )

    @pytest.mark.parametrize(
        "base_url",
        ["not-a-url-at-all", "ftp://example.test", "https://", "://nope"],
        ids=["no_scheme", "wrong_scheme", "no_host", "empty_scheme"],
    )
    def test_malformed_base_url_never_reaches_the_wire(
        self, base_url: str
    ) -> None:
        calls: list[int] = []

        def handler(_req: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json=CHAT_BODY_OK)

        policy = ProviderTransportPolicy(
            provider_kind=ProviderKind.OPENAI,
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=base_url,
            timeout_seconds=120.0,
            connect_timeout_seconds=30.0,
            idle_timeout_seconds=90.0,
            max_connections=10,
            max_keepalive_connections=5,
            max_request_bytes=1024 * 1024,
            max_response_bytes=8 * 1024 * 1024,
        )
        provider = mock_provider(handler, policy=policy)

        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "invalid_base_url"
        assert outcome.recoverability is RecoverabilityClass.PERMANENT
        assert evidence.http_request is None
        assert calls == []
        assert (
            classify_provider_invocation(
                evidence, AcceptAllSemanticResponseClassifier()
            )
            is ProviderInvocationOutcome.NEVER_SENT
        )

    def test_a_url_problem_at_the_wire_boundary_stays_typed_evidence(
        self,
    ) -> None:
        """No URL problem escapes invoke() as an unclassified exception."""

        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.InvalidURL("rejected at dispatch")

        provider = mock_provider(handler)

        evidence = provider.invoke(openai_request())
        outcome = evidence.outcome

        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == "invalid_base_url"
        assert outcome.metadata["exception_type"] == "InvalidURL"
        assert outcome.traceback is not None
        assert (
            classify_provider_invocation(
                evidence, AcceptAllSemanticResponseClassifier()
            )
            is ProviderInvocationOutcome.NEVER_SENT
        )


class TestInvocationWireBoundary:
    def test_one_invocation_makes_one_wire_attempt(self) -> None:
        calls: list[int] = []

        def handler(_req: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(500)

        provider = mock_provider(handler)
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportFailure)
        assert len(calls) == 1


class TestHttpProviderLifecycle:
    def test_one_bounded_owned_client_is_reused_and_closed(self) -> None:
        created: list[httpx.Client] = []
        factory_kwargs: list[dict[str, Any]] = []
        calls: list[int] = []

        def tracking_client(**kwargs: Any) -> httpx.Client:
            factory_kwargs.append(kwargs)
            client = httpx.Client(
                transport=httpx.MockTransport(
                    lambda _req: (
                        calls.append(1)
                        or httpx.Response(200, json=CHAT_BODY_OK)
                    )
                )
            )
            created.append(client)
            return client

        with HttpProvider(
            policy=OPENAI_POLICY,
            api_key="k",
            _client_factory=tracking_client,
        ) as provider:
            first = provider.invoke(openai_request()).outcome
            second = provider.invoke(openai_request()).outcome

        assert isinstance(first, ProviderTransportResponse)
        assert isinstance(second, ProviderTransportResponse)
        assert calls == [1, 1]
        assert len(created) == 1
        assert created[0].is_closed
        limits = factory_kwargs[0]["limits"]
        assert limits.max_connections == OPENAI_POLICY.max_connections
        assert (
            limits.max_keepalive_connections
            == OPENAI_POLICY.max_keepalive_connections
        )
        assert factory_kwargs[0]["follow_redirects"] is False

    def test_invoke_after_close_is_unexpected_misuse(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        provider.close()

        with pytest.raises(RuntimeError, match="closing or closed"):
            provider.invoke(openai_request())
