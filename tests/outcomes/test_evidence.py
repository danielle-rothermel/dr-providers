from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from dr_providers import (
    PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION,
    ApiKeyEnv,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderBaseUrl,
    ProviderCallConfig,
    ProviderCallRequest,
    ProviderHttpRequestEvidence,
    ProviderInvocationEvidence,
    ProviderKind,
    ProviderRetryAfterHint,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RecoverabilityClass,
    Transcript,
    anthropic_messages_config,
    openai_chat_config,
    sanitize_kwargs,
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
    provider_kind=ProviderKind.OPENAI,
    api_key_env=str(ApiKeyEnv.OPENAI),
    base_url=str(ProviderBaseUrl.OPENAI),
    timeout_seconds=120.0,
    connect_timeout_seconds=30.0,
    idle_timeout_seconds=90.0,
    max_connections=10,
    max_keepalive_connections=5,
    max_request_bytes=1024 * 1024,
    max_response_bytes=8 * 1024 * 1024,
)
ANTHROPIC_POLICY = ProviderTransportPolicy(
    provider_kind=ProviderKind.ANTHROPIC,
    api_key_env=str(ApiKeyEnv.ANTHROPIC),
    base_url=str(ProviderBaseUrl.ANTHROPIC),
    timeout_seconds=120.0,
    connect_timeout_seconds=30.0,
    idle_timeout_seconds=90.0,
    max_connections=10,
    max_keepalive_connections=5,
    max_request_bytes=1024 * 1024,
    max_response_bytes=8 * 1024 * 1024,
)

HTTP_REQUEST = ProviderHttpRequestEvidence(
    url="https://example.test/v1",
    headers={"Content-Type": "application/json"},
    body={"model": "m"},
    body_bytes=13,
)
SUCCESS = ProviderTransportResponse(text="hi", response_body={"id": "resp-1"})
FAILURE = ProviderTransportFailure(
    recoverability=RecoverabilityClass.PERMANENT,
    code="invalid_request",
    message="bad request",
    response_body={"error": "bad"},
    status_code=400,
    metadata={"provider": "openai"},
)


def evidence_for(
    *,
    response: ProviderTransportResponse | None = None,
    failure: ProviderTransportFailure | None = None,
) -> ProviderInvocationEvidence:
    return ProviderInvocationEvidence(
        request_identity_hash="1" * 64,
        policy_identity={
            "provider_kind": "openai",
            "policy": "policy-1",
        },
        http_request=HTTP_REQUEST,
        response=response,
        failure=failure,
    )


def expected_document(
    *,
    response: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "dr_providers.provider_invocation_evidence",
        "schema_version": 6,
        "payload": {
            "request_identity_hash": "1" * 64,
            "policy_identity": {
                "provider_kind": "openai",
                "policy": "policy-1",
            },
            "max_request_bytes": None,
            "max_response_bytes": None,
            "http_request": {
                "method": "POST",
                "url": "https://example.test/v1",
                "headers": {"Content-Type": "application/json"},
                "body": {"model": "m"},
                "body_bytes": 13,
            },
            "response_bytes": None,
            "retry_after": None,
            "response": response,
            "failure": failure,
        },
    }


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


class TestInvocationEvidence:
    @pytest.mark.parametrize(
        "policy_identity",
        [{"policy": "v3"}, {"provider_kind": "unsupported"}],
        ids=("missing-provider-kind", "unsupported-provider-kind"),
    )
    def test_policy_identity_requires_supported_provider_kind(
        self,
        policy_identity: dict[str, str],
    ) -> None:
        with pytest.raises(
            ValidationError,
            match="policy_identity requires a supported provider_kind",
        ):
            ProviderInvocationEvidence(
                request_identity_hash="1" * 64,
                policy_identity=policy_identity,
                response=ProviderTransportResponse(text="ok"),
            )

    def test_schema_version_exists_only_on_identity_document(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        evidence = provider.invoke(openai_request())

        assert PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION == 6
        assert "schema_version" not in ProviderInvocationEvidence.model_fields
        properties = ProviderInvocationEvidence.model_json_schema()[
            "properties"
        ]
        assert "schema_version" not in properties
        assert "schema_version" not in evidence.identity_payload()
        assert (
            evidence.identity_document().schema_version
            == PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION
        )

    def test_explicit_schema_version_is_rejected(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        data = provider.invoke(openai_request()).model_dump(mode="python")

        with pytest.raises(ValidationError):
            ProviderInvocationEvidence.model_validate(
                {
                    **data,
                    "schema_version": (
                        PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION
                    ),
                }
            )

    def test_sanitize_kwargs_redacts_credentials(self) -> None:
        assert sanitize_kwargs({"api_key": "secret", "temperature": 0.7}) == {
            "api_key": "<redacted>",
            "temperature": 0.7,
        }

    @pytest.mark.parametrize(
        ("response", "failure"),
        [(None, None), (SUCCESS, FAILURE)],
        ids=["neither", "both"],
    )
    def test_exactly_one_outcome_is_required(
        self,
        response: ProviderTransportResponse | None,
        failure: ProviderTransportFailure | None,
    ) -> None:
        with pytest.raises(ValidationError):
            evidence_for(response=response, failure=failure)

    def test_evidence_binds_request_policy_and_success_body(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        request = openai_request(token_limit=64)
        evidence = provider.invoke(request)

        payload = evidence.identity_payload()
        assert payload["request_identity_hash"] == request.identity_hash
        assert payload["policy_identity"] == OPENAI_POLICY.identity_payload()
        assert isinstance(evidence.outcome, ProviderTransportResponse)
        assert evidence.response is not None
        assert evidence.response.response_body == CHAT_BODY_OK
        assert evidence.http_request is not None
        assert evidence.http_request.body["model"] == "m"

    def test_evidence_retains_complete_failure_body(self) -> None:
        long_message = "x" * 5000
        big_body = {"error": {"message": long_message}}
        provider = mock_provider(
            lambda _req: httpx.Response(400, json=big_body)
        )
        evidence = provider.invoke(openai_request())
        failure = evidence.failure
        assert failure is not None
        assert failure.response_body == big_body
        assert long_message in json.dumps(failure.response_body)

    def test_evidence_never_persists_authorization_header(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        evidence = provider.invoke(openai_request())
        assert evidence.http_request is not None
        headers = evidence.http_request.headers
        assert headers.get("Authorization") == "<redacted>"
        serialized = evidence.identity_document().to_json_dict()
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
        assert "test-key" not in str(
            evidence.identity_document().to_json_dict()
        )

    def test_evidence_retains_wire_path_exception_traceback(self) -> None:
        provider = mock_provider(
            lambda _req: (_ for _ in ()).throw(httpx.ConnectError("wire down"))
        )
        evidence = provider.invoke(openai_request())
        failure = evidence.failure
        assert failure is not None
        assert failure.traceback is not None
        assert "ConnectError" in failure.traceback
        assert "wire down" not in failure.message

    def test_success_document_shape(self) -> None:
        assert evidence_for(
            response=SUCCESS
        ).identity_document().to_json_dict() == (
            expected_document(
                response={
                    "text": "hi",
                    "response_body": {"id": "resp-1"},
                    "usage": None,
                    "cost": None,
                    "warnings": [],
                    "stop_reason": None,
                    "response_id": None,
                    "model": None,
                    "diagnostics": None,
                }
            )
        )

    def test_failure_document_shape(self) -> None:
        assert evidence_for(
            failure=FAILURE
        ).identity_document().to_json_dict() == (
            expected_document(
                failure={
                    "recoverability": "permanent",
                    "code": "invalid_request",
                    "message": "bad request",
                    "traceback": None,
                    "response_body": {"error": "bad"},
                    "status_code": 400,
                    "containment": None,
                    "metadata": {"provider": "openai"},
                }
            )
        )

    @pytest.mark.parametrize(
        "evidence",
        [evidence_for(response=SUCCESS), evidence_for(failure=FAILURE)],
        ids=["success", "failure"],
    )
    def test_document_json_round_trip(
        self, evidence: ProviderInvocationEvidence
    ) -> None:
        document = json.loads(
            json.dumps(evidence.identity_document().to_json_dict())
        )
        restored = ProviderInvocationEvidence.model_validate(
            document["payload"]
        )

        assert restored.identity_document().to_json_dict() == document


def test_retry_after_hint_accepts_bounded_http_date_evidence() -> None:
    hint = ProviderRetryAfterHint(
        kind="http_date",
        value="Wed, 21 Oct 2015 07:28:00 GMT",
    )

    assert hint.model_dump(mode="json") == {
        "kind": "http_date",
        "value": "Wed, 21 Oct 2015 07:28:00 GMT",
    }
