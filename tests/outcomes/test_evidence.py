"""Provider invocation evidence tests."""

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
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    Transcript,
    anthropic_messages_config,
    openai_chat_config,
    sanitize_kwargs,
    token_usage_from_body,
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


class TestInvocationEvidence:
    def test_schema_version_exists_only_on_identity_document(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        evidence = provider.invoke(openai_request())

        assert "schema_version" not in ProviderInvocationEvidence.model_fields
        properties = ProviderInvocationEvidence.model_json_schema()[
            "properties"
        ]
        assert "schema_version" not in properties
        assert "schema_version" not in evidence.stable_payload()
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

    def test_token_usage_absent_when_no_fields(self) -> None:
        assert token_usage_from_body({"usage": {}}) is None
        assert token_usage_from_body({}) is None

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
