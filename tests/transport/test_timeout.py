from __future__ import annotations

from typing import Any

import httpx
import pytest
from _policy import make_transport_policy

from dr_providers import (
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    Transcript,
    TransportTimeoutContainment,
    openai_chat_config,
)
from dr_providers.lifecycle.classifier import (
    AcceptAllSemanticResponseClassifier,
    classify_provider_invocation,
)
from dr_providers.lifecycle.outcomes import ProviderInvocationOutcome
from dr_providers.outcomes.models import (
    POOL_TIMEOUT_CODE,
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
)
from dr_providers.transport.http import HttpProvider, _client_config

MESSAGES = (PromptMessage(role=MessageRole.USER, content="hi"),)


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m", controls=GenerationControls()),
        transcript=Transcript(messages=MESSAGES),
    )


def _policy(**overrides: Any) -> ProviderTransportPolicy:
    return make_transport_policy(
        base_url="https://example.test",
        timeout_seconds=overrides.pop("timeout_seconds", 0.3),
        connect_timeout_seconds=overrides.pop("connect_timeout_seconds", 0.1),
        idle_timeout_seconds=overrides.pop("idle_timeout_seconds", 0.2),
        max_connections=overrides.pop("max_connections", 1),
        max_keepalive_connections=overrides.pop(
            "max_keepalive_connections", 1
        ),
        max_request_bytes=overrides.pop("max_request_bytes", 1024),
        max_response_bytes=overrides.pop("max_response_bytes", 1024),
        **overrides,
    )


def test_client_config_carries_every_policy_bound() -> None:
    policy = _policy(
        timeout_seconds=120.0,
        connect_timeout_seconds=45.0,
        idle_timeout_seconds=90.0,
        max_connections=4,
        max_keepalive_connections=2,
        max_request_bytes=2048,
        max_response_bytes=4096,
    )

    config = _client_config(policy)

    assert config.timeout_seconds == 120.0
    assert config.connect_timeout_seconds == 45.0
    assert config.idle_timeout_seconds == 90.0
    assert config.max_connections == 4
    assert config.max_keepalive_connections == 2
    assert config.max_request_bytes == 2048
    assert config.max_response_bytes == 4096


def test_client_config_receives_policy_clamped_timeouts() -> None:
    """Clamping is identity-bearing, so the policy owns it, not the client."""
    policy = _policy(
        timeout_seconds=10.0,
        connect_timeout_seconds=30.0,
        idle_timeout_seconds=30.0,
    )

    config = _client_config(policy)

    assert policy.connect_timeout_seconds == 10.0
    assert policy.idle_timeout_seconds == 10.0
    assert config.connect_timeout_seconds == 10.0
    assert config.idle_timeout_seconds == 10.0


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (httpx.ConnectTimeout("unbounded provider detail"), TIMEOUT_CODE),
        (
            httpx.ReadTimeout("unbounded provider detail"),
            STALLED_RESPONSE_CODE,
        ),
        (httpx.WriteTimeout("unbounded provider detail"), TIMEOUT_CODE),
        (httpx.PoolTimeout("unbounded provider detail"), POOL_TIMEOUT_CODE),
    ],
    ids=("connect", "read", "write", "pool"),
)
def test_native_timeout_phases_are_contained(
    error: httpx.TimeoutException,
    expected_code: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise error

    provider = HttpProvider(
        policy=_policy(),
        api_key="test-key",
        _client_factory=lambda **_kwargs: httpx.Client(
            transport=httpx.MockTransport(handler)
        ),
    )

    evidence = provider.invoke(_request())
    failure = evidence.failure

    assert isinstance(failure, ProviderTransportFailure)
    assert failure.code == expected_code
    assert failure.containment is TransportTimeoutContainment.CONTAINED
    assert failure.message == "provider transport timeout"
    assert failure.metadata["exception_type"] == type(error).__name__
    assert "phase" not in failure.metadata
    assert "unbounded provider detail" not in failure.message
    assert failure.traceback is not None
    assert type(error).__name__ in failure.traceback
    assert (
        classify_provider_invocation(
            evidence,
            AcceptAllSemanticResponseClassifier(),
        )
        is ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT
    )
