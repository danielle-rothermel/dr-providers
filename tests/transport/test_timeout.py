from __future__ import annotations

import threading
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
from dr_providers.transport.http import (
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
    HttpProvider,
    _httpx_timeout,
)

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
        idle_timeout_seconds=overrides.pop("idle_timeout_seconds", 0.2),
        max_connections=overrides.pop("max_connections", 1),
        max_keepalive_connections=overrides.pop(
            "max_keepalive_connections", 1
        ),
        max_request_bytes=overrides.pop("max_request_bytes", 1024),
        max_response_bytes=overrides.pop("max_response_bytes", 1024),
        **overrides,
    )


def test_native_timeout_phases_are_explicit_and_saturated() -> None:
    policy = _policy(
        timeout_seconds=1e300,
        idle_timeout_seconds=1e300,
    )

    timeout = _httpx_timeout(policy)

    assert timeout.connect == 30.0
    assert timeout.read == threading.TIMEOUT_MAX
    assert timeout.write == threading.TIMEOUT_MAX
    assert timeout.pool == threading.TIMEOUT_MAX
    assert policy.timeout_seconds == 1e300
    assert policy.idle_timeout_seconds == 1e300


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (httpx.ConnectTimeout("unbounded provider detail"), TIMEOUT_CODE),
        (
            httpx.ReadTimeout("unbounded provider detail"),
            STALLED_RESPONSE_CODE,
        ),
        (httpx.WriteTimeout("unbounded provider detail"), TIMEOUT_CODE),
        (httpx.PoolTimeout("unbounded provider detail"), TIMEOUT_CODE),
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
