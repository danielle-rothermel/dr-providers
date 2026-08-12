"""Pin the async entry point against a real HttpProvider."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx
import pytest
from _concurrency import WATCHDOG_SECONDS, DaemonCall
from _policy import make_transport_policy

from dr_providers import (
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportPolicy,
    Transcript,
    openai_chat_config,
)
from dr_providers.lifecycle import (
    ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER,
    AcceptAllSemanticResponseClassifier,
    ProviderCallOutcomeKind,
    ProviderCallState,
    StandardProviderCallRetryPolicy,
    run_local_provider_call_async,
)
from dr_providers.transport.http import HttpProvider

CHAT_BODY_OK: dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "m",
    "choices": [
        {
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
}


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m", controls=GenerationControls()),
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content="hi"),)
        ),
    )


def _state() -> ProviderCallState:
    return ProviderCallState.initial(
        request=_request(),
        retry_policy=StandardProviderCallRetryPolicy(),
        classifier_identifier=ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER,
    )


def _policy(**overrides: Any) -> ProviderTransportPolicy:
    return make_transport_policy(base_url="https://example.test", **overrides)


def _assert_closed(provider: HttpProvider) -> None:
    """A closed provider refuses admission, which is what callers observe."""
    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.offload(lambda: None)


def _provider(
    wire_requests: list[httpx.Request] | None = None,
    **policy_overrides: Any,
) -> HttpProvider:
    recorded = [] if wire_requests is None else wire_requests

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json=CHAT_BODY_OK)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HttpProvider(
        policy=_policy(**policy_overrides),
        api_key="test-key",
        _client_factory=lambda **_kwargs: client,
    )


def test_async_entry_drives_a_real_http_provider() -> None:
    provider = _provider()

    result = asyncio.run(
        run_local_provider_call_async(
            provider=provider,
            state=_state(),
            classifier=AcceptAllSemanticResponseClassifier(),
            cancellation=threading.Event(),
        )
    )
    provider.close()

    assert result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    _assert_closed(provider)


def test_cancelling_the_task_with_a_queued_offload_still_runs_and_closes() -> (
    None
):
    """A queued offload survives task cancellation and still drains."""
    started = threading.Event()
    release = threading.Event()
    wire_requests: list[httpx.Request] = []
    provider = _provider(
        wire_requests,
        max_connections=1,
        max_keepalive_connections=1,
    )

    def gated() -> None:
        started.set()
        if not release.wait(WATCHDOG_SECONDS):
            raise TimeoutError("gate was not released")

    blocker = provider.offload(gated)
    assert started.wait(WATCHDOG_SECONDS)

    async def drive() -> None:
        task = asyncio.create_task(
            run_local_provider_call_async(
                provider=provider,
                state=_state(),
                classifier=AcceptAllSemanticResponseClassifier(),
                cancellation=threading.Event(),
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(drive())
    finally:
        release.set()

    blocker.result(timeout=WATCHDOG_SECONDS)
    closer = DaemonCall.start(provider.close)
    closer.result()

    assert len(wire_requests) == 1
    _assert_closed(provider)
