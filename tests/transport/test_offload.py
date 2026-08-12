"""Provider-level offload delegation over the composed bounded client.

Executor sizing, one-shot drain release, abort semantics, and the close
state machine are verified in dr-wire against the client that owns them.
What is verified here is that the provider delegates to that client, so
offloaded work reaches the same lifecycle a direct invocation does.
"""

from __future__ import annotations

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
    ProviderTransportResponse,
    Transcript,
    openai_chat_config,
)
from dr_providers.transport.http import HttpProvider

MESSAGES = (PromptMessage(role=MessageRole.USER, content="hi"),)
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
OFFLOADED_FAILURE_MSG = "offloaded failure"
OFFLOADED_WORK_NOT_RELEASED = "offloaded work was not released"
OFFLOADED_WORK_DID_NOT_START = "offloaded work did not start"
DRAINED_OFFLOAD_RESULT = "drained"


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m", controls=GenerationControls()),
        transcript=Transcript(messages=MESSAGES),
    )


def _policy(**overrides: Any) -> ProviderTransportPolicy:
    return make_transport_policy(base_url="https://example.test", **overrides)


class _CloseCountingClient(httpx.Client):
    def __init__(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=CHAT_BODY_OK)

        super().__init__(transport=httpx.MockTransport(handler))
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()


def _provider(
    **policy_overrides: Any,
) -> tuple[
    HttpProvider,
    _CloseCountingClient,
]:
    client = _CloseCountingClient()
    provider = HttpProvider(
        policy=_policy(**policy_overrides),
        api_key="test-key",
        _client_factory=lambda **_kwargs: client,
    )
    return provider, client


def _wait_for(event: threading.Event, description: str) -> None:
    if not event.wait(timeout=WATCHDOG_SECONDS):
        raise TimeoutError(description)


def test_offload_runs_work_and_returns_its_result() -> None:
    provider, _client = _provider()

    with provider:
        future = provider.offload(lambda: "offloaded")

        assert future.result(timeout=WATCHDOG_SECONDS) == "offloaded"


def test_offloaded_work_may_invoke_the_provider() -> None:
    provider, _client = _provider()

    with provider:
        future = provider.offload(lambda: provider.invoke(_request()))
        evidence = future.result(timeout=WATCHDOG_SECONDS)

    assert isinstance(evidence.outcome, ProviderTransportResponse)


def test_failing_offloaded_work_propagates_through_the_future() -> None:
    provider, _client = _provider()

    def boom() -> None:
        raise ValueError(OFFLOADED_FAILURE_MSG)

    with provider:
        future = provider.offload(boom)

        with pytest.raises(ValueError, match=OFFLOADED_FAILURE_MSG):
            future.result(timeout=WATCHDOG_SECONDS)


def test_close_drains_offloaded_work_before_closing_the_client() -> None:
    started = threading.Event()
    release = threading.Event()
    provider, client = _provider()

    def gated() -> str:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)
        return DRAINED_OFFLOAD_RESULT

    future = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)

    closer = DaemonCall.start(provider.close)
    closer.wait_until_entered()
    assert client.close_count == 0

    release.set()
    assert future.result(timeout=WATCHDOG_SECONDS) == DRAINED_OFFLOAD_RESULT
    closer.result()

    assert client.close_count == 1
    assert client.is_closed


def test_offload_after_close_is_refused() -> None:
    provider, _client = _provider()
    provider.close()

    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.offload(lambda: None)


def test_sync_only_use_closes_without_offloaded_work() -> None:
    provider, client = _provider()

    evidence = provider.invoke(_request())
    provider.close()

    assert isinstance(evidence.outcome, ProviderTransportResponse)
    assert client.close_count == 1
