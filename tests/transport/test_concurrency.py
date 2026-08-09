from __future__ import annotations

import threading
from typing import Any

import httpx
import pytest
from _concurrency import WATCHDOG_SECONDS, DaemonCall

from dr_providers import (
    ApiKeyEnv,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderKind,
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


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m", controls=GenerationControls()),
        transcript=Transcript(messages=MESSAGES),
    )


def _policy() -> ProviderTransportPolicy:
    return ProviderTransportPolicy(
        provider_kind=ProviderKind.OPENAI,
        api_key_env=str(ApiKeyEnv.OPENAI),
        base_url="https://example.test",
    )


def _wait_for(event: threading.Event, description: str) -> None:
    if not event.wait(timeout=WATCHDOG_SECONDS):
        raise TimeoutError(description)


class RecordingClient(httpx.Client):
    def __init__(self, handler: Any) -> None:
        super().__init__(transport=httpx.MockTransport(handler))
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()


def test_close_stops_admission_drains_and_closes_once() -> None:
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    def handler(_request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            _wait_for(release, "active invocation was not released")
            return httpx.Response(200, json=CHAT_BODY_OK)
        finally:
            exited.set()

    client = RecordingClient(handler)
    provider = HttpProvider(
        policy=_policy(),
        api_key="test-key",
        _client_factory=lambda **_kwargs: client,
    )
    invocation = DaemonCall.start(lambda: provider.invoke(_request()))
    invocation.wait_until_entered()
    _wait_for(entered, "active invocation did not reach the transport")

    closers = [DaemonCall.start(provider.close) for _ in range(3)]
    for closer in closers:
        closer.wait_until_entered()
    with provider._condition:
        reached_closing = provider._condition.wait_for(
            lambda: provider._state.name == "CLOSING",
            timeout=WATCHDOG_SECONDS,
        )
    assert reached_closing
    assert client.close_count == 0
    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.invoke(_request())

    release.set()
    _wait_for(exited, "active invocation did not leave the transport")
    evidence = invocation.result()
    for closer in closers:
        closer.result()

    assert isinstance(evidence.outcome, ProviderTransportResponse)
    assert client.close_count == 1
    assert client.is_closed

    provider.close()
    assert client.close_count == 1
