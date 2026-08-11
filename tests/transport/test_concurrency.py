"""Provider-level admission over the composed bounded client.

The lifecycle state machine itself is verified in dr-http. What matters
here is that a whole invocation is the admitted unit, so a close drains
complete evidence rather than a bare wire call.
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
from dr_providers.transport import http as http_module
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
    return make_transport_policy(base_url="https://example.test")


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


def test_close_drains_the_whole_invocation_and_closes_once() -> None:
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


def test_a_close_waits_for_evidence_built_after_the_wire_call_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admitted unit is the whole invocation, not the wire call.

    The block here is placed after ``call`` has returned, so a provider
    that admitted only its wire exchange would let the close complete
    and shut the client down between the exchange and its evidence.
    """
    in_evidence_phase = threading.Event()
    release = threading.Event()
    real_with_conformance_warnings = http_module.with_conformance_warnings

    def blocking_with_conformance_warnings(
        request: Any,
        outcome: Any,
    ) -> Any:
        in_evidence_phase.set()
        _wait_for(release, "evidence phase was not released")
        return real_with_conformance_warnings(request, outcome)

    monkeypatch.setattr(
        http_module,
        "with_conformance_warnings",
        blocking_with_conformance_warnings,
    )

    client = RecordingClient(
        lambda _request: httpx.Response(200, json=CHAT_BODY_OK)
    )
    provider = HttpProvider(
        policy=_policy(),
        api_key="test-key",
        _client_factory=lambda **_kwargs: client,
    )
    invocation = DaemonCall.start(lambda: provider.invoke(_request()))
    _wait_for(in_evidence_phase, "invocation did not reach its evidence phase")

    closer = DaemonCall.start(provider.close)
    closer.wait_until_entered()
    with provider._client._condition:
        reached_closing = provider._client._condition.wait_for(
            lambda: provider._client._state.name == "CLOSING",
            timeout=WATCHDOG_SECONDS,
        )

    assert reached_closing
    assert not closer.has_returned()
    assert not invocation.has_returned()
    assert client.close_count == 0
    assert not client.is_closed
    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.invoke(_request())

    release.set()
    evidence = invocation.result()
    closer.result()

    assert isinstance(evidence.outcome, ProviderTransportResponse)
    assert client.close_count == 1


def test_invoking_a_closed_provider_is_refused() -> None:
    provider = HttpProvider(
        policy=_policy(),
        api_key="test-key",
        _client_factory=lambda **_kwargs: RecordingClient(
            lambda _request: httpx.Response(200, json=CHAT_BODY_OK)
        ),
    )
    provider.close()

    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.invoke(_request())
