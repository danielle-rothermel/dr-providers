from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

from _concurrency import WATCHDOG_SECONDS, DaemonCall

import dr_providers.transport.http as http_transport
from dr_providers import (
    ApiKeyEnv,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    Transcript,
    openai_chat_config,
)
from dr_providers.transport.http import STALLED_RESPONSE_CODE, HttpProvider

TEST_DEADLINE_MARGIN_SECONDS = 0.05
TEST_TIMEOUT_SECONDS = 0.05
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
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m", controls=GenerationControls()),
        transcript=Transcript(messages=MESSAGES),
    )


def _policy() -> ProviderTransportPolicy:
    return ProviderTransportPolicy(
        api_key_env=str(ApiKeyEnv.OPENAI),
        base_url="https://example.test",
        timeout_seconds=TEST_TIMEOUT_SECONDS,
        idle_timeout_seconds=TEST_TIMEOUT_SECONDS,
    )


def _wait_for(event: threading.Event, description: str) -> None:
    if not event.wait(timeout=WATCHDOG_SECONDS):
        raise TimeoutError(description)


def _blocking_client(
    *,
    entered: threading.Event,
    release: threading.Event,
    exited: threading.Event,
) -> httpx.Client:
    def handler(_request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            _wait_for(release, "blocking client was not released")
            return httpx.Response(200, json=CHAT_BODY_OK)
        finally:
            exited.set()

    return httpx.Client(transport=httpx.MockTransport(handler))


def _deadline_metadata() -> dict[str, float | str]:
    return {
        "url": "https://example.test/chat/completions",
        "timeout_seconds": TEST_TIMEOUT_SECONDS,
        "deadline_seconds": (
            TEST_TIMEOUT_SECONDS + TEST_DEADLINE_MARGIN_SECONDS
        ),
    }


def test_owned_deadline_breach_does_not_disturb_healthy_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_entered = threading.Event()
    blocked_release = threading.Event()
    blocked_exited = threading.Event()
    blocked_client = _blocking_client(
        entered=blocked_entered,
        release=blocked_release,
        exited=blocked_exited,
    )
    healthy_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=CHAT_BODY_OK)
        )
    )
    clients = iter((blocked_client, healthy_client))
    monkeypatch.setattr(
        http_transport, "ATTEMPT_DEADLINE_MARGIN_SECONDS", 0.05
    )
    monkeypatch.setattr(httpx, "Client", lambda: next(clients))
    provider = HttpProvider(policy=_policy(), api_key="test-key")

    blocked_call = DaemonCall.start(
        lambda: provider.invoke(_request()).outcome
    )
    blocked_call.wait_until_entered()
    _wait_for(blocked_entered, "first call did not enter its client")
    healthy_call = DaemonCall.start(
        lambda: provider.invoke(_request()).outcome
    )
    healthy_call.wait_until_entered()
    try:
        healthy_outcome = healthy_call.result()
        blocked_outcome = blocked_call.result()
    finally:
        blocked_release.set()
        _wait_for(blocked_exited, "blocked owned worker did not exit")

    assert isinstance(healthy_outcome, ProviderTransportResponse)
    assert healthy_outcome.text == "ok"
    assert isinstance(blocked_outcome, ProviderTransportFailure)
    assert blocked_outcome.code == STALLED_RESPONSE_CODE
    assert blocked_outcome.metadata == _deadline_metadata()
    assert blocked_client.is_closed
    assert healthy_client.is_closed


def test_injected_client_hard_deadline_preserves_caller_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    client = _blocking_client(
        entered=entered,
        release=release,
        exited=exited,
    )
    worker_threads: list[threading.Thread] = []

    def recording_thread(
        *, target: Callable[[], None], daemon: bool
    ) -> threading.Thread:
        thread = threading.Thread(target=target, daemon=daemon)
        worker_threads.append(thread)
        return thread

    monkeypatch.setattr(
        http_transport,
        "threading",
        SimpleNamespace(
            Event=threading.Event,
            Thread=recording_thread,
            TIMEOUT_MAX=threading.TIMEOUT_MAX,
        ),
    )
    monkeypatch.setattr(
        http_transport,
        "ATTEMPT_DEADLINE_MARGIN_SECONDS",
        TEST_DEADLINE_MARGIN_SECONDS,
    )
    provider = HttpProvider(
        policy=_policy(), client=client, api_key="test-key"
    )
    call = DaemonCall.start(lambda: provider.invoke(_request()).outcome)
    call.wait_until_entered()
    _wait_for(entered, "injected client did not enter its wire call")

    try:
        outcome = call.result()
        assert len(worker_threads) == 1
        worker = worker_threads[0]
        assert worker.is_alive()
        assert not client.is_closed
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code == STALLED_RESPONSE_CODE
        assert outcome.metadata == _deadline_metadata()
    finally:
        release.set()
        _wait_for(exited, "injected client worker did not leave post")

    worker.join(timeout=WATCHDOG_SECONDS)
    assert not worker.is_alive()
    assert not client.is_closed
    client.close()
    assert client.is_closed
