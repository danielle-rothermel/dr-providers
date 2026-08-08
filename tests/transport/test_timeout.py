from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    import socket

    import pytest

from _concurrency import (
    WATCHDOG_SECONDS,
    DaemonCall,
    LocalSocketServer,
)

from dr_providers import (
    ApiKeyEnv,
    FailureClass,
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
from dr_providers.transport.http import (
    ATTEMPT_DEADLINE_MARGIN_SECONDS,
    STALLED_RESPONSE_CODE,
    HttpProvider,
    _httpx_timeout,
    _operational_attempt_deadline_seconds,
)

POLICY_TIMEOUT_SECONDS = 0.3
POLICY_IDLE_SECONDS = 0.2
MESSAGES = (PromptMessage(role=MessageRole.USER, content="hi"),)


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m", controls=GenerationControls()),
        transcript=Transcript(messages=MESSAGES),
    )


def _policy(
    *, idle_timeout_seconds: float, timeout_seconds: float
) -> ProviderTransportPolicy:
    return ProviderTransportPolicy(
        api_key_env=str(ApiKeyEnv.OPENAI),
        base_url="http://placeholder",
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
    )


def _provider(
    server: LocalSocketServer,
    *,
    idle_timeout_seconds: float,
    timeout_seconds: float,
) -> HttpProvider:
    policy = _policy(
        idle_timeout_seconds=idle_timeout_seconds,
        timeout_seconds=timeout_seconds,
    )
    return HttpProvider(
        policy=policy.model_copy(update={"base_url": server.base_url}),
        api_key="test-key",
    )


def test_platform_unsafe_policy_timeouts_are_saturated_operationally() -> None:
    requested_timeout_seconds = 1e300
    policy = ProviderTransportPolicy(
        api_key_env=str(ApiKeyEnv.OPENAI),
        base_url="http://placeholder",
        timeout_seconds=requested_timeout_seconds,
        idle_timeout_seconds=requested_timeout_seconds,
    )

    socket_timeout = _httpx_timeout(policy.idle_timeout_seconds)

    assert socket_timeout.read == threading.TIMEOUT_MAX
    assert socket_timeout.write == threading.TIMEOUT_MAX
    assert socket_timeout.pool == threading.TIMEOUT_MAX
    assert (
        _operational_attempt_deadline_seconds(policy.timeout_seconds)
        == threading.TIMEOUT_MAX
    )
    assert policy.timeout_seconds == requested_timeout_seconds
    assert policy.idle_timeout_seconds == requested_timeout_seconds


def test_provider_uses_saturated_watchdog_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait_timeouts: list[float | None] = []

    class RecordingEvent:
        def __init__(self) -> None:
            self._event = threading.Event()

        def set(self) -> None:
            self._event.set()

        def wait(self, timeout: float | None = None) -> bool:
            wait_timeouts.append(timeout)
            return self._event.wait(timeout=WATCHDOG_SECONDS)

    monkeypatch.setattr(
        "dr_providers.transport.http.threading",
        SimpleNamespace(
            Event=RecordingEvent,
            Thread=threading.Thread,
            TIMEOUT_MAX=threading.TIMEOUT_MAX,
        ),
    )
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )
        )
    )
    policy = ProviderTransportPolicy(
        api_key_env=str(ApiKeyEnv.OPENAI),
        base_url="https://example.test",
        timeout_seconds=1e300,
        idle_timeout_seconds=1e300,
    )

    with HttpProvider(
        policy=policy, client=client, api_key="test-key"
    ) as provider:
        outcome = provider.invoke(_request()).outcome

    assert isinstance(outcome, ProviderTransportResponse)
    assert wait_timeouts == [threading.TIMEOUT_MAX]


def _headers_then_stall(conn: socket.socket, stop: threading.Event) -> None:
    conn.recv(65536)
    conn.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 1000\r\n"
        b"\r\n"
    )
    stop.wait(timeout=WATCHDOG_SECONDS)


def _dribble_forever(conn: socket.socket, stop: threading.Event) -> None:
    conn.recv(65536)
    conn.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 100000\r\n"
        b"\r\n"
        b'{"x":"'
    )
    while not stop.is_set():
        try:
            conn.sendall(b"a")
        except OSError:
            return
        stop.wait(timeout=0.05)


def _steady_stream_then_complete(
    conn: socket.socket, stop: threading.Event
) -> None:
    body = b'{"choices":[{"message":{"content":"ok"}}]}'
    conn.recv(65536)
    conn.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"\r\n"
    )
    for byte in (body[index : index + 1] for index in range(len(body))):
        if stop.is_set():
            return
        with contextlib.suppress(OSError):
            conn.sendall(byte)
        stop.wait(timeout=0.03)


def test_idle_stall_returns_exact_typed_evidence() -> None:
    with LocalSocketServer(_headers_then_stall) as server:
        provider = _provider(
            server,
            idle_timeout_seconds=POLICY_IDLE_SECONDS,
            timeout_seconds=POLICY_TIMEOUT_SECONDS,
        )
        call = DaemonCall.start(lambda: provider.invoke(_request()))
        call.wait_until_entered()
        server.wait_until_entered()
        evidence = call.result()

    failure = evidence.failure
    assert isinstance(failure, ProviderTransportFailure)
    assert failure.code == STALLED_RESPONSE_CODE
    assert failure.failure_class is FailureClass.TRANSIENT
    assert failure.metadata == {
        "url": f"{server.base_url}/chat/completions",
        "timeout_seconds": POLICY_TIMEOUT_SECONDS,
        "idle_timeout_seconds": POLICY_IDLE_SECONDS,
        "phase": "ReadTimeout",
    }


def test_steady_progress_stream_completes() -> None:
    with LocalSocketServer(_steady_stream_then_complete) as server:
        provider = _provider(
            server,
            idle_timeout_seconds=0.2,
            timeout_seconds=5.0,
        )
        call = DaemonCall.start(lambda: provider.invoke(_request()).outcome)
        call.wait_until_entered()
        server.wait_until_entered()
        outcome = call.result()

    assert isinstance(outcome, ProviderTransportResponse)
    assert outcome.text == "ok"


def test_dribble_is_bounded_by_exact_hard_deadline() -> None:
    with LocalSocketServer(_dribble_forever) as server:
        provider = _provider(
            server,
            idle_timeout_seconds=POLICY_TIMEOUT_SECONDS,
            timeout_seconds=POLICY_TIMEOUT_SECONDS,
        )
        call = DaemonCall.start(lambda: provider.invoke(_request()).outcome)
        call.wait_until_entered()
        server.wait_until_entered()
        outcome = call.result()

    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.code == STALLED_RESPONSE_CODE
    assert outcome.failure_class is FailureClass.TRANSIENT
    assert outcome.metadata == {
        "url": f"{server.base_url}/chat/completions",
        "timeout_seconds": POLICY_TIMEOUT_SECONDS,
        "deadline_seconds": (
            POLICY_TIMEOUT_SECONDS + ATTEMPT_DEADLINE_MARGIN_SECONDS
        ),
    }
