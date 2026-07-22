"""Live-fire timeout tests against local stall servers.

Proves the transport enforces the policy ``timeout_seconds`` as a
wall-clock bound: a stalled response (headers sent then bytes stop, or a
socket that dribbles bytes indefinitely, or a connection that never
completes accept/read) returns the typed Provider Transport Failure within
roughly the policy timeout plus its fixed margin — and never hangs.

All policy timeouts here are tiny (well under a second) so the suite stays
fast, and every ``invoke``/``complete`` is additionally wrapped in an
independent wall-clock watchdog so a regression that reintroduces the hang
fails loudly instead of wedging the whole test run.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

from dr_providers import (
    ApiKeyEnv,
    FailureClass,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    Transcript,
    openai_chat_config,
)
from dr_providers.transport import (
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
    HttpProvider,
)

# Tiny budgets so the suite stays fast. The hard deadline the transport
# enforces is ``timeout_seconds + INVOCATION_DEADLINE_MARGIN_SECONDS`` (5s
# margin); we keep timeout_seconds small and give each call a generous but
# finite external watchdog so a real hang is caught.
POLICY_TIMEOUT_SECONDS = 0.5
EXTERNAL_WATCHDOG_SECONDS = 20.0
MESSAGES = (PromptMessage(role=MessageRole.USER, content="hi"),)


def _stall_policy() -> ProviderTransportPolicy:
    return ProviderTransportPolicy(
        api_key_env=str(ApiKeyEnv.OPENAI),
        base_url="http://placeholder",  # overridden per test with the port
        timeout_seconds=POLICY_TIMEOUT_SECONDS,
    )


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m", controls=GenerationControls()),
        transcript=Transcript(messages=MESSAGES),
    )


class _StallServer:
    """A localhost server whose handler is provided per test.

    The handler receives the accepted connection and a stop event. It may
    send partial headers/body and then block, dribble bytes, or simply
    never read — reproducing a stalled edge.
    """

    def __init__(
        self, handler: Callable[[socket.socket, threading.Event], None]
    ) -> None:
        self._handler = handler
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            try:
                self._handler(conn, self._stop)
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    conn.close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> _StallServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._sock.close()
        self._thread.join(timeout=2.0)


def _headers_then_stall(conn: socket.socket, stop: threading.Event) -> None:
    """Send 200 headers promising a body, then never send the body."""
    conn.recv(65536)
    conn.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 1000\r\n"
        b"\r\n"
    )
    stop.wait(timeout=EXTERNAL_WATCHDOG_SECONDS)


def _dribble_forever(conn: socket.socket, stop: threading.Event) -> None:
    """Trickle body bytes slower than a per-read timeout would fire.

    This is the failure mode a bare float timeout cannot bound: each read
    makes progress within the read-timeout window, so httpx's per-read
    timeout never fires and only the wall-clock deadline can stop it.
    """
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


def _accept_then_silent(_conn: socket.socket, stop: threading.Event) -> None:
    """Accept the connection but never read the request or respond.

    Stands in for a wedged handshake/edge that accepts the socket but
    never completes the exchange, blocking a client with zero progress.
    """
    stop.wait(timeout=EXTERNAL_WATCHDOG_SECONDS)


def _invoke_bounded(provider: HttpProvider) -> Any:
    """Run ``invoke`` under an external watchdog so a hang fails loudly."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(provider.invoke, _request())
        try:
            return future.result(timeout=EXTERNAL_WATCHDOG_SECONDS)
        except concurrent.futures.TimeoutError:  # pragma: no cover
            pytest.fail(
                "invoke hung past the external watchdog — the policy "
                "timeout was not enforced as a wall-clock bound"
            )


@pytest.fixture
def make_provider() -> Iterator[Callable[[_StallServer], HttpProvider]]:
    created: list[HttpProvider] = []

    def factory(server: _StallServer) -> HttpProvider:
        policy = _stall_policy().model_copy(
            update={"base_url": server.base_url}
        )
        provider = HttpProvider(policy=policy, api_key="test-key")
        created.append(provider)
        return provider

    yield factory
    for provider in created:
        provider.close()


class TestStalledResponseTimeout:
    def test_headers_then_stall_returns_typed_failure_fast(
        self,
        make_provider: Callable[[_StallServer], HttpProvider],
    ) -> None:
        with _StallServer(_headers_then_stall) as server:
            provider = make_provider(server)
            start = time.monotonic()
            evidence = _invoke_bounded(provider)
            elapsed = time.monotonic() - start

        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}
        # bounded by policy timeout + the fixed deadline margin (5s) with a
        # little slack for thread scheduling.
        assert elapsed < POLICY_TIMEOUT_SECONDS + 8.0

    def test_dribble_forever_is_bounded_by_wall_clock_deadline(
        self,
        make_provider: Callable[[_StallServer], HttpProvider],
    ) -> None:
        # This is the case a bare float read-timeout cannot bound: only the
        # per-invocation wall-clock deadline stops it.
        with _StallServer(_dribble_forever) as server:
            provider = make_provider(server)
            start = time.monotonic()
            evidence = _invoke_bounded(provider)
            elapsed = time.monotonic() - start

        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}
        assert elapsed < POLICY_TIMEOUT_SECONDS + 8.0

    def test_accept_then_silent_returns_typed_failure_fast(
        self,
        make_provider: Callable[[_StallServer], HttpProvider],
    ) -> None:
        with _StallServer(_accept_then_silent) as server:
            provider = make_provider(server)
            start = time.monotonic()
            evidence = _invoke_bounded(provider)
            elapsed = time.monotonic() - start

        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}
        assert elapsed < POLICY_TIMEOUT_SECONDS + 8.0

    def test_complete_also_bounded_and_no_throw(
        self,
        make_provider: Callable[[_StallServer], HttpProvider],
    ) -> None:
        with _StallServer(_headers_then_stall) as server:
            provider = make_provider(server)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(provider.complete, _request())
                try:
                    outcome = future.result(timeout=EXTERNAL_WATCHDOG_SECONDS)
                except concurrent.futures.TimeoutError:  # pragma: no cover
                    pytest.fail("complete hung; timeout not enforced")
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}


class TestTimeoutFailureShape:
    """The typed timeout failure carries transport-level retry evidence.

    No Whetstone semantics here: only that the failure shape marks a stall
    as a transient, retry-appropriate transport condition and preserves the
    raw request evidence.
    """

    def test_timeout_failure_is_transient_and_retryable(
        self,
        make_provider: Callable[[_StallServer], HttpProvider],
    ) -> None:
        with _StallServer(_headers_then_stall) as server:
            provider = make_provider(server)
            evidence = _invoke_bounded(provider)

        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.failure_class is FailureClass.TRANSIENT
        assert failure.retryable is True
        # transport evidence is preserved: the raw request and the target
        # URL survive into the typed failure.
        assert failure.raw_request  # least-processed request retained
        assert failure.metadata.get("url", "").startswith("http://127.0.0.1")
        assert (
            failure.metadata.get("timeout_seconds") == POLICY_TIMEOUT_SECONDS
        )
