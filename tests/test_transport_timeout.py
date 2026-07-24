"""Live-fire timeout tests against local stall servers.

Proves the transport enforces the policy ``timeout_seconds`` as a
wall-clock bound: a stalled response (headers sent then bytes stop, or a
socket that dribbles bytes indefinitely, or a connection that never
completes accept/read) returns the typed Provider Transport Failure within
roughly the policy timeout plus its fixed margin — and never hangs.

Idle-stall cases set a tiny explicit ``idle_timeout_seconds`` so they fire on
the progress/idle timer and finish in well under a second. The forever-dribble
cases are the exception: they deliberately defeat the idle timer (a byte every
0.05s) so ONLY the absolute per-invocation cap can stop them, and that cap is
``timeout_seconds`` + a FIXED 5s margin, so each such case necessarily pays at
least ~5s of wall-clock even with ``timeout_seconds`` minimized. Every
``invoke``/``complete`` is additionally wrapped in an independent wall-clock
watchdog so a regression that reintroduces the hang fails loudly instead of
wedging the whole test run.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

import httpx
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
    ProviderTransportResponse,
    Transcript,
    openai_chat_config,
)
from dr_providers.transport import (
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
    HttpProvider,
)

# Tiny budgets so the suite stays fast. Two distinct timers are exercised:
#   * the IDLE/progress timeout — the primary stall detector — fires promptly
#     for a no-progress response, so idle-stall tests set a tiny explicit
#     ``idle_timeout_seconds`` and finish in well under a second;
#   * the absolute per-invocation cap (``timeout_seconds`` +
#     ``INVOCATION_DEADLINE_MARGIN_SECONDS``, a 5s fixed margin) only bounds a
#     forever-dribble that defeats the idle timer, so those two cases must pay
#     at least the fixed margin — ``timeout_seconds`` is kept minimal there.
# Every ``invoke``/``complete`` is additionally wrapped in a finite external
# watchdog so a real hang is caught loudly instead of wedging the run.
POLICY_TIMEOUT_SECONDS = 0.3
POLICY_IDLE_SECONDS = 0.2
EXTERNAL_WATCHDOG_SECONDS = 20.0
MESSAGES = (PromptMessage(role=MessageRole.USER, content="hi"),)


def _stall_policy() -> ProviderTransportPolicy:
    # Explicit tiny idle timeout so the idle-stall cases fire on the idle timer
    # (not the far-off absolute cap) — deterministic and fast.
    return ProviderTransportPolicy(
        api_key_env=str(ApiKeyEnv.OPENAI),
        base_url="http://placeholder",  # overridden per test with the port
        timeout_seconds=POLICY_TIMEOUT_SECONDS,
        idle_timeout_seconds=POLICY_IDLE_SECONDS,
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


def _steady_stream_then_complete(
    conn: socket.socket, stop: threading.Event
) -> None:
    """A LEGITIMATE long stream: deliver body bytes steadily, then complete.

    Every chunk arrives well within the idle window (never a silent gap longer
    than the idle timeout), so a progress/idle bound must let it finish. Stands
    in for a reasoning model streaming many tokens over a long wall-clock: it
    runs longer than the idle timeout but is always making progress, so it must
    NOT be killed. The total body is small so the test is fast.
    """
    conn.recv(65536)
    body = b'{"choices":[{"message":{"content":"ok"}}]}'
    header = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"\r\n"
    )
    conn.sendall(header)
    # Dribble the body one byte at a time, each within the idle window, for a
    # wall-clock LONGER than the idle timeout but always making progress. A
    # small gap keeps the test fast while still exceeding the idle window.
    for byte in (body[i : i + 1] for i in range(len(body))):
        if stop.is_set():
            return
        with contextlib.suppress(OSError):
            conn.sendall(byte)
        stop.wait(timeout=0.03)


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


def _idle_cap_policy(
    *, idle_timeout_seconds: float, timeout_seconds: float
) -> ProviderTransportPolicy:
    return ProviderTransportPolicy(
        api_key_env=str(ApiKeyEnv.OPENAI),
        base_url="http://placeholder",
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
    )


class TestProgressIdleSemantics:
    """FIX 3: idle/progress timeout is the stall detector, cap is the backstop.

    A LEGITIMATE long stream making steady progress must complete (never capped
    for running long); a genuinely IDLE response fails ``stalled_response`` on
    the idle timeout; a forever-dribble is still caught by the absolute cap.
    """

    def test_steady_progress_stream_is_not_killed_by_running_long(
        self,
    ) -> None:
        # Idle window 0.2s, absolute cap 5.0s. The server dribbles a small body
        # one byte every 0.03s (well within the idle window) over a wall-clock
        # that exceeds the idle timeout — a stream that keeps making progress
        # must SUCCEED, proving idle != flat-deadline. The cap is set well
        # above the stream's total duration so the stream is never cap-killed.
        policy = _idle_cap_policy(
            idle_timeout_seconds=0.2, timeout_seconds=5.0
        )
        with _StallServer(_steady_stream_then_complete) as server:
            provider = HttpProvider(
                policy=policy.model_copy(update={"base_url": server.base_url}),
                api_key="test-key",
            )
            try:
                evidence = _invoke_bounded(provider)
            finally:
                provider.close()
        # It completed with a real response, not a timeout/stall failure.
        assert evidence.failure is None, (
            "a steadily-progressing stream must not be killed for running "
            f"past the idle timeout; got {evidence.failure!r}"
        )

    def test_idle_stall_fails_stalled_response_on_the_idle_timeout(
        self,
    ) -> None:
        # Idle 0.4s, cap 20s: a headers-then-silent response has NO progress,
        # so the IDLE timer (not the far-off cap) fails it promptly as a stall.
        policy = _idle_cap_policy(
            idle_timeout_seconds=0.4, timeout_seconds=20.0
        )
        with _StallServer(_headers_then_stall) as server:
            provider = HttpProvider(
                policy=policy.model_copy(update={"base_url": server.base_url}),
                api_key="test-key",
            )
            start = time.monotonic()
            try:
                evidence = _invoke_bounded(provider)
            finally:
                provider.close()
            elapsed = time.monotonic() - start
        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.code == STALLED_RESPONSE_CODE
        # Fired on the IDLE timeout (~0.4s), FAR before the 20s absolute cap.
        assert elapsed < 5.0

    def test_forever_dribble_is_caught_by_the_absolute_cap(self) -> None:
        # A dribble sends one byte per 0.05s, defeating a naive no-NEW-bytes
        # idle timer (idle 0.3s here never trips). Only the absolute cap
        # (timeout_seconds 0.3 + fixed margin) stops it — bounded, never hangs.
        policy = _idle_cap_policy(
            idle_timeout_seconds=0.3, timeout_seconds=0.3
        )
        with _StallServer(_dribble_forever) as server:
            provider = HttpProvider(
                policy=policy.model_copy(update={"base_url": server.base_url}),
                api_key="test-key",
            )
            start = time.monotonic()
            try:
                evidence = _invoke_bounded(provider)
            finally:
                provider.close()
            elapsed = time.monotonic() - start
        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}
        # Bounded by the absolute cap (0.3 + 5s margin) + slack, not unbounded.
        assert elapsed < 0.3 + 8.0


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


_HEALTHY_BODY = (
    b'{"id":"c1","model":"m","choices":[{"message":'
    b'{"role":"assistant","content":"ok"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":1,"completion_tokens":1}}'
)


def _healthy_response(conn: socket.socket, _stop: threading.Event) -> None:
    """Read the request and return a complete valid chat-completions body."""
    conn.recv(65536)
    conn.sendall(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(_HEALTHY_BODY)}\r\n".encode()
        + b"\r\n"
        + _HEALTHY_BODY
    )


class _ConcurrentServer:
    """A localhost server that handles each connection on its own thread.

    Unlike ``_StallServer`` (which serializes connections), this lets a stalled
    connection and a healthy connection be in flight at the same time — needed
    to prove a deadline breach on one call does not disturb a concurrent call.
    The per-connection handler is chosen by ``select``.
    """

    def __init__(
        self,
        select: Callable[
            [int], Callable[[socket.socket, threading.Event], None]
        ],
    ) -> None:
        self._select = select
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._count = 0
        self._count_lock = threading.Lock()
        self._conn_threads: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            with self._count_lock:
                index = self._count
                self._count += 1
            handler = self._select(index)
            worker = threading.Thread(
                target=self._handle, args=(conn, handler), daemon=True
            )
            worker.start()
            self._conn_threads.append(worker)

    def _handle(
        self,
        conn: socket.socket,
        handler: Callable[[socket.socket, threading.Event], None],
    ) -> None:
        try:
            handler(conn, self._stop)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> _ConcurrentServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._sock.close()
        self._thread.join(timeout=2.0)


class TestClientLifecycleIsolation:
    """A deadline breach must isolate to the timed-out call (owned client).

    The redesign gives each owned wire call its OWN httpx client, so a deadline
    breach closes only that call's client and can never tear down the pool of a
    concurrent healthy call on the same provider.
    """

    def test_deadline_breach_does_not_fail_concurrent_healthy_call(
        self,
    ) -> None:
        # Connection 0 stalls (breaches the deadline); connection 1 is healthy.
        def select(
            index: int,
        ) -> Callable[[socket.socket, threading.Event], None]:
            return _headers_then_stall if index == 0 else _healthy_response

        policy = _idle_cap_policy(
            idle_timeout_seconds=0.2, timeout_seconds=0.3
        )
        with _ConcurrentServer(select) as server:
            provider = HttpProvider(
                policy=policy.model_copy(update={"base_url": server.base_url}),
                api_key="test-key",
            )
            try:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=2
                ) as pool:
                    stall_future = pool.submit(provider.complete, _request())
                    # Give the stall its own connection first, then healthy.
                    time.sleep(0.05)
                    healthy_future = pool.submit(provider.complete, _request())
                    stall_outcome = stall_future.result(
                        timeout=EXTERNAL_WATCHDOG_SECONDS
                    )
                    healthy_outcome = healthy_future.result(
                        timeout=EXTERNAL_WATCHDOG_SECONDS
                    )
            finally:
                provider.close()

        # The stalled call returned a typed failure; the concurrent healthy
        # call SUCCEEDED — it was not corrupted by the stalled call's client
        # teardown.
        assert isinstance(stall_outcome, ProviderTransportFailure)
        assert stall_outcome.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}
        assert isinstance(healthy_outcome, ProviderTransportResponse)
        assert healthy_outcome.text == "ok"

    def test_no_client_leak_on_concurrent_first_calls(self) -> None:
        # Concurrent first calls on a fresh owned provider each use their own
        # per-call client; all must succeed and the provider must accumulate
        # no shared client state.
        with _ConcurrentServer(lambda _index: _healthy_response) as server:
            provider = HttpProvider(
                policy=_stall_policy().model_copy(
                    update={"base_url": server.base_url}
                ),
                api_key="test-key",
            )
            try:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=4
                ) as pool:
                    futures = [
                        pool.submit(provider.complete, _request())
                        for _ in range(4)
                    ]
                    outcomes = [
                        f.result(timeout=EXTERNAL_WATCHDOG_SECONDS)
                        for f in futures
                    ]
            finally:
                provider.close()
        assert all(isinstance(o, ProviderTransportResponse) for o in outcomes)
        # An owned provider holds no shared client at any point.
        assert provider._client is None

    def test_injected_client_timeout_returns_typed_outcome_on_schedule(
        self,
    ) -> None:
        # With a caller-owned (injected) client a deadline breach cannot force
        # the wedged worker to unblock, but the deadline must STILL return the
        # typed outcome on schedule and never hang.
        with _StallServer(_headers_then_stall) as server:
            client = httpx.Client()
            policy = _idle_cap_policy(
                idle_timeout_seconds=30.0, timeout_seconds=0.3
            )
            provider = HttpProvider(
                policy=policy.model_copy(update={"base_url": server.base_url}),
                client=client,
                api_key="test-key",
            )
            start = time.monotonic()
            outcome = provider.complete(_request())
            elapsed = time.monotonic() - start
            # the transport must NOT close a client it does not own; the
            # deadline breach cannot forcibly unblock the wedged worker, but
            # the caller's client is left open for the caller to manage.
            client_open_after = not client.is_closed
            client.close()
        # idle is 30s (won't fire); the absolute cap (0.3 + 5s margin) returns
        # the typed failure on schedule without hanging.
        assert isinstance(outcome, ProviderTransportFailure)
        assert outcome.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}
        assert elapsed < 0.3 + 8.0
        assert client_open_after
