from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, override

import httpx
import pytest
from _concurrency import WATCHDOG_SECONDS, DaemonCall
from _policy import TEST_MAX_CONNECTIONS, make_transport_policy

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
from dr_providers.transport.http import (
    OFFLOAD_THREAD_NAME_PREFIX,
    HttpProvider,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
SHUTDOWN_FAILED_MSG = "shutdown failed"
CLOSE_WAIT_NOT_REACHED = "close did not reach the drain wait"


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m", controls=GenerationControls()),
        transcript=Transcript(messages=MESSAGES),
    )


def _policy(**overrides: Any) -> ProviderTransportPolicy:
    return make_transport_policy(base_url="https://example.test", **overrides)


def _ok_client() -> httpx.Client:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_BODY_OK)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _provider(**policy_overrides: Any) -> HttpProvider:
    client = _ok_client()
    return HttpProvider(
        policy=_policy(**policy_overrides),
        api_key="test-key",
        _client_factory=lambda **_kwargs: client,
    )


def _wait_for(event: threading.Event, description: str) -> None:
    if not event.wait(timeout=WATCHDOG_SECONDS):
        raise TimeoutError(description)


class _InterruptingCondition(threading.Condition):
    """Raise out of the drain wait taken in one named provider state.

    This stands in for a keyboard interrupt delivered to the closing
    thread, which CPython raises out of exactly this wait. Setting
    ``only_thread_ident`` restricts the interrupt to one thread so a
    test can target a specific waiter.
    """

    def __init__(self, interrupt_in_state: str) -> None:
        super().__init__()
        self._interrupt_in_state = interrupt_in_state
        self.state_name: Callable[[], str] = lambda: ""
        self.interrupted = threading.Event()
        self.only_thread_ident: int | None = None

    @override
    def wait(self, timeout: float | None = None) -> bool:
        targeted = (
            self.only_thread_ident is None
            or self.only_thread_ident == threading.get_ident()
        )
        if targeted and self.state_name() == self._interrupt_in_state:
            self.interrupted.set()
            raise KeyboardInterrupt(self._interrupt_in_state)
        return super().wait(timeout)


def _provider_interrupted_in(
    state_name: str,
) -> tuple[HttpProvider, _InterruptingCondition]:
    """Build a provider whose close aborts while waiting in ``state_name``."""
    provider = _provider()
    condition = _InterruptingCondition(state_name)
    condition.state_name = lambda: provider._state.name
    provider._condition = condition
    return provider, condition


def test_sync_only_use_never_creates_the_executor() -> None:
    provider = _provider()

    evidence = provider.invoke(_request())
    provider.close()

    assert isinstance(evidence.outcome, ProviderTransportResponse)
    assert provider._executor is None


def test_first_offload_creates_executor_sized_from_max_connections() -> None:
    provider = _provider()

    with provider:
        assert (
            provider.offload(lambda: "offloaded").result(
                timeout=WATCHDOG_SECONDS
            )
            == "offloaded"
        )
        executor = provider._executor

    assert executor is not None
    assert executor._max_workers == TEST_MAX_CONNECTIONS


def test_executor_size_follows_max_connections() -> None:
    provider = _provider(max_connections=3, max_keepalive_connections=1)

    with provider:
        provider.offload(lambda: None).result(timeout=WATCHDOG_SECONDS)
        executor = provider._executor

    assert executor is not None
    assert executor._max_workers == 3


def test_offload_reuses_one_executor() -> None:
    provider = _provider()

    with provider:
        provider.offload(lambda: None).result(timeout=WATCHDOG_SECONDS)
        first = provider._executor
        provider.offload(lambda: None).result(timeout=WATCHDOG_SECONDS)

        assert provider._executor is first


def test_failing_offloaded_work_releases_the_drain() -> None:
    provider = _provider()

    def boom() -> None:
        raise ValueError(OFFLOADED_FAILURE_MSG)

    with provider:
        future = provider.offload(boom)
        with pytest.raises(ValueError, match=OFFLOADED_FAILURE_MSG):
            future.result(timeout=WATCHDOG_SECONDS)

        assert provider._active_offloads == 0


def test_cancelling_a_queued_offload_releases_the_drain() -> None:
    started = threading.Event()
    release = threading.Event()
    provider = _provider(max_connections=1, max_keepalive_connections=1)

    def gated() -> None:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)

    running = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)
    queued = provider.offload(lambda: "never runs")

    assert queued.cancel()
    assert queued.cancelled()

    release.set()
    running.result(timeout=WATCHDOG_SECONDS)

    with provider._condition:
        drained = provider._condition.wait_for(
            lambda: provider._active_offloads == 0,
            timeout=WATCHDOG_SECONDS,
        )
    assert drained

    closer = DaemonCall.start(provider.close)
    closer.result()
    assert provider._state.name == "CLOSED"


def test_failed_submit_releases_the_drain_and_close_completes() -> None:
    provider = _provider()

    provider.offload(lambda: None).result(timeout=WATCHDOG_SECONDS)
    executor = provider._executor
    assert executor is not None
    executor.shutdown(wait=True)

    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        provider.offload(lambda: None)
    assert provider._active_offloads == 0

    closer = DaemonCall.start(provider.close)
    closer.result()
    assert provider._state.name == "CLOSED"


def test_close_shuts_down_the_executor() -> None:
    provider = _provider()

    provider.offload(lambda: None).result(timeout=WATCHDOG_SECONDS)
    executor = provider._executor
    assert executor is not None

    provider.close()

    assert executor._shutdown is True
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(OFFLOAD_THREAD_NAME_PREFIX)
        and thread.is_alive()
    ]


class _RaisingShutdownExecutor(ThreadPoolExecutor):
    """Fail shutdown so close must still reach the client."""

    @override
    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        raise RuntimeError(SHUTDOWN_FAILED_MSG)


class _CloseCountingClient(httpx.Client):
    """Count how many times the transport client is closed."""

    close_count: int = 0

    @override
    def close(self) -> None:
        self.close_count += 1
        super().close()


def test_failing_executor_shutdown_still_closes_the_client_once() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_BODY_OK)

    client = _CloseCountingClient(transport=httpx.MockTransport(handler))
    provider = HttpProvider(
        policy=_policy(),
        api_key="test-key",
        _client_factory=lambda **_kwargs: client,
    )
    provider._executor = _RaisingShutdownExecutor(max_workers=1)

    with pytest.raises(RuntimeError, match=SHUTDOWN_FAILED_MSG):
        provider.close()

    assert client.close_count == 1
    assert client.is_closed is True
    assert provider._state.name == "CLOSED"


def test_close_drains_offloaded_work_before_completing() -> None:
    started = threading.Event()
    release = threading.Event()
    provider = _provider()

    def gated() -> str:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)
        return DRAINED_OFFLOAD_RESULT

    future = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)

    closer = DaemonCall.start(provider.close)
    closer.wait_until_entered()
    with provider._condition:
        reached_draining = provider._condition.wait_for(
            lambda: provider._state.name == "DRAINING_OFFLOADS",
            timeout=WATCHDOG_SECONDS,
        )
    assert reached_draining
    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.offload(lambda: None)
    assert not closer._done.is_set()
    executor = provider._executor
    assert executor is not None

    release.set()
    assert future.result(timeout=WATCHDOG_SECONDS) == DRAINED_OFFLOAD_RESULT
    closer.result()

    assert provider._state.name == "CLOSED"
    assert executor._shutdown is True


def test_any_caller_may_invoke_while_offloaded_work_drains() -> None:
    started = threading.Event()
    release = threading.Event()
    provider = _provider()

    def gated() -> None:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)

    future = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)

    closer = DaemonCall.start(provider.close)
    closer.wait_until_entered()
    with provider._condition:
        reached_draining = provider._condition.wait_for(
            lambda: provider._state.name == "DRAINING_OFFLOADS",
            timeout=WATCHDOG_SECONDS,
        )
    assert reached_draining

    external = DaemonCall.start(lambda: provider.invoke(_request()))
    evidence = external.result()

    release.set()
    future.result(timeout=WATCHDOG_SECONDS)
    closer.result()

    assert isinstance(evidence.outcome, ProviderTransportResponse)
    assert provider._state.name == "CLOSED"


def test_draining_offload_can_still_invoke_the_provider() -> None:
    started = threading.Event()
    release = threading.Event()
    provider = _provider()

    def gated() -> Any:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)
        assert provider._client.is_closed is False
        return provider.invoke(_request())

    future = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)

    closer = DaemonCall.start(provider.close)
    closer.wait_until_entered()
    with provider._condition:
        reached_draining = provider._condition.wait_for(
            lambda: provider._state.name == "DRAINING_OFFLOADS",
            timeout=WATCHDOG_SECONDS,
        )
    assert reached_draining
    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.offload(lambda: None)

    release.set()
    evidence = future.result(timeout=WATCHDOG_SECONDS)
    closer.result()

    assert isinstance(evidence.outcome, ProviderTransportResponse)
    assert provider._state.name == "CLOSED"
    assert provider._client.is_closed is True


def test_offload_after_close_raises_and_concurrent_closers_return() -> None:
    started = threading.Event()
    release = threading.Event()
    provider = _provider()

    def gated() -> None:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)

    future = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)

    closers = [DaemonCall.start(provider.close) for _ in range(2)]
    for closer in closers:
        closer.wait_until_entered()
    with provider._condition:
        reached_draining = provider._condition.wait_for(
            lambda: provider._state.name == "DRAINING_OFFLOADS",
            timeout=WATCHDOG_SECONDS,
        )
    assert reached_draining

    release.set()
    future.result(timeout=WATCHDOG_SECONDS)
    for closer in closers:
        closer.result()

    assert provider._state.name == "CLOSED"
    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.offload(lambda: None)
    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.invoke(_request())


def _assert_aborted_close_released_everything(
    provider: HttpProvider,
    executor: ThreadPoolExecutor,
) -> None:
    """An aborted close is terminal, released, and refuses admission."""
    assert provider._state.name == "CLOSED"
    assert provider._client.is_closed is True
    assert executor._shutdown is True

    second = DaemonCall.start(provider.close)
    second.result()

    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.offload(lambda: None)
    with pytest.raises(RuntimeError, match="closing or closed"):
        provider.invoke(_request())


def test_interrupted_offload_drain_leaves_the_provider_terminal() -> None:
    started = threading.Event()
    release = threading.Event()
    provider, condition = _provider_interrupted_in("DRAINING_OFFLOADS")

    def gated() -> None:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)

    future = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)
    executor = provider._executor
    assert executor is not None

    closer = DaemonCall.start(provider.close)
    _wait_for(condition.interrupted, CLOSE_WAIT_NOT_REACHED)
    with pytest.raises(KeyboardInterrupt, match="DRAINING_OFFLOADS"):
        closer.result()

    release.set()
    future.result(timeout=WATCHDOG_SECONDS)
    _assert_aborted_close_released_everything(provider, executor)


def test_interrupted_invocation_drain_leaves_the_provider_terminal() -> None:
    started = threading.Event()
    release = threading.Event()
    provider, condition = _provider_interrupted_in("CLOSING")
    provider.offload(lambda: None).result(timeout=WATCHDOG_SECONDS)
    executor = provider._executor
    assert executor is not None

    def gated_invocation() -> Any:
        provider._begin_invocation()
        try:
            started.set()
            _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)
        finally:
            provider._end_invocation()

    invoker = DaemonCall.start(gated_invocation)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)

    closer = DaemonCall.start(provider.close)
    _wait_for(condition.interrupted, CLOSE_WAIT_NOT_REACHED)
    with pytest.raises(KeyboardInterrupt, match="CLOSING"):
        closer.result()

    release.set()
    invoker.result()
    _assert_aborted_close_released_everything(provider, executor)


NO_THREAD_IDENT = 0
SECONDARY_CLOSE_NOT_REACHED = "secondary closer did not reach its wait"


def test_interrupted_secondary_closer_leaves_the_primary_close_intact() -> (
    None
):
    started = threading.Event()
    release = threading.Event()
    provider, condition = _provider_interrupted_in("DRAINING_OFFLOADS")
    condition.only_thread_ident = NO_THREAD_IDENT

    def gated() -> None:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)

    future = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)

    primary = DaemonCall.start(provider.close)
    with condition:
        assert condition.wait_for(
            lambda: provider._state.name == "DRAINING_OFFLOADS",
            timeout=WATCHDOG_SECONDS,
        ), CLOSE_WAIT_NOT_REACHED

    def secondary_close() -> None:
        condition.only_thread_ident = threading.get_ident()
        provider.close()

    secondary = DaemonCall.start(secondary_close)
    _wait_for(condition.interrupted, SECONDARY_CLOSE_NOT_REACHED)
    with pytest.raises(KeyboardInterrupt, match="DRAINING_OFFLOADS"):
        secondary.result()

    assert provider._client.is_closed is False
    assert provider._state.name == "DRAINING_OFFLOADS"

    release.set()
    future.result(timeout=WATCHDOG_SECONDS)
    primary.result()
    assert provider._state.name == "CLOSED"
    assert provider._client.is_closed is True


CLIENT_CLOSE_FAILED_MSG = "client close failed"


class _RaisingCloseClient(httpx.Client):
    @override
    def close(self) -> None:
        raise RuntimeError(CLIENT_CLOSE_FAILED_MSG)


def test_abort_close_release_errors_do_not_mask_the_interrupt() -> None:
    started = threading.Event()
    release = threading.Event()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_BODY_OK)

    client = _RaisingCloseClient(transport=httpx.MockTransport(handler))
    provider = HttpProvider(
        policy=_policy(),
        api_key="test-key",
        _client_factory=lambda **_kwargs: client,
    )
    condition = _InterruptingCondition("DRAINING_OFFLOADS")
    condition.state_name = lambda: provider._state.name
    provider._condition = condition

    def gated() -> None:
        started.set()
        _wait_for(release, OFFLOADED_WORK_NOT_RELEASED)

    future = provider.offload(gated)
    _wait_for(started, OFFLOADED_WORK_DID_NOT_START)
    executor = provider._executor
    assert executor is not None

    closer = DaemonCall.start(provider.close)
    _wait_for(condition.interrupted, CLOSE_WAIT_NOT_REACHED)
    with pytest.raises(KeyboardInterrupt, match="DRAINING_OFFLOADS"):
        closer.result()

    assert provider._state.name == "CLOSED"
    assert executor._shutdown is True

    release.set()
    future.result(timeout=WATCHDOG_SECONDS)
