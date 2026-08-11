from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

import pytest

from dr_providers import (
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportFailure,
    RecoverabilityClass,
    ScriptedOutcome,
    ScriptedProvider,
    Transcript,
    openai_chat_config,
)
from dr_providers.lifecycle import (
    ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER,
    AcceptAllSemanticResponseClassifier,
    ProviderCallOutcomeKind,
    ProviderCallResult,
    ProviderCallState,
    StandardProviderCallRetryPolicy,
    run_local_provider_call,
    run_local_provider_call_async,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dr_providers.outcomes.evidence import ProviderInvocationEvidence

WATCHDOG_SECONDS = 5.0


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openai_chat_config(model="m"),
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


class _OffloadingScriptedProvider:
    """Give a scripted provider the offload half of the async surface."""

    def __init__(self, outcomes: list[ScriptedOutcome] | None = None) -> None:
        self.scripted = ScriptedProvider(outcomes)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        return self.scripted.invoke(request)

    def offload[ResultT](self, fn: Callable[[], ResultT]) -> Future[ResultT]:
        return self._executor.submit(fn)

    @property
    def executor_thread_ident(self) -> int:
        """Identify the single worker thread backing this offload seam."""
        return self._executor.submit(threading.get_ident).result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def test_async_entry_matches_sync_driver_for_one_scripted_exchange() -> None:
    classifier = AcceptAllSemanticResponseClassifier()
    outcomes = [ScriptedOutcome(text="accepted")]

    sync_result = run_local_provider_call(
        provider=ScriptedProvider(list(outcomes)),
        state=_state(),
        classifier=classifier,
        cancellation=Event(),
    )

    provider = _OffloadingScriptedProvider(list(outcomes))
    try:
        async_result = asyncio.run(
            run_local_provider_call_async(
                provider=provider,
                state=_state(),
                classifier=classifier,
                cancellation=Event(),
            )
        )
    finally:
        provider.shutdown()

    assert async_result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    assert async_result.identity_hash == sync_result.identity_hash
    assert len(provider.scripted.requests) == 1


def test_async_entry_runs_the_driver_off_the_awaiting_thread() -> None:
    class _ThreadRecordingProvider(_OffloadingScriptedProvider):
        def __init__(self) -> None:
            super().__init__()
            self.invoked_on: list[int] = []

        def invoke(
            self, request: ProviderCallRequest
        ) -> ProviderInvocationEvidence:
            self.invoked_on.append(threading.get_ident())
            return super().invoke(request)

    provider = _ThreadRecordingProvider()
    worker_thread = provider.executor_thread_ident

    async def drive() -> int:
        await run_local_provider_call_async(
            provider=provider,
            state=_state(),
            classifier=AcceptAllSemanticResponseClassifier(),
            cancellation=Event(),
        )
        return threading.get_ident()

    try:
        awaiting_thread = asyncio.run(drive())
    finally:
        provider.shutdown()

    assert provider.invoked_on == [worker_thread]
    assert worker_thread != awaiting_thread


def test_async_entry_follows_retry_instructions_like_the_sync_driver() -> None:
    from _retry_fixtures import two_invocation_transient_retry_policy

    class _RecordingWait:
        def __init__(self) -> None:
            self.delays: list[float] = []

        def wait(self, delay_seconds: float, cancellation: Event) -> None:
            assert not cancellation.is_set()
            self.delays.append(delay_seconds)

    outcomes = [
        ScriptedOutcome(
            failure=ProviderTransportFailure(
                recoverability=RecoverabilityClass.TRANSIENT,
                message="try again",
            )
        ),
        ScriptedOutcome(text="accepted"),
    ]
    state = ProviderCallState.initial(
        request=_request(),
        retry_policy=two_invocation_transient_retry_policy(),
        classifier_identifier=ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER,
    )
    provider = _OffloadingScriptedProvider(outcomes)
    wait = _RecordingWait()

    try:
        result = asyncio.run(
            run_local_provider_call_async(
                provider=provider,
                state=state,
                classifier=AcceptAllSemanticResponseClassifier(),
                cancellation=Event(),
                retry_wait=wait,
            )
        )
    finally:
        provider.shutdown()

    assert result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    assert wait.delays == [1.0]
    assert len(provider.scripted.requests) == 2


def test_async_entry_does_not_block_the_event_loop() -> None:
    offload_started = Event()
    loop_ran = Event()

    class _LoopGatedProvider(_OffloadingScriptedProvider):
        def invoke(
            self, request: ProviderCallRequest
        ) -> ProviderInvocationEvidence:
            offload_started.set()
            if not loop_ran.wait(WATCHDOG_SECONDS):
                raise TimeoutError("event loop never progressed")
            return super().invoke(request)

    provider = _LoopGatedProvider()

    async def drive() -> ProviderCallResult:
        async def unblock() -> None:
            await asyncio.to_thread(offload_started.wait, WATCHDOG_SECONDS)
            loop_ran.set()

        task = asyncio.create_task(
            run_local_provider_call_async(
                provider=provider,
                state=_state(),
                classifier=AcceptAllSemanticResponseClassifier(),
                cancellation=Event(),
            )
        )
        await unblock()
        return await task

    try:
        result = asyncio.run(drive())
    finally:
        provider.shutdown()

    assert loop_ran.is_set()
    assert result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED


def test_cancelling_the_task_does_not_interrupt_a_running_offload() -> None:
    started = Event()
    release = Event()
    completed = Event()

    class _GatedProvider(_OffloadingScriptedProvider):
        def invoke(
            self, request: ProviderCallRequest
        ) -> ProviderInvocationEvidence:
            started.set()
            if not release.wait(WATCHDOG_SECONDS):
                raise TimeoutError("offloaded call was not released")
            evidence = super().invoke(request)
            completed.set()
            return evidence

    provider = _GatedProvider()

    async def drive() -> None:
        task = asyncio.create_task(
            run_local_provider_call_async(
                provider=provider,
                state=_state(),
                classifier=AcceptAllSemanticResponseClassifier(),
                cancellation=Event(),
            )
        )
        await asyncio.to_thread(started.wait, WATCHDOG_SECONDS)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(drive())
        release.set()
        assert completed.wait(WATCHDOG_SECONDS)
    finally:
        release.set()
        provider.shutdown()

    assert len(provider.scripted.requests) == 1


def test_cancelled_await_reports_an_orphaned_offload_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A driver defect after cancellation is reported, not swallowed.

    Nobody awaits the shielded future once the task is cancelled, so the
    offloaded exception would otherwise reach no output at all.
    """
    started = Event()
    release = Event()
    failure = RuntimeError("systematic driver defect")

    class _RaisingProvider(_OffloadingScriptedProvider):
        def __init__(self) -> None:
            super().__init__()
            self.offloaded_exceptions: list[
                Callable[[float], BaseException | None]
            ] = []

        def invoke(
            self, request: ProviderCallRequest
        ) -> ProviderInvocationEvidence:
            del request
            started.set()
            if not release.wait(WATCHDOG_SECONDS):
                raise TimeoutError("offloaded call was not released")
            raise failure

        def offload[ResultT](
            self, fn: Callable[[], ResultT]
        ) -> Future[ResultT]:
            future = super().offload(fn)
            self.offloaded_exceptions.append(
                lambda timeout: future.exception(timeout=timeout)
            )
            return future

    provider = _RaisingProvider()

    async def drive() -> None:
        task = asyncio.create_task(
            run_local_provider_call_async(
                provider=provider,
                state=_state(),
                classifier=AcceptAllSemanticResponseClassifier(),
                cancellation=Event(),
            )
        )
        await asyncio.to_thread(started.wait, WATCHDOG_SECONDS)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        with caplog.at_level(logging.ERROR, logger="dr_providers"):
            asyncio.run(drive())
            release.set()
            # Synchronize on the offloaded call reaching its terminal
            # state, then on the drain, never on elapsed time.
            assert len(provider.offloaded_exceptions) == 1
            assert isinstance(
                provider.offloaded_exceptions[0](WATCHDOG_SECONDS),
                RuntimeError,
            )
            provider.shutdown()
    finally:
        release.set()
        provider.shutdown()

    errors = [
        record for record in caplog.records if record.levelno == logging.ERROR
    ]
    assert len(errors) == 1
    assert errors[0].name == "dr_providers.lifecycle.driver"
    assert errors[0].exc_info is not None
    assert errors[0].exc_info[1] is failure


def test_async_entry_returns_the_sync_cancelled_result_shape() -> None:
    classifier = AcceptAllSemanticResponseClassifier()
    cancellation = Event()
    cancellation.set()

    sync_result = run_local_provider_call(
        provider=ScriptedProvider(),
        state=_state(),
        classifier=classifier,
        cancellation=cancellation,
    )

    provider = _OffloadingScriptedProvider()
    try:
        async_result = asyncio.run(
            run_local_provider_call_async(
                provider=provider,
                state=_state(),
                classifier=classifier,
                cancellation=cancellation,
            )
        )
    finally:
        provider.shutdown()

    assert async_result.outcome.kind is (
        ProviderCallOutcomeKind.DRAINING_CANCELLATION
    )
    assert async_result.completed_invocations == ()
    assert provider.scripted.requests == []
    assert async_result.identity_hash == sync_result.identity_hash
