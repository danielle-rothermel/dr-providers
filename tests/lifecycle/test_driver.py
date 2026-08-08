from __future__ import annotations

from threading import Event, Thread

from dr_providers import (
    FailureClass,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportFailure,
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
)

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


class _RecordingWait:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def wait(self, delay_seconds: float, cancellation: Event) -> None:
        assert not cancellation.is_set()
        self.delays.append(delay_seconds)


def test_driver_follows_reducer_retry_instruction() -> None:
    provider = ScriptedProvider(
        [
            ScriptedOutcome(
                failure=ProviderTransportFailure(
                    failure_class=FailureClass.TRANSIENT,
                    message="try again",
                )
            ),
            ScriptedOutcome(text="accepted"),
        ]
    )
    wait = _RecordingWait()

    result = run_local_provider_call(
        provider=provider,
        state=_state(),
        classifier=AcceptAllSemanticResponseClassifier(),
        cancellation=Event(),
        retry_wait=wait,
    )

    assert result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    assert len(result.completed_invocations) == 2
    assert wait.delays == [1.0]
    assert len(provider.requests) == 2


def test_cancellation_before_invocation_starts_no_work() -> None:
    provider = ScriptedProvider()
    cancellation = Event()
    cancellation.set()

    result = run_local_provider_call(
        provider=provider,
        state=_state(),
        classifier=AcceptAllSemanticResponseClassifier(),
        cancellation=cancellation,
    )

    assert result.outcome.kind is ProviderCallOutcomeKind.DRAINING_CANCELLATION
    assert result.completed_invocations == ()
    assert provider.requests == []


def test_cancellation_interrupts_controlled_retry_wait() -> None:
    entered_wait = Event()
    cancellation = Event()
    results: list[ProviderCallResult] = []

    class GateWait:
        def wait(self, delay_seconds: float, cancellation: Event) -> None:
            assert delay_seconds == 1.0
            assert cancellation is cancellation_signal
            entered_wait.set()
            assert cancellation.wait(WATCHDOG_SECONDS)

    cancellation_signal = cancellation
    provider = ScriptedProvider(
        [
            ScriptedOutcome(
                failure=ProviderTransportFailure(
                    failure_class=FailureClass.TRANSIENT,
                    message="try again",
                )
            ),
            ScriptedOutcome(text="must not run"),
        ]
    )
    thread = Thread(
        target=lambda: results.append(
            run_local_provider_call(
                provider=provider,
                state=_state(),
                classifier=AcceptAllSemanticResponseClassifier(),
                cancellation=cancellation,
                retry_wait=GateWait(),
            )
        ),
        daemon=True,
    )
    thread.start()
    assert entered_wait.wait(WATCHDOG_SECONDS)
    cancellation.set()
    thread.join(WATCHDOG_SECONDS)

    assert not thread.is_alive()
    assert results[0].outcome.kind is (
        ProviderCallOutcomeKind.DRAINING_CANCELLATION
    )
    assert len(results[0].completed_invocations) == 1
    assert len(provider.requests) == 1


def test_cancellation_after_active_invocation_records_observation() -> None:
    started = Event()
    release = Event()
    cancellation = Event()
    results: list[ProviderCallResult] = []

    class GateProvider:
        def __init__(self) -> None:
            self.scripted = ScriptedProvider([ScriptedOutcome(text="paid")])

        def invoke(self, request: ProviderCallRequest):
            started.set()
            assert release.wait(WATCHDOG_SECONDS)
            return self.scripted.invoke(request)

    provider = GateProvider()
    thread = Thread(
        target=lambda: results.append(
            run_local_provider_call(
                provider=provider,
                state=_state(),
                classifier=AcceptAllSemanticResponseClassifier(),
                cancellation=cancellation,
            )
        ),
        daemon=True,
    )
    thread.start()
    assert started.wait(WATCHDOG_SECONDS)
    cancellation.set()
    release.set()
    thread.join(WATCHDOG_SECONDS)

    assert not thread.is_alive()
    assert results[0].outcome.kind is (
        ProviderCallOutcomeKind.DRAINING_CANCELLATION
    )
    assert len(results[0].completed_invocations) == 1
    assert len(provider.scripted.requests) == 1
