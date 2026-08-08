from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dr_providers.lifecycle.classifier import (
    SemanticResponseClassifier,
    classify_provider_invocation,
)
from dr_providers.lifecycle.models import (
    CompletedProviderInvocationObservation,
    ProviderCallResult,
    ProviderCallState,
    ProviderRetryInstruction,
)
from dr_providers.lifecycle.reducer import (
    cancel_provider_call,
    transition_provider_call,
)

if TYPE_CHECKING:
    from threading import Event

    from dr_providers.core.provider import Provider


class ProviderRetryWait(Protocol):
    """Controlled boundary for one reducer-declared retry delay."""

    def wait(self, delay_seconds: float, cancellation: Event) -> None: ...


@dataclass(frozen=True, slots=True)
class EventProviderRetryWait:
    """Production retry wait interruptible by a cancellation event."""

    def wait(self, delay_seconds: float, cancellation: Event) -> None:
        cancellation.wait(delay_seconds)


def run_local_provider_call(
    *,
    provider: Provider,
    state: ProviderCallState,
    classifier: SemanticResponseClassifier,
    cancellation: Event,
    retry_wait: ProviderRetryWait | None = None,
) -> ProviderCallResult:
    """Drive one provider call by following reducer transition outputs."""
    if classifier.identifier != state.classifier_identifier:
        msg = "semantic classifier identifier does not match provider call"
        raise ValueError(msg)
    wait = retry_wait or EventProviderRetryWait()
    current_state = state
    while True:
        if cancellation.is_set():
            return cancel_provider_call(current_state)

        evidence = provider.invoke(current_state.request)
        outcome = classify_provider_invocation(evidence, classifier)
        observation = CompletedProviderInvocationObservation(
            invocation_ordinal=current_state.next_invocation_ordinal,
            request_identity_hash=current_state.request_identity_hash,
            evidence=evidence,
            evidence_identity_hash=evidence.identity_hash,
            outcome=outcome,
        )

        if cancellation.is_set():
            return cancel_provider_call(current_state, observation)

        transition = transition_provider_call(current_state, observation)
        if isinstance(transition, ProviderCallResult):
            return transition
        assert isinstance(transition, ProviderRetryInstruction)
        wait.wait(transition.delay_seconds, cancellation)
        current_state = transition.next_state
        if cancellation.is_set():
            return cancel_provider_call(current_state)
