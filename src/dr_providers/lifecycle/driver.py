from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import TIMEOUT_MAX
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
    from collections.abc import Callable
    from concurrent.futures import Future
    from threading import Event

    from dr_providers.core.provider import Provider
    from dr_providers.modeling.request import ProviderCallRequest
    from dr_providers.outcomes.evidence import ProviderInvocationEvidence


class ProviderRetryWait(Protocol):
    """Controlled boundary for one reducer-declared retry delay."""

    def wait(self, delay_seconds: float, cancellation: Event) -> None: ...


class OffloadingProvider(Protocol):
    """A provider that also runs caller work on its own executor."""

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence: ...

    def offload[ResultT](
        self, fn: Callable[[], ResultT]
    ) -> Future[ResultT]: ...


@dataclass(frozen=True, slots=True)
class EventProviderRetryWait:
    """Production retry wait interruptible by a cancellation event."""

    def wait(self, delay_seconds: float, cancellation: Event) -> None:
        full_chunk_count, final_delay_seconds = divmod(
            delay_seconds, TIMEOUT_MAX
        )
        for _ in range(int(full_chunk_count)):
            if cancellation.wait(TIMEOUT_MAX):
                return
        if final_delay_seconds > 0 or full_chunk_count == 0:
            cancellation.wait(final_delay_seconds)


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


async def run_local_provider_call_async(
    *,
    provider: OffloadingProvider,
    state: ProviderCallState,
    classifier: SemanticResponseClassifier,
    cancellation: Event,
    retry_wait: ProviderRetryWait | None = None,
) -> ProviderCallResult:
    """Await one local provider call offloaded onto the provider's executor.

    The synchronous driver runs unchanged on a provider-owned thread.
    Cancelling the awaiting asyncio task does not interrupt the offloaded
    call: the offloaded future is shielded, so cancellation flows through
    the cancellation event, and admitted offloaded work always drains
    through the provider's close.
    """
    future = provider.offload(
        lambda: run_local_provider_call(
            provider=provider,
            state=state,
            classifier=classifier,
            cancellation=cancellation,
            retry_wait=retry_wait,
        )
    )
    return await asyncio.shield(asyncio.wrap_future(future))
