from __future__ import annotations

from dr_providers.lifecycle.models import (
    CompletedProviderInvocationObservation,
    DecidedProviderInvocationRecord,
    ProviderCallResult,
    ProviderCallState,
    ProviderRetryDecision,
    ProviderRetryInstruction,
)
from dr_providers.lifecycle.outcomes import (
    ProviderCallOutcome,
    ProviderCallOutcomeKind,
    ProviderInvocationOutcome,
)


def transition_provider_call(
    state: ProviderCallState,
    observation: CompletedProviderInvocationObservation,
) -> ProviderRetryInstruction | ProviderCallResult:
    """Decide one completed invocation and advance the provider call."""
    _validate_observation_for_state(state, observation)
    outcome = observation.outcome
    if outcome is ProviderInvocationOutcome.SUCCESS:
        return _terminal_result(
            state,
            DecidedProviderInvocationRecord(observation=observation),
            ProviderCallOutcome(
                kind=ProviderCallOutcomeKind.ACCEPTED,
                invocation_outcome=outcome,
            ),
        )

    is_eligible = outcome in state.retry_policy.eligible_outcomes
    has_capacity = (
        observation.invocation_ordinal < state.retry_policy.maximum_invocations
    )
    if is_eligible and has_capacity:
        delay = state.retry_policy.retry_delay_after(
            observation.invocation_ordinal
        )
        record = DecidedProviderInvocationRecord(
            observation=observation,
            retry_decision=ProviderRetryDecision(delay_seconds=delay),
        )
        records = (*state.completed_invocations, record)
        record_hashes = (
            *state.completed_invocation_record_hashes,
            record.identity_hash,
        )
        next_state = ProviderCallState(
            request=state.request,
            request_identity_hash=state.request_identity_hash,
            retry_policy=state.retry_policy,
            retry_policy_identity_hash=state.retry_policy_identity_hash,
            classifier_identifier=state.classifier_identifier,
            call_identity_hash=state.call_identity_hash,
            completed_invocations=records,
            completed_invocation_record_hashes=record_hashes,
            next_invocation_ordinal=observation.invocation_ordinal + 1,
        )
        return ProviderRetryInstruction(
            delay_seconds=delay,
            next_invocation_ordinal=next_state.next_invocation_ordinal,
            next_state=next_state,
        )

    terminal_kind = (
        ProviderCallOutcomeKind.POLICY_EXHAUSTION
        if is_eligible
        else ProviderCallOutcomeKind.INVOCATION_OUTCOME
    )
    return _terminal_result(
        state,
        DecidedProviderInvocationRecord(observation=observation),
        ProviderCallOutcome(
            kind=terminal_kind,
            invocation_outcome=outcome,
        ),
    )


def cancel_provider_call(
    state: ProviderCallState,
    observation: CompletedProviderInvocationObservation | None = None,
) -> ProviderCallResult:
    """Terminalize cancellation without serializing runtime state."""
    if observation is None:
        records = state.completed_invocations
        record_hashes = state.completed_invocation_record_hashes
    else:
        _validate_observation_for_state(state, observation)
        record = DecidedProviderInvocationRecord(observation=observation)
        records = (*state.completed_invocations, record)
        record_hashes = (
            *state.completed_invocation_record_hashes,
            record.identity_hash,
        )
    return ProviderCallResult(
        request=state.request,
        request_identity_hash=state.request_identity_hash,
        retry_policy=state.retry_policy,
        retry_policy_identity_hash=state.retry_policy_identity_hash,
        classifier_identifier=state.classifier_identifier,
        call_identity_hash=state.call_identity_hash,
        completed_invocations=records,
        completed_invocation_record_hashes=record_hashes,
        outcome=ProviderCallOutcome(
            kind=ProviderCallOutcomeKind.DRAINING_CANCELLATION
        ),
    )


def _validate_observation_for_state(
    state: ProviderCallState,
    observation: CompletedProviderInvocationObservation,
) -> None:
    if observation.invocation_ordinal != state.next_invocation_ordinal:
        msg = "completed observation ordinal does not match next invocation"
        raise ValueError(msg)
    if observation.request_identity_hash != state.request_identity_hash:
        msg = "completed observation request does not match provider call"
        raise ValueError(msg)
    if (
        observation.evidence.request_identity_hash
        != state.request_identity_hash
    ):
        msg = "completed evidence request does not match provider call"
        raise ValueError(msg)


def _terminal_result(
    state: ProviderCallState,
    record: DecidedProviderInvocationRecord,
    outcome: ProviderCallOutcome,
) -> ProviderCallResult:
    records = (*state.completed_invocations, record)
    record_hashes = (
        *state.completed_invocation_record_hashes,
        record.identity_hash,
    )
    return ProviderCallResult(
        request=state.request,
        request_identity_hash=state.request_identity_hash,
        retry_policy=state.retry_policy,
        retry_policy_identity_hash=state.retry_policy_identity_hash,
        classifier_identifier=state.classifier_identifier,
        call_identity_hash=state.call_identity_hash,
        completed_invocations=records,
        completed_invocation_record_hashes=record_hashes,
        outcome=outcome,
    )
