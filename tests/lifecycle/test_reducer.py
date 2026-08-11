from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dr_providers import (
    FailureClass,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderHttpRequestEvidence,
    ProviderInvocationEvidence,
    ProviderTransportFailure,
    ProviderTransportResponse,
    Transcript,
    openai_chat_config,
)
from dr_providers.lifecycle import (
    CompletedProviderInvocationObservation,
    DecidedProviderInvocationRecord,
    ProviderCallOutcomeKind,
    ProviderCallResult,
    ProviderCallState,
    ProviderInvocationOutcome,
    ProviderRetryDecision,
    ProviderRetryInstruction,
    SemanticResponseClassifierIdentifier,
    StandardProviderCallRetryPolicy,
    cancel_provider_call,
    transition_provider_call,
)
from dr_providers.outcomes.models import (
    STALLED_RESPONSE_CODE,
    TransportTimeoutContainment,
)
from dr_providers.translation.responses import RESPONSE_REFUSAL_CODE

CLASSIFIER_ID = SemanticResponseClassifierIdentifier("semantic-v1")
HTTP_REQUEST = ProviderHttpRequestEvidence(
    url="https://example.test/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    body_bytes=57,
)


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
        classifier_identifier=CLASSIFIER_ID,
    )


def _observation(
    state: ProviderCallState,
    outcome: ProviderInvocationOutcome,
    *,
    marker: str = "one",
) -> CompletedProviderInvocationObservation:
    response_outcomes = {
        ProviderInvocationOutcome.SUCCESS,
        ProviderInvocationOutcome.EMPTY_GENERATION,
        ProviderInvocationOutcome.SEMANTIC_REJECTION,
    }
    if outcome in response_outcomes:
        evidence = ProviderInvocationEvidence(
            request_identity_hash=state.request_identity_hash,
            policy_identity={
                "provider_kind": "openai",
                "transport_fixture": "v1",
            },
            http_request=HTTP_REQUEST,
            response=ProviderTransportResponse(
                text=(
                    ""
                    if outcome is ProviderInvocationOutcome.EMPTY_GENERATION
                    else marker
                ),
                response_body={"id": marker},
            ),
        )
    else:
        failure_class = {
            ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE: (
                FailureClass.TRANSIENT
            ),
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT: (
                FailureClass.TRANSIENT
            ),
            ProviderInvocationOutcome.RATE_LIMITING: FailureClass.RATE_LIMITED,
            ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION: (
                FailureClass.TRANSIENT
            ),
            ProviderInvocationOutcome.RESOURCE_EXHAUSTION: (
                FailureClass.RESOURCE_EXHAUSTION
            ),
            ProviderInvocationOutcome.PROVIDER_REJECTION: (
                FailureClass.PERMANENT
            ),
            ProviderInvocationOutcome.UNKNOWN_TRANSPORT_FAILURE: (
                FailureClass.UNKNOWN
            ),
        }[outcome]
        code = outcome.value
        metadata = {"marker": marker}
        containment = None
        if outcome in {
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
            ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION,
        }:
            code = STALLED_RESPONSE_CODE
            if outcome is (
                ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT
            ):
                containment = TransportTimeoutContainment.CONTAINED
        elif outcome is ProviderInvocationOutcome.PROVIDER_REJECTION:
            code = RESPONSE_REFUSAL_CODE
        evidence = ProviderInvocationEvidence(
            request_identity_hash=state.request_identity_hash,
            policy_identity={
                "provider_kind": "openai",
                "transport_fixture": "v1",
            },
            http_request=HTTP_REQUEST,
            failure=ProviderTransportFailure(
                failure_class=failure_class,
                code=code,
                message=marker,
                containment=containment,
                metadata=metadata,
            ),
        )
    return CompletedProviderInvocationObservation(
        invocation_ordinal=state.next_invocation_ordinal,
        request_identity_hash=state.request_identity_hash,
        evidence=evidence,
        evidence_identity_hash=evidence.identity_hash,
        outcome=outcome,
    )


def _restored_state(state: ProviderCallState) -> ProviderCallState:
    return ProviderCallState.model_validate_json(state.model_dump_json())


def test_initial_state_carries_full_identity_components() -> None:
    state = _state()

    assert state.request == _request()
    assert isinstance(state.retry_policy, StandardProviderCallRetryPolicy)
    assert state.classifier_identifier == CLASSIFIER_ID
    assert state.request_identity_hash == state.request.identity_hash
    assert state.retry_policy_identity_hash == state.retry_policy.identity_hash
    assert state.next_invocation_ordinal == 1
    assert _restored_state(state) == state


def test_two_round_restore_matches_uninterrupted_transition() -> None:
    initial = _state()
    first = _observation(
        initial,
        ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE,
    )

    uninterrupted_instruction = transition_provider_call(initial, first)
    assert isinstance(uninterrupted_instruction, ProviderRetryInstruction)
    restored_instruction = ProviderRetryInstruction.model_validate_json(
        uninterrupted_instruction.model_dump_json()
    )
    restored_state = _restored_state(restored_instruction.next_state)
    second = _observation(
        restored_state,
        ProviderInvocationOutcome.SUCCESS,
        marker="two",
    )

    restored_result = transition_provider_call(restored_state, second)
    uninterrupted_result = transition_provider_call(
        uninterrupted_instruction.next_state,
        second,
    )

    assert isinstance(restored_result, ProviderCallResult)
    assert isinstance(uninterrupted_result, ProviderCallResult)
    assert restored_result == uninterrupted_result
    assert restored_result.identity_hash == uninterrupted_result.identity_hash
    assert restored_result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    assert len(restored_result.completed_invocations) == 2
    retry_decision = restored_result.completed_invocations[0].retry_decision
    assert retry_decision is not None
    assert retry_decision.delay_seconds == 1.0
    assert restored_result.completed_invocations[1].retry_decision is None
    assert "identity_hash" in restored_result.__dict__


@pytest.mark.parametrize(
    "outcome",
    [
        ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE,
        ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
    ],
)
def test_standard_policy_retries_only_its_exact_eligible_outcomes(
    outcome: ProviderInvocationOutcome,
) -> None:
    state = _state()
    instruction = transition_provider_call(
        state,
        _observation(state, outcome),
    )

    assert isinstance(instruction, ProviderRetryInstruction)
    assert instruction.delay_seconds == 1.0
    assert instruction.next_invocation_ordinal == 2


@pytest.mark.parametrize(
    "outcome",
    [
        ProviderInvocationOutcome.RATE_LIMITING,
        ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION,
        ProviderInvocationOutcome.RESOURCE_EXHAUSTION,
        ProviderInvocationOutcome.PROVIDER_REJECTION,
        ProviderInvocationOutcome.UNKNOWN_TRANSPORT_FAILURE,
    ],
)
def test_standard_policy_terminal_outcomes_start_no_successor(
    outcome: ProviderInvocationOutcome,
) -> None:
    state = _state()
    result = transition_provider_call(state, _observation(state, outcome))

    assert isinstance(result, ProviderCallResult)
    assert result.outcome.kind is ProviderCallOutcomeKind.INVOCATION_OUTCOME
    assert result.outcome.invocation_outcome is outcome
    assert result.completed_invocations[-1].retry_decision is None


def test_final_eligible_invocation_is_policy_exhaustion() -> None:
    state = _state()
    first = transition_provider_call(
        state,
        _observation(
            state,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
        ),
    )
    assert isinstance(first, ProviderRetryInstruction)
    second_state = first.next_state

    result = transition_provider_call(
        second_state,
        _observation(
            second_state,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
            marker="two",
        ),
    )

    assert isinstance(result, ProviderCallResult)
    assert result.outcome.kind is ProviderCallOutcomeKind.POLICY_EXHAUSTION
    assert len(result.completed_invocations) == 2


def test_transition_rejects_invalid_observation_sequence() -> None:
    state = _state()
    observation = _observation(
        state,
        ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
    )
    data = observation.model_dump(mode="python")

    skipped = CompletedProviderInvocationObservation.model_validate(
        {**data, "invocation_ordinal": 2}
    )
    with pytest.raises(ValueError, match="ordinal"):
        transition_provider_call(state, skipped)

    other_request = ProviderCallRequest(
        config=openai_chat_config(model="other"),
        transcript=state.request.transcript,
    )
    mismatched = CompletedProviderInvocationObservation.model_validate(
        {**data, "request_identity_hash": other_request.identity_hash}
    )
    with pytest.raises(ValueError, match="request"):
        transition_provider_call(state, mismatched)

    instruction = transition_provider_call(state, observation)
    assert isinstance(instruction, ProviderRetryInstruction)
    with pytest.raises(ValueError, match="ordinal"):
        transition_provider_call(instruction.next_state, observation)


def test_observation_rejects_bad_hash_and_decision_input() -> None:
    state = _state()
    observation = _observation(state, ProviderInvocationOutcome.SUCCESS)
    data = observation.model_dump(mode="python")

    with pytest.raises(ValidationError, match="evidence identity hash"):
        CompletedProviderInvocationObservation.model_validate(
            {**data, "evidence_identity_hash": "0" * 64}
        )
    with pytest.raises(ValidationError):
        CompletedProviderInvocationObservation.model_validate(
            {**data, "retry_decision": {"delay_seconds": 1.0}}
        )


def test_failure_outcome_is_recomputed_during_json_restore() -> None:
    state = _state()
    evidence = ProviderInvocationEvidence(
        request_identity_hash=state.request_identity_hash,
        failure=ProviderTransportFailure(
            failure_class=FailureClass.PERMANENT,
            code="missing_api_key",
            message="missing API key",
        ),
    )
    observation = CompletedProviderInvocationObservation(
        invocation_ordinal=state.next_invocation_ordinal,
        request_identity_hash=state.request_identity_hash,
        evidence=evidence,
        evidence_identity_hash=evidence.identity_hash,
        outcome=ProviderInvocationOutcome.MISSING_CREDENTIAL,
    )
    payload = json.loads(observation.model_dump_json())
    payload["outcome"] = (
        ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE.value
    )

    with pytest.raises(
        ValidationError,
        match="failure outcome does not match failure evidence",
    ):
        CompletedProviderInvocationObservation.model_validate_json(
            json.dumps(payload)
        )

    result = transition_provider_call(state, observation)
    assert isinstance(result, ProviderCallResult)
    assert result.outcome.invocation_outcome is (
        ProviderInvocationOutcome.MISSING_CREDENTIAL
    )


def test_evidence_identity_cache_is_protected_by_deep_freezing() -> None:
    state = _state()
    observation = _observation(state, ProviderInvocationOutcome.SUCCESS)
    evidence = observation.evidence
    assert evidence.response is not None
    cached_hash = evidence.identity_hash

    with pytest.raises(TypeError, match="immutable"):
        evidence.response.response_body["id"] = "changed"
    assert evidence.http_request is not None
    with pytest.raises(TypeError, match="immutable"):
        evidence.http_request.body["messages"][0]["content"] = "changed"

    assert evidence.identity_hash == cached_hash


def test_state_rejects_bad_record_hash_and_terminal_record() -> None:
    state = _state()
    instruction = transition_provider_call(
        state,
        _observation(
            state,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
        ),
    )
    assert isinstance(instruction, ProviderRetryInstruction)
    data = instruction.next_state.model_dump(mode="python")

    with pytest.raises(ValidationError, match="record hash"):
        ProviderCallState.model_validate(
            {**data, "completed_invocation_record_hashes": ("0" * 64,)}
        )
    retry_record = instruction.next_state.completed_invocations[0]
    terminal_record = retry_record.model_copy(update={"retry_decision": None})
    with pytest.raises(ValidationError, match="terminal record"):
        ProviderCallState.model_validate(
            {
                **data,
                "completed_invocations": (terminal_record,),
                "completed_invocation_record_hashes": (
                    terminal_record.identity_hash,
                ),
            }
        )


def test_state_and_result_reject_history_beyond_policy_limit() -> None:
    initial = _state()
    instruction = transition_provider_call(
        initial,
        _observation(
            initial,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
        ),
    )
    assert isinstance(instruction, ProviderRetryInstruction)
    result = transition_provider_call(
        instruction.next_state,
        _observation(
            instruction.next_state,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
            marker="two",
        ),
    )
    assert isinstance(result, ProviderCallResult)
    records = (*result.completed_invocations, result.completed_invocations[-1])
    record_hashes = (*result.completed_invocation_record_hashes, "0" * 64)

    state_data = instruction.next_state.model_dump(mode="python")
    with pytest.raises(ValidationError, match="history exceeds"):
        ProviderCallState.model_validate(
            {
                **state_data,
                "completed_invocations": records,
                "completed_invocation_record_hashes": record_hashes,
                "next_invocation_ordinal": 4,
            }
        )

    result_data = result.model_dump(mode="python")
    with pytest.raises(ValidationError, match="history exceeds"):
        ProviderCallResult.model_validate(
            {
                **result_data,
                "completed_invocations": records,
                "completed_invocation_record_hashes": record_hashes,
            }
        )


def test_result_rejects_retry_decision_at_maximum_ordinal() -> None:
    initial = _state()
    instruction = transition_provider_call(
        initial,
        _observation(
            initial,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
        ),
    )
    assert isinstance(instruction, ProviderRetryInstruction)
    result = transition_provider_call(
        instruction.next_state,
        _observation(
            instruction.next_state,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
            marker="two",
        ),
    )
    assert isinstance(result, ProviderCallResult)
    final_observation = result.completed_invocations[-1].observation
    invalid_final_record = DecidedProviderInvocationRecord(
        observation=final_observation,
        retry_decision=ProviderRetryDecision(delay_seconds=1.0),
    )
    result_data = result.model_dump(mode="python")

    with pytest.raises(ValidationError, match="maximum permitted ordinal"):
        ProviderCallResult.model_validate(
            {
                **result_data,
                "completed_invocations": (
                    result.completed_invocations[0],
                    invalid_final_record,
                ),
                "completed_invocation_record_hashes": (
                    result.completed_invocation_record_hashes[0],
                    invalid_final_record.identity_hash,
                ),
            }
        )


def test_cancellation_terminalizes_idle_pending_and_active_states() -> None:
    initial = _state()
    idle = cancel_provider_call(initial)
    assert idle.outcome.kind is ProviderCallOutcomeKind.DRAINING_CANCELLATION
    assert idle.completed_invocations == ()

    instruction = transition_provider_call(
        initial,
        _observation(
            initial,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
        ),
    )
    assert isinstance(instruction, ProviderRetryInstruction)
    pending = cancel_provider_call(instruction.next_state)
    assert pending.completed_invocations[-1].retry_decision is not None

    active = cancel_provider_call(
        initial,
        _observation(initial, ProviderInvocationOutcome.SUCCESS),
    )
    assert active.completed_invocations[-1].retry_decision is None
    assert (
        active.completed_invocations[-1].observation.outcome
        is ProviderInvocationOutcome.SUCCESS
    )


def test_result_identity_payload_binds_ordered_decided_record_hashes() -> None:
    initial = _state()
    instruction = transition_provider_call(
        initial,
        _observation(
            initial,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
        ),
    )
    assert isinstance(instruction, ProviderRetryInstruction)
    result = transition_provider_call(
        instruction.next_state,
        _observation(
            instruction.next_state,
            ProviderInvocationOutcome.SUCCESS,
            marker="two",
        ),
    )
    assert isinstance(result, ProviderCallResult)

    assert result.identity_payload()[
        "completed_invocation_record_hashes"
    ] == list(result.completed_invocation_record_hashes)
    restored = ProviderCallResult.model_validate_json(result.model_dump_json())
    assert restored.identity_hash == result.identity_hash


def test_persisted_state_instruction_and_result_keys_are_pinned() -> None:
    state = _state()
    instruction = transition_provider_call(
        state,
        _observation(
            state,
            ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
        ),
    )
    assert isinstance(instruction, ProviderRetryInstruction)
    result = cancel_provider_call(instruction.next_state)

    assert set(json.loads(state.model_dump_json())) == {
        "schema_version",
        "request",
        "request_identity_hash",
        "retry_policy",
        "retry_policy_identity_hash",
        "classifier_identifier",
        "call_identity_hash",
        "completed_invocations",
        "completed_invocation_record_hashes",
        "next_invocation_ordinal",
    }
    assert set(json.loads(instruction.model_dump_json())) == {
        "schema_version",
        "source",
        "delay_seconds",
        "next_invocation_ordinal",
        "next_state",
    }
    assert set(json.loads(result.model_dump_json())) == {
        "schema_version",
        "request",
        "request_identity_hash",
        "retry_policy",
        "retry_policy_identity_hash",
        "classifier_identifier",
        "call_identity_hash",
        "completed_invocations",
        "completed_invocation_record_hashes",
        "outcome",
    }
