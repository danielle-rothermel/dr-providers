from __future__ import annotations

import json
from threading import Event
from typing import Any

import pytest
from _retry_fixtures import two_invocation_transient_retry_policy
from pydantic import ValidationError

from dr_providers import (
    MessageRole,
    PromptMessage,
    ProviderCallKind,
    ProviderGenerateRequest,
    ProviderHttpRequestEvidence,
    ProviderInvocationEvidence,
    ProviderTransportFailure,
    ProviderTransportResponse,
    RecoverabilityClass,
    ScriptedOutcome,
    ScriptedProvider,
    Transcript,
    TransportTimeoutContainment,
    openai_chat_config,
)
from dr_providers.lifecycle import (
    ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER,
    AcceptAllSemanticResponseClassifier,
    CompletedProviderInvocationObservation,
    ProviderCallOutcomeKind,
    ProviderCallResult,
    ProviderCallState,
    ProviderInvocationOutcome,
    ProviderRetryInstruction,
    SemanticResponseClassifierIdentifier,
    StandardProviderCallRetryPolicy,
    cancel_provider_call,
    classify_provider_invocation,
    run_local_provider_call,
    transition_provider_call,
)

REQUEST_HASH = (
    "9632eb598dd9f55aff338f51ecfb8793b15d43ac58263c1e53148652f56695fa"
)
RETRY_POLICY_HASH = (
    "a465bcf528ec87cfacd1fe842849ee13880bc236f3693268caf6555aeba7c4cc"
)
CALL_HASH = "64b91c55bf59813087ad626de28e4976c938d97aef87c362aa829381a5fadf51"
FIRST_EVIDENCE_HASH = (
    "4f1abf090574e0d468f9112d16688a982c9463c02a8b1e8d25783375d2d02700"
)
FIRST_OBSERVATION_IDENTITY_HASH = (
    "d6dd44c2de9130781960f01aa62ab664944760dd4f4f6304d2977f23ee975e8e"
)
FIRST_RECORD_IDENTITY_HASH = (
    "1109796cf3483814949f6d033e393aa49236aa016a45d47c0bc2442e068b17af"
)
SECOND_RECORD_IDENTITY_HASH = (
    "35b3d3619555b3da2f608c5040de0e452e5779e4f4f4d8fb7d247ec648b3e70d"
)
RESULT_IDENTITY_HASH = (
    "540c27b92b41ffa78598099007988f8e86d93202bff181613209fdb0d1b3c664"
)
CANCELLATION_IDENTITY_HASH = (
    "d87b0575d7d4d59b64537c8cda4924efa7b9850dddcce853def32e34d8c35827"
)


def _request() -> ProviderGenerateRequest:
    return ProviderGenerateRequest(
        config=openai_chat_config(model="m"),
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content="hi"),)
        ),
    )


def _state(
    classifier_identifier: SemanticResponseClassifierIdentifier = (
        ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER
    ),
) -> ProviderCallState:
    return ProviderCallState.initial(
        request=_request(),
        retry_policy=StandardProviderCallRetryPolicy(),
        classifier_identifier=classifier_identifier,
    )


def _retry_state(
    classifier_identifier: SemanticResponseClassifierIdentifier = (
        ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER
    ),
) -> ProviderCallState:
    return ProviderCallState.initial(
        request=_request(),
        retry_policy=two_invocation_transient_retry_policy(),
        classifier_identifier=classifier_identifier,
    )


def _scripted_outcomes() -> list[ScriptedOutcome]:
    return [
        ScriptedOutcome(
            failure=ProviderTransportFailure(
                recoverability=RecoverabilityClass.TRANSIENT,
                code="connection_reset",
                message="retryable failure",
            )
        ),
        ScriptedOutcome(
            text="accepted",
            response_body={"response_body_marker": "second"},
        ),
    ]


def _observation(
    state: ProviderCallState,
    evidence: ProviderInvocationEvidence,
) -> CompletedProviderInvocationObservation:
    return CompletedProviderInvocationObservation(
        invocation_ordinal=state.next_invocation_ordinal,
        request_hash=state.request_hash,
        evidence=evidence,
        evidence_hash=evidence.identity_hash,
        outcome=classify_provider_invocation(
            evidence, AcceptAllSemanticResponseClassifier()
        ),
    )


class _NoWait:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def wait(self, delay_seconds: float, cancellation: Event) -> None:
        assert not cancellation.is_set()
        self.delays.append(delay_seconds)


def test_serialized_two_invocation_handoff_matches_local_driver() -> None:
    initial_state = _retry_state()
    durable_provider = ScriptedProvider(_scripted_outcomes())

    first_evidence = durable_provider.invoke(initial_state.request)
    first_observation = _observation(initial_state, first_evidence)
    first_transition = transition_provider_call(
        initial_state, first_observation
    )
    assert isinstance(first_transition, ProviderRetryInstruction)
    assert first_transition.delay_seconds == 1.0
    assert first_transition.next_invocation_ordinal == 2

    restored_instruction = ProviderRetryInstruction.model_validate_json(
        first_transition.model_dump_json()
    )
    restored_state = ProviderCallState.model_validate_json(
        restored_instruction.next_state.model_dump_json()
    )
    second_evidence = durable_provider.invoke(restored_state.request)
    second_observation = _observation(restored_state, second_evidence)
    durable_result = transition_provider_call(
        restored_state, second_observation
    )
    assert isinstance(durable_result, ProviderCallResult)
    restored_result = ProviderCallResult.model_validate_json(
        durable_result.model_dump_json()
    )

    wait = _NoWait()
    local_result = run_local_provider_call(
        provider=ScriptedProvider(_scripted_outcomes()),
        state=_retry_state(),
        classifier=AcceptAllSemanticResponseClassifier(),
        cancellation=Event(),
        retry_wait=wait,
    )

    assert restored_result == local_result
    assert restored_result.identity_hash == local_result.identity_hash
    assert restored_result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    assert wait.delays == [1.0]
    assert len(durable_provider.requests) == 2


def _golden_trace() -> tuple[
    ProviderCallState,
    ProviderRetryInstruction,
    ProviderCallResult,
]:
    state = _retry_state(SemanticResponseClassifierIdentifier("semantic-v1"))
    http_request = ProviderHttpRequestEvidence(
        url="https://example.test/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        body={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        },
        body_bytes=57,
    )
    first_evidence = ProviderInvocationEvidence(
        kind=ProviderCallKind.GENERATE,
        request_hash=state.request_hash,
        policy_identity={
            "provider_kind": "openai",
            "transport_fixture": "v1",
        },
        max_request_bytes=1024,
        max_response_bytes=2048,
        http_request=http_request,
        response_bytes=41,
        failure=ProviderTransportFailure(
            recoverability=RecoverabilityClass.TRANSIENT,
            code="connection_reset",
            message="retryable failure",
            response_body={"failure_body_marker": "first"},
        ),
    )
    first_observation = CompletedProviderInvocationObservation(
        invocation_ordinal=1,
        request_hash=state.request_hash,
        evidence=first_evidence,
        evidence_hash=first_evidence.identity_hash,
        outcome=(
            ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE
        ),
    )
    instruction = transition_provider_call(state, first_observation)
    assert isinstance(instruction, ProviderRetryInstruction)

    second_evidence = ProviderInvocationEvidence(
        kind=ProviderCallKind.GENERATE,
        request_hash=state.request_hash,
        policy_identity={
            "provider_kind": "openai",
            "transport_fixture": "v1",
        },
        max_request_bytes=1024,
        max_response_bytes=2048,
        http_request=http_request,
        response_bytes=42,
        response=ProviderTransportResponse(
            text="accepted",
            response_body={"response_body_marker": "second"},
        ),
    )
    second_observation = CompletedProviderInvocationObservation(
        invocation_ordinal=2,
        request_hash=state.request_hash,
        evidence=second_evidence,
        evidence_hash=second_evidence.identity_hash,
        outcome=ProviderInvocationOutcome.SUCCESS,
    )
    result = transition_provider_call(
        instruction.next_state, second_observation
    )
    assert isinstance(result, ProviderCallResult)
    return state, instruction, result


def test_lifecycle_wire_dictionaries_and_identities_are_pinned() -> None:
    state, instruction, result = _golden_trace()
    first_record = instruction.next_state.completed_invocations[0]
    first_observation = first_record.observation
    cancellation = cancel_provider_call(instruction.next_state)

    request_payload = state.request.model_dump(mode="json")
    policy_payload = state.retry_policy.model_dump(mode="json")
    evidence_payload = first_observation.evidence.model_dump(mode="json")
    observation_payload = {
        "schema_version": 3,
        "invocation_ordinal": 1,
        "request_hash": REQUEST_HASH,
        "evidence": evidence_payload,
        "evidence_hash": FIRST_EVIDENCE_HASH,
        "outcome": "transient_provider_or_network_failure",
    }
    first_record_payload = {
        "schema_version": 3,
        "observation": observation_payload,
        "retry_decision": {
            "source": "provider_call_retry_policy",
            "delay_seconds": 1.0,
        },
    }
    initial_state_payload = {
        "schema_version": 3,
        "request": request_payload,
        "request_hash": REQUEST_HASH,
        "retry_policy": policy_payload,
        "retry_policy_hash": RETRY_POLICY_HASH,
        "classifier_identifier": "semantic-v1",
        "call_hash": CALL_HASH,
        "completed_invocations": [],
        "completed_invocation_record_hashes": [],
        "next_invocation_ordinal": 1,
    }
    next_state_payload = {
        **initial_state_payload,
        "completed_invocations": [first_record_payload],
        "completed_invocation_record_hashes": [FIRST_RECORD_IDENTITY_HASH],
        "next_invocation_ordinal": 2,
    }

    assert state.model_dump(mode="json") == initial_state_payload
    assert first_observation.model_dump(mode="json") == observation_payload
    assert first_record.model_dump(mode="json") == first_record_payload
    assert instruction.model_dump(mode="json") == {
        "schema_version": 3,
        "source": "provider_call_retry_policy",
        "delay_seconds": 1.0,
        "next_invocation_ordinal": 2,
        "next_state": next_state_payload,
    }
    assert result.model_dump(mode="json") == {
        "schema_version": 3,
        "request": request_payload,
        "request_hash": REQUEST_HASH,
        "retry_policy": policy_payload,
        "retry_policy_hash": RETRY_POLICY_HASH,
        "classifier_identifier": "semantic-v1",
        "call_hash": CALL_HASH,
        "completed_invocations": [
            first_record_payload,
            result.completed_invocations[1].model_dump(mode="json"),
        ],
        "completed_invocation_record_hashes": [
            FIRST_RECORD_IDENTITY_HASH,
            SECOND_RECORD_IDENTITY_HASH,
        ],
        "outcome": {"kind": "accepted", "invocation_outcome": "success"},
    }
    assert cancellation.model_dump(mode="json") == {
        "schema_version": 3,
        "request": request_payload,
        "request_hash": REQUEST_HASH,
        "retry_policy": policy_payload,
        "retry_policy_hash": RETRY_POLICY_HASH,
        "classifier_identifier": "semantic-v1",
        "call_hash": CALL_HASH,
        "completed_invocations": [first_record_payload],
        "completed_invocation_record_hashes": [FIRST_RECORD_IDENTITY_HASH],
        "outcome": {
            "kind": "draining_cancellation",
            "invocation_outcome": None,
        },
    }

    assert state.request_hash == REQUEST_HASH
    assert state.retry_policy_hash == RETRY_POLICY_HASH
    assert state.call_hash == CALL_HASH
    assert first_observation.evidence_hash == (FIRST_EVIDENCE_HASH)
    assert first_observation.identity_hash == FIRST_OBSERVATION_IDENTITY_HASH
    assert first_record.identity_hash == FIRST_RECORD_IDENTITY_HASH
    assert result.identity_hash == RESULT_IDENTITY_HASH
    assert cancellation.identity_hash == CANCELLATION_IDENTITY_HASH


def test_restored_pending_state_cancels_identically() -> None:
    _state_value, instruction, _result = _golden_trace()
    restored_state = ProviderCallState.model_validate_json(
        instruction.next_state.model_dump_json()
    )

    restored_cancellation = ProviderCallResult.model_validate_json(
        cancel_provider_call(restored_state).model_dump_json()
    )
    uninterrupted_cancellation = cancel_provider_call(instruction.next_state)

    assert restored_cancellation == uninterrupted_cancellation
    assert (
        restored_cancellation.identity_hash
        == uninterrupted_cancellation.identity_hash
    )
    assert restored_cancellation.outcome.kind is (
        ProviderCallOutcomeKind.DRAINING_CANCELLATION
    )
    assert len(restored_cancellation.completed_invocations) == 1


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("request_hash", "0" * 64, "request identity hash"),
        ("retry_policy_hash", "0" * 64, "policy identity hash"),
        ("call_hash", "0" * 64, "call identity hash"),
        ("classifier_identifier", "semantic-v2", "call identity hash"),
    ],
)
def test_restored_state_rejects_mismatched_identity_components(
    field_name: str,
    replacement: str,
    message: str,
) -> None:
    payload = json.loads(_state().model_dump_json())
    payload[field_name] = replacement

    with pytest.raises(ValidationError, match=message):
        ProviderCallState.model_validate(payload)


def test_restored_state_rejects_altered_decided_history() -> None:
    _state_value, instruction, _result = _golden_trace()
    payload = json.loads(instruction.next_state.model_dump_json())
    payload["completed_invocations"][0]["retry_decision"]["delay_seconds"] = (
        0.0
    )

    with pytest.raises(ValidationError, match="record hash"):
        ProviderCallState.model_validate(payload)


@pytest.mark.parametrize(
    "outcome",
    [
        ProviderInvocationOutcome.SUCCESS,
        ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION,
    ],
)
def test_terminal_history_cannot_be_restored_as_continuable_state(
    outcome: ProviderInvocationOutcome,
) -> None:
    state = _state()
    if outcome is ProviderInvocationOutcome.SUCCESS:
        evidence = ProviderInvocationEvidence(
            kind=ProviderCallKind.GENERATE,
            request_hash=state.request_hash,
            response=ProviderTransportResponse(text="accepted"),
        )
    else:
        evidence = ProviderInvocationEvidence(
            kind=ProviderCallKind.GENERATE,
            request_hash=state.request_hash,
            failure=ProviderTransportFailure(
                recoverability=RecoverabilityClass.TRANSIENT,
                code="stalled_response",
                message="local work may remain active",
                containment=TransportTimeoutContainment.UNCONTAINED,
            ),
        )
    observation = CompletedProviderInvocationObservation(
        invocation_ordinal=1,
        request_hash=state.request_hash,
        evidence=evidence,
        evidence_hash=evidence.identity_hash,
        outcome=outcome,
    )
    terminal = transition_provider_call(state, observation)
    assert isinstance(terminal, ProviderCallResult)

    payload = terminal.model_dump(mode="json")
    payload.pop("outcome")
    payload["next_invocation_ordinal"] = 2
    with pytest.raises(ValidationError, match="exceeds retry policy"):
        ProviderCallState.model_validate(payload)


def _count_equal_nodes(value: object, target: object) -> int:
    count = int(value == target)
    if isinstance(value, dict):
        return count + sum(
            _count_equal_nodes(child, target) for child in value.values()
        )
    if isinstance(value, list):
        return count + sum(
            _count_equal_nodes(child, target) for child in value
        )
    return count


def test_two_invocation_result_does_not_amplify_large_evidence() -> None:
    state, _instruction, result = _golden_trace()
    payload: dict[str, Any] = json.loads(result.model_dump_json())
    request_payload = state.request.model_dump(mode="json")
    http_body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    failure_body = {"failure_body_marker": "first"}
    response_body = {"response_body_marker": "second"}

    assert _count_equal_nodes(payload, request_payload) == 1
    assert _count_equal_nodes(payload, http_body) == 2
    assert _count_equal_nodes(payload, failure_body) == 1
    assert _count_equal_nodes(payload, response_body) == 1
    first_evidence = payload["completed_invocations"][0]["observation"][
        "evidence"
    ]
    assert "http_request" not in first_evidence["failure"]
    assert "request_body" not in first_evidence["failure"]
