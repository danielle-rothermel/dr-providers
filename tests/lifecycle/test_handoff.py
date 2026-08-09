from __future__ import annotations

import json
from threading import Event
from typing import Any

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
    ScriptedOutcome,
    ScriptedProvider,
    Transcript,
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

REQUEST_IDENTITY_HASH = (
    "5e7f8d07340accee58cc0a3d6570cf7966295b1a7f1102858b2fd254a7c2b27c"
)
POLICY_IDENTITY_HASH = (
    "a53e481b6f1be12c4f23785315463fb53193c5e674364a860907369730c2f797"
)
CALL_IDENTITY_HASH = (
    "a1ca086506691c936b6f19342935a27b592eaf0f7e2f4642038df9a7c5658c91"
)
FIRST_EVIDENCE_IDENTITY_HASH = (
    "eb01af647d148ecf4294ea54a87308c64beccb2d799b100e587ad05612691cf8"
)
FIRST_OBSERVATION_IDENTITY_HASH = (
    "2a024f2af5114e82ffb54ec2c6238813dbc0ba45a2eb9992063997cba8febc28"
)
FIRST_RECORD_IDENTITY_HASH = (
    "f22529b6a2c010711ff32dfa47be06e4619a2468fe2ff5fd6b59225da46d613b"
)
SECOND_RECORD_IDENTITY_HASH = (
    "962a198db09e39ba58d31ce5de5c6b23a160159f8d2b52561774e0590d295cbd"
)
RESULT_IDENTITY_HASH = (
    "fefa7cbaab7685f587092156f547ffa4742fd7bffebac6ca4e7dcebee6f1422c"
)
CANCELLATION_IDENTITY_HASH = (
    "391b87d6df7f2ffdda79d80a063f8af051e287380669af08f6b1afe836c6fb18"
)


def _request() -> ProviderCallRequest:
    return ProviderCallRequest(
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


def _scripted_outcomes() -> list[ScriptedOutcome]:
    return [
        ScriptedOutcome(
            failure=ProviderTransportFailure(
                failure_class=FailureClass.TRANSIENT,
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
        request_identity_hash=state.request_identity_hash,
        evidence=evidence,
        evidence_identity_hash=evidence.identity_hash,
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
    initial_state = _state()
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
        state=_state(),
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
    state = _state(SemanticResponseClassifierIdentifier("semantic-v1"))
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
        request_identity_hash=state.request_identity_hash,
        policy_identity={
            "provider_kind": "openai",
            "transport_fixture": "v1",
        },
        max_request_bytes=1024,
        max_response_bytes=2048,
        http_request=http_request,
        response_bytes=41,
        failure=ProviderTransportFailure(
            failure_class=FailureClass.TRANSIENT,
            code="connection_reset",
            message="retryable failure",
            response_body={"failure_body_marker": "first"},
            metadata={"phase": "dispatch"},
        ),
    )
    first_observation = CompletedProviderInvocationObservation(
        invocation_ordinal=1,
        request_identity_hash=state.request_identity_hash,
        evidence=first_evidence,
        evidence_identity_hash=first_evidence.identity_hash,
        outcome=(
            ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE
        ),
    )
    instruction = transition_provider_call(state, first_observation)
    assert isinstance(instruction, ProviderRetryInstruction)

    second_evidence = ProviderInvocationEvidence(
        request_identity_hash=state.request_identity_hash,
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
        request_identity_hash=state.request_identity_hash,
        evidence=second_evidence,
        evidence_identity_hash=second_evidence.identity_hash,
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
        "schema_version": 1,
        "invocation_ordinal": 1,
        "request_identity_hash": REQUEST_IDENTITY_HASH,
        "evidence": evidence_payload,
        "evidence_identity_hash": FIRST_EVIDENCE_IDENTITY_HASH,
        "outcome": "transient_provider_or_network_failure",
    }
    first_record_payload = {
        "schema_version": 1,
        "observation": observation_payload,
        "retry_decision": {
            "source": "provider_call_retry_policy",
            "delay_seconds": 1.0,
        },
    }
    initial_state_payload = {
        "schema_version": 1,
        "request": request_payload,
        "request_identity_hash": REQUEST_IDENTITY_HASH,
        "retry_policy": policy_payload,
        "retry_policy_identity_hash": POLICY_IDENTITY_HASH,
        "classifier_identifier": "semantic-v1",
        "call_identity_hash": CALL_IDENTITY_HASH,
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
        "schema_version": 1,
        "source": "provider_call_retry_policy",
        "delay_seconds": 1.0,
        "next_invocation_ordinal": 2,
        "next_state": next_state_payload,
    }
    assert result.model_dump(mode="json") == {
        "schema_version": 1,
        "request": request_payload,
        "request_identity_hash": REQUEST_IDENTITY_HASH,
        "retry_policy": policy_payload,
        "retry_policy_identity_hash": POLICY_IDENTITY_HASH,
        "classifier_identifier": "semantic-v1",
        "call_identity_hash": CALL_IDENTITY_HASH,
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
        "schema_version": 1,
        "request": request_payload,
        "request_identity_hash": REQUEST_IDENTITY_HASH,
        "retry_policy": policy_payload,
        "retry_policy_identity_hash": POLICY_IDENTITY_HASH,
        "classifier_identifier": "semantic-v1",
        "call_identity_hash": CALL_IDENTITY_HASH,
        "completed_invocations": [first_record_payload],
        "completed_invocation_record_hashes": [FIRST_RECORD_IDENTITY_HASH],
        "outcome": {
            "kind": "draining_cancellation",
            "invocation_outcome": None,
        },
    }

    assert state.request_identity_hash == REQUEST_IDENTITY_HASH
    assert state.retry_policy_identity_hash == POLICY_IDENTITY_HASH
    assert state.call_identity_hash == CALL_IDENTITY_HASH
    assert first_observation.evidence_identity_hash == (
        FIRST_EVIDENCE_IDENTITY_HASH
    )
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
        ("request_identity_hash", "0" * 64, "request identity hash"),
        ("retry_policy_identity_hash", "0" * 64, "policy identity hash"),
        ("call_identity_hash", "0" * 64, "call identity hash"),
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
            request_identity_hash=state.request_identity_hash,
            response=ProviderTransportResponse(text="accepted"),
        )
    else:
        evidence = ProviderInvocationEvidence(
            request_identity_hash=state.request_identity_hash,
            failure=ProviderTransportFailure(
                failure_class=FailureClass.TRANSIENT,
                code="stalled_response",
                message="local work may remain active",
            ),
        )
    observation = CompletedProviderInvocationObservation(
        invocation_ordinal=1,
        request_identity_hash=state.request_identity_hash,
        evidence=evidence,
        evidence_identity_hash=evidence.identity_hash,
        outcome=outcome,
    )
    terminal = transition_provider_call(state, observation)
    assert isinstance(terminal, ProviderCallResult)

    payload = terminal.model_dump(mode="json")
    payload.pop("outcome")
    payload["next_invocation_ordinal"] = 2
    with pytest.raises(ValidationError, match="terminal record"):
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
