from __future__ import annotations

import json
from threading import Event
from typing import Any

import pytest
from _retry_fixtures import two_invocation_transient_retry_policy
from pydantic import ValidationError

from dr_providers import (
    ContinuationScore,
    CostInfo,
    LocalExecutionEvidence,
    MessageRole,
    PromptMessage,
    ProviderCallConfig,
    ProviderCallKind,
    ProviderCallRequest,
    ProviderGenerateRequest,
    ProviderHttpRequestEvidence,
    ProviderInvocationEvidence,
    ProviderScoreRequest,
    ProviderTransportFailure,
    ProviderTransportResponse,
    RecoverabilityClass,
    ScriptedOutcome,
    ScriptedProvider,
    TokenUsage,
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
    "1b74ea11e5bbf266c3d281b11c0e1ea506d881f58f25e826c6f9aa59c681e5db"
)
RETRY_POLICY_HASH = (
    "a465bcf528ec87cfacd1fe842849ee13880bc236f3693268caf6555aeba7c4cc"
)
CALL_HASH = "06c15ee09d7ecc51ad5964b70885c2dab998c7d6732b87773bcd9d4100691504"
FIRST_EVIDENCE_HASH = (
    "cfe4262873cd893a715c308709a7db7191c5d94d9f8009949500cd4a280363e2"
)
FIRST_OBSERVATION_IDENTITY_HASH = (
    "1a88bb66e58b6c267cae98582f88362957b0dacf3b664c4ddf16c43067221d96"
)
FIRST_RECORD_IDENTITY_HASH = (
    "78751c3f39e94e705cfae82a6144c6c67eea7382cef43b322541f3abe033a49d"
)
SECOND_RECORD_IDENTITY_HASH = (
    "574a4c724b0b37405f81c923cc251da919b59ca18930a79398e63835eb71365b"
)
RESULT_IDENTITY_HASH = (
    "d6d5b98c279d8cd5d326d232d906bc87872b0d97dae71eee4d6f9a992b8eb4f1"
)
CANCELLATION_IDENTITY_HASH = (
    "b48781399233d5e5e81ff48d527010e74680110173de2ab710ee0b1a8478e8da"
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
        "schema_version": 4,
        "invocation_ordinal": 1,
        "request_hash": REQUEST_HASH,
        "evidence": evidence_payload,
        "evidence_hash": FIRST_EVIDENCE_HASH,
        "outcome": "transient_provider_or_network_failure",
    }
    first_record_payload = {
        "schema_version": 4,
        "observation": observation_payload,
        "retry_decision": {
            "source": "provider_call_retry_policy",
            "delay_seconds": 1.0,
        },
    }
    initial_state_payload = {
        "schema_version": 4,
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
        "schema_version": 4,
        "source": "provider_call_retry_policy",
        "delay_seconds": 1.0,
        "next_invocation_ordinal": 2,
        "next_state": next_state_payload,
    }
    assert result.model_dump(mode="json") == {
        "schema_version": 4,
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
        "schema_version": 4,
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


class _NoScoreSemanticClassification(AcceptAllSemanticResponseClassifier):
    def classify(
        self, response: ProviderTransportResponse
    ) -> ProviderInvocationOutcome:
        del response
        raise AssertionError(
            "score handoff must not invoke semantic classification"
        )


def test_score_handoff_wire_dictionaries_match_local_driver(
    score_request: ProviderScoreRequest,
) -> None:
    request = ProviderScoreRequest(
        config=score_request.config,
        context="context",
        continuations=("answer",),
        token_logprobs=True,
    )
    classifier = _NoScoreSemanticClassification()
    state = ProviderCallState.initial(
        request=request,
        retry_policy=two_invocation_transient_retry_policy(),
        classifier_identifier=classifier.identifier,
    )
    scripted = [
        ScriptedOutcome(
            failure=ProviderTransportFailure(
                recoverability=RecoverabilityClass.TRANSIENT,
                code="connection_reset",
                message="retryable failure",
            )
        ),
        ScriptedOutcome(
            scores=(
                ContinuationScore(
                    log_likelihood=-2.0,
                    token_count=1,
                    char_count=6,
                    token_logprobs=(-2.0,),
                ),
            )
        ),
    ]
    with ScriptedProvider(scripted) as durable_provider:
        first = durable_provider.invoke(state.request)
        instruction = transition_provider_call(
            state, _observation(state, first)
        )
        assert isinstance(instruction, ProviderRetryInstruction)
        restored = ProviderRetryInstruction.model_validate_json(
            instruction.model_dump_json()
        )
        second = durable_provider.invoke(restored.next_state.request)
        result = transition_provider_call(
            restored.next_state,
            _observation(restored.next_state, second),
        )
        assert isinstance(result, ProviderCallResult)
        restored_result = ProviderCallResult.model_validate_json(
            result.model_dump_json()
        )
        assert durable_provider.requests == [request, request]
        assert durable_provider.payloads == []
    wait = _NoWait()
    with ScriptedProvider(scripted) as local_provider:
        local_result = run_local_provider_call(
            provider=local_provider,
            state=state,
            classifier=classifier,
            cancellation=Event(),
            retry_wait=wait,
        )
    assert restored_result == local_result
    assert wait.delays == [1.0]

    # Handwritten payloads and independent hashes pin the wire shape.
    request_hash = (
        "49898c77be48c16f4917f26cd65995d892ddb42931cf549177e2cdbf3f921a56"
    )
    call_hash = (
        "0c85a9266a700c712f42497940b7ce3d2686bad2b65ba6da5809720e53c39a62"
    )
    first_evidence_hash = (
        "faf73ba322b20270dfe6b9b8072518329d81722467ef97a718a451be838fca22"
    )
    first_record_hash = (
        "44d330abef6bb7568e7e4694ac21e37bff30efec16da1a342249f5c40610469a"
    )
    second_evidence_hash = (
        "041dde943dbbcb72a5e9fdc2096f040dd1ab30d88f8758451d3ca9b075856560"
    )
    second_record_hash = (
        "9de11de64a1b6e00a271f8978180149b0078edd02b56dd14eaee188116649292"
    )
    request_payload = {
        "kind": "score",
        "config": {
            "definition": {
                "definition_id": "test.score",
                "route": {
                    "provider": "openai",
                    "protocol": "chat_completions",
                    "model": "m",
                },
                "local": None,
                "supported_kinds": ["score"],
                "constraints": {
                    "supported_controls": [
                        "temperature",
                        "token_limit",
                        "top_p",
                    ],
                    "token_limit_parameter": "max_completion_tokens",
                    "reasoning_shape": "none",
                },
                "prompt_rendering": "role_messages",
                "required_controls": [],
                "extension_keys": [],
            },
            "controls": {
                "temperature": None,
                "top_p": None,
                "token_limit": None,
                "reasoning": None,
                "seed": None,
                "verbosity": None,
            },
            "extensions": {"extra_body": {}},
        },
        "context": "context",
        "continuations": ["answer"],
        "token_logprobs": True,
    }
    retry_payload = {
        "policy_type": "custom",
        "maximum_invocations": 2,
        "eligible_outcomes": [
            "contained_transport_timeout",
            "transient_provider_or_network_failure",
        ],
        "declared_delays_seconds": [1.0],
    }
    initial_payload = {
        "schema_version": 4,
        "request": request_payload,
        "request_hash": request_hash,
        "retry_policy": retry_payload,
        "retry_policy_hash": RETRY_POLICY_HASH,
        "classifier_identifier": (
            "dr_providers.accept_all_semantic_response.v1"
        ),
        "call_hash": call_hash,
        "completed_invocations": [],
        "completed_invocation_record_hashes": [],
        "next_invocation_ordinal": 1,
    }
    first_evidence_payload = {
        "request_hash": request_hash,
        "kind": "score",
        "policy_identity": None,
        "max_request_bytes": None,
        "max_response_bytes": None,
        "http_request": None,
        "response_bytes": None,
        "retry_after": None,
        "local_execution": None,
        "wall_time_seconds": None,
        "response": None,
        "score_response": None,
        "failure": {
            "recoverability": "transient",
            "code": "connection_reset",
            "message": "retryable failure",
            "traceback": None,
            "response_body": None,
            "status_code": None,
            "containment": None,
            "metadata": {},
        },
    }
    first_record_payload = {
        "schema_version": 4,
        "observation": {
            "schema_version": 4,
            "invocation_ordinal": 1,
            "request_hash": request_hash,
            "evidence": first_evidence_payload,
            "evidence_hash": first_evidence_hash,
            "outcome": "transient_provider_or_network_failure",
        },
        "retry_decision": {
            "source": "provider_call_retry_policy",
            "delay_seconds": 1.0,
        },
    }
    second_evidence_payload = {
        **first_evidence_payload,
        "failure": None,
        "score_response": {
            "scores": [
                {
                    "log_likelihood": -2.0,
                    "token_count": 1,
                    "char_count": 6,
                    "token_logprobs": [-2.0],
                }
            ],
            "usage": None,
            "cost": None,
            "warnings": [],
            "model": "m",
        },
    }
    second_record_payload = {
        "schema_version": 4,
        "observation": {
            "schema_version": 4,
            "invocation_ordinal": 2,
            "request_hash": request_hash,
            "evidence": second_evidence_payload,
            "evidence_hash": second_evidence_hash,
            "outcome": "success",
        },
        "retry_decision": None,
    }
    assert state.model_dump(mode="json") == initial_payload
    assert instruction.model_dump(mode="json") == {
        "schema_version": 4,
        "source": "provider_call_retry_policy",
        "delay_seconds": 1.0,
        "next_invocation_ordinal": 2,
        "next_state": {
            **initial_payload,
            "completed_invocations": [first_record_payload],
            "completed_invocation_record_hashes": [first_record_hash],
            "next_invocation_ordinal": 2,
        },
    }
    result_payload = {
        key: value
        for key, value in initial_payload.items()
        if key != "next_invocation_ordinal"
    }
    assert restored_result.model_dump(mode="json") == {
        **result_payload,
        "completed_invocations": [first_record_payload, second_record_payload],
        "completed_invocation_record_hashes": [
            first_record_hash,
            second_record_hash,
        ],
        "outcome": {"kind": "accepted", "invocation_outcome": "success"},
    }
    assert restored_result.identity_hash == (
        "38bacd16e1200e76c41e475816db2d31d6ea80220d34705735afb673fa3a369e"
    )


class _LocalEvidenceScriptedProvider(ScriptedProvider):
    def __init__(
        self,
        outcomes: list[ScriptedOutcome],
        *,
        execution: LocalExecutionEvidence,
        wall_time: float,
    ) -> None:
        super().__init__(outcomes)
        self.execution = execution
        self.wall_time = wall_time

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        evidence = super().invoke(request)
        if evidence.score_response is None:
            return evidence
        return ProviderInvocationEvidence.build(
            request=request,
            policy=None,
            http_request=None,
            outcome=evidence.score_response,
            local_execution=self.execution,
            wall_time_seconds=self.wall_time,
        )


def test_local_score_handoff_preserves_execution_and_excludes_wall_time(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    request = ProviderScoreRequest(
        config=local_config,
        context="",
        continuations=("answer",),
        token_logprobs=True,
    )
    classifier = AcceptAllSemanticResponseClassifier()
    state = ProviderCallState.initial(
        request=request,
        retry_policy=two_invocation_transient_retry_policy(),
        classifier_identifier=classifier.identifier,
    )
    outcomes = [
        ScriptedOutcome(
            failure=ProviderTransportFailure(
                code="connection_reset",
                recoverability=RecoverabilityClass.TRANSIENT,
                message="scripted retry",
            )
        ),
        ScriptedOutcome(
            scores=(
                ContinuationScore(
                    log_likelihood=-1.0,
                    token_count=1,
                    char_count=6,
                    token_logprobs=(-1.0,),
                ),
            ),
            cost=CostInfo(total_cost=0.0),
            usage=TokenUsage(prompt_tokens=2, total_tokens=2),
        ),
    ]
    with _LocalEvidenceScriptedProvider(
        outcomes, execution=local_execution, wall_time=1.25
    ) as durable:
        instruction = transition_provider_call(
            state, _observation(state, durable.invoke(request))
        )
        assert isinstance(instruction, ProviderRetryInstruction)
        restored = ProviderRetryInstruction.model_validate_json(
            instruction.model_dump_json()
        )
        next_state = restored.next_state
        result = transition_provider_call(
            next_state,
            _observation(next_state, durable.invoke(next_state.request)),
        )
        assert isinstance(result, ProviderCallResult)
        restored_result = ProviderCallResult.model_validate_json(
            result.model_dump_json()
        )
        assert durable.payloads == []
    with _LocalEvidenceScriptedProvider(
        outcomes, execution=local_execution, wall_time=1.25
    ) as local:
        local_result = run_local_provider_call(
            provider=local,
            state=state,
            classifier=classifier,
            cancellation=Event(),
            retry_wait=_NoWait(),
        )
    assert restored_result == local_result
    evidence = restored_result.completed_invocations[-1].observation.evidence
    assert evidence.local_execution == local_execution
    assert evidence.wall_time_seconds == 1.25
    data = restored_result.model_dump(mode="json")
    data["completed_invocations"][-1]["observation"]["evidence"][
        "wall_time_seconds"
    ] = 99.0
    changed = ProviderCallResult.model_validate(data)
    assert changed != restored_result
    assert (
        changed.identity_hash
        == restored_result.identity_hash
        == "6ae8ddef40f217eb7c913ba71b44084df6ca0250ec12b2c49ce98e817a5419a1"
    )
