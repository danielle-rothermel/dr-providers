from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from typing import Annotated, Literal

from dr_serialize import (
    IdentityDocument,
    build_identity_document,
    identity_document_hash,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from dr_providers.lifecycle.classifier import (
    SemanticResponseClassifierIdentifier,  # noqa: TC001 -- pydantic field
)
from dr_providers.lifecycle.outcomes import (
    ProviderCallOutcome,
    ProviderCallOutcomeKind,
    ProviderInvocationOutcome,
)
from dr_providers.lifecycle.policy import (  # noqa: TC001 -- pydantic field
    ProviderCallRetryPolicy,
)
from dr_providers.modeling.request import (  # noqa: TC001 -- pydantic field
    ProviderCallRequest,
)
from dr_providers.outcomes.evidence import (  # noqa: TC001 -- pydantic field
    ProviderInvocationEvidence,
)

PROVIDER_CALL_SCHEMA = "dr_providers.provider_call"
PROVIDER_CALL_SCHEMA_VERSION = 1
COMPLETED_INVOCATION_OBSERVATION_SCHEMA = (
    "dr_providers.completed_invocation_observation"
)
COMPLETED_INVOCATION_OBSERVATION_SCHEMA_VERSION = 1
DECIDED_INVOCATION_RECORD_SCHEMA = "dr_providers.decided_invocation_record"
DECIDED_INVOCATION_RECORD_SCHEMA_VERSION = 1
PROVIDER_CALL_STATE_SCHEMA_VERSION = 1
PROVIDER_RETRY_INSTRUCTION_SCHEMA_VERSION = 1
PROVIDER_CALL_RESULT_SCHEMA = "dr_providers.provider_call_result"
PROVIDER_CALL_RESULT_SCHEMA_VERSION = 1

ContentIdentityHash = Annotated[
    StrictStr,
    Field(pattern=r"^[0-9a-f]{64}$"),
]


def provider_call_identity_document(
    *,
    request_identity_hash: str,
    retry_policy_identity_hash: str,
    classifier_identifier: SemanticResponseClassifierIdentifier,
) -> IdentityDocument:
    return build_identity_document(
        schema=PROVIDER_CALL_SCHEMA,
        schema_version=PROVIDER_CALL_SCHEMA_VERSION,
        payload={
            "provider_call_schema_version": PROVIDER_CALL_SCHEMA_VERSION,
            "request_identity_hash": request_identity_hash,
            "retry_policy_identity_hash": retry_policy_identity_hash,
            "classifier_identifier": classifier_identifier.root,
        },
    )


def provider_call_identity_hash(
    *,
    request_identity_hash: str,
    retry_policy_identity_hash: str,
    classifier_identifier: SemanticResponseClassifierIdentifier,
) -> str:
    return identity_document_hash(
        provider_call_identity_document(
            request_identity_hash=request_identity_hash,
            retry_policy_identity_hash=retry_policy_identity_hash,
            classifier_identifier=classifier_identifier,
        )
    )


class ProviderRetryDelaySource(StrEnum):
    PROVIDER_CALL_RETRY_POLICY = "provider_call_retry_policy"


class ProviderRetryDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal[ProviderRetryDelaySource.PROVIDER_CALL_RETRY_POLICY] = (
        ProviderRetryDelaySource.PROVIDER_CALL_RETRY_POLICY
    )
    delay_seconds: float = Field(ge=0, allow_inf_nan=False, strict=True)


class CompletedProviderInvocationObservation(BaseModel):
    """Completed evidence and classification before any retry decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = (
        COMPLETED_INVOCATION_OBSERVATION_SCHEMA_VERSION
    )
    invocation_ordinal: StrictInt = Field(gt=0)
    request_identity_hash: ContentIdentityHash
    evidence: ProviderInvocationEvidence
    evidence_identity_hash: ContentIdentityHash
    outcome: ProviderInvocationOutcome

    @model_validator(mode="after")
    def _validate_observation(self) -> CompletedProviderInvocationObservation:
        if self.evidence_identity_hash != self.evidence.identity_hash:
            msg = "evidence identity hash does not match embedded evidence"
            raise ValueError(msg)
        response_outcomes = {
            ProviderInvocationOutcome.SUCCESS,
            ProviderInvocationOutcome.BLANK_RESPONSE,
            ProviderInvocationOutcome.SEMANTIC_REJECTION,
        }
        if self.evidence.response is not None:
            if self.outcome not in response_outcomes:
                msg = "provider response evidence has a failure outcome"
                raise ValueError(msg)
            is_blank = not self.evidence.response.text.strip()
            if (
                self.outcome is ProviderInvocationOutcome.BLANK_RESPONSE
            ) is not is_blank:
                msg = "blank response outcome must match response text"
                raise ValueError(msg)
        elif self.outcome in response_outcomes:
            msg = "provider failure evidence has a response outcome"
            raise ValueError(msg)
        return self

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "invocation_ordinal": self.invocation_ordinal,
            "request_identity_hash": self.request_identity_hash,
            "evidence_identity_hash": self.evidence_identity_hash,
            "outcome": self.outcome.value,
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=COMPLETED_INVOCATION_OBSERVATION_SCHEMA,
            schema_version=COMPLETED_INVOCATION_OBSERVATION_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())


class DecidedProviderInvocationRecord(BaseModel):
    """A completed observation plus the reducer-selected retry decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = DECIDED_INVOCATION_RECORD_SCHEMA_VERSION
    observation: CompletedProviderInvocationObservation
    retry_decision: ProviderRetryDecision | None = None

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "observation_identity_hash": self.observation.identity_hash,
            "retry_decision": (
                None
                if self.retry_decision is None
                else self.retry_decision.model_dump(mode="json")
            ),
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=DECIDED_INVOCATION_RECORD_SCHEMA,
            schema_version=DECIDED_INVOCATION_RECORD_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())


def _validate_components(  # noqa: PLR0913 -- validates one component set
    *,
    request: ProviderCallRequest,
    request_identity_hash: str,
    retry_policy: ProviderCallRetryPolicy,
    retry_policy_identity_hash: str,
    classifier_identifier: SemanticResponseClassifierIdentifier,
    call_identity_hash: str,
) -> None:
    if request_identity_hash != request.identity_hash:
        msg = "request identity hash does not match embedded request"
        raise ValueError(msg)
    if retry_policy_identity_hash != retry_policy.identity_hash:
        msg = "retry policy identity hash does not match embedded policy"
        raise ValueError(msg)
    expected_call_identity_hash = provider_call_identity_hash(
        request_identity_hash=request_identity_hash,
        retry_policy_identity_hash=retry_policy_identity_hash,
        classifier_identifier=classifier_identifier,
    )
    if call_identity_hash != expected_call_identity_hash:
        msg = "provider call identity hash does not match call components"
        raise ValueError(msg)


def _validate_records(
    *,
    records: tuple[DecidedProviderInvocationRecord, ...],
    record_hashes: tuple[str, ...],
    request: ProviderCallRequest,
    request_identity_hash: str,
    retry_policy: ProviderCallRetryPolicy,
) -> None:
    if len(records) != len(record_hashes):
        msg = "completed records and record hashes must have equal length"
        raise ValueError(msg)
    if len(records) > retry_policy.maximum_invocations:
        msg = "completed history exceeds retry policy invocation limit"
        raise ValueError(msg)
    for expected_ordinal, (record, declared_hash) in enumerate(
        zip(records, record_hashes, strict=True),
        start=1,
    ):
        observation = record.observation
        if observation.invocation_ordinal != expected_ordinal:
            msg = "completed invocation ordinals must be contiguous from one"
            raise ValueError(msg)
        if observation.request_identity_hash != request_identity_hash:
            msg = "completed observation has a different request identity"
            raise ValueError(msg)
        evidence_request_identity = observation.evidence.model_dump(
            mode="json"
        )["request_identity"]
        if evidence_request_identity != request.identity_payload():
            msg = "invocation evidence has a different request identity"
            raise ValueError(msg)
        if declared_hash != record.identity_hash:
            msg = "completed record hash does not match embedded record"
            raise ValueError(msg)
        decision = record.retry_decision
        if decision is None:
            continue
        if observation.outcome not in retry_policy.eligible_outcomes:
            msg = "retry decision follows an ineligible invocation outcome"
            raise ValueError(msg)
        if (
            observation.outcome
            is ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION
        ):
            msg = (
                "uncontained deadline expiration cannot have a retry decision"
            )
            raise ValueError(msg)
        if expected_ordinal >= retry_policy.maximum_invocations:
            msg = (
                "retry decision is forbidden at the maximum permitted ordinal"
            )
            raise ValueError(msg)
        expected_delay = retry_policy.retry_delay_after(expected_ordinal)
        if decision.delay_seconds != expected_delay:
            msg = "retry decision delay does not match retry policy"
            raise ValueError(msg)


class ProviderCallState(BaseModel):
    """Serializable nonterminal provider-call reducer state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = PROVIDER_CALL_STATE_SCHEMA_VERSION
    request: ProviderCallRequest
    request_identity_hash: ContentIdentityHash
    retry_policy: ProviderCallRetryPolicy
    retry_policy_identity_hash: ContentIdentityHash
    classifier_identifier: SemanticResponseClassifierIdentifier
    call_identity_hash: ContentIdentityHash
    completed_invocations: tuple[DecidedProviderInvocationRecord, ...] = ()
    completed_invocation_record_hashes: tuple[ContentIdentityHash, ...] = ()
    next_invocation_ordinal: StrictInt = Field(gt=0)

    @classmethod
    def initial(
        cls,
        *,
        request: ProviderCallRequest,
        retry_policy: ProviderCallRetryPolicy,
        classifier_identifier: SemanticResponseClassifierIdentifier,
    ) -> ProviderCallState:
        request_hash = request.identity_hash
        retry_policy_hash = retry_policy.identity_hash
        return cls(
            request=request,
            request_identity_hash=request_hash,
            retry_policy=retry_policy,
            retry_policy_identity_hash=retry_policy_hash,
            classifier_identifier=classifier_identifier,
            call_identity_hash=provider_call_identity_hash(
                request_identity_hash=request_hash,
                retry_policy_identity_hash=retry_policy_hash,
                classifier_identifier=classifier_identifier,
            ),
            next_invocation_ordinal=1,
        )

    @model_validator(mode="after")
    def _validate_state(self) -> ProviderCallState:
        _validate_components(
            request=self.request,
            request_identity_hash=self.request_identity_hash,
            retry_policy=self.retry_policy,
            retry_policy_identity_hash=self.retry_policy_identity_hash,
            classifier_identifier=self.classifier_identifier,
            call_identity_hash=self.call_identity_hash,
        )
        _validate_records(
            records=self.completed_invocations,
            record_hashes=self.completed_invocation_record_hashes,
            request=self.request,
            request_identity_hash=self.request_identity_hash,
            retry_policy=self.retry_policy,
        )
        if self.next_invocation_ordinal != len(self.completed_invocations) + 1:
            msg = "next invocation ordinal must follow completed records"
            raise ValueError(msg)
        if (
            self.next_invocation_ordinal
            > self.retry_policy.maximum_invocations
        ):
            msg = "nonterminal state exceeds retry policy invocation limit"
            raise ValueError(msg)
        if any(
            record.retry_decision is None
            for record in self.completed_invocations
        ):
            msg = "nonterminal state cannot contain a terminal record"
            raise ValueError(msg)
        return self


class ProviderRetryInstruction(BaseModel):
    """Serializable declaration of the next permitted invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = PROVIDER_RETRY_INSTRUCTION_SCHEMA_VERSION
    source: Literal[ProviderRetryDelaySource.PROVIDER_CALL_RETRY_POLICY] = (
        ProviderRetryDelaySource.PROVIDER_CALL_RETRY_POLICY
    )
    delay_seconds: float = Field(ge=0, allow_inf_nan=False, strict=True)
    next_invocation_ordinal: StrictInt = Field(gt=0)
    next_state: ProviderCallState

    @model_validator(mode="after")
    def _validate_instruction(self) -> ProviderRetryInstruction:
        if (
            self.next_invocation_ordinal
            != self.next_state.next_invocation_ordinal
        ):
            msg = "retry instruction ordinal must match next state"
            raise ValueError(msg)
        if not self.next_state.completed_invocations:
            msg = "retry instruction requires one completed invocation"
            raise ValueError(msg)
        decision = self.next_state.completed_invocations[-1].retry_decision
        if decision is None:
            msg = "retry instruction requires a decided retry"
            raise ValueError(msg)
        if self.source is not decision.source:
            msg = "retry instruction source must match decided record"
            raise ValueError(msg)
        if self.delay_seconds != decision.delay_seconds:
            msg = "retry instruction delay must match decided record"
            raise ValueError(msg)
        return self


class ProviderCallResult(BaseModel):
    """Self-contained terminal call result with compositional identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = PROVIDER_CALL_RESULT_SCHEMA_VERSION
    request: ProviderCallRequest
    request_identity_hash: ContentIdentityHash
    retry_policy: ProviderCallRetryPolicy
    retry_policy_identity_hash: ContentIdentityHash
    classifier_identifier: SemanticResponseClassifierIdentifier
    call_identity_hash: ContentIdentityHash
    completed_invocations: tuple[DecidedProviderInvocationRecord, ...]
    completed_invocation_record_hashes: tuple[ContentIdentityHash, ...]
    outcome: ProviderCallOutcome

    @model_validator(mode="after")
    def _validate_result(self) -> ProviderCallResult:
        _validate_components(
            request=self.request,
            request_identity_hash=self.request_identity_hash,
            retry_policy=self.retry_policy,
            retry_policy_identity_hash=self.retry_policy_identity_hash,
            classifier_identifier=self.classifier_identifier,
            call_identity_hash=self.call_identity_hash,
        )
        _validate_records(
            records=self.completed_invocations,
            record_hashes=self.completed_invocation_record_hashes,
            request=self.request,
            request_identity_hash=self.request_identity_hash,
            retry_policy=self.retry_policy,
        )
        self._validate_terminal_shape()
        return self

    def _validate_terminal_shape(self) -> None:
        if self.outcome.kind is ProviderCallOutcomeKind.DRAINING_CANCELLATION:
            self._validate_cancellation_records()
            return
        if not self.completed_invocations:
            msg = "non-cancellation result requires a completed invocation"
            raise ValueError(msg)
        if any(
            record.retry_decision is None
            for record in self.completed_invocations[:-1]
        ):
            msg = "only the final invocation record may be terminal"
            raise ValueError(msg)
        final_record = self.completed_invocations[-1]
        if final_record.retry_decision is not None:
            msg = "normal terminal result requires a terminal final record"
            raise ValueError(msg)
        final_outcome = final_record.observation.outcome
        if self.outcome.invocation_outcome is not final_outcome:
            msg = "call outcome must match the final invocation outcome"
            raise ValueError(msg)
        if self.outcome.kind is ProviderCallOutcomeKind.ACCEPTED:
            return
        if self.outcome.kind is ProviderCallOutcomeKind.POLICY_EXHAUSTION:
            if final_outcome not in self.retry_policy.eligible_outcomes:
                msg = "policy exhaustion requires an eligible final outcome"
                raise ValueError(msg)
            if (
                len(self.completed_invocations)
                != self.retry_policy.maximum_invocations
            ):
                msg = "policy exhaustion requires the invocation limit"
                raise ValueError(msg)
            return
        if final_outcome in self.retry_policy.eligible_outcomes:
            msg = "policy-stopped outcome cannot be retry eligible"
            raise ValueError(msg)

    def _validate_cancellation_records(self) -> None:
        terminal_indexes = [
            index
            for index, record in enumerate(self.completed_invocations)
            if record.retry_decision is None
        ]
        if terminal_indexes and terminal_indexes != [
            len(self.completed_invocations) - 1
        ]:
            msg = "cancellation may add only one final terminal record"
            raise ValueError(msg)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "call_identity_hash": self.call_identity_hash,
            "request_identity_hash": self.request_identity_hash,
            "retry_policy_identity_hash": self.retry_policy_identity_hash,
            "classifier_identifier": self.classifier_identifier.root,
            "completed_invocation_record_hashes": list(
                self.completed_invocation_record_hashes
            ),
            "outcome": self.outcome.model_dump(mode="json"),
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_CALL_RESULT_SCHEMA,
            schema_version=PROVIDER_CALL_RESULT_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())
