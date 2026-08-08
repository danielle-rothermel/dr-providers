from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class ProviderInvocationOutcome(StrEnum):
    """Closed, evaluation-neutral outcome of one provider invocation."""

    SUCCESS = "success"
    BLANK_RESPONSE = "blank_response"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER_REJECTION = "provider_rejection"
    SEMANTIC_REJECTION = "semantic_rejection"
    PERMANENT_PROVIDER_OR_TRANSPORT_FAILURE = (
        "permanent_provider_or_transport_failure"
    )
    TRANSIENT_PROVIDER_OR_NETWORK_FAILURE = (
        "transient_provider_or_network_failure"
    )
    RATE_LIMITING = "rate_limiting"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONTAINED_TRANSPORT_TIMEOUT = "contained_transport_timeout"
    UNCONTAINED_DEADLINE_EXPIRATION = "uncontained_deadline_expiration"
    UNKNOWN_TRANSPORT_FAILURE = "unknown_transport_failure"


class ProviderCallOutcomeKind(StrEnum):
    ACCEPTED = "accepted"
    INVOCATION_OUTCOME = "invocation_outcome"
    DRAINING_CANCELLATION = "draining_cancellation"
    POLICY_EXHAUSTION = "policy_exhaustion"


class ProviderCallOutcome(BaseModel):
    """Closed terminal classification of a provider call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ProviderCallOutcomeKind
    invocation_outcome: ProviderInvocationOutcome | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> ProviderCallOutcome:
        if self.kind is ProviderCallOutcomeKind.DRAINING_CANCELLATION:
            if self.invocation_outcome is not None:
                msg = "draining cancellation has no invocation outcome"
                raise ValueError(msg)
            return self
        if self.invocation_outcome is None:
            msg = f"{self.kind.value} requires an invocation outcome"
            raise ValueError(msg)
        if (
            self.kind is ProviderCallOutcomeKind.ACCEPTED
            and self.invocation_outcome
            is not ProviderInvocationOutcome.SUCCESS
        ):
            msg = "accepted provider calls require a successful invocation"
            raise ValueError(msg)
        if (
            self.kind is not ProviderCallOutcomeKind.ACCEPTED
            and self.invocation_outcome is ProviderInvocationOutcome.SUCCESS
        ):
            msg = "successful invocations require an accepted call outcome"
            raise ValueError(msg)
        return self
