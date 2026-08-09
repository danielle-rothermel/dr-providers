from __future__ import annotations

import math
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
    field_serializer,
    model_validator,
)

from dr_providers.lifecycle.outcomes import ProviderInvocationOutcome

PROVIDER_CALL_RETRY_POLICY_SCHEMA = "dr_providers.provider_call_retry_policy"
PROVIDER_CALL_RETRY_POLICY_SCHEMA_VERSION = 1

STANDARD_RETRY_ELIGIBLE_OUTCOMES = (
    ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE,
    ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT,
)
STANDARD_RETRY_DELAYS_SECONDS = (1.0,)

type NonNegativeFiniteFloat = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False, strict=True),
]


class StandardProviderCallRetryPolicy(BaseModel):
    """The exact standard two-invocation, one-second retry policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_type: Literal["standard"] = "standard"
    maximum_invocations: Literal[2] = 2
    eligible_outcomes: tuple[ProviderInvocationOutcome, ...] = (
        STANDARD_RETRY_ELIGIBLE_OUTCOMES
    )
    declared_delays_seconds: tuple[NonNegativeFiniteFloat, ...] = (
        STANDARD_RETRY_DELAYS_SECONDS
    )
    maximum_cumulative_delay_seconds: NonNegativeFiniteFloat = 1.0

    @model_validator(mode="after")
    def _require_exact_standard_policy(
        self,
    ) -> StandardProviderCallRetryPolicy:
        if self.eligible_outcomes != STANDARD_RETRY_ELIGIBLE_OUTCOMES:
            msg = "standard retry eligibility is fixed"
            raise ValueError(msg)
        if self.declared_delays_seconds != STANDARD_RETRY_DELAYS_SECONDS:
            msg = "standard retry delays are fixed"
            raise ValueError(msg)
        if self.maximum_cumulative_delay_seconds != 1.0:
            msg = "standard maximum cumulative delay is fixed"
            raise ValueError(msg)
        return self

    def identity_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_CALL_RETRY_POLICY_SCHEMA,
            schema_version=PROVIDER_CALL_RETRY_POLICY_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())

    def retry_delay_after(self, invocation_ordinal: int) -> float:
        return self.declared_delays_seconds[invocation_ordinal - 1]


class CustomProviderCallRetryPolicy(BaseModel):
    """Closed deterministic retry data selected explicitly by a caller."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_type: Literal["custom"] = "custom"
    maximum_invocations: StrictInt = Field(gt=0)
    eligible_outcomes: frozenset[ProviderInvocationOutcome]
    declared_delays_seconds: tuple[NonNegativeFiniteFloat, ...]

    @field_serializer("eligible_outcomes", when_used="json")
    def _serialize_eligible_outcomes(
        self, outcomes: frozenset[ProviderInvocationOutcome]
    ) -> list[str]:
        return sorted(outcome.value for outcome in outcomes)

    @model_validator(mode="after")
    def _validate_policy(self) -> CustomProviderCallRetryPolicy:
        expected_delay_count = self.maximum_invocations - 1
        if len(self.declared_delays_seconds) != expected_delay_count:
            msg = (
                "custom retry policy requires exactly one declared delay "
                "between each permitted invocation"
            )
            raise ValueError(msg)
        if not math.isfinite(self.maximum_cumulative_delay_seconds):
            msg = "custom retry policy cumulative delay must be finite"
            raise ValueError(msg)
        forbidden = {
            ProviderInvocationOutcome.SUCCESS,
            ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION,
        }
        selected_forbidden = self.eligible_outcomes & forbidden
        if selected_forbidden:
            values = sorted(outcome.value for outcome in selected_forbidden)
            msg = f"custom retry eligibility forbids outcomes {values!r}"
            raise ValueError(msg)
        return self

    @property
    def maximum_cumulative_delay_seconds(self) -> float:
        return sum(self.declared_delays_seconds)

    def identity_payload(self) -> dict[str, object]:
        payload = self.model_dump(mode="json")
        payload["maximum_cumulative_delay_seconds"] = (
            self.maximum_cumulative_delay_seconds
        )
        return payload

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_CALL_RETRY_POLICY_SCHEMA,
            schema_version=PROVIDER_CALL_RETRY_POLICY_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())

    def retry_delay_after(self, invocation_ordinal: int) -> float:
        return self.declared_delays_seconds[invocation_ordinal - 1]


type ProviderCallRetryPolicy = Annotated[
    StandardProviderCallRetryPolicy | CustomProviderCallRetryPolicy,
    Field(discriminator="policy_type"),
]
