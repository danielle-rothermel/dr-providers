from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, RootModel, StrictStr

from dr_providers.core.failures import FailureClass
from dr_providers.lifecycle.outcomes import ProviderInvocationOutcome
from dr_providers.outcomes.models import (
    INVALID_JSON_CODE,
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
    ProviderTransportFailure,
    ProviderTransportResponse,
)
from dr_providers.translation.common import (
    PARSE_ERROR_CODE,
    RESPONSE_NO_TEXT_CODE,
)
from dr_providers.translation.responses import (
    RESPONSE_FAILED_CODE,
    RESPONSE_INCOMPLETE_NO_TEXT_CODE,
    RESPONSE_REFUSAL_CODE,
)

if TYPE_CHECKING:
    from dr_providers.outcomes.evidence import ProviderInvocationEvidence


class SemanticResponseClassifierIdentifier(
    RootModel[Annotated[StrictStr, Field(min_length=1)]]
):
    """Opaque caller-owned stable identifier for classifier behavior."""

    model_config = ConfigDict(frozen=True)


ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER = (
    SemanticResponseClassifierIdentifier(
        "dr_providers.accept_all_semantic_response.v1"
    )
)


@runtime_checkable
class SemanticResponseClassifier(Protocol):
    @property
    def identifier(self) -> SemanticResponseClassifierIdentifier: ...

    def classify(
        self, response: ProviderTransportResponse
    ) -> ProviderInvocationOutcome: ...


@dataclass(frozen=True, slots=True)
class AcceptAllSemanticResponseClassifier:
    """Accept every nonblank protocol-valid provider response."""

    identifier: SemanticResponseClassifierIdentifier = (
        ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER
    )

    def classify(
        self, response: ProviderTransportResponse
    ) -> ProviderInvocationOutcome:
        del response
        return ProviderInvocationOutcome.SUCCESS


def classify_semantic_response(
    classifier: SemanticResponseClassifier,
    response: ProviderTransportResponse,
) -> ProviderInvocationOutcome:
    """Apply the classifier while rejecting outcomes outside its boundary."""
    outcome = classifier.classify(response)
    allowed = {
        ProviderInvocationOutcome.SUCCESS,
        ProviderInvocationOutcome.SEMANTIC_REJECTION,
    }
    if outcome not in allowed:
        msg = (
            "semantic response classifier must return success or semantic "
            f"rejection, got {outcome!r}"
        )
        raise ValueError(msg)
    return outcome


def classify_provider_invocation(
    evidence: ProviderInvocationEvidence,
    classifier: SemanticResponseClassifier,
) -> ProviderInvocationOutcome:
    """Classify transport and protocol evidence before semantic response."""
    if evidence.response is not None:
        if not evidence.response.text.strip():
            return ProviderInvocationOutcome.BLANK_RESPONSE
        return classify_semantic_response(classifier, evidence.response)
    assert evidence.failure is not None
    return classify_provider_failure(evidence.failure)


def classify_provider_failure(
    failure: ProviderTransportFailure,
) -> ProviderInvocationOutcome:
    """Deterministically classify provider failure evidence."""
    if failure.code in {
        RESPONSE_NO_TEXT_CODE,
        RESPONSE_INCOMPLETE_NO_TEXT_CODE,
    }:
        return ProviderInvocationOutcome.BLANK_RESPONSE
    if failure.code in {PARSE_ERROR_CODE, INVALID_JSON_CODE}:
        return ProviderInvocationOutcome.MALFORMED_RESPONSE
    if failure.code in {RESPONSE_REFUSAL_CODE, RESPONSE_FAILED_CODE}:
        return ProviderInvocationOutcome.PROVIDER_REJECTION
    if failure.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}:
        if "phase" in failure.metadata:
            return ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT
        return ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION
    by_failure_class = {
        FailureClass.PERMANENT: (
            ProviderInvocationOutcome.PERMANENT_PROVIDER_OR_TRANSPORT_FAILURE
        ),
        FailureClass.TRANSIENT: (
            ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE
        ),
        FailureClass.RATE_LIMITED: ProviderInvocationOutcome.RATE_LIMITING,
        FailureClass.RESOURCE_EXHAUSTION: (
            ProviderInvocationOutcome.RESOURCE_EXHAUSTION
        ),
        FailureClass.UNKNOWN: (
            ProviderInvocationOutcome.UNKNOWN_TRANSPORT_FAILURE
        ),
    }
    return by_failure_class[failure.failure_class]
