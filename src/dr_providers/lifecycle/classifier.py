from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, RootModel, StrictStr

from dr_providers.core.failures import RecoverabilityClass
from dr_providers.lifecycle.outcomes import ProviderInvocationOutcome
from dr_providers.outcomes.models import (
    INVALID_JSON_CODE,
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
    ProviderTransportFailure,
    ProviderTransportResponse,
    TransportTimeoutContainment,
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

MISSING_API_KEY_CODE = "missing_api_key"
MISSING_BASE_URL_CODE = "missing_base_url"
HTTP_STATUS_402_CODE = "http_status_402"

CODE_TO_OUTCOME = {
    RESPONSE_INCOMPLETE_NO_TEXT_CODE: (
        ProviderInvocationOutcome.TRUNCATED_NO_TEXT
    ),
    RESPONSE_NO_TEXT_CODE: ProviderInvocationOutcome.MISSING_GENERATION_TEXT,
    MISSING_API_KEY_CODE: ProviderInvocationOutcome.MISSING_CREDENTIAL,
    MISSING_BASE_URL_CODE: ProviderInvocationOutcome.MISSING_TRANSPORT_CONFIG,
    HTTP_STATUS_402_CODE: ProviderInvocationOutcome.BUDGET_EXHAUSTED,
    PARSE_ERROR_CODE: ProviderInvocationOutcome.MALFORMED_RESPONSE,
    INVALID_JSON_CODE: ProviderInvocationOutcome.MALFORMED_RESPONSE,
    RESPONSE_REFUSAL_CODE: ProviderInvocationOutcome.PROVIDER_REJECTION,
    RESPONSE_FAILED_CODE: ProviderInvocationOutcome.PROVIDER_REJECTION,
}


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
            return ProviderInvocationOutcome.EMPTY_GENERATION
        return classify_semantic_response(classifier, evidence.response)
    assert evidence.failure is not None
    return classify_provider_failure(evidence.failure)


def classify_provider_failure(
    failure: ProviderTransportFailure,
) -> ProviderInvocationOutcome:
    """Deterministically classify provider failure evidence."""
    if failure.code in CODE_TO_OUTCOME:
        return CODE_TO_OUTCOME[failure.code]
    if failure.code in {TIMEOUT_CODE, STALLED_RESPONSE_CODE}:
        if failure.containment is TransportTimeoutContainment.CONTAINED:
            return ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT
        return ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION
    by_recoverability = {
        RecoverabilityClass.PERMANENT: (
            ProviderInvocationOutcome.PERMANENT_PROVIDER_OR_TRANSPORT_FAILURE
        ),
        RecoverabilityClass.TRANSIENT: (
            ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE
        ),
        RecoverabilityClass.RATE_LIMITED: (
            ProviderInvocationOutcome.RATE_LIMITING
        ),
        RecoverabilityClass.RESOURCE_EXHAUSTION: (
            ProviderInvocationOutcome.RESOURCE_EXHAUSTION
        ),
        RecoverabilityClass.UNKNOWN: (
            ProviderInvocationOutcome.UNKNOWN_TRANSPORT_FAILURE
        ),
    }
    return by_recoverability[failure.recoverability]
