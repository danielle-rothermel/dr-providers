from __future__ import annotations

import pytest

from dr_providers.core.failures import RecoverabilityClass
from dr_providers.lifecycle import (
    ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER,
    AcceptAllSemanticResponseClassifier,
    ProviderInvocationOutcome,
    SemanticResponseClassifierIdentifier,
    classify_provider_invocation,
)
from dr_providers.outcomes.evidence import ProviderInvocationEvidence
from dr_providers.outcomes.models import (
    INVALID_JSON_CODE,
    STALLED_RESPONSE_CODE,
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

REQUEST_HASH = "1" * 64
CLASSIFIER = AcceptAllSemanticResponseClassifier()


def _failure_evidence(
    *,
    recoverability: RecoverabilityClass,
    code: str | None = None,
    metadata: dict[str, object] | None = None,
    containment: TransportTimeoutContainment | None = None,
) -> ProviderInvocationEvidence:
    return ProviderInvocationEvidence(
        request_identity_hash=REQUEST_HASH,
        failure=ProviderTransportFailure(
            recoverability=recoverability,
            code=code,
            message="classified failure",
            containment=containment,
            metadata=metadata or {},
        ),
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            RESPONSE_NO_TEXT_CODE,
            ProviderInvocationOutcome.MISSING_GENERATION_TEXT,
        ),
        (
            RESPONSE_INCOMPLETE_NO_TEXT_CODE,
            ProviderInvocationOutcome.TRUNCATED_NO_TEXT,
        ),
        (PARSE_ERROR_CODE, ProviderInvocationOutcome.MALFORMED_RESPONSE),
        (INVALID_JSON_CODE, ProviderInvocationOutcome.MALFORMED_RESPONSE),
        (
            RESPONSE_REFUSAL_CODE,
            ProviderInvocationOutcome.PROVIDER_REJECTION,
        ),
        (
            RESPONSE_FAILED_CODE,
            ProviderInvocationOutcome.PROVIDER_REJECTION,
        ),
        ("missing_api_key", ProviderInvocationOutcome.MISSING_CREDENTIAL),
        (
            "missing_base_url",
            ProviderInvocationOutcome.MISSING_TRANSPORT_CONFIG,
        ),
        ("http_status_402", ProviderInvocationOutcome.BUDGET_EXHAUSTED),
    ],
)
def test_protocol_outcome_precedes_recoverability(
    code: str,
    expected: ProviderInvocationOutcome,
) -> None:
    evidence = _failure_evidence(
        recoverability=RecoverabilityClass.PERMANENT,
        code=code,
    )

    assert classify_provider_invocation(evidence, CLASSIFIER) is expected


@pytest.mark.parametrize(
    ("recoverability", "expected"),
    [
        (
            RecoverabilityClass.PERMANENT,
            ProviderInvocationOutcome.PERMANENT_PROVIDER_OR_TRANSPORT_FAILURE,
        ),
        (
            RecoverabilityClass.TRANSIENT,
            ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE,
        ),
        (
            RecoverabilityClass.RATE_LIMITED,
            ProviderInvocationOutcome.RATE_LIMITING,
        ),
        (
            RecoverabilityClass.RESOURCE_EXHAUSTION,
            ProviderInvocationOutcome.RESOURCE_EXHAUSTION,
        ),
        (
            RecoverabilityClass.UNKNOWN,
            ProviderInvocationOutcome.UNKNOWN_TRANSPORT_FAILURE,
        ),
    ],
)
def test_recoverability_maps_to_closed_outcome(
    recoverability: RecoverabilityClass,
    expected: ProviderInvocationOutcome,
) -> None:
    evidence = _failure_evidence(recoverability=recoverability)

    assert classify_provider_invocation(evidence, CLASSIFIER) is expected


def test_timeout_classification_exposes_containment() -> None:
    contained = _failure_evidence(
        recoverability=RecoverabilityClass.TRANSIENT,
        code=STALLED_RESPONSE_CODE,
        containment=TransportTimeoutContainment.CONTAINED,
    )
    uncontained = _failure_evidence(
        recoverability=RecoverabilityClass.TRANSIENT,
        code=STALLED_RESPONSE_CODE,
        metadata={"deadline_seconds": 5.0},
    )

    assert (
        classify_provider_invocation(contained, CLASSIFIER)
        is ProviderInvocationOutcome.CONTAINED_TRANSPORT_TIMEOUT
    )
    assert (
        classify_provider_invocation(uncontained, CLASSIFIER)
        is ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION
    )


def test_empty_generation_precedes_semantic_classifier() -> None:
    class RejectingClassifier:
        identifier = SemanticResponseClassifierIdentifier("rejecting-v1")

        def classify(
            self, response: ProviderTransportResponse
        ) -> ProviderInvocationOutcome:
            del response
            raise AssertionError(
                "empty generation reached semantic classifier"
            )

    evidence = ProviderInvocationEvidence(
        request_identity_hash=REQUEST_HASH,
        response=ProviderTransportResponse(text="  "),
    )

    assert (
        classify_provider_invocation(evidence, RejectingClassifier())
        is ProviderInvocationOutcome.EMPTY_GENERATION
    )


def test_accept_all_classifier_has_stable_identifier() -> None:
    assert CLASSIFIER.identifier == ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER
    assert CLASSIFIER.identifier.root == (
        "dr_providers.accept_all_semantic_response.v1"
    )
