from __future__ import annotations

import pytest

from dr_providers.core.failures import FailureClass
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
)
from dr_providers.translation.common import (
    PARSE_ERROR_CODE,
    RESPONSE_NO_TEXT_CODE,
)
from dr_providers.translation.responses import (
    RESPONSE_FAILED_CODE,
    RESPONSE_REFUSAL_CODE,
)

REQUEST_HASH = "1" * 64
CLASSIFIER = AcceptAllSemanticResponseClassifier()


def _failure_evidence(
    *,
    failure_class: FailureClass,
    code: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ProviderInvocationEvidence:
    return ProviderInvocationEvidence(
        request_identity_hash=REQUEST_HASH,
        failure=ProviderTransportFailure(
            failure_class=failure_class,
            code=code,
            message="classified failure",
            metadata=metadata or {},
        ),
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (RESPONSE_NO_TEXT_CODE, ProviderInvocationOutcome.BLANK_RESPONSE),
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
    ],
)
def test_protocol_outcome_precedes_failure_class(
    code: str,
    expected: ProviderInvocationOutcome,
) -> None:
    evidence = _failure_evidence(
        failure_class=FailureClass.PERMANENT,
        code=code,
    )

    assert classify_provider_invocation(evidence, CLASSIFIER) is expected


@pytest.mark.parametrize(
    ("failure_class", "expected"),
    [
        (
            FailureClass.PERMANENT,
            ProviderInvocationOutcome.PERMANENT_PROVIDER_OR_TRANSPORT_FAILURE,
        ),
        (
            FailureClass.TRANSIENT,
            ProviderInvocationOutcome.TRANSIENT_PROVIDER_OR_NETWORK_FAILURE,
        ),
        (FailureClass.RATE_LIMITED, ProviderInvocationOutcome.RATE_LIMITING),
        (
            FailureClass.RESOURCE_EXHAUSTION,
            ProviderInvocationOutcome.RESOURCE_EXHAUSTION,
        ),
        (
            FailureClass.UNKNOWN,
            ProviderInvocationOutcome.UNKNOWN_TRANSPORT_FAILURE,
        ),
    ],
)
def test_failure_class_maps_to_closed_outcome(
    failure_class: FailureClass,
    expected: ProviderInvocationOutcome,
) -> None:
    evidence = _failure_evidence(failure_class=failure_class)

    assert classify_provider_invocation(evidence, CLASSIFIER) is expected


def test_timeout_classification_exposes_containment() -> None:
    contained = _failure_evidence(
        failure_class=FailureClass.TRANSIENT,
        code=STALLED_RESPONSE_CODE,
        metadata={"phase": "ReadTimeout"},
    )
    uncontained = _failure_evidence(
        failure_class=FailureClass.TRANSIENT,
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


def test_blank_response_precedes_semantic_classifier() -> None:
    class RejectingClassifier:
        identifier = SemanticResponseClassifierIdentifier("rejecting-v1")

        def classify(
            self, response: ProviderTransportResponse
        ) -> ProviderInvocationOutcome:
            del response
            raise AssertionError("blank response reached semantic classifier")

    evidence = ProviderInvocationEvidence(
        request_identity_hash=REQUEST_HASH,
        response=ProviderTransportResponse(text="  "),
    )

    assert (
        classify_provider_invocation(evidence, RejectingClassifier())
        is ProviderInvocationOutcome.BLANK_RESPONSE
    )


def test_accept_all_classifier_has_stable_identifier() -> None:
    assert CLASSIFIER.identifier == ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER
    assert CLASSIFIER.identifier.root == (
        "dr_providers.accept_all_semantic_response.v1"
    )
