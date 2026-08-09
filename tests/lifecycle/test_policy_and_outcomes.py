from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from dr_providers.lifecycle import (
    STANDARD_RETRY_DELAYS_SECONDS,
    STANDARD_RETRY_ELIGIBLE_OUTCOMES,
    CustomProviderCallRetryPolicy,
    ProviderCallOutcome,
    ProviderCallOutcomeKind,
    ProviderInvocationOutcome,
    SemanticResponseClassifierIdentifier,
    StandardProviderCallRetryPolicy,
    classify_semantic_response,
)
from dr_providers.outcomes.models import ProviderTransportResponse

EXPECTED_INVOCATION_OUTCOME_LITERALS = [
    "success",
    "blank_response",
    "malformed_response",
    "provider_rejection",
    "semantic_rejection",
    "permanent_provider_or_transport_failure",
    "transient_provider_or_network_failure",
    "rate_limiting",
    "resource_exhaustion",
    "contained_transport_timeout",
    "uncontained_deadline_expiration",
    "unknown_transport_failure",
]
EXPECTED_CALL_OUTCOME_LITERALS = [
    "accepted",
    "invocation_outcome",
    "draining_cancellation",
    "policy_exhaustion",
]
EXPECTED_STANDARD_POLICY = {
    "policy_type": "standard",
    "maximum_invocations": 2,
    "eligible_outcomes": [
        "transient_provider_or_network_failure",
        "contained_transport_timeout",
    ],
    "declared_delays_seconds": [1.0],
    "maximum_cumulative_delay_seconds": 1.0,
}
GOLDEN_STANDARD_POLICY_HASH = (
    "a53e481b6f1be12c4f23785315463fb53193c5e674364a860907369730c2f797"
)


def test_persisted_outcome_literals_are_pinned() -> None:
    assert [outcome.value for outcome in ProviderInvocationOutcome] == (
        EXPECTED_INVOCATION_OUTCOME_LITERALS
    )
    assert [outcome.value for outcome in ProviderCallOutcomeKind] == (
        EXPECTED_CALL_OUTCOME_LITERALS
    )


def test_standard_policy_shape_and_identity_are_pinned() -> None:
    policy = StandardProviderCallRetryPolicy()

    assert policy.model_dump(mode="json") == EXPECTED_STANDARD_POLICY
    assert policy.identity_hash == GOLDEN_STANDARD_POLICY_HASH
    assert policy.maximum_invocations == 2
    assert policy.eligible_outcomes == STANDARD_RETRY_ELIGIBLE_OUTCOMES
    assert policy.declared_delays_seconds == STANDARD_RETRY_DELAYS_SECONDS
    assert policy.retry_delay_after(1) == 1.0
    assert "identity_hash" in policy.__dict__


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_invocations": 3},
        {"eligible_outcomes": ()},
        {"declared_delays_seconds": (0.5,)},
        {"maximum_cumulative_delay_seconds": 2.0},
    ],
)
def test_standard_policy_rejects_variants(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StandardProviderCallRetryPolicy.model_validate(change)


def test_custom_policy_is_closed_deterministic_data() -> None:
    policy = CustomProviderCallRetryPolicy(
        maximum_invocations=3,
        eligible_outcomes=frozenset(
            {
                ProviderInvocationOutcome.BLANK_RESPONSE,
                ProviderInvocationOutcome.RATE_LIMITING,
            }
        ),
        declared_delays_seconds=(0.0, 2.5),
    )

    assert policy.maximum_cumulative_delay_seconds == 2.5
    assert policy.retry_delay_after(1) == 0.0
    assert policy.retry_delay_after(2) == 2.5
    assert policy.identity_payload()["eligible_outcomes"] == [
        "blank_response",
        "rate_limiting",
    ]


def test_custom_policy_rejects_non_finite_cumulative_delay() -> None:
    with pytest.raises(
        ValidationError,
        match="custom retry policy cumulative delay must be finite",
    ):
        CustomProviderCallRetryPolicy(
            maximum_invocations=3,
            eligible_outcomes=frozenset(),
            declared_delays_seconds=(1e308, 1e308),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "maximum_invocations": 0,
            "eligible_outcomes": frozenset(),
            "declared_delays_seconds": (),
        },
        {
            "maximum_invocations": 2,
            "eligible_outcomes": frozenset(),
            "declared_delays_seconds": (),
        },
        {
            "maximum_invocations": 2,
            "eligible_outcomes": frozenset(),
            "declared_delays_seconds": (-1.0,),
        },
        {
            "maximum_invocations": 2,
            "eligible_outcomes": frozenset(),
            "declared_delays_seconds": (math.inf,),
        },
        {
            "maximum_invocations": 2,
            "eligible_outcomes": frozenset(
                {ProviderInvocationOutcome.SUCCESS}
            ),
            "declared_delays_seconds": (0.0,),
        },
        {
            "maximum_invocations": 2,
            "eligible_outcomes": frozenset(
                {ProviderInvocationOutcome.UNCONTAINED_DEADLINE_EXPIRATION}
            ),
            "declared_delays_seconds": (0.0,),
        },
    ],
)
def test_custom_policy_rejects_invalid_declarations(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CustomProviderCallRetryPolicy.model_validate(kwargs)


class _Classifier:
    def __init__(self, outcome: ProviderInvocationOutcome) -> None:
        self.identifier = SemanticResponseClassifierIdentifier("classifier-v1")
        self._outcome = outcome

    def classify(
        self, response: ProviderTransportResponse
    ) -> ProviderInvocationOutcome:
        del response
        return self._outcome


@pytest.mark.parametrize(
    "outcome",
    [
        ProviderInvocationOutcome.SUCCESS,
        ProviderInvocationOutcome.SEMANTIC_REJECTION,
    ],
)
def test_semantic_classifier_boundary_accepts_only_semantic_outcomes(
    outcome: ProviderInvocationOutcome,
) -> None:
    classifier = _Classifier(outcome)
    response = ProviderTransportResponse(text="valid")

    assert classify_semantic_response(classifier, response) is outcome


def test_semantic_classifier_boundary_rejects_protocol_outcome() -> None:
    classifier = _Classifier(ProviderInvocationOutcome.MALFORMED_RESPONSE)

    with pytest.raises(ValueError, match="must return success"):
        classify_semantic_response(
            classifier,
            ProviderTransportResponse(text="valid"),
        )


def test_classifier_identifier_must_be_nonempty() -> None:
    with pytest.raises(ValidationError):
        SemanticResponseClassifierIdentifier("")


def test_call_outcome_shape_is_closed() -> None:
    with pytest.raises(ValidationError):
        ProviderCallOutcome(kind=ProviderCallOutcomeKind.ACCEPTED)
    with pytest.raises(ValidationError):
        ProviderCallOutcome(
            kind=ProviderCallOutcomeKind.DRAINING_CANCELLATION,
            invocation_outcome=ProviderInvocationOutcome.SUCCESS,
        )
