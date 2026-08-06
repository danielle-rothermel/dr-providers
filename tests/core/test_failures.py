import pytest
from pydantic import ValidationError

from dr_providers import (
    FAILURE_ERROR_TYPES,
    RECOVERABLE_FAILURE_CLASSES,
    RETRYABLE_FAILURE_CLASSES,
    FailureClass,
    PermanentProviderError,
    ProviderFailure,
    ProviderFailureError,
    RateLimitedProviderError,
    ResourceExhaustionProviderError,
    TransientProviderError,
    UnknownProviderError,
    failure_record,
    raise_failure,
)

FAILURE_CASES: tuple[
    tuple[
        FailureClass,
        tuple[bool, bool],
        type[ProviderFailureError],
    ],
    ...,
] = (
    (FailureClass.PERMANENT, (False, False), PermanentProviderError),
    (FailureClass.TRANSIENT, (True, True), TransientProviderError),
    (FailureClass.RATE_LIMITED, (True, True), RateLimitedProviderError),
    (
        FailureClass.RESOURCE_EXHAUSTION,
        (True, False),
        ResourceExhaustionProviderError,
    ),
    (FailureClass.UNKNOWN, (False, False), UnknownProviderError),
)


@pytest.mark.parametrize(
    (
        "failure_class",
        "expected_membership",
        "error_type",
    ),
    FAILURE_CASES,
)
def test_failure_taxonomy_and_carrier_mapping(
    failure_class: FailureClass,
    expected_membership: tuple[bool, bool],
    error_type: type[ProviderFailureError],
) -> None:
    recoverable, retryable = expected_membership
    underlying = RuntimeError("sentinel")
    failure = failure_record(
        failure_class=failure_class,
        code="test_failure",
        message="provider failed",
        metadata={"attempt": 2},
    )

    assert (failure_class in RECOVERABLE_FAILURE_CLASSES) is recoverable
    assert (failure_class in RETRYABLE_FAILURE_CLASSES) is retryable
    assert failure.retryable is retryable
    assert FAILURE_ERROR_TYPES[failure_class] is error_type

    error = raise_failure(failure, underlying=underlying)

    assert type(error) is error_type
    assert str(error) == failure.message
    assert error.failure is failure
    assert error.underlying is underlying


@pytest.mark.parametrize(
    ("failure_class", "error_type"),
    [(case[0], case[2]) for case in FAILURE_CASES],
)
def test_failure_carrier_rejects_mismatched_classification(
    failure_class: FailureClass,
    error_type: type[ProviderFailureError],
) -> None:
    mismatched_class = next(
        candidate
        for candidate in FailureClass
        if candidate is not failure_class
    )
    failure = failure_record(
        failure_class=mismatched_class,
        message="contradictory classification",
    )

    with pytest.raises(ValueError, match="requires failure class"):
        error_type(failure)


@pytest.mark.parametrize(
    "case",
    [
        (FailureClass.PERMANENT, True),
        (FailureClass.TRANSIENT, False),
    ],
)
def test_failure_record_rejects_inconsistent_retryability(
    case: tuple[FailureClass, bool],
) -> None:
    failure_class, retryable = case
    with pytest.raises(ValidationError, match="requires retryable"):
        ProviderFailure(
            failure_class=failure_class,
            message="contradictory retryability",
            retryable=retryable,
        )
