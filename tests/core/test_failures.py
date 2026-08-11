import pytest

from dr_providers import (
    ControlValidationError,
    ProviderFailureError,
    RecoverabilityClass,
    failure_record,
)


@pytest.mark.parametrize("recoverability", list(RecoverabilityClass))
def test_failure_record_builds_provider_failure(
    recoverability: RecoverabilityClass,
) -> None:
    failure = failure_record(
        recoverability=recoverability,
        code="test_failure",
        message="provider failed",
        metadata={"attempt": 2},
    )

    assert failure.recoverability is recoverability
    assert failure.code == "test_failure"
    assert failure.message == "provider failed"
    assert failure.metadata == {"attempt": 2}


def test_control_validation_error_carries_failure_record() -> None:
    underlying = RuntimeError("sentinel")
    failure = failure_record(
        recoverability=RecoverabilityClass.PERMANENT,
        code="invalid_control",
        message="bad control",
    )
    error = ControlValidationError(failure, underlying=underlying)

    assert isinstance(error, ProviderFailureError)
    assert str(error) == failure.message
    assert error.failure is failure
    assert error.underlying is underlying
