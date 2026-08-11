from dr_providers import (
    ControlValidationError,
    ProviderFailureError,
    RecoverabilityClass,
    failure_record,
)


def test_control_validation_error_carries_failure_record() -> None:
    underlying = RuntimeError("sentinel")
    failure = failure_record(
        recoverability=RecoverabilityClass.PERMANENT,
        code="invalid_control",
        message="bad control",
        metadata={"attempt": 2},
    )
    error = ControlValidationError(failure, underlying=underlying)

    assert isinstance(error, ProviderFailureError)
    assert str(error) == failure.message
    assert error.failure is failure
    assert error.failure.recoverability is RecoverabilityClass.PERMANENT
    assert error.failure.metadata == {"attempt": 2}
    assert error.underlying is underlying
