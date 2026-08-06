from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    model_validator,
)


class FailureClass(StrEnum):
    PERMANENT = "permanent"
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


RECOVERABLE_FAILURE_CLASSES = frozenset(
    {
        FailureClass.TRANSIENT,
        FailureClass.RATE_LIMITED,
        FailureClass.RESOURCE_EXHAUSTION,
    }
)
RETRYABLE_FAILURE_CLASSES = frozenset(
    {
        FailureClass.TRANSIENT,
        FailureClass.RATE_LIMITED,
    }
)


class ProviderFailure(BaseModel):
    """Classification record carried only by ``ProviderFailureError``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: FailureClass
    code: StrictStr | None = None
    message: StrictStr
    retryable: StrictBool
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_retryability(self) -> ProviderFailure:
        expected = self.failure_class in RETRYABLE_FAILURE_CLASSES
        if self.retryable is not expected:
            msg = (
                f"failure class {self.failure_class.value!r} requires "
                f"retryable={expected!r}"
            )
            raise ValueError(msg)
        return self


def failure_record(
    *,
    failure_class: FailureClass,
    message: str,
    code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProviderFailure:
    return ProviderFailure(
        failure_class=failure_class,
        code=code,
        message=message,
        retryable=failure_class in RETRYABLE_FAILURE_CLASSES,
        metadata=dict(metadata or {}),
    )


class ProviderFailureError(Exception):
    """Raised carrier for a :class:`ProviderFailure` record."""

    failure_class: ClassVar[FailureClass] = FailureClass.UNKNOWN

    def __init__(
        self,
        failure: ProviderFailure,
        *,
        underlying: BaseException | None = None,
    ) -> None:
        if failure.failure_class is not self.failure_class:
            msg = (
                f"{type(self).__name__} requires failure class "
                f"{self.failure_class.value!r}, got "
                f"{failure.failure_class.value!r}"
            )
            raise ValueError(msg)
        super().__init__(failure.message)
        self.failure = failure
        self.underlying = underlying


class PermanentProviderError(ProviderFailureError):
    failure_class = FailureClass.PERMANENT


class TransientProviderError(ProviderFailureError):
    failure_class = FailureClass.TRANSIENT


class RateLimitedProviderError(ProviderFailureError):
    failure_class = FailureClass.RATE_LIMITED


class ResourceExhaustionProviderError(ProviderFailureError):
    failure_class = FailureClass.RESOURCE_EXHAUSTION


class UnknownProviderError(ProviderFailureError):
    failure_class = FailureClass.UNKNOWN


class ControlValidationError(PermanentProviderError):
    pass


FAILURE_ERROR_TYPES: dict[FailureClass, type[ProviderFailureError]] = {
    FailureClass.PERMANENT: PermanentProviderError,
    FailureClass.TRANSIENT: TransientProviderError,
    FailureClass.RATE_LIMITED: RateLimitedProviderError,
    FailureClass.RESOURCE_EXHAUSTION: ResourceExhaustionProviderError,
    FailureClass.UNKNOWN: UnknownProviderError,
}


def raise_failure(
    failure: ProviderFailure,
    *,
    underlying: BaseException | None = None,
) -> ProviderFailureError:
    error_type = FAILURE_ERROR_TYPES[failure.failure_class]
    return error_type(failure, underlying=underlying)
