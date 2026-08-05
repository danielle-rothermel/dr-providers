"""Shared failure taxonomy and raised-error carriers.

dr-providers classifies transport-level failure only. There are two
distinct shapes and they are NOT nested in each other:

  * ``ProviderTransportFailure`` (``outcomes.models``) is the FLAT expected
    transport-outcome value. It carries the failure fields (``failure_class``,
    ``code``, ``message``, ``retryable``) directly on itself alongside the raw
    request/response evidence — it does not embed a ``ProviderFailure``.
  * ``ProviderFailure`` (below) is a compact classification record carried
    ONLY by the raised-error path: ``ProviderFailureError.failure`` holds one,
    and it is raised solely for unexpected programming/infrastructure errors,
    never returned as an expected transport outcome.

Whetstone owns semantic failure taxonomy and retry policy.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


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
    """Transport failure classification record.

    Carried inside a Provider Transport Failure and by the exception
    types used only for unexpected raises.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: FailureClass
    code: StrictStr | None = None
    message: StrictStr
    retryable: StrictBool
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    """Carrier for a :class:`ProviderFailure` record.

    Raised only for unexpected programming/infrastructure errors, never
    for expected transport outcomes (which are returned, not thrown).
    """

    failure_class: ClassVar[FailureClass] = FailureClass.UNKNOWN

    def __init__(
        self,
        failure: ProviderFailure,
        *,
        underlying: BaseException | None = None,
    ) -> None:
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
    """A Definition/Config assignment violates a control invariant.

    Raised at Definition or Config validation time for an unsupported
    control, a missing required control, an undeclared extension key, or
    an extension that would overwrite a reserved core wire field. These
    are construction-time programming errors, so they raise rather than
    returning a transport outcome.
    """


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
    """Build the carrier exception matching the record's class."""
    error_type = FAILURE_ERROR_TYPES[failure.failure_class]
    return error_type(failure, underlying=underlying)
