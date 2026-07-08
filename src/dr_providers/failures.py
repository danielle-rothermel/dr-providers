"""Failure taxonomy: records first, exceptions as carriers.

dr-providers is the canonical home of ``FailureClass`` and the failure
record (whetstone's platform imports them; the graph runner
interoperates via a structural protocol). The library classifies —
callers own retry policy.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

SANITIZE_KEYS = frozenset(
    {"api_key", "api_base", "base_url", "model_list", "authorization"}
)

RATE_LIMIT_STATUS = 429
TRANSIENT_STATUS_CODES = frozenset({408, 409, 425})
SERVER_ERROR_FLOOR = 500


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
    """Primary failure artifact; exceptions carry it."""

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
    """Carrier for a :class:`ProviderFailure` record."""

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


class UnsupportedControlError(PermanentProviderError):
    """A request set a knob the provider config cannot transport."""


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


def classify_status_code(status_code: int) -> FailureClass:
    """HTTP status → failure class (raw-httpx transport, one place)."""
    if status_code == RATE_LIMIT_STATUS:
        return FailureClass.RATE_LIMITED
    if (
        status_code >= SERVER_ERROR_FLOOR
        or status_code in TRANSIENT_STATUS_CODES
    ):
        return FailureClass.TRANSIENT
    return FailureClass.PERMANENT


def sanitize_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Strip credential-like keys from a kwargs dict before logging."""
    if not kwargs:
        return {}
    return {
        k: ("<redacted>" if k.lower() in SANITIZE_KEYS else v)
        for k, v in kwargs.items()
    }
