from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class RecoverabilityClass(StrEnum):
    PERMANENT = "permanent"
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


class ProviderFailure(BaseModel):
    """Classification record carried by raised provider errors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recoverability: RecoverabilityClass
    code: StrictStr | None = None
    message: StrictStr
    metadata: dict[str, Any] = Field(default_factory=dict)


def failure_record(
    *,
    recoverability: RecoverabilityClass,
    message: str,
    code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProviderFailure:
    return ProviderFailure(
        recoverability=recoverability,
        code=code,
        message=message,
        metadata=dict(metadata or {}),
    )


class ProviderFailureError(Exception):
    """Raised carrier for a :class:`ProviderFailure` record."""

    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class ControlValidationError(ProviderFailureError):
    pass
