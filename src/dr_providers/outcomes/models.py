from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_providers.core.failures import (  # noqa: TC001 -- pydantic field
    RecoverabilityClass,
)
from dr_providers.core.frozen import _freeze_json

INVALID_JSON_CODE = "invalid_response_json"
TIMEOUT_CODE = "timeout"
STALLED_RESPONSE_CODE = "stalled_response"


class ProviderStopReason(StrEnum):
    """Typed protocol stop reason for a successful transport response."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


class TransportTimeoutContainment(StrEnum):
    """Whether a transport timeout ended the local HTTP operation."""

    CONTAINED = "contained"
    UNCONTAINED = "uncontained"


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ProviderTransportWarning(BaseModel):
    """Conformance or parse observation; the caller decides fatality."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: StrictStr
    message: StrictStr
    severity: WarningSeverity = WarningSeverity.WARNING
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: StrictInt | None = None
    completion_tokens: StrictInt | None = None
    total_tokens: StrictInt | None = None
    reasoning_tokens: StrictInt | None = None


class CostInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_cost: float
    currency: StrictStr = "USD"


class ResponsesDiagnostics(BaseModel):
    """Safe, content-free observations from an OpenAI Responses body.

    Provider-controlled enums are retained only when explicitly
    allowlisted; all other string values are coalesced into ``unknown``
    count categories. ``response_id_hash`` is a truncated, unsalted
    SHA-256 digest retained only for correlating high-entropy provider
    IDs (diagnostic-only, never domain identity).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    response_status: StrictStr | None = None
    incomplete_reason: StrictStr | None = None
    output_item_types: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    content_part_types: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )
    output_text_len: StrictInt = 0
    refusal_len: StrictInt | None = None
    response_id_hash: StrictStr | None = None

    def model_post_init(self, _context: Any) -> None:
        object.__setattr__(
            self,
            "output_item_types",
            _freeze_json(self.output_item_types),
        )
        object.__setattr__(
            self,
            "content_part_types",
            _freeze_json(self.content_part_types),
        )


class ProviderTransportResponse(BaseModel):
    """Transport success with the parsed JSON response mapping.

    It does not assert semantic acceptance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr
    response_body: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage | None = None
    cost: CostInfo | None = None
    warnings: tuple[ProviderTransportWarning, ...] = ()
    stop_reason: ProviderStopReason | None = None
    response_id: StrictStr | None = None
    model: StrictStr | None = None
    diagnostics: ResponsesDiagnostics | None = None

    def model_post_init(self, _context: Any) -> None:
        object.__setattr__(
            self,
            "response_body",
            _freeze_json(self.response_body),
        )


class ProviderTransportFailure(BaseModel):
    """Expected provider transport or protocol failure.

    Response evidence is decoded as JSON when possible or retained as text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    recoverability: RecoverabilityClass
    code: StrictStr | None = None
    message: StrictStr
    response_body: Any | None = None
    status_code: StrictInt | None = None
    containment: TransportTimeoutContainment | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        object.__setattr__(
            self,
            "response_body",
            _freeze_json(self.response_body),
        )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


ProviderTransportOutcome = ProviderTransportResponse | ProviderTransportFailure


def is_response(
    outcome: ProviderTransportOutcome,
) -> TypeGuard[ProviderTransportResponse]:
    # Python 3.12 lacks TypeIs; avoid a typing_extensions dependency.
    return isinstance(outcome, ProviderTransportResponse)


def is_failure(
    outcome: ProviderTransportOutcome,
) -> TypeGuard[ProviderTransportFailure]:
    return isinstance(outcome, ProviderTransportFailure)
