"""Typed no-throw Provider Transport Outcome.

The transport returns a closed union of exactly one Provider Transport
Response or one Provider Transport Failure. Expected outcomes never
raise; only unexpected programming/infrastructure errors raise. Neither
value asserts Whetstone semantic acceptance — Whetstone projects a
Generation or classifies a Provider Semantic Failure downstream.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_providers.core.failures import (  # noqa: TC001 -- pydantic field
    FailureClass,
)


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


class ProviderTransportResponse(BaseModel):
    """Typed successful transport value.

    Carries the least-processed raw response body plus the provider
    identifiers, usage, cost, warnings, and diagnostics available at the
    transport boundary. It does not assert semantic acceptance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr
    raw_body: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage | None = None
    cost: CostInfo | None = None
    warnings: tuple[ProviderTransportWarning, ...] = ()
    finish_reason: StrictStr | None = None
    response_id: StrictStr | None = None
    model: StrictStr | None = None
    diagnostics: ResponsesDiagnostics | None = None


class ProviderTransportFailure(BaseModel):
    """Typed expected transport failure value.

    Retains the complete least-processed raw request plus failure
    response evidence and transport diagnostics, without semantic
    classification or retry decisions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: FailureClass
    code: StrictStr | None = None
    message: StrictStr
    retryable: bool
    raw_request: dict[str, Any] = Field(default_factory=dict)
    raw_response_body: Any | None = None
    status_code: StrictInt | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ProviderTransportOutcome = ProviderTransportResponse | ProviderTransportFailure


def is_response(
    outcome: ProviderTransportOutcome,
) -> TypeGuard[ProviderTransportResponse]:
    # ``TypeGuard`` (not ``TypeIs``): the stdlib ``typing.TypeIs`` lands in
    # 3.13, and this package targets 3.12 without a typing_extensions
    # dependency, so ``TypeGuard`` is the portable choice here.
    return isinstance(outcome, ProviderTransportResponse)


def is_failure(
    outcome: ProviderTransportOutcome,
) -> TypeGuard[ProviderTransportFailure]:
    return isinstance(outcome, ProviderTransportFailure)
