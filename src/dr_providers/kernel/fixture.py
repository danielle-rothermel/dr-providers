"""FixtureProvider: a testing peer implementing the real interface.

Public API — consumers exercise full request/response flows with no
network by scripting outcomes (text, usage, cost, warnings, or a
failure) per call. Outcomes are consumed in order; the last outcome
repeats for subsequent calls.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dr_providers.kernel.failures import (
    ProviderFailure,
    raise_failure,
)
from dr_providers.kernel.request import LlmRequest, build_payload
from dr_providers.kernel.response import (
    CostInfo,
    LlmResponse,
    LlmWarning,
    TokenUsage,
)

FIXTURE_RESPONSE_ID_PREFIX = "fixture-response"


class Provider(Protocol):
    """The single-shot provider call interface."""

    def complete(self, request: LlmRequest) -> LlmResponse: ...


class FixtureOutcome(BaseModel):
    """One scripted call result: either text parts or a failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr = ""
    usage: TokenUsage | None = None
    cost: CostInfo | None = None
    warnings: tuple[LlmWarning, ...] = ()
    finish_reason: StrictStr | None = "stop"
    failure: ProviderFailure | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class FixtureProvider:
    """Scripted provider; records every request it serves."""

    def __init__(self, outcomes: list[FixtureOutcome] | None = None) -> None:
        self._outcomes = list(
            outcomes or [FixtureOutcome(text="fixture output")]
        )
        self.requests: list[LlmRequest] = []
        self.payloads: list[dict[str, Any]] = []

    def complete(self, request: LlmRequest) -> LlmResponse:
        payload = build_payload(request)
        self.requests.append(request)
        self.payloads.append(payload)
        index = min(len(self.requests) - 1, len(self._outcomes) - 1)
        outcome = self._outcomes[index]
        if outcome.failure is not None:
            raise raise_failure(outcome.failure)
        response_id = f"{FIXTURE_RESPONSE_ID_PREFIX}-{len(self.requests)}"
        return LlmResponse(
            text=outcome.text,
            usage=outcome.usage,
            cost=outcome.cost,
            warnings=outcome.warnings,
            finish_reason=outcome.finish_reason,
            response_id=response_id,
            model=request.provider_config.model,
            continuation_handle=response_id,
            payload=payload,
            provider_metadata=dict(outcome.provider_metadata),
        )
