"""ScriptedProvider: a testing peer implementing the real interface.

Public API — consumers exercise full request/response flows with no
network by scripting outcomes (text, usage, cost, warnings, or a
failure) per call. Outcomes are consumed in order; the last outcome
repeats for subsequent calls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dr_providers.conformance import with_conformance_warnings
from dr_providers.failures import (
    ProviderFailure,
    raise_failure,
)
from dr_providers.provider import Provider
from dr_providers.request import LlmRequest, build_payload
from dr_providers.response import (
    CostInfo,
    LlmResponse,
    LlmWarning,
    TokenUsage,
)

__all__ = [
    "SCRIPTED_RESPONSE_ID_PREFIX",
    "Provider",
    "ScriptedOutcome",
    "ScriptedProvider",
]

SCRIPTED_RESPONSE_ID_PREFIX = "scripted-response"


class ScriptedOutcome(BaseModel):
    """One scripted call result: either text parts or a failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr = ""
    usage: TokenUsage | None = None
    cost: CostInfo | None = None
    warnings: tuple[LlmWarning, ...] = ()
    finish_reason: StrictStr | None = "stop"
    failure: ProviderFailure | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class ScriptedProvider:
    """Scripted provider; records every request it serves."""

    def __init__(self, outcomes: list[ScriptedOutcome] | None = None) -> None:
        self._outcomes = list(
            outcomes or [ScriptedOutcome(text="scripted output")]
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
        response_id = f"{SCRIPTED_RESPONSE_ID_PREFIX}-{len(self.requests)}"
        response = LlmResponse(
            text=outcome.text,
            usage=outcome.usage,
            cost=outcome.cost,
            warnings=outcome.warnings,
            finish_reason=outcome.finish_reason,
            response_id=response_id,
            model=request.provider_config.model,
            provider_metadata=dict(outcome.provider_metadata),
        )
        return with_conformance_warnings(request, response)
