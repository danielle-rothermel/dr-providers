from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dr_providers.outcomes.conformance import with_conformance_warnings
from dr_providers.outcomes.models import (
    CostInfo,
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
    ProviderTransportWarning,
    TokenUsage,
)
from dr_providers.translation.request import build_payload

if TYPE_CHECKING:
    from dr_providers.modeling.request import ProviderCallRequest

__all__ = [
    "SCRIPTED_RESPONSE_ID_PREFIX",
    "ScriptedOutcome",
    "ScriptedProvider",
]

SCRIPTED_RESPONSE_ID_PREFIX = "scripted-response"


class ScriptedOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr = ""
    usage: TokenUsage | None = None
    cost: CostInfo | None = None
    warnings: tuple[ProviderTransportWarning, ...] = ()
    finish_reason: StrictStr | None = "stop"
    failure: ProviderTransportFailure | None = None
    raw_body: dict[str, Any] = Field(default_factory=dict)


class ScriptedProvider:
    """Consume outcomes in order, then repeat the last.

    Every request is recorded.
    """

    def __init__(self, outcomes: list[ScriptedOutcome] | None = None) -> None:
        self._outcomes = list(
            outcomes or [ScriptedOutcome(text="scripted output")]
        )
        self.requests: list[ProviderCallRequest] = []
        self.payloads: list[dict[str, Any]] = []

    def complete(
        self, request: ProviderCallRequest
    ) -> ProviderTransportOutcome:
        payload = build_payload(request)
        self.requests.append(request)
        self.payloads.append(payload)
        index = min(len(self.requests) - 1, len(self._outcomes) - 1)
        outcome = self._outcomes[index]
        if outcome.failure is not None:
            return outcome.failure.model_copy(
                update={"raw_request": dict(payload)}
            )
        response_id = f"{SCRIPTED_RESPONSE_ID_PREFIX}-{len(self.requests)}"
        response = ProviderTransportResponse(
            text=outcome.text,
            raw_body=dict(outcome.raw_body),
            usage=outcome.usage,
            cost=outcome.cost,
            warnings=outcome.warnings,
            finish_reason=outcome.finish_reason,
            response_id=response_id,
            model=request.config.route.model,
        )
        return with_conformance_warnings(request, response)
