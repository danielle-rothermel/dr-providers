from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dr_providers.outcomes.conformance import with_conformance_warnings
from dr_providers.outcomes.evidence import ProviderInvocationEvidence
from dr_providers.outcomes.models import (
    CostInfo,
    ProviderStopReason,
    ProviderTransportFailure,
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
    stop_reason: ProviderStopReason | None = ProviderStopReason.STOP
    failure: ProviderTransportFailure | None = None
    response_body: dict[str, Any] = Field(default_factory=dict)


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

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        payload = build_payload(request)
        self.requests.append(request)
        self.payloads.append(payload)
        index = min(len(self.requests) - 1, len(self._outcomes) - 1)
        outcome = self._outcomes[index]
        if outcome.failure is not None:
            return ProviderInvocationEvidence.build(
                request=request,
                policy=None,
                http_request=None,
                outcome=outcome.failure,
            )
        response_id = f"{SCRIPTED_RESPONSE_ID_PREFIX}-{len(self.requests)}"
        response = ProviderTransportResponse(
            text=outcome.text,
            response_body=dict(outcome.response_body),
            usage=outcome.usage,
            cost=outcome.cost,
            warnings=outcome.warnings,
            stop_reason=outcome.stop_reason,
            response_id=response_id,
            model=request.config.route.model,
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=None,
            http_request=None,
            outcome=with_conformance_warnings(request, response),
        )
