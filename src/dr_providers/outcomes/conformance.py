from __future__ import annotations

from typing import TYPE_CHECKING

from dr_providers.modeling.controls import ReasoningEffort
from dr_providers.outcomes.models import (
    ProviderTransportResponse,
    ProviderTransportWarning,
)

if TYPE_CHECKING:
    from dr_providers.modeling.request import ProviderCallRequest

REASONING_NOT_OBSERVED_CODE = "reasoning_not_observed"
MODEL_SUBSTITUTION_CODE = "model_substitution"


def conformance_warnings(
    request: ProviderCallRequest,
    response: ProviderTransportResponse,
) -> tuple[ProviderTransportWarning, ...]:
    warnings: list[ProviderTransportWarning] = []
    usage = response.usage
    controls = request.config.controls

    reasoning_tokens = usage.reasoning_tokens if usage else None
    reasoning_requested = (
        controls.reasoning is not None
        and controls.reasoning is not ReasoningEffort.NONE
    )
    if reasoning_requested and not reasoning_tokens:
        assert controls.reasoning is not None
        warnings.append(
            ProviderTransportWarning(
                code=REASONING_NOT_OBSERVED_CODE,
                message=(
                    "reasoning was requested but the response reports "
                    "no reasoning tokens"
                ),
                metadata={"requested": controls.reasoning.value},
            )
        )

    requested_model = request.config.route.model
    if response.model is not None and response.model != requested_model:
        warnings.append(
            ProviderTransportWarning(
                code=MODEL_SUBSTITUTION_CODE,
                message=(
                    f"requested model {requested_model!r} but response "
                    f"reports {response.model!r}"
                ),
                metadata={
                    "requested_model": requested_model,
                    "response_model": response.model,
                },
            )
        )
    return tuple(warnings)


def with_conformance_warnings(
    request: ProviderCallRequest,
    response: ProviderTransportResponse,
) -> ProviderTransportResponse:
    warnings = conformance_warnings(request, response)
    if not warnings:
        return response
    return response.model_copy(
        update={"warnings": (*response.warnings, *warnings)}
    )
