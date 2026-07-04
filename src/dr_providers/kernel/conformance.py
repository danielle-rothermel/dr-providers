"""Post-response conformance: check observed evidence, never predict.

Violations are warnings with severity; the caller decides what is
fatal. Library default severity is WARNING for every check (recorded
conservative choice — see the open-questions section of whetstone's
``llm_provider.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dr_providers.kernel.response import LlmResponse, LlmWarning

if TYPE_CHECKING:
    from dr_providers.kernel.request import LlmRequest

REASONING_NOT_OBSERVED_CODE = "reasoning_not_observed"
TOKEN_LIMIT_EXCEEDED_CODE = "token_limit_exceeded"  # noqa: S105
MODEL_SUBSTITUTION_CODE = "model_substitution"


def conformance_warnings(
    request: LlmRequest,
    response: LlmResponse,
) -> tuple[LlmWarning, ...]:
    warnings: list[LlmWarning] = []
    usage = response.usage

    reasoning_tokens = usage.reasoning_tokens if usage else None
    if request.reasoning and not reasoning_tokens:
        warnings.append(
            LlmWarning(
                code=REASONING_NOT_OBSERVED_CODE,
                message=(
                    "reasoning was requested but the response reports "
                    "no reasoning tokens"
                ),
                metadata={"requested": dict(request.reasoning)},
            )
        )

    completion_tokens = usage.completion_tokens if usage else None
    if (
        request.token_limit is not None
        and completion_tokens is not None
        and completion_tokens > request.token_limit
    ):
        warnings.append(
            LlmWarning(
                code=TOKEN_LIMIT_EXCEEDED_CODE,
                message=(
                    f"response used {completion_tokens} completion tokens "
                    f"despite a {request.token_limit}-token cap"
                ),
                metadata={
                    "token_limit": request.token_limit,
                    "completion_tokens": completion_tokens,
                },
            )
        )

    requested_model = request.provider_config.model
    if response.model is not None and response.model != requested_model:
        warnings.append(
            LlmWarning(
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
    request: LlmRequest,
    response: LlmResponse,
) -> LlmResponse:
    warnings = conformance_warnings(request, response)
    if not warnings:
        return response
    return response.model_copy(
        update={"warnings": (*response.warnings, *warnings)}
    )
