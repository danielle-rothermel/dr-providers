from __future__ import annotations

import json
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from dr_providers.query.errors import (
    ProviderSemanticError,
    ProviderTransportError,
)
from dr_providers.query.reasoning import ReasoningWarning  # noqa: TC001

if TYPE_CHECKING:
    import httpx

    from dr_providers.query.request import LlmRequest


class HttpStatusCode(IntEnum):
    TRANSIENT_ERROR = 500
    TIMEOUT = 408
    TOO_MANY_REQUESTS = 429
    BAD_REQUEST = 400


def validate_http_response(
    *,
    provider_label: str,
    status_code: int,
    response_text_preview: str,
    json_error: str | None,
    response_shape_error: str | None,
) -> None:
    if status_code >= HttpStatusCode.TRANSIENT_ERROR or status_code in {
        HttpStatusCode.TIMEOUT,
        HttpStatusCode.TOO_MANY_REQUESTS,
    }:
        raise ProviderTransportError(
            f"{provider_label} transient error status={status_code} "
            f"body={response_text_preview}"
        )
    if status_code >= HttpStatusCode.BAD_REQUEST:
        raise ProviderSemanticError(
            f"{provider_label} rejected request status={status_code} "
            f"body={response_text_preview}"
        )
    if json_error is not None:
        raise ProviderTransportError(
            f"{provider_label} invalid JSON response: {json_error}"
        )
    if response_shape_error is not None:
        raise ProviderSemanticError(
            f"{provider_label} response shape invalid: {response_shape_error}"
        )


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_json: dict[str, Any] = Field(default_factory=dict)
    provider: str
    model: str
    latency_ms: int
    text: str
    finish_reason: str | None = None
    warnings: list[Any] = Field(default_factory=list)


def _parse_response_json(
    response: httpx.Response,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        body_raw = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, str(exc), None
    if not isinstance(body_raw, dict):
        return None, None, "expected JSON object"
    return body_raw, None, None


def _extract_completion_fields(
    body: dict[str, Any], *, provider_label: str
) -> tuple[str, str | None]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderSemanticError(
            f"{provider_label} response missing choices"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderSemanticError(
            f"{provider_label} response choice invalid"
        )
    finish_reason = choice.get("finish_reason")
    finish_reason_str = (
        finish_reason if isinstance(finish_reason, str) else None
    )
    message = choice.get("message")
    if not isinstance(message, dict):
        return "", finish_reason_str
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    return text, finish_reason_str


def llm_response_from_http(
    response: httpx.Response,
    request: LlmRequest,
    *,
    latency_ms: int,
    warnings: list[ReasoningWarning],
) -> LlmResponse:
    provider_label = str(request.provider)
    response_text_preview = response.text[:500]
    body_raw, json_error, shape_error = _parse_response_json(response)
    validate_http_response(
        provider_label=provider_label,
        status_code=response.status_code,
        response_text_preview=response_text_preview,
        json_error=json_error,
        response_shape_error=shape_error,
    )
    if body_raw is None:
        raise ProviderSemanticError(
            f"{provider_label} response shape invalid: missing body"
        )
    text, finish_reason = _extract_completion_fields(
        body_raw, provider_label=provider_label
    )
    return LlmResponse(
        text=text,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        raw_json=body_raw,
        provider=provider_label,
        model=request.model,
        warnings=warnings,
    )
