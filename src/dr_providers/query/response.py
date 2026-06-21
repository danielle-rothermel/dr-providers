from __future__ import annotations

import json
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dr_providers.query.provider_config import (
    ProviderSemanticError,
    ProviderTransportError,
    ReasoningWarning,
)

if TYPE_CHECKING:
    import httpx

    from dr_providers.query.request import LlmRequest, OpenAICompatRequest


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


def parse_http_response_body[T: BaseModel](
    response: httpx.Response,
    payload_model_cls: type[T],
) -> tuple[Any, T | None, str | None, str | None]:
    try:
        body_raw = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, None, str(exc), None
    if not isinstance(body_raw, dict):
        return body_raw, None, None, "expected JSON object"
    try:
        parsed = payload_model_cls(**body_raw)
    except ValidationError as exc:
        return body_raw, None, None, str(exc)
    return body_raw, parsed, None, None


class CallError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error_type: str
    message: str
    retryable: bool = False
    raw_json: dict[str, Any] | None = None


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_json: dict[str, Any] = Field(default_factory=dict)
    provider: str
    model: str
    latency_ms: int
    text: str
    finish_reason: str | None = None
    warnings: list[Any] = Field(default_factory=list)


class _OpenAICompatUsageDetails(BaseModel):
    reasoning_tokens: int | None = None


class _OpenAICompatUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: _OpenAICompatUsageDetails | None = None
    output_tokens_details: _OpenAICompatUsageDetails | None = None


class _OpenAICompatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: str | None = None
    reasoning: str | int | float | None = None
    reasoning_content: str | int | float | None = None
    reasoning_details: list[dict[str, Any]] | None = None


class _OpenAICompatChoice(BaseModel):
    finish_reason: str | None = None
    message: _OpenAICompatMessage


class _OpenAICompatResponseBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: list[_OpenAICompatChoice] = Field(default_factory=list)
    usage: _OpenAICompatUsage | None = None


class OpenAICompatResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status_code: int
    response_text_preview: str = Field(exclude=True)
    raw_json: Any | None = Field(default=None, exclude=True, repr=False)
    choices: list[_OpenAICompatChoice] = Field(default_factory=list)
    usage: _OpenAICompatUsage | None = None
    json_error: str | None = Field(default=None, exclude=True, repr=False)
    response_shape_error: str | None = Field(
        default=None, exclude=True, repr=False
    )

    @classmethod
    def from_http_response(
        cls, response: httpx.Response
    ) -> OpenAICompatResponse:
        raw_json, parsed, json_error, shape_error = parse_http_response_body(
            response, _OpenAICompatResponseBody
        )
        return cls(
            status_code=response.status_code,
            response_text_preview=response.text[:500],
            raw_json=raw_json,
            choices=parsed.choices if parsed else [],
            usage=parsed.usage if parsed else None,
            json_error=json_error,
            response_shape_error=shape_error,
        )

    def event_payload(self, request: OpenAICompatRequest) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "endpoint": request.endpoint(),
            "idempotency_key": request.idempotency_key,
            "response_text_preview": self.response_text_preview,
            "request_shape": {
                "model": request.model,
                "message_count": len(request.messages),
            },
        }

    def _validated_choice(self, *, provider_name: str) -> _OpenAICompatChoice:
        validate_http_response(
            provider_label=provider_name,
            status_code=self.status_code,
            response_text_preview=self.response_text_preview,
            json_error=self.json_error,
            response_shape_error=self.response_shape_error,
        )
        if not self.choices:
            raise ProviderSemanticError(
                f"{provider_name} response missing choices"
            )
        return self.choices[0]

    def to_llm_response(
        self,
        request: LlmRequest,
        *,
        latency_ms: int,
        warnings: list[ReasoningWarning],
    ) -> LlmResponse:
        choice = self._validated_choice(provider_name=request.provider)
        message = choice.message
        return LlmResponse(
            text=message.content or "",
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            raw_json=self.raw_json if isinstance(self.raw_json, dict) else {},
            provider=request.provider,
            model=request.model,
            warnings=warnings,
        )
