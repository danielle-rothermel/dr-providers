"""Response as materialized typed parts, plus the wire parsers.

``LlmResponse`` composes text, usage, cost, warnings, finish reason,
and provider metadata — so a future streaming mode can emit the same
parts incrementally without a breaking redesign.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
)

from dr_providers.config import EndpointKind, ProviderConfig
from dr_providers.failures import (
    FailureClass,
    failure_record,
    raise_failure,
)

OUTPUT_TEXT_PART_TYPE = "output_text"
RESPONSES_INCOMPLETE_REASON_LENGTH = "max_output_tokens"
RESPONSES_INCOMPLETE_REASON_CONTENT_FILTER = "content_filter"
RESPONSES_STATUS_COMPLETED = "completed"
FINISH_REASON_LENGTH = "length"
FINISH_REASON_CONTENT_FILTER = "content_filter"
FINISH_REASON_STOP = "stop"
PARSE_ERROR_CODE = "response_parse_error"
RESPONSE_PREVIEW_LIMIT = 512


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class LlmWarning(BaseModel):
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


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr
    usage: TokenUsage | None = None
    cost: CostInfo | None = None
    warnings: tuple[LlmWarning, ...] = ()
    finish_reason: StrictStr | None = None
    response_id: StrictStr | None = None
    model: StrictStr | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


def parse_response(
    body: Mapping[str, Any],
    *,
    config: ProviderConfig,
) -> LlmResponse:
    if config.endpoint_kind is EndpointKind.CHAT_COMPLETIONS:
        return parse_chat_completions_body(body, config=config)
    return parse_responses_body(body, config=config)


def parse_chat_completions_body(
    body: Mapping[str, Any],
    *,
    config: ProviderConfig,
) -> LlmResponse:
    choices = body.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
        raise _parse_error("provider response missing choices", body, config)
    if not choices:
        raise _parse_error("provider response has empty choices", body, config)
    choice = choices[0]
    message = _get(choice, "message")
    text = _content_to_text(_get(message, "content"))
    if text is None:
        value = _get(choice, "text")
        text = value if isinstance(value, str) else None
    if text is None or not text.strip():
        raise _parse_error(
            "provider response produced no generation text", body, config
        )
    return LlmResponse(
        text=text,
        usage=token_usage_from_body(body),
        cost=cost_from_body(body),
        finish_reason=_optional_str(_get(choice, "finish_reason")),
        response_id=_optional_str(body.get("id")),
        model=_optional_str(body.get("model")) or config.model,
        provider_metadata=dict(body),
    )


def parse_responses_body(
    body: Mapping[str, Any],
    *,
    config: ProviderConfig,
) -> LlmResponse:
    text = _optional_str(body.get("output_text"))
    if text is None:
        text = _text_from_responses_output(body.get("output"))
    if text is None or not text.strip():
        raise _parse_error(
            "provider response produced no generation text", body, config
        )
    response_id = _optional_str(body.get("id"))
    return LlmResponse(
        text=text,
        usage=token_usage_from_body(body),
        cost=cost_from_body(body),
        finish_reason=_finish_reason_from_responses_body(body),
        response_id=response_id,
        model=_optional_str(body.get("model")) or config.model,
        provider_metadata=dict(body),
    )


def token_usage_from_body(body: Mapping[str, Any]) -> TokenUsage | None:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return None
    prompt = _optional_int(
        usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    completion = _optional_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total = _optional_int(usage.get("total_tokens"))
    reasoning = None
    for details_key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, Mapping):
            reasoning = _optional_int(details.get("reasoning_tokens"))
            if reasoning is not None:
                break
    if (prompt, completion, total, reasoning) == (None, None, None, None):
        return None
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
    )


def cost_from_body(body: Mapping[str, Any]) -> CostInfo | None:
    for key in ("cost", "total_cost"):
        value = body.get(key)
        if isinstance(value, int | float):
            return CostInfo(total_cost=float(value))
    usage = body.get("usage")
    if isinstance(usage, Mapping):
        value = usage.get("cost")
        if isinstance(value, int | float):
            return CostInfo(total_cost=float(value))
    return None


def _finish_reason_from_responses_body(
    body: Mapping[str, Any],
) -> str | None:
    incomplete_details = body.get("incomplete_details")
    if isinstance(incomplete_details, Mapping):
        reason = _optional_str(incomplete_details.get("reason"))
        if reason == RESPONSES_INCOMPLETE_REASON_LENGTH:
            return FINISH_REASON_LENGTH
        if reason == RESPONSES_INCOMPLETE_REASON_CONTENT_FILTER:
            return FINISH_REASON_CONTENT_FILTER
    status = _optional_str(body.get("status"))
    if status == RESPONSES_STATUS_COMPLETED:
        return FINISH_REASON_STOP
    return None


def _text_from_responses_output(output: Any) -> str | None:
    if not isinstance(output, Sequence) or isinstance(output, str | bytes):
        return None
    parts: list[str] = []
    for item in output:
        content = _get(item, "content")
        if not isinstance(content, Sequence) or isinstance(
            content, str | bytes
        ):
            continue
        for part in content:
            if _get(part, "type") != OUTPUT_TEXT_PART_TYPE:
                continue
            text = _get(part, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts) or None


def _content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return None
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts) or None


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _parse_error(
    message: str,
    body: Mapping[str, Any],
    config: ProviderConfig,
) -> Exception:
    failure = failure_record(
        failure_class=FailureClass.PERMANENT,
        code=PARSE_ERROR_CODE,
        message=message,
        metadata={
            "provider_kind": config.provider_kind.value,
            "endpoint_kind": config.endpoint_kind.value,
            "response_preview": repr(dict(body))[:RESPONSE_PREVIEW_LIMIT],
        },
    )
    return raise_failure(failure)
