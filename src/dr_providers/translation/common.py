from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from dr_providers.core.failures import FailureClass
from dr_providers.outcomes.models import (
    CostInfo,
    ProviderStopReason,
    ProviderTransportFailure,
    ProviderTransportResponse,
    TokenUsage,
)

if TYPE_CHECKING:
    from dr_providers.modeling.call import ProviderCallConfig

PARSE_ERROR_CODE = "response_parse_error"
RESPONSE_NO_TEXT_CODE = "response_no_text"

ParseOutcome = ProviderTransportResponse | ProviderTransportFailure


def token_usage_from_body(body: Mapping[str, Any]) -> TokenUsage | None:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return None
    prompt = optional_int(
        usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    completion = optional_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total = optional_int(usage.get("total_tokens"))
    reasoning = None
    for details_key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, Mapping):
            reasoning = optional_int(details.get("reasoning_tokens"))
            if reasoning is not None:
                break
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
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
        if isinstance(value, int | float) and not isinstance(value, bool):
            return CostInfo(total_cost=float(value))
    usage = body.get("usage")
    if isinstance(usage, Mapping):
        value = usage.get("cost")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return CostInfo(total_cost=float(value))
    return None


def content_to_text(content: Any) -> str | None:
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


def get_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def stop_reason_from_chat_completions(
    value: str | None,
) -> ProviderStopReason | None:
    if value == "stop":
        return ProviderStopReason.STOP
    if value == "length":
        return ProviderStopReason.LENGTH
    if value == "content_filter":
        return ProviderStopReason.CONTENT_FILTER
    return None


def stop_reason_from_anthropic(value: str | None) -> ProviderStopReason | None:
    if value in {"end_turn", "stop_sequence"}:
        return ProviderStopReason.STOP
    if value == "max_tokens":
        return ProviderStopReason.LENGTH
    return None


def parse_failure(
    message: str,
    body: Mapping[str, Any],
    config: ProviderCallConfig,
    *,
    code: str = PARSE_ERROR_CODE,
) -> ProviderTransportFailure:
    return ProviderTransportFailure(
        failure_class=FailureClass.PERMANENT,
        code=code,
        message=message,
        response_body=dict(body),
        metadata={
            "provider": config.route.provider.value,
            "protocol": config.route.protocol.value,
        },
    )
