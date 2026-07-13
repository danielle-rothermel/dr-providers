"""Response as materialized typed parts, plus the wire parsers.

``LlmResponse`` composes text, usage, cost, warnings, finish reason,
and provider metadata — so a future streaming mode can emit the same
parts incrementally without a breaking redesign.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
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
REFUSAL_PART_TYPE = "refusal"
MESSAGE_ITEM_TYPE = "message"
RESPONSES_INCOMPLETE_REASON_LENGTH = "max_output_tokens"
RESPONSES_INCOMPLETE_REASON_CONTENT_FILTER = "content_filter"
RESPONSES_STATUS_COMPLETED = "completed"
RESPONSES_STATUS_FAILED = "failed"
FINISH_REASON_LENGTH = "length"
FINISH_REASON_CONTENT_FILTER = "content_filter"
FINISH_REASON_STOP = "stop"
PARSE_ERROR_CODE = "response_parse_error"
RESPONSE_REFUSAL_CODE = "response_refusal"
RESPONSE_INCOMPLETE_NO_TEXT_CODE = "response_incomplete_no_text"
RESPONSE_FAILED_CODE = "response_failed"
RESPONSE_NO_TEXT_CODE = "response_no_text"
RESPONSE_PREVIEW_LIMIT = 512
RESPONSE_ID_HASH_LENGTH = 16
UNKNOWN_DIAGNOSTIC_CATEGORY = "unknown"
RESPONSES_STATUS_VALUES = frozenset(
    {"cancelled", "completed", "failed", "in_progress", "incomplete", "queued"}
)
RESPONSES_INCOMPLETE_REASON_VALUES = frozenset(
    {
        RESPONSES_INCOMPLETE_REASON_CONTENT_FILTER,
        RESPONSES_INCOMPLETE_REASON_LENGTH,
    }
)
RESPONSES_OUTPUT_ITEM_TYPE_VALUES = frozenset(
    {MESSAGE_ITEM_TYPE, "function_call", "reasoning"}
)
RESPONSES_CONTENT_PART_TYPE_VALUES = frozenset(
    {OUTPUT_TEXT_PART_TYPE, REFUSAL_PART_TYPE}
)


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


class ResponsesDiagnostics(BaseModel):
    """Safe, content-free observations from an OpenAI Responses body.

    Provider-controlled enums are retained only when explicitly allowlisted;
    all other string values are coalesced into ``unknown`` count categories.
    ``response_id_hash`` is a truncated, unsalted SHA-256 digest retained for
    correlating high-entropy provider IDs. It does not protect low-entropy
    values from dictionary attacks, so callers must never treat hashing as a
    substitute for excluding arbitrary provider content.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    response_status: StrictStr | None = None
    incomplete_reason: StrictStr | None = None
    output_item_types: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    content_part_types: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )
    output_text_len: StrictInt = 0
    refusal_len: StrictInt | None = None
    response_id_hash: StrictStr | None = None


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr
    usage: TokenUsage | None = None
    cost: CostInfo | None = None
    warnings: tuple[LlmWarning, ...] = ()
    finish_reason: StrictStr | None = None
    response_id: StrictStr | None = None
    model: StrictStr | None = None
    diagnostics: ResponsesDiagnostics | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class _ResponsesWalk:
    text: str
    diagnostics: ResponsesDiagnostics
    parse_error: str | None = None


@dataclass(frozen=True)
class _ResponsesOutputWalk:
    text: str
    refusal_len: int | None
    item_types: dict[str, int]
    content_types: dict[str, int]
    parse_error: str | None = None


@dataclass(frozen=True)
class _ResponsesContentWalk:
    text: str
    refusal_len: int | None
    content_types: dict[str, int]
    parse_error: str | None = None


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
    walk = _walk_responses_body(body)
    if walk.parse_error is not None:
        raise _responses_failure(
            code=PARSE_ERROR_CODE,
            message=walk.parse_error,
            diagnostics=walk.diagnostics,
            config=config,
        )
    if walk.diagnostics.response_status == RESPONSES_STATUS_FAILED:
        raise _responses_failure(
            code=RESPONSE_FAILED_CODE,
            message="provider response failed",
            diagnostics=walk.diagnostics,
            config=config,
        )
    if not walk.text.strip():
        if walk.diagnostics.refusal_len is not None:
            code = RESPONSE_REFUSAL_CODE
            message = "provider response contained a refusal and no text"
        elif walk.diagnostics.response_status == "incomplete":
            code = RESPONSE_INCOMPLETE_NO_TEXT_CODE
            message = "provider response was incomplete and contained no text"
        elif walk.diagnostics.response_status is None:
            code = PARSE_ERROR_CODE
            message = "provider response missing status and generation text"
        else:
            code = RESPONSE_NO_TEXT_CODE
            message = "provider response contained no generation text"
        raise _responses_failure(
            code=code,
            message=message,
            diagnostics=walk.diagnostics,
            config=config,
        )
    response_id = _optional_str(body.get("id"))
    return LlmResponse(
        text=walk.text,
        usage=token_usage_from_body(body),
        cost=cost_from_body(body),
        finish_reason=_finish_reason_from_responses_body(body),
        response_id=response_id,
        model=_optional_str(body.get("model")) or config.model,
        diagnostics=walk.diagnostics,
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


def _walk_responses_body(body: Mapping[str, Any]) -> _ResponsesWalk:
    status = _allowlisted_diagnostic_value(
        body.get("status"), RESPONSES_STATUS_VALUES
    )
    incomplete_details = body.get("incomplete_details")
    incomplete_reason = None
    if isinstance(incomplete_details, Mapping):
        incomplete_reason = _allowlisted_diagnostic_value(
            incomplete_details.get("reason"),
            RESPONSES_INCOMPLETE_REASON_VALUES,
        )

    output_walk = _walk_responses_output(body.get("output"))
    response_id = _optional_str(body.get("id"))
    diagnostics = ResponsesDiagnostics(
        response_status=status,
        incomplete_reason=incomplete_reason,
        output_item_types=output_walk.item_types,
        content_part_types=output_walk.content_types,
        output_text_len=len(output_walk.text),
        refusal_len=output_walk.refusal_len,
        response_id_hash=_hash_response_id(response_id),
    )
    return _ResponsesWalk(
        text=output_walk.text,
        diagnostics=diagnostics,
        parse_error=output_walk.parse_error,
    )


def _walk_responses_output(output: Any) -> _ResponsesOutputWalk:
    parse_error = None
    text_parts: list[str] = []
    refusal_len = 0
    saw_refusal = False
    item_types: Counter[str] = Counter()
    content_types: Counter[str] = Counter()

    if not _is_sequence(output):
        parse_error = "provider response output is not a list"
    else:
        for item in output:
            if not isinstance(item, Mapping):
                parse_error = "provider response output item is not an object"
                break
            item_type = _optional_str(item.get("type"))
            if item_type is None:
                parse_error = "provider response output item missing type"
                break
            item_types[
                _allowlisted_diagnostic_category(
                    item_type, RESPONSES_OUTPUT_ITEM_TYPE_VALUES
                )
            ] += 1

            if item_type != MESSAGE_ITEM_TYPE:
                continue
            content_walk = _walk_responses_content(item.get("content"))
            text_parts.append(content_walk.text)
            content_types.update(content_walk.content_types)
            if content_walk.refusal_len is not None:
                saw_refusal = True
                refusal_len += content_walk.refusal_len
            if content_walk.parse_error is not None:
                parse_error = content_walk.parse_error
                break

    text = "".join(text_parts)
    return _ResponsesOutputWalk(
        text=text,
        refusal_len=refusal_len if saw_refusal else None,
        item_types=dict(sorted(item_types.items())),
        content_types=dict(sorted(content_types.items())),
        parse_error=parse_error,
    )


def _walk_responses_content(content: Any) -> _ResponsesContentWalk:
    if not _is_sequence(content):
        return _ResponsesContentWalk(
            text="",
            refusal_len=None,
            content_types={},
            parse_error="provider response message content is not a list",
        )

    text_parts: list[str] = []
    refusal_len = 0
    saw_refusal = False
    content_types: Counter[str] = Counter()
    for part in content:
        if not isinstance(part, Mapping):
            return _ResponsesContentWalk(
                text="".join(text_parts),
                refusal_len=refusal_len if saw_refusal else None,
                content_types=dict(sorted(content_types.items())),
                parse_error="provider response content part is not an object",
            )
        part_type = _optional_str(part.get("type"))
        if part_type is None:
            return _ResponsesContentWalk(
                text="".join(text_parts),
                refusal_len=refusal_len if saw_refusal else None,
                content_types=dict(sorted(content_types.items())),
                parse_error="provider response content part missing type",
            )
        content_types[
            _allowlisted_diagnostic_category(
                part_type, RESPONSES_CONTENT_PART_TYPE_VALUES
            )
        ] += 1
        if part_type == OUTPUT_TEXT_PART_TYPE:
            text = part.get("text")
            if not isinstance(text, str):
                return _content_parse_error(
                    text_parts=text_parts,
                    refusal_len=refusal_len if saw_refusal else None,
                    content_types=content_types,
                    message="response output_text value is not a string",
                )
            text_parts.append(text)
        elif part_type == REFUSAL_PART_TYPE:
            saw_refusal = True
            refusal = part.get("refusal")
            if not isinstance(refusal, str):
                return _content_parse_error(
                    text_parts=text_parts,
                    refusal_len=refusal_len,
                    content_types=content_types,
                    message="response refusal value is not a string",
                )
            refusal_len += len(refusal)
    return _ResponsesContentWalk(
        text="".join(text_parts),
        refusal_len=refusal_len if saw_refusal else None,
        content_types=dict(sorted(content_types.items())),
    )


def _content_parse_error(
    *,
    text_parts: list[str],
    refusal_len: int | None,
    content_types: Counter[str],
    message: str,
) -> _ResponsesContentWalk:
    return _ResponsesContentWalk(
        text="".join(text_parts),
        refusal_len=refusal_len,
        content_types=dict(sorted(content_types.items())),
        parse_error=message,
    )


def _hash_response_id(response_id: str | None) -> str | None:
    if response_id is None:
        return None
    return sha256(response_id.encode()).hexdigest()[:RESPONSE_ID_HASH_LENGTH]


def _allowlisted_diagnostic_value(
    value: Any, allowed_values: frozenset[str]
) -> str | None:
    raw_value = _optional_str(value)
    if raw_value is None:
        return None
    if raw_value in allowed_values:
        return raw_value
    return UNKNOWN_DIAGNOSTIC_CATEGORY


def _allowlisted_diagnostic_category(
    value: str, allowed_values: frozenset[str]
) -> str:
    if value in allowed_values:
        return value
    return UNKNOWN_DIAGNOSTIC_CATEGORY


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


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


def _responses_failure(
    *,
    code: str,
    message: str,
    diagnostics: ResponsesDiagnostics,
    config: ProviderConfig,
) -> Exception:
    failure = failure_record(
        failure_class=FailureClass.PERMANENT,
        code=code,
        message=message,
        metadata={
            "provider_kind": config.provider_kind.value,
            "endpoint_kind": config.endpoint_kind.value,
            "diagnostics": diagnostics.model_dump(),
        },
    )
    return raise_failure(failure)
