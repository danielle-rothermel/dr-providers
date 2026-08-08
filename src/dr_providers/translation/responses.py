from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from dr_providers.core.failures import FailureClass
from dr_providers.outcomes.models import (
    ProviderTransportFailure,
    ProviderTransportResponse,
    ResponsesDiagnostics,
)
from dr_providers.translation.common import (
    PARSE_ERROR_CODE,
    RESPONSE_NO_TEXT_CODE,
    ParseOutcome,
    cost_from_body,
    optional_str,
    token_usage_from_body,
)

if TYPE_CHECKING:
    from dr_providers.modeling.call import ProviderCallConfig

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
RESPONSE_REFUSAL_CODE = "response_refusal"
RESPONSE_INCOMPLETE_NO_TEXT_CODE = "response_incomplete_no_text"
RESPONSE_FAILED_CODE = "response_failed"
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


def parse_responses_body(
    body: Mapping[str, Any],
    *,
    config: ProviderCallConfig,
) -> ParseOutcome:
    walk = _walk_responses_body(body)
    if walk.parse_error is not None:
        return _responses_failure(
            code=PARSE_ERROR_CODE,
            message=walk.parse_error,
            diagnostics=walk.diagnostics,
            body=body,
            config=config,
        )
    if walk.diagnostics.response_status == RESPONSES_STATUS_FAILED:
        return _responses_failure(
            code=RESPONSE_FAILED_CODE,
            message="provider response failed",
            diagnostics=walk.diagnostics,
            body=body,
            config=config,
        )
    if not walk.text.strip():
        code, message = _responses_no_text_reason(walk.diagnostics)
        return _responses_failure(
            code=code,
            message=message,
            diagnostics=walk.diagnostics,
            body=body,
            config=config,
        )
    return ProviderTransportResponse(
        text=walk.text,
        response_body=dict(body),
        usage=token_usage_from_body(body),
        cost=cost_from_body(body),
        finish_reason=_finish_reason_from_responses_body(body),
        response_id=optional_str(body.get("id")),
        model=optional_str(body.get("model")) or config.route.model,
        diagnostics=walk.diagnostics,
    )


def _responses_no_text_reason(
    diagnostics: ResponsesDiagnostics,
) -> tuple[str, str]:
    if diagnostics.refusal_len is not None:
        return (
            RESPONSE_REFUSAL_CODE,
            "provider response contained a refusal and no text",
        )
    if diagnostics.response_status == "incomplete":
        return (
            RESPONSE_INCOMPLETE_NO_TEXT_CODE,
            "provider response was incomplete and contained no text",
        )
    if diagnostics.response_status is None:
        return (
            PARSE_ERROR_CODE,
            "provider response missing status and generation text",
        )
    return (
        RESPONSE_NO_TEXT_CODE,
        "provider response contained no generation text",
    )


def _finish_reason_from_responses_body(
    body: Mapping[str, Any],
) -> str | None:
    incomplete_details = body.get("incomplete_details")
    if isinstance(incomplete_details, Mapping):
        reason = optional_str(incomplete_details.get("reason"))
        if reason == RESPONSES_INCOMPLETE_REASON_LENGTH:
            return FINISH_REASON_LENGTH
        if reason == RESPONSES_INCOMPLETE_REASON_CONTENT_FILTER:
            return FINISH_REASON_CONTENT_FILTER
    status = optional_str(body.get("status"))
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
    response_id = optional_str(body.get("id"))
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
            item_type = optional_str(item.get("type"))
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
        part_type = optional_str(part.get("type"))
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
    raw_value = optional_str(value)
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


def _responses_failure(
    *,
    code: str,
    message: str,
    diagnostics: ResponsesDiagnostics,
    body: Mapping[str, Any],
    config: ProviderCallConfig,
) -> ProviderTransportFailure:
    return ProviderTransportFailure(
        failure_class=FailureClass.PERMANENT,
        code=code,
        message=message,
        response_body=dict(body),
        metadata={
            "provider": config.route.provider.value,
            "protocol": config.route.protocol.value,
            "diagnostics": diagnostics.model_dump(),
        },
    )
