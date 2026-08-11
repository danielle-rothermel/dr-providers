from collections.abc import Mapping, Sequence
from typing import Any

from dr_providers.modeling.call import ProviderCallConfig
from dr_providers.outcomes.models import (
    ProviderStopReason,
    ProviderTransportResponse,
)
from dr_providers.translation.common import (
    PROVIDER_ERROR_ENVELOPE_CODE,
    RESPONSE_INCOMPLETE_NO_TEXT_CODE,
    RESPONSE_NO_TEXT_CODE,
    ParseOutcome,
    cost_from_body,
    optional_str,
    parse_failure,
    stop_reason_from_anthropic,
    token_usage_from_body,
)

ANTHROPIC_ERROR_MESSAGE_TYPE = "error"


def parse_anthropic_messages_body(
    body: Mapping[str, Any],
    *,
    config: ProviderCallConfig,
) -> ParseOutcome:
    """Parse an Anthropic Messages body into one outcome.

    Blank generation text is classified by the stop reason the body already
    carries: a ``max_tokens`` stop is truncation before any text, which a
    thinking-capable model reaches whenever its thinking consumes the whole
    token limit; ``end_turn`` or ``stop_sequence`` is a successful response
    whose generation is empty; and an absent or unknown stop reason is
    missing generation text.

    A body carrying an error envelope alongside text content remains a
    success: the provider produced the generation the caller asked for, and
    the envelope describes a partial upstream condition rather than a
    refusal to answer. An error envelope with no usable generation text,
    whether the content is absent, empty, or blank, is the envelope's own
    failure and outranks any stop reason the body also carries.
    """
    content = body.get("content")
    envelope = _has_error_envelope(body)
    if envelope and not content:
        return parse_failure(
            "anthropic returned an error envelope and no content",
            body,
            config,
            code=PROVIDER_ERROR_ENVELOPE_CODE,
        )
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return parse_failure(
            "anthropic response missing content list", body, config
        )
    text, parse_error = _anthropic_text(content)
    if parse_error is not None:
        return parse_failure(parse_error, body, config)
    stop_reason = stop_reason_from_anthropic(
        optional_str(body.get("stop_reason"))
    )
    if text is None or not text.strip():
        if envelope:
            return parse_failure(
                "anthropic returned an error envelope and no generation text",
                body,
                config,
                code=PROVIDER_ERROR_ENVELOPE_CODE,
            )
        blank = _blank_text_failure(body, config, stop_reason)
        if blank is not None:
            return blank
        text = text or ""
    return ProviderTransportResponse(
        text=text,
        response_body=dict(body),
        usage=token_usage_from_body(body),
        cost=cost_from_body(body),
        stop_reason=stop_reason,
        response_id=optional_str(body.get("id")),
        model=optional_str(body.get("model")) or config.route.model,
    )


def _blank_text_failure(
    body: Mapping[str, Any],
    config: ProviderCallConfig,
    stop_reason: ProviderStopReason | None,
) -> ParseOutcome | None:
    """Classify blank generation text, or admit a genuinely empty response.

    Returning ``None`` admits the response so the invocation classifier owns
    the empty-generation outcome for a provider that genuinely finished.
    """
    if stop_reason is ProviderStopReason.LENGTH:
        return parse_failure(
            "anthropic response was truncated before producing text",
            body,
            config,
            code=RESPONSE_INCOMPLETE_NO_TEXT_CODE,
        )
    if stop_reason is ProviderStopReason.STOP:
        return None
    return parse_failure(
        "anthropic response produced no generation text",
        body,
        config,
        code=RESPONSE_NO_TEXT_CODE,
    )


def _has_error_envelope(body: Mapping[str, Any]) -> bool:
    """Detect an upstream error delivered inside a success status code."""
    return body.get("type") == ANTHROPIC_ERROR_MESSAGE_TYPE and isinstance(
        body.get("error"), Mapping
    )


def _anthropic_text(content: Sequence[Any]) -> tuple[str | None, str | None]:
    """Reject malformed text blocks rather than silently truncating text."""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            return None, "anthropic response content block is not an object"
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str):
            return None, "anthropic response text block value is not a string"
        parts.append(text)
    return ("".join(parts) or None), None
