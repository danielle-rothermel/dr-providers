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
    content_to_text,
    cost_from_body,
    get_value,
    optional_str,
    parse_failure,
    stop_reason_from_chat_completions,
    token_usage_from_body,
)


def parse_chat_completions_body(
    body: Mapping[str, Any],
    *,
    config: ProviderCallConfig,
) -> ParseOutcome:
    """Parse an OpenAI-compatible chat completion body into one outcome.

    Blank generation text is classified by the stop reason the body already
    carries: a length stop is truncation before any text, a genuine stop is a
    successful response whose generation is empty, and an absent or unknown
    stop reason is missing generation text.

    A body carrying an error envelope alongside a text-bearing choice remains
    a success: the provider produced the generation the caller asked for, and
    the envelope describes a partial upstream condition rather than a refusal
    to answer.
    """
    choices = body.get("choices")
    if _is_error_envelope(body, choices):
        return parse_failure(
            "provider returned an error envelope and no choices",
            body,
            config,
            code=PROVIDER_ERROR_ENVELOPE_CODE,
        )
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
        return parse_failure("provider response missing choices", body, config)
    if not choices:
        return parse_failure(
            "provider response has empty choices", body, config
        )
    choice = choices[0]
    message = get_value(choice, "message")
    text = content_to_text(get_value(message, "content"))
    stop_reason = stop_reason_from_chat_completions(
        optional_str(get_value(choice, "finish_reason"))
    )
    if text is None or not text.strip():
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
            "provider response was truncated before producing text",
            body,
            config,
            code=RESPONSE_INCOMPLETE_NO_TEXT_CODE,
        )
    if stop_reason is ProviderStopReason.STOP:
        return None
    return parse_failure(
        "provider response produced no generation text",
        body,
        config,
        code=RESPONSE_NO_TEXT_CODE,
    )


def _is_error_envelope(body: Mapping[str, Any], choices: Any) -> bool:
    """Detect an upstream error delivered inside a success status code."""
    return isinstance(body.get("error"), Mapping) and choices is None
