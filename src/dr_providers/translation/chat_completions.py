"""Translate Chat Completions wire bodies into typed outcomes."""

from collections.abc import Mapping, Sequence
from typing import Any

from dr_providers.modeling.call import ProviderCallConfig
from dr_providers.outcomes.models import ProviderTransportResponse
from dr_providers.translation.common import (
    ParseOutcome,
    content_to_text,
    cost_from_body,
    get_value,
    optional_str,
    parse_failure,
    token_usage_from_body,
)


def parse_chat_completions_body(
    body: Mapping[str, Any],
    *,
    config: ProviderCallConfig,
) -> ParseOutcome:
    choices = body.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
        return parse_failure("provider response missing choices", body, config)
    if not choices:
        return parse_failure(
            "provider response has empty choices", body, config
        )
    choice = choices[0]
    message = get_value(choice, "message")
    text = content_to_text(get_value(message, "content"))
    if text is None or not text.strip():
        return parse_failure(
            "provider response produced no generation text", body, config
        )
    return ProviderTransportResponse(
        text=text,
        raw_body=dict(body),
        usage=token_usage_from_body(body),
        cost=cost_from_body(body),
        finish_reason=optional_str(get_value(choice, "finish_reason")),
        response_id=optional_str(body.get("id")),
        model=optional_str(body.get("model")) or config.route.model,
    )
