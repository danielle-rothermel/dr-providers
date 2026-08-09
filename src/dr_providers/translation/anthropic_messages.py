from collections.abc import Mapping, Sequence
from typing import Any

from dr_providers.modeling.call import ProviderCallConfig
from dr_providers.outcomes.models import ProviderTransportResponse
from dr_providers.translation.common import (
    RESPONSE_NO_TEXT_CODE,
    ParseOutcome,
    cost_from_body,
    optional_str,
    parse_failure,
    token_usage_from_body,
)


def parse_anthropic_messages_body(
    body: Mapping[str, Any],
    *,
    config: ProviderCallConfig,
) -> ParseOutcome:
    content = body.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return parse_failure(
            "anthropic response missing content list", body, config
        )
    text, parse_error = _anthropic_text(content)
    if parse_error is not None:
        return parse_failure(parse_error, body, config)
    if text is None or not text.strip():
        return parse_failure(
            "anthropic response produced no generation text",
            body,
            config,
            code=RESPONSE_NO_TEXT_CODE,
        )
    return ProviderTransportResponse(
        text=text,
        response_body=dict(body),
        usage=token_usage_from_body(body),
        cost=cost_from_body(body),
        finish_reason=optional_str(body.get("stop_reason")),
        response_id=optional_str(body.get("id")),
        model=optional_str(body.get("model")) or config.route.model,
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
