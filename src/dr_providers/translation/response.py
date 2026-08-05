"""Dispatch response translation by provider wire protocol."""

from collections.abc import Mapping
from typing import Any, assert_never

from dr_providers.modeling.call import ProviderCallConfig
from dr_providers.modeling.route import Protocol
from dr_providers.translation.anthropic_messages import (
    parse_anthropic_messages_body,
)
from dr_providers.translation.chat_completions import (
    parse_chat_completions_body,
)
from dr_providers.translation.common import ParseOutcome
from dr_providers.translation.responses import parse_responses_body


def parse_response(
    body: Mapping[str, Any],
    *,
    config: ProviderCallConfig,
) -> ParseOutcome:
    protocol = config.route.protocol
    if protocol is Protocol.CHAT_COMPLETIONS:
        return parse_chat_completions_body(body, config=config)
    if protocol is Protocol.ANTHROPIC_MESSAGES:
        return parse_anthropic_messages_body(body, config=config)
    if protocol is Protocol.RESPONSES:
        return parse_responses_body(body, config=config)
    assert_never(protocol)
