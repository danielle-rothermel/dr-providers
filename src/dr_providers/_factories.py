"""Shared config-factory registry for the CLI and serve surfaces.

The CLI (``ProviderChoice``) and serve (``ServeProviderKind``) surfaces both
map a provider selection to one of the same five preset Config factories.
Historically each redeclared its own enum→factory map with a DIVERGENT
spelling for the OpenAI Responses member (the CLI uses the hyphenated
``"openai-responses"`` and serve uses ``"openai_responses"``). Those two
public spellings are load-bearing wire/API surface and must NOT change, so
the divergence is deliberately preserved. What is unified here is the single
source of truth for the factory registry: both surfaces derive their factory
from :data:`FACTORY_BY_KIND` via their own enum's mapping to
:class:`ProviderFactoryKind`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from dr_providers.config import (
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dr_providers.config import ProviderCallConfig


class ProviderFactoryKind(StrEnum):
    """Canonical (snake_case) identifier for each preset Config factory."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


FACTORY_BY_KIND: dict[
    ProviderFactoryKind, Callable[..., ProviderCallConfig]
] = {
    ProviderFactoryKind.OPENROUTER: openrouter_chat_config,
    ProviderFactoryKind.OPENAI: openai_chat_config,
    ProviderFactoryKind.OPENAI_RESPONSES: openai_responses_config,
    ProviderFactoryKind.GEMINI: gemini_chat_config,
    ProviderFactoryKind.ANTHROPIC: anthropic_messages_config,
}
