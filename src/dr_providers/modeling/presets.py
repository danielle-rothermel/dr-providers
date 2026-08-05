"""Preset Provider Call Config builders and their shared registry."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from dr_providers.modeling.call import (
    ProviderCallConfig,
    ProviderCallDefinition,
)
from dr_providers.modeling.controls import (
    ControlConstraints,
    GenerationControls,
    ProviderBodyExtensions,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
)
from dr_providers.modeling.route import ModelRoute, Protocol, ProviderKind

if TYPE_CHECKING:
    from collections.abc import Callable


def _chat_constraints(
    *,
    token_limit_parameter: TokenLimitParameter,
    reasoning_shape: ReasoningRequestShape,
) -> ControlConstraints:
    return ControlConstraints(
        token_limit_parameter=token_limit_parameter,
        reasoning_shape=reasoning_shape,
    )


def _config_from_route(  # noqa: PLR0913 -- explicit keyword-only builder
    *,
    definition_id: str,
    route: ModelRoute,
    constraints: ControlConstraints,
    controls: GenerationControls | None,
    extensions: ProviderBodyExtensions | None,
    required_controls: frozenset[RequestControl] = frozenset(),
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    # When ``extension_keys`` is given the caller declares the exact set of
    # extensions the Definition exposes and passed extensions are validated
    # strictly against it. When omitted, the declared set is derived from the
    # extensions actually passed; either way the declared set is captured in
    # the Definition identity, so the undeclared-extension check is never
    # vacuous.
    if extension_keys is None:
        declared_keys = (
            frozenset(extensions.extra_body) if extensions else frozenset()
        )
    else:
        declared_keys = frozenset(extension_keys)
    definition = ProviderCallDefinition(
        definition_id=definition_id,
        route=route,
        constraints=constraints,
        required_controls=required_controls,
        extension_keys=declared_keys,
    )
    return definition.materialize(controls=controls, extensions=extensions)


def openrouter_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    route = ModelRoute(
        provider=ProviderKind.OPENROUTER,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )
    return _config_from_route(
        definition_id="openrouter.chat_completions",
        route=route,
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
            reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        ),
        controls=controls,
        extensions=extensions,
        extension_keys=extension_keys,
    )


def openai_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    route = ModelRoute(
        provider=ProviderKind.OPENAI,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )
    return _config_from_route(
        definition_id="openai.chat_completions",
        route=route,
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
            reasoning_shape=ReasoningRequestShape.EFFORT_FIELD,
        ),
        controls=controls,
        extensions=extensions,
        extension_keys=extension_keys,
    )


def openai_responses_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    route = ModelRoute(
        provider=ProviderKind.OPENAI,
        protocol=Protocol.RESPONSES,
        model=model,
    )
    return _config_from_route(
        definition_id="openai.responses",
        route=route,
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_OUTPUT_TOKENS,
            reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        ),
        controls=controls,
        extensions=extensions,
        extension_keys=extension_keys,
    )


def gemini_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    """Gemini via Google's OpenAI-compatible endpoint (AI Studio key)."""
    route = ModelRoute(
        provider=ProviderKind.GEMINI,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )
    return _config_from_route(
        definition_id="gemini.openai_compat",
        route=route,
        # The compat endpoint takes a flat reasoning_effort field, not an
        # OpenAI-style nested reasoning object; thinking budgets need the
        # native API (deferred until compat gaps require it).
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
            reasoning_shape=ReasoningRequestShape.EFFORT_FIELD,
        ),
        controls=controls,
        extensions=extensions,
        extension_keys=extension_keys,
    )


def anthropic_messages_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    """Anthropic Messages protocol over a (custom-capable) base URL.

    Anthropic requires ``max_tokens``, so ``TOKEN_LIMIT`` is a required
    control: a Config without a token limit cannot materialize.
    """
    route = ModelRoute(
        provider=ProviderKind.ANTHROPIC,
        protocol=Protocol.ANTHROPIC_MESSAGES,
        model=model,
    )
    return _config_from_route(
        definition_id="anthropic.messages",
        route=route,
        # Anthropic Messages requires max_tokens and serializes reasoning
        # as a native effort object.
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_TOKENS,
            reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        ),
        controls=controls,
        extensions=extensions,
        required_controls=frozenset({RequestControl.TOKEN_LIMIT}),
        extension_keys=extension_keys,
    )


class ProviderFactoryKind(StrEnum):
    """Canonical identifier for each preset Config factory."""

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
