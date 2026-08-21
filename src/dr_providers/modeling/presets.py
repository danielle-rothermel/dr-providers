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


_OPENAI_COMPAT_CONTROLS: frozenset[RequestControl] = frozenset(
    {
        RequestControl.TEMPERATURE,
        RequestControl.TOP_P,
        RequestControl.TOKEN_LIMIT,
        RequestControl.REASONING,
        RequestControl.SEED,
    }
)
_ANTHROPIC_CONTROLS: frozenset[RequestControl] = frozenset(
    {
        RequestControl.TEMPERATURE,
        RequestControl.TOP_P,
        RequestControl.TOKEN_LIMIT,
        RequestControl.REASONING,
    }
)


def _chat_constraints(
    *,
    token_limit_parameter: TokenLimitParameter,
    reasoning_shape: ReasoningRequestShape,
    supported_controls: frozenset[RequestControl],
) -> ControlConstraints:
    return ControlConstraints(
        supported_controls=supported_controls,
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
    # Definition identity includes the explicit or derived extension-key set.
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
            supported_controls=_OPENAI_COMPAT_CONTROLS,
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
            supported_controls=_OPENAI_COMPAT_CONTROLS,
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
            supported_controls=_OPENAI_COMPAT_CONTROLS,
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
    """Use Google's OpenAI-compatible endpoint with an AI Studio key."""
    route = ModelRoute(
        provider=ProviderKind.GEMINI,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )
    return _config_from_route(
        definition_id="gemini.openai_compat",
        route=route,
        # The compatibility endpoint uses a flat reasoning_effort field.
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
            reasoning_shape=ReasoningRequestShape.EFFORT_FIELD,
            supported_controls=_OPENAI_COMPAT_CONTROLS,
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
    """Require ``TOKEN_LIMIT`` for Anthropic's ``max_tokens`` field."""
    route = ModelRoute(
        provider=ProviderKind.ANTHROPIC,
        protocol=Protocol.ANTHROPIC_MESSAGES,
        model=model,
    )
    return _config_from_route(
        definition_id="anthropic.messages",
        route=route,
        # Anthropic uses max_tokens and a native reasoning effort object.
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_TOKENS,
            reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
            supported_controls=_ANTHROPIC_CONTROLS,
        ),
        controls=controls,
        extensions=extensions,
        required_controls=frozenset({RequestControl.TOKEN_LIMIT}),
        extension_keys=extension_keys,
    )


class ProviderFactoryKind(StrEnum):
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
