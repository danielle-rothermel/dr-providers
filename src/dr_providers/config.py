"""Provider Call Config: complete validated assignment with Identity Hash.

A Provider Call Config is materialized by assigning every required
Variable of exactly one Provider Call Definition. It carries its typed
Definition reference and its Identity Hash (full 64-char SHA-256 via
dr-serialize). Its identity-bearing fields are the Model Route plus
every output-affecting generation control, control-mapping constraint,
and provider body extension. Transport policy is excluded from identity.

Preset builders ship for OpenRouter, OpenAI, Gemini (OpenAI-compat), and
Anthropic Messages.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from dr_serialize import (
    IdentityDocument,
    build_identity_document,
    identity_document_hash,
)
from pydantic import BaseModel, ConfigDict

from dr_providers.controls import (
    ControlConstraints,
    GenerationControls,
    ProviderBodyExtensions,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
)
from dr_providers.definition import ProviderCallDefinition
from dr_providers.route import (
    ApiKeyEnv,
    ModelRoute,
    Protocol,
    ProviderBaseUrl,
    ProviderKind,
    ProviderQuotaIdentity,
)

PROVIDER_CALL_CONFIG_SCHEMA = "dr_providers.provider_call_config"
PROVIDER_CALL_CONFIG_SCHEMA_VERSION = 1

_ALL_CONTROLS = frozenset(
    {
        RequestControl.TEMPERATURE,
        RequestControl.TOP_P,
        RequestControl.TOKEN_LIMIT,
        RequestControl.REASONING,
    }
)


class ProviderCallConfig(BaseModel):
    """A Definition with every required Variable set, plus Identity Hash.

    Construct via :meth:`ProviderCallDefinition.materialize` or a preset
    builder so the assignment is always validated against its owner.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    definition: ProviderCallDefinition
    controls: GenerationControls = GenerationControls()
    extensions: ProviderBodyExtensions = ProviderBodyExtensions()

    @property
    def route(self) -> ModelRoute:
        return self.definition.route

    @property
    def quota_identity(self) -> ProviderQuotaIdentity:
        return self.definition.route.quota_identity

    def identity_payload(self) -> dict[str, Any]:
        """Model Route + output-affecting controls/extensions; transport
        policy excluded."""
        return {
            "definition_id": self.definition.definition_id,
            "definition_schema_version": self.definition.schema_version,
            "route": self.route.identity_payload(),
            "constraints": self.definition.constraints.identity_payload(),
            "controls": self.controls.identity_payload(),
            "extensions": self.extensions.identity_payload(),
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_CALL_CONFIG_SCHEMA,
            schema_version=PROVIDER_CALL_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        """Full 64-char lowercase SHA-256 Config Identity Hash."""
        return identity_document_hash(self.identity_document())


# --- Presets ---------------------------------------------------------------


def _chat_constraints(
    *,
    token_limit_parameter: TokenLimitParameter,
    reasoning_shape: ReasoningRequestShape,
) -> ControlConstraints:
    return ControlConstraints(
        supported_controls=_ALL_CONTROLS,
        token_limit_parameter=token_limit_parameter,
        reasoning_shape=reasoning_shape,
    )


def _config_from_route(
    *,
    definition_id: str,
    route: ModelRoute,
    constraints: ControlConstraints,
    controls: GenerationControls | None,
    extensions: ProviderBodyExtensions | None,
) -> ProviderCallConfig:
    extension_keys = (
        frozenset(extensions.extra_body) if extensions else frozenset()
    )
    definition = ProviderCallDefinition(
        definition_id=definition_id,
        route=route,
        constraints=constraints,
        extension_keys=extension_keys,
    )
    return definition.materialize(controls=controls, extensions=extensions)


def openrouter_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
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
    )


def openai_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
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
    )


def openai_responses_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
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
    )


def gemini_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
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
    )


def anthropic_messages_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
) -> ProviderCallConfig:
    """Anthropic Messages protocol over a (custom-capable) base URL."""
    route = ModelRoute(
        provider=ProviderKind.ANTHROPIC,
        protocol=Protocol.ANTHROPIC_MESSAGES,
        model=model,
    )
    return _config_from_route(
        definition_id="anthropic.messages",
        route=route,
        # Anthropic Messages requires max_tokens and serializes reasoning
        # as a native thinking object.
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_TOKENS,
            reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        ),
        controls=controls,
        extensions=extensions,
    )


DEFAULT_BASE_URLS: dict[ProviderKind, ProviderBaseUrl] = {
    ProviderKind.OPENROUTER: ProviderBaseUrl.OPENROUTER,
    ProviderKind.OPENAI: ProviderBaseUrl.OPENAI,
    ProviderKind.GEMINI: ProviderBaseUrl.GEMINI_OPENAI_COMPAT,
    ProviderKind.ANTHROPIC: ProviderBaseUrl.ANTHROPIC,
}

DEFAULT_API_KEY_ENVS: dict[ProviderKind, ApiKeyEnv] = {
    ProviderKind.OPENROUTER: ApiKeyEnv.OPENROUTER,
    ProviderKind.OPENAI: ApiKeyEnv.OPENAI,
    ProviderKind.GEMINI: ApiKeyEnv.GEMINI,
    ProviderKind.ANTHROPIC: ApiKeyEnv.ANTHROPIC,
}
