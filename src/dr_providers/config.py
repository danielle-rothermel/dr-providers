"""Provider configuration as data records, not classes.

Provider differences are config: base URL, API key env var, endpoint
kind, reasoning request shape, token-limit parameter name, and the set
of controls the provider can transport. Presets ship for OpenRouter,
OpenAI, and Gemini (via Google's OpenAI-compatible endpoint).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
)


class ApiKeyEnv(StrEnum):
    """Environment variables the preset configs read API keys from.

    Member names follow ``{name}_API_KEY`` by convention (enforced in
    tests); values stay explicit literals so they remain greppable.
    """

    OPENROUTER = "OPENROUTER_API_KEY"
    OPENAI = "OPENAI_API_KEY"
    GEMINI = "GEMINI_API_KEY"


class ProviderBaseUrl(StrEnum):
    """Default base URLs used by the preset provider configs."""

    OPENROUTER = "https://openrouter.ai/api/v1"
    OPENAI = "https://api.openai.com/v1"
    # The OpenAI-compat surface, not "Gemini's URL": a future native
    # Gemini endpoint would be a sibling member, not this one.
    GEMINI_OPENAI_COMPAT = (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )


class ProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"


class EndpointKind(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TokenLimitParameter(StrEnum):
    MAX_TOKENS = "max_tokens"
    MAX_COMPLETION_TOKENS = "max_completion_tokens"
    MAX_OUTPUT_TOKENS = "max_output_tokens"


class ReasoningEffort(StrEnum):
    """Typed cross-provider reasoning level (see ADR 0001)."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ReasoningRequestShape(StrEnum):
    """How a config serializes reasoning effort on the wire."""

    NONE = "none"
    EFFORT_FIELD = "effort_field"
    REASONING_OBJECT = "reasoning_object"


class RequestControl(StrEnum):
    """Knobs a request may set; configs declare which they transport."""

    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    TOKEN_LIMIT = "token_limit"  # noqa: S105 -- knob name, not a secret
    REASONING = "reasoning"


class PromptMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: StrictStr

    def provider_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class ProviderConfig(BaseModel):
    """Runtime provider call configuration (pure data)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_kind: ProviderKind
    endpoint_kind: EndpointKind
    model: StrictStr
    api_key_env: StrictStr
    base_url: StrictStr | None = None
    supported_controls: frozenset[RequestControl] = frozenset(
        {
            RequestControl.TEMPERATURE,
            RequestControl.TOP_P,
            RequestControl.TOKEN_LIMIT,
            RequestControl.REASONING,
        }
    )
    reasoning_shape: ReasoningRequestShape = ReasoningRequestShape.NONE
    token_limit_parameter: TokenLimitParameter
    extra_body: dict[str, Any] = Field(default_factory=dict)
    throttle_key: StrictStr | None = None
    allow_unsupported_control_drop: StrictBool = False

    @property
    def throttle_identity(self) -> str:
        if self.throttle_key:
            return self.throttle_key
        return (
            f"{self.provider_kind.value}:"
            f"{self.endpoint_kind.value}:{self.model}"
        )

    def supports(self, control: RequestControl) -> bool:
        return control in self.supported_controls


def openrouter_chat_config(
    *,
    model: str,
    base_url: str = ProviderBaseUrl.OPENROUTER,
) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENROUTER,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        api_key_env=ApiKeyEnv.OPENROUTER,
        base_url=base_url,
        reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )


def openai_chat_config(*, model: str) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        api_key_env=ApiKeyEnv.OPENAI,
        base_url=ProviderBaseUrl.OPENAI,
        reasoning_shape=ReasoningRequestShape.EFFORT_FIELD,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )


def openai_responses_config(*, model: str) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model=model,
        api_key_env=ApiKeyEnv.OPENAI,
        base_url=ProviderBaseUrl.OPENAI,
        reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        token_limit_parameter=TokenLimitParameter.MAX_OUTPUT_TOKENS,
    )


def gemini_chat_config(*, model: str) -> ProviderConfig:
    """Gemini via Google's OpenAI-compatible endpoint (AI Studio key)."""
    return ProviderConfig(
        provider_kind=ProviderKind.GEMINI,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        api_key_env=ApiKeyEnv.GEMINI,
        base_url=ProviderBaseUrl.GEMINI_OPENAI_COMPAT,
        # The compat endpoint takes a flat reasoning_effort field, not an
        # OpenAI-style nested reasoning object; thinking budgets need the
        # native API (deferred until compat gaps require it).
        reasoning_shape=ReasoningRequestShape.EFFORT_FIELD,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )
