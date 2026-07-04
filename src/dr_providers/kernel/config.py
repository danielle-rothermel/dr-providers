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

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_OPENAI_COMPAT_BASE_URL = (
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


class ReasoningRequestShape(StrEnum):
    NONE = "none"
    EXTRA_BODY = "extra_body"
    TOP_LEVEL = "top_level"


class RequestControl(StrEnum):
    """Knobs a request may set; configs declare which they transport."""

    TEMPERATURE = "temperature"
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
    base_url: str = OPENROUTER_BASE_URL,
) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENROUTER,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        api_key_env=OPENROUTER_API_KEY_ENV,
        base_url=base_url,
        reasoning_shape=ReasoningRequestShape.EXTRA_BODY,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )


def openai_chat_config(*, model: str) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        api_key_env=OPENAI_API_KEY_ENV,
        base_url=OPENAI_BASE_URL,
        reasoning_shape=ReasoningRequestShape.TOP_LEVEL,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )


def openai_responses_config(*, model: str) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model=model,
        api_key_env=OPENAI_API_KEY_ENV,
        base_url=OPENAI_BASE_URL,
        reasoning_shape=ReasoningRequestShape.TOP_LEVEL,
        token_limit_parameter=TokenLimitParameter.MAX_OUTPUT_TOKENS,
    )


def gemini_chat_config(*, model: str) -> ProviderConfig:
    """Gemini via Google's OpenAI-compatible endpoint (AI Studio key)."""
    return ProviderConfig(
        provider_kind=ProviderKind.GEMINI,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        api_key_env=GEMINI_API_KEY_ENV,
        base_url=GEMINI_OPENAI_COMPAT_BASE_URL,
        # The compat endpoint takes reasoning_effort in the body, not an
        # OpenAI-style top-level reasoning object; thinking budgets need
        # the native API (deferred until compat gaps require it).
        reasoning_shape=ReasoningRequestShape.EXTRA_BODY,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )
