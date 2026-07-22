"""Model Route, Protocol, and Provider Quota Identity.

The Model Route is the executable target tuple ``(provider, protocol,
model)`` — no credentials, accounts, or generation controls. The
Provider Quota Identity is exactly that tuple with no credential,
account, or override component; Whetstone derives collision-free Stage
labels from it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, StrictStr


class ProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class Protocol(StrEnum):
    """Wire protocol of a Model Route (a Model Route component).

    ``chat_completions`` is the OpenAI-compatible / OpenRouter surface,
    ``responses`` is the OpenAI Responses surface, and
    ``anthropic_messages`` is the Anthropic Messages surface. All three
    are first-class transport paths.
    """

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


class ApiKeyEnv(StrEnum):
    """Environment variables the preset routes read API keys from."""

    OPENROUTER = "OPENROUTER_API_KEY"
    OPENAI = "OPENAI_API_KEY"
    GEMINI = "GEMINI_API_KEY"
    ANTHROPIC = "ANTHROPIC_API_KEY"


class ProviderBaseUrl(StrEnum):
    """Default base URLs used by the preset provider configs."""

    OPENROUTER = "https://openrouter.ai/api/v1"
    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com/v1"
    # The OpenAI-compat surface, not "Gemini's URL": a future native
    # Gemini endpoint would be a sibling member, not this one.
    GEMINI_OPENAI_COMPAT = (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )


class ModelRoute(BaseModel):
    """Executable target tuple ``(provider, protocol, model)``.

    Excludes credentials, accounts, and generation controls. This is
    the identity-bearing route component of a Provider Call Config.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderKind
    protocol: Protocol
    model: StrictStr

    def identity_payload(self) -> dict[str, str]:
        return {
            "provider": self.provider.value,
            "protocol": self.protocol.value,
            "model": self.model,
        }

    @property
    def quota_identity(self) -> ProviderQuotaIdentity:
        return ProviderQuotaIdentity(
            provider=self.provider,
            protocol=self.protocol,
            model=self.model,
        )


class ProviderQuotaIdentity(BaseModel):
    """Best-effort quota identity: exactly ``(provider, protocol, model)``.

    No credential, account, or override component. Whetstone derives
    collision-free Stage labels from this tuple and owns concurrency.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderKind
    protocol: Protocol
    model: StrictStr

    def label(self) -> str:
        """Stable collision-free label string for Stage admission."""
        return f"{self.provider.value}:{self.protocol.value}:{self.model}"
