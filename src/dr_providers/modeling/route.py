from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, StrictStr


class ProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class Protocol(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


class ModelRoute(BaseModel):
    """Identity excludes credentials, accounts, and generation controls."""

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
    """Quota identity excludes credentials, accounts, and overrides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderKind
    protocol: Protocol
    model: StrictStr

    def label(self) -> str:
        return f"{self.provider.value}:{self.protocol.value}:{self.model}"
