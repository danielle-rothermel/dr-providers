from __future__ import annotations

from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from dr_providers.modeling.route import ProviderKind

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 90.0
DEFAULT_MAX_CONNECTIONS = 10
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 5
DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class ApiKeyEnv(StrEnum):
    OPENROUTER = "OPENROUTER_API_KEY"
    OPENAI = "OPENAI_API_KEY"
    GEMINI = "GEMINI_API_KEY"
    ANTHROPIC = "ANTHROPIC_API_KEY"


class ProviderBaseUrl(StrEnum):
    OPENROUTER = "https://openrouter.ai/api/v1"
    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com/v1"
    GEMINI_OPENAI_COMPAT = (
        "https://generativelanguage.googleapis.com/v1beta/openai"
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


class ProviderTransportPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_kind: ProviderKind
    api_key_env: StrictStr
    base_url: StrictStr | None = None
    """Retained verbatim in evidence after URL userinfo is rejected."""
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        allow_inf_nan=False,
        strict=True,
    )
    """Native connect, write, and pool timeout bound."""
    idle_timeout_seconds: float = Field(
        default=DEFAULT_IDLE_TIMEOUT_SECONDS,
        gt=0,
        allow_inf_nan=False,
        strict=True,
    )
    """Native response-read idle bound, clamped to ``timeout_seconds``."""
    max_connections: int = Field(
        default=DEFAULT_MAX_CONNECTIONS,
        gt=0,
        strict=True,
    )
    """Maximum open connections in the provider-owned client pool."""
    max_keepalive_connections: int = Field(
        default=DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        gt=0,
        strict=True,
    )
    """Maximum idle connections retained by the provider-owned client."""
    max_request_bytes: int = Field(
        default=DEFAULT_MAX_REQUEST_BYTES,
        gt=0,
        strict=True,
    )
    """Maximum exact UTF-8 JSON request-body bytes dispatched."""
    max_response_bytes: int = Field(
        default=DEFAULT_MAX_RESPONSE_BYTES,
        gt=0,
        strict=True,
    )
    """Maximum decompressed response-body bytes retained and decoded."""

    @field_validator("base_url")
    @classmethod
    def _reject_url_userinfo(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            msg = "base_url must not contain URL userinfo"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _normalize_and_validate_limits(self) -> ProviderTransportPolicy:
        if self.idle_timeout_seconds > self.timeout_seconds:
            object.__setattr__(
                self, "idle_timeout_seconds", self.timeout_seconds
            )
        if self.max_keepalive_connections > self.max_connections:
            msg = "max_keepalive_connections must not exceed max_connections"
            raise ValueError(msg)
        return self

    def identity_payload(self) -> dict[str, Any]:
        """Include the credential variable name, never its value.

        ``base_url`` is retained verbatim after URL userinfo is rejected.
        """
        return {
            "provider_kind": self.provider_kind.value,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "max_connections": self.max_connections,
            "max_keepalive_connections": self.max_keepalive_connections,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
        }


def policy_for(  # noqa: PLR0913 -- one explicit transport policy surface
    kind: ProviderKind,
    *,
    api_key_env: ApiKeyEnv | str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> ProviderTransportPolicy:
    resolved_key_env = (
        DEFAULT_API_KEY_ENVS[kind] if api_key_env is None else api_key_env
    )
    resolved_base_url = (
        str(DEFAULT_BASE_URLS[kind]) if base_url is None else base_url
    )
    return ProviderTransportPolicy(
        provider_kind=kind,
        api_key_env=str(resolved_key_env),
        base_url=resolved_base_url,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
    )
