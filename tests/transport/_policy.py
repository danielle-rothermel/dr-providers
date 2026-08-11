from __future__ import annotations

from typing import Any

from dr_providers import (
    ApiKeyEnv,
    ProviderBaseUrl,
    ProviderKind,
    ProviderTransportPolicy,
)

TEST_TIMEOUT_SECONDS = 120.0
TEST_CONNECT_TIMEOUT_SECONDS = 30.0
TEST_IDLE_TIMEOUT_SECONDS = 90.0
TEST_MAX_CONNECTIONS = 10
TEST_MAX_KEEPALIVE_CONNECTIONS = 5
TEST_MAX_REQUEST_BYTES = 1024 * 1024
TEST_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

_DEFAULT_API_KEY_ENVS: dict[ProviderKind, ApiKeyEnv] = {
    ProviderKind.OPENROUTER: ApiKeyEnv.OPENROUTER,
    ProviderKind.OPENAI: ApiKeyEnv.OPENAI,
    ProviderKind.GEMINI: ApiKeyEnv.GEMINI,
    ProviderKind.ANTHROPIC: ApiKeyEnv.ANTHROPIC,
}

_DEFAULT_BASE_URLS: dict[ProviderKind, ProviderBaseUrl] = {
    ProviderKind.OPENROUTER: ProviderBaseUrl.OPENROUTER,
    ProviderKind.OPENAI: ProviderBaseUrl.OPENAI,
    ProviderKind.GEMINI: ProviderBaseUrl.GEMINI_OPENAI_COMPAT,
    ProviderKind.ANTHROPIC: ProviderBaseUrl.ANTHROPIC,
}


def make_transport_policy(  # noqa: PLR0913 -- test helper mirrors explicit sizing
    *,
    provider_kind: ProviderKind = ProviderKind.OPENAI,
    api_key_env: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = TEST_TIMEOUT_SECONDS,
    connect_timeout_seconds: float = TEST_CONNECT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = TEST_IDLE_TIMEOUT_SECONDS,
    max_connections: int = TEST_MAX_CONNECTIONS,
    max_keepalive_connections: int = TEST_MAX_KEEPALIVE_CONNECTIONS,
    max_request_bytes: int = TEST_MAX_REQUEST_BYTES,
    max_response_bytes: int = TEST_MAX_RESPONSE_BYTES,
    **overrides: Any,
) -> ProviderTransportPolicy:
    resolved_api_key_env = (
        str(_DEFAULT_API_KEY_ENVS[provider_kind])
        if api_key_env is None
        else api_key_env
    )
    resolved_base_url = (
        str(_DEFAULT_BASE_URLS[provider_kind])
        if base_url is None
        else base_url
    )
    return ProviderTransportPolicy(
        provider_kind=provider_kind,
        api_key_env=resolved_api_key_env,
        base_url=resolved_base_url,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
        **overrides,
    )
