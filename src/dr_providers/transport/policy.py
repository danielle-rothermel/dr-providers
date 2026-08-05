"""Provider Transport Policy: credentials, timeout, native retry, wire.

Operational transport-only policy. Under Whetstone durable execution the
native retry count is zero (its default), and the policy holds no semantic
failure classification, logical-attempt bound, backoff, or DBOS policy —
those belong to Whetstone's Provider Execution Policy. Transport policy is
excluded from every Definition/Config/Request identity.

This module also owns the transport-facing credential/base-URL vocabulary
(``ApiKeyEnv``, ``ProviderBaseUrl`` and the ``DEFAULT_*`` provider maps),
which are transport-policy concerns rather than identity concerns.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from dr_providers.modeling.route import ProviderKind

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 90.0


class ApiKeyEnv(StrEnum):
    """Environment variables the transport reads provider API keys from."""

    OPENROUTER = "OPENROUTER_API_KEY"
    OPENAI = "OPENAI_API_KEY"
    GEMINI = "GEMINI_API_KEY"
    ANTHROPIC = "ANTHROPIC_API_KEY"


class ProviderBaseUrl(StrEnum):
    """Default base URLs used by the preset provider transport policies."""

    OPENROUTER = "https://openrouter.ai/api/v1"
    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com/v1"
    # The OpenAI-compat surface, not "Gemini's URL": a future native
    # Gemini endpoint would be a sibling member, not this one.
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
    """Wire execution concerns only. Native retry defaults to zero."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_env: StrictStr
    base_url: StrictStr | None = None
    """Transport base URL retained in invocation evidence.

    The supplied value is retained verbatim in policy identity and as the base
    of the raw request URL. Callers must not embed credentials in it.
    """
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    """Absolute wall-clock CAP for any single wire call (the hard backstop).

    ``timeout_seconds`` is the maximum total wall-clock a single invocation
    may take before the transport interrupts it and returns a typed failure.
    It is NOT a per-socket-read timeout, and -- crucially -- it is NOT the
    primary stall detector: a LEGITIMATE long streaming response (e.g. a
    reasoning model emitting 18k tokens over many minutes) is making steady
    progress and must NOT be killed merely for running long. The primary
    stall detector is the effective idle timeout (below); ``timeout_seconds``
    is the absolute cap that also catches a pathological dribble (a wedged
    edge sending one byte per idle window forever, which defeats a naive
    no-NEW-bytes idle timer).

    Enforcement layers:

      * httpx timeout discipline -- ``connect = min(30, effective idle)``
        (a stalled TCP/TLS handshake fails fast); the streaming READ phase is
        bounded by the effective idle timeout (per-read, i.e. per inter-byte
        gap), so a genuine idle stall fails as ``stalled_response`` promptly
        WITHOUT capping a progressing stream.
      * an overall per-invocation deadline (hard watchdog) of
        ``timeout_seconds`` plus a small fixed margin -- the absolute cap. It
        guarantees NO single call (including a forever-dribbling response that
        resets the idle timer one byte at a time) can exceed the budget; on
        breach the transport interrupts the wedged socket and returns a typed
        Provider Transport Failure
        (``code='timeout'``/``'stalled_response'``), never hangs.
    """
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS
    """Progress/idle timeout: fail ``stalled_response`` if no bytes arrive
    for this many seconds (default 90s).

    This is the PRIMARY stall detector and the semantic replacement for the
    old flat deadline: a response that keeps producing bytes (a legitimate
    long stream) never trips it, while a response that goes silent for longer
    than the idle timeout fails promptly. It is applied as httpx's per-read
    (per inter-byte) timeout on the streaming read phase. A pathological
    dribble that sends a single byte just inside every idle window defeats
    this timer by design -- that case is caught by the absolute
    ``timeout_seconds`` cap.

    The effective idle timeout is clamped to at most ``timeout_seconds``: an
    idle window wider than the absolute cap could never fire before the cap
    interrupts the call, so it is silently narrowed to ``timeout_seconds`` at
    construction.
    """
    native_retry_count: StrictInt = 0

    @model_validator(mode="after")
    def _clamp_idle_to_timeout(self) -> ProviderTransportPolicy:
        # The idle timeout is the primary stall detector; if it is set wider
        # than the absolute cap it could never fire, so clamp it down rather
        # than reject an otherwise-coherent policy.
        if self.idle_timeout_seconds > self.timeout_seconds:
            object.__setattr__(
                self, "idle_timeout_seconds", self.timeout_seconds
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        """Policy identity for Invocation Evidence binding.

        Includes only the *name* of the credential env var, the base URL,
        timeout, idle timeout, and native retry count; it never resolves or
        reads the credential value. The base URL is retained verbatim, so
        callers must not embed credentials in it.
        """
        return {
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "native_retry_count": self.native_retry_count,
        }


def policy_for(  # noqa: PLR0913 -- explicit keyword-only overrides
    kind: ProviderKind,
    *,
    api_key_env: ApiKeyEnv | str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    native_retry_count: int = 0,
) -> ProviderTransportPolicy:
    """Build a transport policy for a provider kind from the DEFAULT maps.

    ``api_key_env`` and ``base_url`` default to the provider's standard
    env-var name and base URL, and either may be overridden (e.g. a proxy
    base URL or a non-standard key env var).
    """
    resolved_key_env = (
        DEFAULT_API_KEY_ENVS[kind] if api_key_env is None else api_key_env
    )
    resolved_base_url = (
        str(DEFAULT_BASE_URLS[kind]) if base_url is None else base_url
    )
    return ProviderTransportPolicy(
        api_key_env=str(resolved_key_env),
        base_url=resolved_base_url,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        native_retry_count=native_retry_count,
    )
