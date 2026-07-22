"""Provider Transport Policy: credentials, timeout, native retry, wire.

Operational transport-only policy. Under Whetstone durable execution the
native retry count is zero (its default), and the policy holds no semantic
failure classification, logical-attempt bound, backoff, or DBOS policy —
those belong to Whetstone's Provider Execution Policy. Transport policy is
excluded from every Definition/Config/Request identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_providers.failures import sanitize_headers

if TYPE_CHECKING:
    from dr_providers.route import ApiKeyEnv

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 90.0


class ProviderTransportPolicy(BaseModel):
    """Wire execution concerns only. Native retry defaults to zero."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_env: StrictStr
    base_url: StrictStr | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    """Absolute wall-clock CAP for any single wire call (the hard backstop).

    ``timeout_seconds`` is the maximum total wall-clock a single invocation
    may take before the transport interrupts it and returns a typed failure.
    It is NOT a per-socket-read timeout, and -- crucially -- it is NOT the
    primary stall detector: a LEGITIMATE long streaming response (e.g. a
    reasoning model emitting 18k tokens over many minutes) is making steady
    progress and must NOT be killed merely for running long. The primary
    stall detector is ``idle_timeout_seconds`` (below); ``timeout_seconds``
    is the absolute cap that also catches a pathological dribble (a wedged
    edge sending one byte per idle window forever, which defeats a naive
    no-NEW-bytes idle timer).

    Enforcement layers:

      * httpx timeout discipline -- ``connect = min(30, timeout_seconds)``
        (a stalled TCP/TLS handshake fails fast); the streaming READ phase is
        bounded by ``idle_timeout_seconds`` (per-read, i.e. per inter-byte
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
    than ``idle_timeout_seconds`` fails promptly. It is applied as httpx's
    per-read (per inter-byte) timeout on the streaming read phase. A
    pathological dribble that sends a single byte just inside every idle
    window defeats this timer by design -- that case is caught by the
    absolute ``timeout_seconds`` cap.
    """
    native_retry_count: StrictInt = 0

    def identity_payload(self) -> dict[str, Any]:
        """Policy identity for Invocation Evidence binding.

        Never includes credential material: only the *name* of the env
        var, the base URL, timeout, and native retry count. This binds
        the policy to evidence without persisting any secret.
        """
        return {
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "native_retry_count": self.native_retry_count,
        }


def policy_for(
    *,
    api_key_env: ApiKeyEnv | str,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    native_retry_count: int = 0,
) -> ProviderTransportPolicy:
    return ProviderTransportPolicy(
        api_key_env=str(api_key_env),
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        native_retry_count=native_retry_count,
    )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "ProviderTransportPolicy",
    "policy_for",
    "sanitize_headers",
]
