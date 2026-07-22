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


class ProviderTransportPolicy(BaseModel):
    """Wire execution concerns only. Native retry defaults to zero."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_env: StrictStr
    base_url: StrictStr | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    """Effective wall-clock bound for any single wire call.

    ``timeout_seconds`` is the total wall-clock budget the transport
    enforces against a single invocation, not merely a per-socket-read
    timeout. It is applied on two independent layers:

      * httpx timeout discipline — every phase is bounded so no phase can
        stall silently: ``connect = min(30, timeout_seconds)`` (a stalled
        TCP/TLS handshake must fail fast), and ``read = write = pool =
        timeout_seconds``.
      * an overall per-invocation deadline (a hard watchdog) of
        ``timeout_seconds`` plus a small fixed margin. This backstops the
        httpx read timeout, which is *per-read-operation* and therefore
        does not bound wall-clock: a stalled response that trickles bytes
        (e.g. a wedged Cloudflare edge) can reset the per-read timer
        indefinitely and never trigger httpx's read timeout. The deadline
        guarantees a single stall can never exceed the budget; on breach
        the transport returns a typed Provider Transport Failure
        (``code='timeout'``/``'stalled_response'``), never hangs.
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
            "native_retry_count": self.native_retry_count,
        }


def policy_for(
    *,
    api_key_env: ApiKeyEnv | str,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    native_retry_count: int = 0,
) -> ProviderTransportPolicy:
    return ProviderTransportPolicy(
        api_key_env=str(api_key_env),
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        native_retry_count=native_retry_count,
    )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "ProviderTransportPolicy",
    "policy_for",
    "sanitize_headers",
]
