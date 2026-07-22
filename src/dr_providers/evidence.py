"""Provider Invocation Evidence: the stable serializable transport record.

One completed transport invocation, binding the exact Provider Call
Request and Provider Transport Policy identities to the typed Provider
Transport Outcome and the complete least-processed raw request plus
success or failure evidence.

Two invariants are load-bearing and tested:
  * No silent truncation — the complete raw request and success/failure
    bodies are retained verbatim (no preview limit).
  * No credential material — authorization headers and credentials are
    never persisted; request headers are redacted before binding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dr_providers.failures import sanitize_headers
from dr_providers.outcome import (
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
)

if TYPE_CHECKING:
    from dr_providers.policy import ProviderTransportPolicy
    from dr_providers.request import ProviderCallRequest

PROVIDER_INVOCATION_EVIDENCE_SCHEMA = (
    "dr_providers.provider_invocation_evidence"
)
PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION = 1


class RawHttpRequest(BaseModel):
    """The complete least-processed wire request, with headers redacted.

    ``headers`` never contains authorization or credential material.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: StrictStr = "POST"
    url: StrictStr
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        method: str = "POST",
    ) -> RawHttpRequest:
        return cls(
            method=method,
            url=url,
            headers=sanitize_headers(headers),
            body=dict(body),
        )


class ProviderInvocationEvidence(BaseModel):
    """Stable serializable artifact for one completed transport call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION
    request_identity: dict[str, Any]
    policy_identity: dict[str, Any]
    raw_request: RawHttpRequest
    response: ProviderTransportResponse | None = None
    failure: ProviderTransportFailure | None = None

    @property
    def outcome(self) -> ProviderTransportOutcome:
        if self.response is not None:
            return self.response
        assert self.failure is not None
        return self.failure

    @classmethod
    def build(
        cls,
        *,
        request: ProviderCallRequest,
        policy: ProviderTransportPolicy,
        raw_request: RawHttpRequest,
        outcome: ProviderTransportOutcome,
    ) -> ProviderInvocationEvidence:
        response = (
            outcome if isinstance(outcome, ProviderTransportResponse) else None
        )
        failure = (
            outcome if isinstance(outcome, ProviderTransportFailure) else None
        )
        return cls(
            request_identity=request.identity_payload(),
            policy_identity=policy.identity_payload(),
            raw_request=raw_request,
            response=response,
            failure=failure,
        )

    def to_stable_dict(self) -> dict[str, Any]:
        """Stable serialized form for persistence/checkpointing."""
        return self.model_dump(mode="json")
