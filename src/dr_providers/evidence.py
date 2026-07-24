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

from collections.abc import Mapping  # noqa: TC003 -- pydantic field type
from typing import TYPE_CHECKING, Any

from dr_serialize import (
    IdentityDocument,
    build_identity_document,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_serializer,
    model_validator,
)

from dr_providers._frozen import _deep_freeze, _thaw
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
    ``headers`` and ``body`` are deeply immutable so a persisted evidence
    record can never be mutated after construction — e.g. an ``Authorization``
    header cannot be re-added after redaction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: StrictStr = "POST"
    url: StrictStr
    headers: Mapping[str, str] = Field(default_factory=dict)
    body: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _freeze_maps(self) -> RawHttpRequest:
        object.__setattr__(self, "headers", _deep_freeze(dict(self.headers)))
        object.__setattr__(self, "body", _deep_freeze(dict(self.body)))
        return self

    @field_serializer("headers")
    def _serialize_headers(self, value: Mapping[str, str]) -> dict[str, str]:
        return _thaw(value)

    @field_serializer("body")
    def _serialize_body(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw(value)

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
    """Stable serializable artifact for one completed transport call.

    Exactly one of ``response``/``failure`` is set (enforced), and the
    identity payloads are deeply immutable so the persisted record can never
    be tampered with after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION
    request_identity: Mapping[str, Any]
    policy_identity: Mapping[str, Any]
    raw_request: RawHttpRequest
    response: ProviderTransportResponse | None = None
    failure: ProviderTransportFailure | None = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> ProviderInvocationEvidence:
        if (self.response is None) == (self.failure is None):
            msg = (
                "ProviderInvocationEvidence requires exactly one of "
                "response/failure to be set"
            )
            raise ValueError(msg)
        object.__setattr__(
            self, "request_identity", _deep_freeze(dict(self.request_identity))
        )
        object.__setattr__(
            self, "policy_identity", _deep_freeze(dict(self.policy_identity))
        )
        return self

    @field_serializer("request_identity")
    def _serialize_request_identity(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        return _thaw(value)

    @field_serializer("policy_identity")
    def _serialize_policy_identity(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        return _thaw(value)

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

    def stable_payload(self) -> dict[str, Any]:
        """The bare JSON payload (no schema envelope)."""
        return self.model_dump(mode="json")

    def identity_document(self) -> IdentityDocument:
        """The persisted ``{schema, schema_version, payload}`` document.

        The exported schema/version constants govern this envelope,
        consistent with how ``config.py`` builds identity documents via
        ``dr_serialize.build_identity_document``.
        """
        return build_identity_document(
            schema=PROVIDER_INVOCATION_EVIDENCE_SCHEMA,
            schema_version=PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION,
            payload=self.stable_payload(),
        )

    def to_stable_dict(self) -> dict[str, Any]:
        """Stable serialized form for persistence/checkpointing.

        The schema-wrapped ``{schema, schema_version, payload}`` envelope so
        the persisted artifact is self-describing and versioned.
        """
        return self.identity_document().to_json_dict()
