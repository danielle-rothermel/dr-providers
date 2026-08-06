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

from dr_providers.core.frozen import _deep_freeze, _thaw
from dr_providers.outcomes.models import (
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
)

if TYPE_CHECKING:
    from dr_providers.modeling.request import ProviderCallRequest
    from dr_providers.transport.policy import ProviderTransportPolicy

SANITIZE_KEYS = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "model_list",
        "authorization",
        "x-api-key",
        "x-goog-api-key",
    }
)


def sanitize_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Remove credential-keyed fields before evidence persistence."""
    if not kwargs:
        return {}
    return {
        key: ("<redacted>" if key.lower() in SANITIZE_KEYS else value)
        for key, value in kwargs.items()
    }


def sanitize_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Redact credential-bearing values before evidence persistence."""
    if not headers:
        return {}
    return {
        key: ("<redacted>" if key.lower() in SANITIZE_KEYS else value)
        for key, value in headers.items()
    }


PROVIDER_INVOCATION_EVIDENCE_SCHEMA = (
    "dr_providers.provider_invocation_evidence"
)
PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION = 2


class ProviderHttpRequestEvidence(BaseModel):
    """``build()`` redacts known credential headers; direct construction and
    deserialization do not sanitize. Immutability does not prove redaction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: StrictStr = "POST"
    url: StrictStr
    headers: Mapping[str, str] = Field(default_factory=dict)
    body: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _freeze_maps(self) -> ProviderHttpRequestEvidence:
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
    ) -> ProviderHttpRequestEvidence:
        return cls(
            method=method,
            url=url,
            headers=sanitize_headers(headers),
            body=dict(body),
        )


class ProviderInvocationEvidence(BaseModel):
    """Freeze identities only; outcome dictionaries remain mutable.

    Schema metadata belongs to ``identity_document()``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_identity: Mapping[str, Any]
    policy_identity: Mapping[str, Any]
    http_request: ProviderHttpRequestEvidence
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
        http_request: ProviderHttpRequestEvidence,
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
            http_request=http_request,
            response=response,
            failure=failure,
        )

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_INVOCATION_EVIDENCE_SCHEMA,
            schema_version=PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )
