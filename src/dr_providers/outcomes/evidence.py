from __future__ import annotations

import os.path
from collections.abc import Mapping  # noqa: TC003 -- pydantic field type
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Literal

from dr_serialize import (
    IdentityDocument,
    build_identity_document,
    identity_document_hash,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_serializer,
    model_validator,
)

from dr_providers.core.frozen import _deep_freeze, _thaw
from dr_providers.modeling.controls import ProviderCallKind
from dr_providers.modeling.local import (  # noqa: TC001 -- pydantic fields
    Float32MatmulPrecision,
    LocalDevice,
    LocalDtype,
    Quantization,
)
from dr_providers.modeling.request import ProviderScoreRequest
from dr_providers.modeling.route import HTTP_PROVIDER_KINDS, ProviderKind
from dr_providers.outcomes.models import (
    ProviderScoreResponse,
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


HOME_DIRECTORY_PLACEHOLDER = "~"


def scrub_traceback(traceback: str | None) -> str | None:
    """Replace the user home-directory prefix in captured traceback text.

    A traceback captured on the wire path names absolute source paths
    under the running user's home directory. Those paths are machine
    state, not provider behavior, so they are collapsed to ``~`` before
    the text enters the model.

    This applies on the build path only, mirroring header redaction:
    deserializing third-party evidence does not re-scrub it.
    """
    if not traceback:
        return traceback
    home_directory = os.path.expanduser("~")  # noqa: PTH111 -- exact prefix
    if not home_directory:
        return traceback
    return traceback.replace(home_directory, HOME_DIRECTORY_PLACEHOLDER)


PROVIDER_INVOCATION_EVIDENCE_SCHEMA = (
    "dr_providers.provider_invocation_evidence"
)
PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION = 11
ContentIdentityHash = Annotated[
    StrictStr,
    Field(pattern=r"^[0-9a-f]{64}$"),
]
MAX_RETRY_AFTER_HEADER_BYTES = 128
MAX_RETRY_AFTER_DELTA_SECONDS = 31_536_000

EVIDENCE_IDENTITY_EXCLUDED_FAILURE_FIELDS = frozenset({"traceback", "message"})
"""Persisted transport-failure fields kept out of the identity preimage.

These names are a hash-preimage format, not merely field names: changing
the set changes every evidence identity hash and therefore requires a
schema-version bump.
"""


EVIDENCE_IDENTITY_EXCLUDED_FIELDS = frozenset({"wall_time_seconds"})
"""Persisted evidence fields kept out of the identity preimage.

These names are a hash-preimage format, not merely field names: changing
this set changes evidence identity and requires a schema-version bump.
"""


class LocalExecutionEvidence(BaseModel):
    """Selected realized local execution conditions, all identity-bearing.

    A different GPU model or torch version is a different measurement
    condition and forks evidence identity. This record is not a complete
    execution-environment fingerprint or a reproducibility guarantee.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: LocalDevice
    device_name: StrictStr | None
    dtype: LocalDtype
    float32_matmul_precision: Float32MatmulPrecision
    quantization: Quantization
    revision_commit: StrictStr | None
    torch_version: StrictStr
    transformers_version: StrictStr
    chat_template_applied: StrictBool | None
    add_bos_token: StrictBool | None


class ProviderHttpRequestEvidence(BaseModel):
    """``build()`` redacts known credential headers; direct construction and
    deserialization do not sanitize. Immutability does not prove redaction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: StrictStr = "POST"
    url: StrictStr
    headers: Mapping[str, str] = Field(default_factory=dict)
    body: Mapping[str, Any] = Field(default_factory=dict)
    body_bytes: StrictInt = Field(ge=0)

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
        body_bytes: int,
        method: str = "POST",
    ) -> ProviderHttpRequestEvidence:
        return cls(
            method=method,
            url=url,
            headers=sanitize_headers(headers),
            body=dict(body),
            body_bytes=body_bytes,
        )


class ProviderRetryAfterHint(BaseModel):
    """Bounded recorded form of a response ``Retry-After`` header.

    The HTTP producer normalizes values at capture time. dr-providers records
    the hint as evidence only; no retry policy reads it today.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["delta_seconds", "http_date"]
    value: StrictInt | StrictStr

    @model_validator(mode="after")
    def _value_matches_kind(self) -> ProviderRetryAfterHint:
        if self.kind == "delta_seconds":
            if (
                not isinstance(self.value, int)
                or not 0 <= self.value <= MAX_RETRY_AFTER_DELTA_SECONDS
            ):
                msg = "delta_seconds Retry-After value is outside its bound"
                raise ValueError(msg)
        elif (
            not isinstance(self.value, str)
            or len(self.value.encode("utf-8")) > MAX_RETRY_AFTER_HEADER_BYTES
        ):
            msg = "http_date Retry-After value is outside its bound"
            raise ValueError(msg)
        return self


class ProviderInvocationEvidence(BaseModel):
    """Freeze nested identity-bearing JSON and identity components.

    Schema metadata belongs to ``identity_document()``. All persisted fields
    bear identity except wall time, failure message, and traceback. ``build()``
    checks score correspondence; direct construction and restoration
    are trusted paths and do not repeat those request-dependent checks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_hash: ContentIdentityHash
    kind: ProviderCallKind
    policy_identity: Mapping[str, Any] | None = None
    max_request_bytes: StrictInt | None = Field(default=None, gt=0)
    max_response_bytes: StrictInt | None = Field(default=None, gt=0)
    http_request: ProviderHttpRequestEvidence | None = None
    response_bytes: StrictInt | None = Field(default=None, ge=0)
    retry_after: ProviderRetryAfterHint | None = None
    local_execution: LocalExecutionEvidence | None = None
    response: ProviderTransportResponse | None = None
    score_response: ProviderScoreResponse | None = None
    failure: ProviderTransportFailure | None = None
    wall_time_seconds: float | None = Field(
        default=None,
        strict=True,
        allow_inf_nan=False,
        ge=0,
    )

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> ProviderInvocationEvidence:
        if self.local_execution is not None and (
            self.policy_identity is not None or self.http_request is not None
        ):
            raise ValueError(
                "evidence carries at most one execution record kind"
            )
        if (
            sum(
                value is not None
                for value in (self.response, self.score_response, self.failure)
            )
            != 1
        ):
            msg = (
                "ProviderInvocationEvidence requires exactly one of "
                "response/score_response/failure to be set"
            )
            raise ValueError(msg)
        if (
            self.kind is ProviderCallKind.GENERATE
            and self.score_response is not None
        ) or (
            self.kind is ProviderCallKind.SCORE and self.response is not None
        ):
            raise ValueError("response slot must match invocation kind")
        if self.policy_identity is not None:
            object.__setattr__(
                self,
                "policy_identity",
                _deep_freeze(dict(self.policy_identity)),
            )
            try:
                provider_kind = ProviderKind(
                    self.policy_identity["provider_kind"]
                )
            except (KeyError, TypeError, ValueError):
                msg = "policy_identity requires a supported provider_kind"
                raise ValueError(msg) from None
            if provider_kind not in HTTP_PROVIDER_KINDS:
                raise ValueError(
                    "policy_identity requires a supported HTTP provider_kind"
                )
            for field_name in ("max_request_bytes", "max_response_bytes"):
                if field_name in self.policy_identity and self.policy_identity[
                    field_name
                ] != getattr(self, field_name):
                    msg = f"{field_name} must match policy_identity evidence"
                    raise ValueError(msg)
        if self.response is not None:
            object.__setattr__(
                self,
                "response",
                ProviderTransportResponse.model_validate(
                    self.response.model_dump(mode="python")
                ),
            )
        if self.score_response is not None:
            object.__setattr__(
                self,
                "score_response",
                ProviderScoreResponse.model_validate(
                    self.score_response.model_dump(mode="python")
                ),
            )
        if self.failure is not None:
            object.__setattr__(
                self,
                "failure",
                ProviderTransportFailure.model_validate(
                    self.failure.model_dump(mode="python")
                ),
            )
        return self

    @field_serializer("policy_identity")
    def _serialize_policy_identity(
        self, value: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        return None if value is None else _thaw(value)

    @property
    def outcome(self) -> ProviderTransportOutcome:
        if self.response is not None:
            return self.response
        if self.score_response is not None:
            return self.score_response
        assert self.failure is not None
        return self.failure

    @classmethod
    def build(  # noqa: PLR0913 -- normalized evidence components
        cls,
        *,
        request: ProviderCallRequest,
        policy: ProviderTransportPolicy | None,
        http_request: ProviderHttpRequestEvidence | None,
        outcome: ProviderTransportOutcome,
        response_bytes: int | None = None,
        retry_after: ProviderRetryAfterHint | None = None,
        local_execution: LocalExecutionEvidence | None = None,
        wall_time_seconds: float | None = None,
    ) -> ProviderInvocationEvidence:
        response = (
            outcome if isinstance(outcome, ProviderTransportResponse) else None
        )
        score_response = (
            outcome if isinstance(outcome, ProviderScoreResponse) else None
        )
        if score_response is not None:
            if not isinstance(request, ProviderScoreRequest):
                raise ValueError("score response requires a score request")
            if len(score_response.scores) != len(request.continuations):
                raise ValueError("score count must equal continuation count")
            for score, continuation in zip(
                score_response.scores, request.continuations, strict=True
            ):
                if score.char_count != len(continuation):
                    raise ValueError(
                        "score char_count must equal continuation length"
                    )
                if request.token_logprobs and score.token_logprobs is None:
                    raise ValueError(
                        "requested token_logprobs missing from a score"
                    )
        failure = (
            outcome if isinstance(outcome, ProviderTransportFailure) else None
        )
        if failure is not None and failure.traceback is not None:
            failure = failure.model_copy(
                update={"traceback": scrub_traceback(failure.traceback)}
            )
        return cls(
            request_hash=request.identity_hash,
            kind=request.kind,
            policy_identity=(
                None if policy is None else policy.identity_payload()
            ),
            max_request_bytes=(
                None if policy is None else policy.max_request_bytes
            ),
            max_response_bytes=(
                None if policy is None else policy.max_response_bytes
            ),
            http_request=http_request,
            response_bytes=response_bytes,
            retry_after=retry_after,
            local_execution=local_execution,
            wall_time_seconds=wall_time_seconds,
            response=response,
            score_response=score_response,
            failure=failure,
        )

    def identity_payload(self) -> dict[str, Any]:
        """Project the persisted record onto its identity-bearing fields.

        Wall time and two failure fields stay persisted but are excluded
        from the hash preimage:

        ``failure.traceback`` names absolute source paths of the machine
        that captured it, so including it would fork the identity of one
        provider behavior across machines and interpreter versions.

        ``failure.message`` is human prose whose every fact already
        appears in a typed field beside it — ``code``, ``recoverability``,
        ``status_code``, ``containment``, and ``metadata``. Excluding it
        is a decision, not an oversight: a summary message may be
        reworded without rehashing already-recorded evidence. Every
        failure this package builds sets a typed ``code``; a directly
        constructed failure that relies on ``message`` alone to
        distinguish itself from another shares that other's identity.

        ``wall_time_seconds`` measures duration, not invocation meaning.

        Every other persisted field is identity-bearing, including
        ``response.warnings[]`` messages, which this package generates
        deterministically and which discriminate rather than fork identity.
        """
        payload = {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if key not in EVIDENCE_IDENTITY_EXCLUDED_FIELDS
        }
        failure = payload.get("failure")
        if isinstance(failure, dict):
            payload["failure"] = {
                key: value
                for key, value in failure.items()
                if key not in EVIDENCE_IDENTITY_EXCLUDED_FAILURE_FIELDS
            }
        return payload

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_INVOCATION_EVIDENCE_SCHEMA,
            schema_version=PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())
