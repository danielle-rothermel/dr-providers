from __future__ import annotations

from functools import cached_property
from typing import Annotated, Any, Literal

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
    StrictStr,
    model_validator,
)

from dr_providers.core.failures import (
    ControlValidationError,
    RecoverabilityClass,
    failure_record,
)
from dr_providers.modeling.call import (  # noqa: TC001 -- pydantic field
    ProviderCallConfig,
)
from dr_providers.modeling.controls import CONTROL_ATTR, ProviderCallKind
from dr_providers.modeling.transcript import Transcript  # noqa: TC001

PROVIDER_GENERATE_REQUEST_SCHEMA = "dr_providers.provider_generate_request"
PROVIDER_GENERATE_REQUEST_SCHEMA_VERSION = 1
PROVIDER_SCORE_REQUEST_SCHEMA = "dr_providers.provider_score_request"
PROVIDER_SCORE_REQUEST_SCHEMA_VERSION = 1


def _validate_supported_kind(
    config: ProviderCallConfig, kind: ProviderCallKind
) -> None:
    if kind not in config.definition.supported_kinds:
        raise ControlValidationError(
            failure_record(
                recoverability=RecoverabilityClass.PERMANENT,
                code="unsupported_call_kind",
                message=(f"definition does not support kind {kind.value!r}"),
                metadata={
                    "kind": kind.value,
                    "definition_id": config.definition.definition_id,
                },
            )
        )


class ProviderGenerateRequest(BaseModel):
    """Identity contains the generate kind, config hash, and transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ProviderCallKind.GENERATE] = ProviderCallKind.GENERATE
    config: ProviderCallConfig
    transcript: Transcript

    @model_validator(mode="after")
    def _validate_kind(self) -> ProviderGenerateRequest:
        _validate_supported_kind(self.config, self.kind)
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "config_hash": self.config.identity_hash,
            "transcript": self.transcript.identity_payload(),
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_GENERATE_REQUEST_SCHEMA,
            schema_version=PROVIDER_GENERATE_REQUEST_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())


class ProviderScoreRequest(BaseModel):
    """Identity binds kind, config hash, ordered strings, and detail flag.

    Empty context and duplicate continuations are allowed. A score-capable
    definition must have no required controls to admit a score request.
    ``token_logprobs=True`` requires per-token values in every returned score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[ProviderCallKind.SCORE] = ProviderCallKind.SCORE
    config: ProviderCallConfig
    context: StrictStr
    continuations: tuple[Annotated[StrictStr, Field(min_length=1)], ...] = (
        Field(min_length=1)
    )
    token_logprobs: StrictBool = False

    @model_validator(mode="after")
    def _validate_assignment(self) -> ProviderScoreRequest:
        _validate_supported_kind(self.config, self.kind)
        controls = sorted(
            control.value
            for control, attr in CONTROL_ATTR.items()
            if getattr(self.config.controls, attr) is not None
        )
        if controls:
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="score_request_rejects_controls",
                    message="score requests require all controls to be unset",
                    metadata={"controls": controls},
                )
            )
        if self.config.extensions.extra_body:
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="score_request_rejects_extensions",
                    message="score requests require empty body extensions",
                )
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "config_hash": self.config.identity_hash,
            "context": self.context,
            "continuations": list(self.continuations),
            "token_logprobs": self.token_logprobs,
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_SCORE_REQUEST_SCHEMA,
            schema_version=PROVIDER_SCORE_REQUEST_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())


ProviderCallRequest = Annotated[
    ProviderGenerateRequest | ProviderScoreRequest, Field(discriminator="kind")
]
