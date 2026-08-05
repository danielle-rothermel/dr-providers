"""Provider Call Request identity model."""

from __future__ import annotations

from functools import cached_property
from typing import Any

from dr_serialize import (
    IdentityDocument,
    build_identity_document,
    identity_document_hash,
)
from pydantic import BaseModel, ConfigDict

from dr_providers.modeling.call import (  # noqa: TC001 -- pydantic field
    ProviderCallConfig,
)
from dr_providers.modeling.transcript import Transcript  # noqa: TC001

PROVIDER_CALL_REQUEST_SCHEMA = "dr_providers.provider_call_request"
PROVIDER_CALL_REQUEST_SCHEMA_VERSION = 1


class ProviderCallRequest(BaseModel):
    """Immutable identity-bearing request: one Config + one Transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: ProviderCallConfig
    transcript: Transcript

    def identity_payload(self) -> dict[str, Any]:
        """Config reference (by Identity Hash) plus Transcript. No copied
        controls, no transport policy."""
        return {
            "config_identity_hash": self.config.identity_hash,
            "transcript": self.transcript.identity_payload(),
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_CALL_REQUEST_SCHEMA,
            schema_version=PROVIDER_CALL_REQUEST_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        """Full 64-char lowercase SHA-256 Request Identity Hash."""
        return identity_document_hash(self.identity_document())
