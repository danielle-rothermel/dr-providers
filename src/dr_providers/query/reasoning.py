from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from dr_providers.config import ReasoningSpec  # noqa: TC001
from dr_providers.names import EffortLevel  # noqa: TC001


class ReasoningPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool | None = None
    effort: EffortLevel | None = None
    reasoning: bool | None = None

    @classmethod
    def from_spec(cls, spec: ReasoningSpec) -> ReasoningPayload:
        if spec.enabled is not None:
            return cls(enabled=spec.enabled)
        if spec.effort is not None:
            return cls(effort=spec.effort)
        return cls(reasoning=True)


class ReasoningExtraBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning: ReasoningPayload
