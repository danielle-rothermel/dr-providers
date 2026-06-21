from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_providers.config import ReasoningSpec  # noqa: TC001


class ReasoningWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    provider: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def _reasoning_extra_body(spec: ReasoningSpec) -> dict[str, Any]:
    if spec.enabled is not None:
        payload: dict[str, Any] = {"enabled": spec.enabled}
    elif spec.effort is not None:
        payload = {"effort": spec.effort}
    else:
        payload = {"reasoning": True}
    return {"reasoning": payload}


class RequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    extra_body: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls, reasoning: ReasoningSpec | None
    ) -> RequestControls:
        if reasoning is None:
            return cls()
        return cls(extra_body=_reasoning_extra_body(reasoning))
