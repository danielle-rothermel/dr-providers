from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from dr_providers.names import EffortLevel, ProviderName  # noqa: TC001


class ReasoningSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool | None = None
    effort: EffortLevel | None = None


class SamplingControls(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = None
    top_p: float | None = None

    def is_empty(self) -> bool:
        return self.temperature is None and self.top_p is None


class LlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName
    model: str
    max_tokens: int | None = None
    effort: EffortLevel | None = None
    reasoning: ReasoningSpec | None = None
    sampling: SamplingControls | None = None
