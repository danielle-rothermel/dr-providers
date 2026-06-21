from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

if TYPE_CHECKING:
    from dr_providers.config import ReasoningSpec, SamplingControls
    from dr_providers.names import MessageRole, ProviderName


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: str


class LlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    effort: Any | None = None
    reasoning: ReasoningSpec | None = None
    sampling: SamplingControls | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_sampling_controls(self) -> bool:
        return self.sampling is not None and not self.sampling.is_empty()

    @property
    def sampling_temperature(self) -> float | None:
        return self.sampling.temperature if self.sampling is not None else None

    @property
    def sampling_top_p(self) -> float | None:
        return self.sampling.top_p if self.sampling is not None else None
