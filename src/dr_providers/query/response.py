from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CallError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error_type: str
    message: str
    retryable: bool = False
    raw_json: dict[str, Any] | None = None


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_json: dict[str, Any] = Field(default_factory=dict)
    provider: str
    model: str
    text: str
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: Any | None = None
    cost: Any | None = None
    latency_ms: int = 0
    warnings: list[Any] = Field(default_factory=list)
