from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from dr_providers.config import ReasoningSpec, SamplingControls  # noqa: TC001
from dr_providers.names import MessageRole, ProviderName  # noqa: TC001
from dr_providers.query.reasoning import ReasoningWarning, RequestControls
from dr_providers.query.transport_config import ProviderConfig, resolve_api_key


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
    reasoning: ReasoningSpec | None = None
    sampling: SamplingControls | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    base_url: str | None = Field(default=None, exclude=True)
    chat_path: str | None = Field(default=None, exclude=True)
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    idempotency_key: str | None = Field(default=None, exclude=True)
    extra_body: dict[str, Any] = Field(default_factory=dict, exclude=True)
    warnings: list[ReasoningWarning] = Field(
        default_factory=list, exclude=True
    )

    @property
    def has_sampling_controls(self) -> bool:
        return self.sampling is not None and not self.sampling.is_empty()

    @property
    def sampling_temperature(self) -> float | None:
        return self.sampling.temperature if self.sampling is not None else None

    @property
    def sampling_top_p(self) -> float | None:
        return self.sampling.top_p if self.sampling is not None else None

    def prepare(
        self,
        config: ProviderConfig,
        *,
        controls: RequestControls | None = None,
    ) -> LlmRequest:
        request_controls = controls or RequestControls.from_reasoning(
            self.reasoning
        )
        return self.model_copy(
            update={
                "base_url": config.base_url,
                "chat_path": config.chat_path,
                "api_key": resolve_api_key(config, label=self.provider),
                "idempotency_key": self._resolve_idempotency_key(),
                "extra_body": request_controls.extra_body,
                "warnings": request_controls.warnings,
            }
        )

    def _resolve_idempotency_key(self) -> str:
        raw_idempotency_key = self.metadata.get("idempotency_key")
        if isinstance(raw_idempotency_key, str) and raw_idempotency_key:
            return raw_idempotency_key
        return uuid4().hex

    def _require_prepared(self) -> None:
        if self.api_key is None or self.base_url is None:
            raise ValueError("LlmRequest is not prepared for transport")

    def endpoint(self) -> str:
        self._require_prepared()
        base_url = self.base_url
        if base_url is None:
            raise ValueError("LlmRequest is not prepared for transport")
        if self.chat_path is None:
            return base_url
        return base_url.rstrip("/") + self.chat_path

    def headers(self) -> dict[str, str]:
        self._require_prepared()
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": self.idempotency_key or "",
        }

    def json_payload(self) -> dict[str, Any]:
        self._require_prepared()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self.messages
            ],
        }
        if self.sampling_temperature is not None:
            payload["temperature"] = self.sampling_temperature
        if self.sampling_top_p is not None:
            payload["top_p"] = self.sampling_top_p
        if self.max_tokens is not None:
            # TODO: will this break things?
            payload["max_tokens"] = self.max_tokens
            payload["max_completion_tokens"] = self.max_tokens

        overlapping_keys = sorted(set(self.extra_body) & set(payload))
        if overlapping_keys:
            conflicts = ", ".join(overlapping_keys)
            raise ValueError(
                "extra_body conflicts with "
                f"validated payload keys: {conflicts}"
            )
        payload.update(self.extra_body)
        return payload
