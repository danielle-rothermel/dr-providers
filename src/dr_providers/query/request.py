from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from dr_providers.config import SamplingControls  # noqa: TC001
from dr_providers.names import MessageRole, ProviderName  # noqa: TC001
from dr_providers.query.provider_config import (
    ProviderConfig,
    ReasoningWarning,
    resolve_api_key,
)
from dr_providers.query.reasoning import ReasoningSpec  # noqa: TC001


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


class OpenAICompatRequest(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    provider: str = Field(exclude=True)
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    base_url: str = Field(exclude=True)
    chat_path: str | None = Field(exclude=True)
    api_key_env: str = Field(exclude=True)
    api_key: str = Field(exclude=True, repr=False)
    idempotency_key: str = Field(exclude=True)
    warnings: list[ReasoningWarning] = Field(
        default_factory=list, exclude=True
    )

    @classmethod
    def from_llm_request(
        cls,
        request: LlmRequest,
        config: ProviderConfig,
        *,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        warnings: list[ReasoningWarning] | None = None,
    ) -> OpenAICompatRequest:
        return cls(
            provider=str(request.provider),
            model=request.model,
            messages=cls._to_openai_messages(request.messages),
            temperature=request.sampling_temperature,
            top_p=request.sampling_top_p,
            max_tokens=request.max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body or {},
            base_url=config.base_url,
            chat_path=config.chat_path,
            api_key_env=config.api_key_env,
            api_key=resolve_api_key(config, label=request.provider),
            idempotency_key=cls._resolve_idempotency_key(request=request),
            warnings=warnings or [],
        )

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    @staticmethod
    def _resolve_idempotency_key(*, request: LlmRequest) -> str:
        raw_idempotency_key = request.metadata.get("idempotency_key")
        if isinstance(raw_idempotency_key, str) and raw_idempotency_key:
            return raw_idempotency_key
        return uuid4().hex

    def endpoint(self) -> str:
        if self.chat_path is None:
            return self.base_url
        return self.base_url.rstrip("/") + self.chat_path

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": self.idempotency_key,
        }

    def json_payload(self) -> dict[str, Any]:
        payload = self.model_dump(
            mode="json", exclude_none=True, exclude={"extra_body"}
        )
        if "max_tokens" in payload:
            # TODO: will this break things?
            payload["max_completion_tokens"] = payload["max_tokens"]

        overlapping_keys = sorted(set(self.extra_body) & set(payload))
        if overlapping_keys:
            conflicts = ", ".join(overlapping_keys)
            raise ValueError(
                "extra_body conflicts with "
                f"validated payload keys: {conflicts}"
            )
        payload.update(self.extra_body)
        return payload
