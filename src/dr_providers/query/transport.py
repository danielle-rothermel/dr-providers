from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, Self

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dr_providers.config import ReasoningSpec, SamplingControls  # noqa: TC001
from dr_providers.names import MessageRole, ProviderName
from dr_providers.query.errors import ProviderTransportError
from dr_providers.query.reasoning import ReasoningWarning, RequestControls
from dr_providers.query.request import LlmRequest, Message, OpenAICompatRequest
from dr_providers.query.response import LlmResponse, llm_response_from_http
from dr_providers.query.transport_config import ProviderConfig

if TYPE_CHECKING:
    from dr_providers.query.transport_config import ProviderAvailabilityStatus


class ProviderTransport(ABC):
    _config: ProviderConfig

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def availability_status(self) -> ProviderAvailabilityStatus:
        return self._config.availability_status()

    def is_available(self) -> bool:
        return self.availability_status().available

    @abstractmethod
    def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ApiProviderRequest(Protocol):
    warnings: list[ReasoningWarning]

    def endpoint(self) -> str: ...

    def headers(self) -> dict[str, str]: ...

    def json_payload(self) -> dict[str, Any]: ...


class ApiProvider(ProviderTransport):
    _config: ProviderConfig

    def __init__(
        self,
        *,
        config: ProviderConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @abstractmethod
    def _build_request(self, request: LlmRequest) -> ApiProviderRequest:
        """Translate an ``LlmRequest`` into an ``ApiProviderRequest``."""

    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.TransportError)
        ),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _post_with_retry(
        self, provider_request: ApiProviderRequest
    ) -> httpx.Response:
        return self._client.post(
            provider_request.endpoint(),
            headers=provider_request.headers(),
            json=provider_request.json_payload(),
        )

    def generate(self, request: LlmRequest) -> LlmResponse:
        provider_request = self._build_request(request)
        started = time.perf_counter()
        try:
            response = self._post_with_retry(provider_request)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderTransportError(
                f"{self.name} HTTP request failed: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        return llm_response_from_http(
            response,
            request,
            latency_ms=latency_ms,
            warnings=provider_request.warnings,
        )


class OpenRouterProvider(ApiProvider):
    _config: ProviderConfig

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=ProviderConfig(
                name=ProviderName.OPENROUTER,
                base_url=ProviderName.OPENROUTER.api_base_url,
                api_key_env=ProviderName.OPENROUTER.api_key_env(),
            ),
            client=client,
        )

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def _build_request(self, request: LlmRequest) -> OpenAICompatRequest:
        request_controls = RequestControls.from_reasoning(request.reasoning)
        return OpenAICompatRequest.from_llm_request(
            request,
            self._config,
            extra_body=request_controls.extra_body,
            warnings=request_controls.warnings,
        )


def execute_query(  # noqa: PLR0913
    provider: ApiProvider,
    provider_name: ProviderName,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningSpec | None = None,
    sampling: SamplingControls | None = None,
    metadata: dict[str, Any] | None = None,
) -> LlmResponse:
    messages: list[Message] = []
    if system is not None:
        messages.append(Message(role=MessageRole.SYSTEM, content=system))
    messages.append(Message(role=MessageRole.USER, content=prompt))
    request = LlmRequest(
        provider=provider_name,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        reasoning=reasoning,
        sampling=sampling,
        metadata=metadata or {},
    )
    return provider.generate(request)
