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

from dr_providers.names import ProviderName
from dr_providers.query.provider_config import (
    OpenAICompatConfig,
    ProviderTransportError,
    ReasoningWarning,
)
from dr_providers.query.request import (
    LlmRequest,
    OpenAICompatRequest,
    RequestControls,
)
from dr_providers.query.response import (
    LlmResponse,
    OpenAICompatResponse,
)

if TYPE_CHECKING:
    from dr_providers.query.provider_config import (
        ProviderAvailabilityStatus,
        ProviderConfig,
    )


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


class ApiProviderResponse(Protocol):
    def event_payload(self, request: Any) -> dict[str, Any]: ...

    def to_llm_response(
        self,
        request: LlmRequest,
        *,
        latency_ms: int,
        warnings: list[ReasoningWarning],
    ) -> LlmResponse: ...


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

    @abstractmethod
    def _parse_response(self, response: httpx.Response) -> ApiProviderResponse:
        """Decode ``httpx.Response`` into provider-specific response shape."""

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
        provider_response = self._parse_response(response)
        return provider_response.to_llm_response(
            request,
            latency_ms=latency_ms,
            warnings=provider_request.warnings,
        )


class OpenRouterProvider(ApiProvider):
    _config: OpenAICompatConfig

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            config=OpenAICompatConfig(
                name=ProviderName.OPENROUTER,
                base_url=ProviderName.OPENROUTER.api_base_url,
                api_key_env=ProviderName.OPENROUTER.api_key_env(),
            ),
            client=client,
        )

    @property
    def config(self) -> OpenAICompatConfig:
        return self._config

    def _build_request(self, request: LlmRequest) -> OpenAICompatRequest:
        request_controls = RequestControls.from_reasoning(request.reasoning)
        return OpenAICompatRequest.from_llm_request(
            request,
            self._config,
            extra_body=request_controls.extra_body,
            warnings=request_controls.warnings,
        )

    def _parse_response(
        self, response: httpx.Response
    ) -> OpenAICompatResponse:
        return OpenAICompatResponse.from_http_response(response)
