from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dr_providers.query.errors import ProviderTransportError
from dr_providers.query.reasoning import RequestControls
from dr_providers.query.response import llm_response_from_http

if TYPE_CHECKING:
    from dr_providers.query.request import LlmRequest
    from dr_providers.query.response import LlmResponse
    from dr_providers.query.transport_config import (
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

    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.TransportError)
        ),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _post_with_retry(self, prepared: LlmRequest) -> httpx.Response:
        return self._client.post(
            prepared.endpoint(),
            headers=prepared.headers(),
            json=prepared.json_payload(),
        )

    def generate(self, request: LlmRequest) -> LlmResponse:
        controls = RequestControls.from_reasoning(request.reasoning)
        prepared = request.prepare(self._config, controls=controls)
        started = time.perf_counter()
        try:
            response = self._post_with_retry(prepared)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderTransportError(
                f"{self.name} HTTP request failed: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        return llm_response_from_http(
            response,
            request,
            latency_ms=latency_ms,
            warnings=prepared.warnings,
        )
