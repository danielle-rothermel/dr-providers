from __future__ import annotations

import time
from secrets import randbelow
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

import httpx

from dr_providers.query.errors import ProviderTransportError
from dr_providers.query.reasoning import _reasoning_extra_body
from dr_providers.query.response import llm_response_from_http
from dr_providers.query.transport_config import resolve_api_key

if TYPE_CHECKING:
    from dr_providers.query.request import LlmRequest
    from dr_providers.query.response import LlmResponse
    from dr_providers.query.transport_config import ProviderConfig


POST_ATTEMPTS = 3
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 8.0
RETRY_JITTER_SECONDS = 1.0
RETRY_JITTER_STEPS = 1000
AUTHORIZATION_HEADER = "Authorization"
CONTENT_TYPE_HEADER = "Content-Type"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
JSON_CONTENT_TYPE = "application/json"
METADATA_IDEMPOTENCY_KEY = "idempotency_key"


def _retry_delay_seconds(attempt_index: int) -> float:
    base_delay = INITIAL_RETRY_DELAY_SECONDS * (2**attempt_index)
    jitter = randbelow(RETRY_JITTER_STEPS) / RETRY_JITTER_STEPS
    delay_with_jitter = base_delay + (jitter * RETRY_JITTER_SECONDS)
    return min(delay_with_jitter, MAX_RETRY_DELAY_SECONDS)


def _resolve_idempotency_key(request: LlmRequest) -> str:
    raw_idempotency_key = request.metadata.get(METADATA_IDEMPOTENCY_KEY)
    if isinstance(raw_idempotency_key, str) and raw_idempotency_key:
        return raw_idempotency_key
    return uuid4().hex


def _endpoint(config: ProviderConfig) -> str:
    if config.chat_path is None:
        return config.base_url
    return config.base_url.rstrip("/") + config.chat_path


def _headers(*, api_key: str, idempotency_key: str) -> dict[str, str]:
    return {
        AUTHORIZATION_HEADER: f"Bearer {api_key}",
        CONTENT_TYPE_HEADER: JSON_CONTENT_TYPE,
        IDEMPOTENCY_KEY_HEADER: idempotency_key,
    }


def _json_payload(request: LlmRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
    }
    if (
        request.sampling is not None
        and request.sampling.temperature is not None
    ):
        payload["temperature"] = request.sampling.temperature
    if request.sampling is not None and request.sampling.top_p is not None:
        payload["top_p"] = request.sampling.top_p
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
        payload["max_completion_tokens"] = request.max_tokens
    payload.update(_reasoning_extra_body(request.reasoning))
    return payload


class ApiProvider:
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

    @property
    def name(self) -> str:
        return self._config.name

    def _post_with_retry(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        for attempt_index in range(POST_ATTEMPTS):
            try:
                return self._client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt_index == POST_ATTEMPTS - 1:
                    raise
                time.sleep(_retry_delay_seconds(attempt_index))
        raise RuntimeError("unreachable post retry state")

    def generate(self, request: LlmRequest) -> LlmResponse:
        api_key = resolve_api_key(self._config, label=self.name)
        endpoint = _endpoint(self._config)
        headers = _headers(
            api_key=api_key,
            idempotency_key=_resolve_idempotency_key(request),
        )
        payload = _json_payload(request)
        started = time.perf_counter()
        try:
            response = self._post_with_retry(
                endpoint=endpoint,
                headers=headers,
                payload=payload,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderTransportError(
                f"{self.name} HTTP request failed: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        return llm_response_from_http(
            response,
            request,
            latency_ms=latency_ms,
        )
