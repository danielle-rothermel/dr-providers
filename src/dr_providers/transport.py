"""Raw-httpx transport implementing the Provider protocol.

The library classifies failures precisely and never sleeps unless the
caller opts in: ``TransportPolicy.max_retries`` defaults to 0 because
durable callers (DBOS steps) own retry; script/CLI use can enable a
bounded, jittered backoff. Timeouts are always explicit.
"""

from __future__ import annotations

import os
import random
import time
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, ConfigDict, StrictInt

from dr_providers.conformance import with_conformance_warnings

if TYPE_CHECKING:
    from collections.abc import Callable

    from dr_providers.config import ProviderConfig
from dr_providers.failures import (
    FailureClass,
    ProviderFailureError,
    classify_status_code,
    failure_record,
    raise_failure,
)
from dr_providers.request import (
    LlmRequest,
    build_payload,
    endpoint_path,
)
from dr_providers.response import LlmResponse, parse_response

DEFAULT_TIMEOUT_SECONDS = 120.0
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
MISSING_API_KEY_CODE = "missing_api_key"
MISSING_BASE_URL_CODE = "missing_base_url"
HTTP_STATUS_CODE_PREFIX = "http_status_"
TRANSPORT_ERROR_CODE = "transport_error"
INVALID_JSON_CODE = "invalid_response_json"


class TransportPolicy(BaseModel):
    """Explicit, bounded resilience knobs. Retry is opt-in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: StrictInt = 0
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 30.0


class HttpProvider:
    """Single-shot provider calls over raw httpx.

    ``sleep`` and ``rng`` are injectable for deterministic tests.
    """

    def __init__(
        self,
        *,
        policy: TransportPolicy | None = None,
        client: httpx.Client | None = None,
        api_key: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._policy = policy or TransportPolicy()
        self._client = client
        self._owns_client = client is None
        self._api_key = api_key
        self._sleep = sleep
        self._rng = rng

    def close(self) -> None:
        """Close the httpx client only if this provider created it.

        Injected clients belong to the caller and are left open.
        """
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> HttpProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(self, request: LlmRequest) -> LlmResponse:
        payload = build_payload(request)
        attempts = self._policy.max_retries + 1
        last_error: ProviderFailureError | None = None
        for attempt in range(attempts):
            try:
                response = self._complete_once(request, payload)
            except ProviderFailureError as error:
                last_error = error
                should_retry = (
                    error.failure.retryable and attempt + 1 < attempts
                )
                if not should_retry:
                    raise
                self._sleep(self._backoff_delay(attempt))
                continue
            return with_conformance_warnings(request, response)
        raise last_error or AssertionError("unreachable")

    def _complete_once(
        self,
        request: LlmRequest,
        payload: dict[str, Any],
    ) -> LlmResponse:
        config = request.provider_config
        url = self._request_url(config)
        headers = self._headers(request)
        try:
            http_response = self._httpx_client().post(
                url,
                json=payload,
                headers=headers,
                timeout=self._policy.timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise raise_failure(
                failure_record(
                    failure_class=FailureClass.TRANSIENT,
                    code=TRANSPORT_ERROR_CODE,
                    message=f"{type(error).__name__}: {error}",
                    metadata={"url": url},
                ),
                underlying=error,
            ) from error
        if http_response.status_code >= 400:  # noqa: PLR2004
            raise raise_failure(
                failure_record(
                    failure_class=classify_status_code(
                        http_response.status_code
                    ),
                    code=(
                        f"{HTTP_STATUS_CODE_PREFIX}{http_response.status_code}"
                    ),
                    message=http_response.text[:512],
                    metadata={
                        "status_code": http_response.status_code,
                        "url": url,
                    },
                )
            )
        try:
            body = http_response.json()
        except ValueError as error:
            raise raise_failure(
                failure_record(
                    failure_class=FailureClass.PERMANENT,
                    code=INVALID_JSON_CODE,
                    message="provider response body is not valid JSON",
                    metadata={
                        "url": url,
                        "body_preview": http_response.text[:512],
                    },
                ),
                underlying=error,
            ) from error
        return parse_response(body, config=config)

    def _request_url(self, config: ProviderConfig) -> str:
        if not config.base_url:
            raise raise_failure(
                failure_record(
                    failure_class=FailureClass.PERMANENT,
                    code=MISSING_BASE_URL_CODE,
                    message=(
                        f"provider config {config.throttle_identity!r} "
                        "has no base_url"
                    ),
                )
            )
        return config.base_url.rstrip("/") + endpoint_path(config)

    def _headers(self, request: LlmRequest) -> dict[str, str]:
        config = request.provider_config
        api_key = self._api_key or os.environ.get(config.api_key_env)
        if not api_key:
            raise raise_failure(
                failure_record(
                    failure_class=FailureClass.PERMANENT,
                    code=MISSING_API_KEY_CODE,
                    message=(
                        f"environment variable {config.api_key_env!r} "
                        "is not set"
                    ),
                )
            )
        headers = {"Authorization": f"Bearer {api_key}"}
        if request.idempotency_key is not None:
            headers[IDEMPOTENCY_KEY_HEADER] = request.idempotency_key
        return headers

    def _httpx_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def _backoff_delay(self, attempt: int) -> float:
        base = min(
            self._policy.backoff_base_seconds * (2**attempt),
            self._policy.backoff_max_seconds,
        )
        return base * (0.5 + self._rng() / 2)
