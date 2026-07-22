"""Raw-httpx transport returning the no-throw Provider Transport Outcome.

``complete`` returns a closed union of Provider Transport Response or
Provider Transport Failure — expected outcomes never raise. Only
unexpected programming/infrastructure errors raise. ``invoke`` returns a
stable Provider Invocation Evidence artifact binding request + policy
identities to the outcome and the complete least-processed raw request.

Native retry count defaults to zero and is honored as literal retries of
the same wire call; there is no backoff, sleep, or semantic
classification here (Whetstone's Provider Execution Policy owns those).

Two protocol families are first-class:
  * OpenAI-compatible / OpenRouter ``chat_completions`` (and OpenAI
    ``responses``) over a Bearer token, with a custom base URL.
  * Anthropic ``anthropic_messages`` over ``x-api-key`` +
    ``anthropic-version`` headers, with a custom base URL.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx

from dr_providers.conformance import with_conformance_warnings
from dr_providers.evidence import ProviderInvocationEvidence, RawHttpRequest
from dr_providers.failures import (
    FailureClass,
    classify_status_code,
)
from dr_providers.outcome import (
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
)
from dr_providers.request import (
    ProviderCallRequest,
    build_payload,
    protocol_path,
)
from dr_providers.response import parse_response
from dr_providers.route import Protocol

if TYPE_CHECKING:
    from dr_providers.config import ProviderCallConfig
    from dr_providers.policy import ProviderTransportPolicy

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_VERSION_HEADER = "anthropic-version"
ANTHROPIC_API_KEY_HEADER = "x-api-key"
AUTHORIZATION_HEADER = "Authorization"
MISSING_API_KEY_CODE = "missing_api_key"
MISSING_BASE_URL_CODE = "missing_base_url"
HTTP_STATUS_CODE_PREFIX = "http_status_"
TRANSPORT_ERROR_CODE = "transport_error"
INVALID_JSON_CODE = "invalid_response_json"


class HttpProvider:
    """Single-shot provider calls over raw httpx.

    The Provider Transport Policy supplies credentials env var, base URL,
    timeout, and native retry count. An explicit ``api_key`` overrides
    the env lookup (for tests). An injected client is left open on close.
    """

    def __init__(
        self,
        *,
        policy: ProviderTransportPolicy,
        client: httpx.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        self._policy = policy
        self._client = client
        self._owns_client = client is None
        self._api_key = api_key

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> HttpProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(
        self, request: ProviderCallRequest
    ) -> ProviderTransportOutcome:
        """Return the typed no-throw outcome for one request."""
        payload = build_payload(request)
        outcome = self._complete_with_retries(request, payload)
        if isinstance(outcome, ProviderTransportResponse):
            return with_conformance_warnings(request, outcome)
        return outcome

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        """Complete the call and bind it into stable Invocation Evidence."""
        payload = build_payload(request)
        url = self._request_url(request.config)
        headers = self._headers(request.config)
        raw_request = RawHttpRequest.build(
            url=url or "<missing_base_url>",
            headers=headers or {},
            body=payload,
        )
        outcome = self._complete_with_retries(request, payload)
        if isinstance(outcome, ProviderTransportResponse):
            outcome = with_conformance_warnings(request, outcome)
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self._policy,
            raw_request=raw_request,
            outcome=outcome,
        )

    def _complete_with_retries(
        self,
        request: ProviderCallRequest,
        payload: dict[str, Any],
    ) -> ProviderTransportOutcome:
        attempts = self._policy.native_retry_count + 1
        outcome: ProviderTransportOutcome = self._complete_once(
            request, payload
        )
        for _ in range(attempts - 1):
            if isinstance(outcome, ProviderTransportResponse):
                return outcome
            if not outcome.retryable:
                return outcome
            outcome = self._complete_once(request, payload)
        return outcome

    def _complete_once(
        self,
        request: ProviderCallRequest,
        payload: dict[str, Any],
    ) -> ProviderTransportOutcome:
        config = request.config
        url = self._request_url(config)
        if url is None:
            return self._missing_base_url_failure(config, payload)
        headers = self._headers(config)
        if headers is None:
            return self._missing_api_key_failure(payload)
        try:
            http_response = self._httpx_client().post(
                url,
                json=payload,
                headers=headers,
                timeout=self._policy.timeout_seconds,
            )
        except httpx.HTTPError as error:
            return ProviderTransportFailure(
                failure_class=FailureClass.TRANSIENT,
                code=TRANSPORT_ERROR_CODE,
                message=f"{type(error).__name__}: {error}",
                retryable=True,
                raw_request=dict(payload),
                metadata={"url": url},
            )
        return self._outcome_from_response(
            request, http_response, url, payload
        )

    def _outcome_from_response(
        self,
        request: ProviderCallRequest,
        http_response: httpx.Response,
        url: str,
        payload: dict[str, Any],
    ) -> ProviderTransportOutcome:
        if http_response.status_code >= 400:  # noqa: PLR2004
            return self._http_status_failure(http_response, url, payload)
        try:
            body = http_response.json()
        except ValueError:
            return ProviderTransportFailure(
                failure_class=FailureClass.PERMANENT,
                code=INVALID_JSON_CODE,
                message="provider response body is not valid JSON",
                retryable=False,
                raw_request=dict(payload),
                raw_response_body=http_response.text,
                metadata={"url": url},
            )
        outcome = parse_response(body, config=request.config)
        if isinstance(outcome, ProviderTransportFailure):
            return outcome.model_copy(update={"raw_request": dict(payload)})
        return outcome

    def _http_status_failure(
        self,
        http_response: httpx.Response,
        url: str,
        payload: dict[str, Any],
    ) -> ProviderTransportFailure:
        failure_class = classify_status_code(http_response.status_code)
        raw_body: Any
        try:
            raw_body = http_response.json()
        except ValueError:
            raw_body = http_response.text
        return ProviderTransportFailure(
            failure_class=failure_class,
            code=f"{HTTP_STATUS_CODE_PREFIX}{http_response.status_code}",
            message=http_response.text,
            retryable=failure_class in _RETRYABLE,
            raw_request=dict(payload),
            raw_response_body=raw_body,
            status_code=http_response.status_code,
            metadata={"url": url},
        )

    def _missing_base_url_failure(
        self,
        config: ProviderCallConfig,
        payload: dict[str, Any],
    ) -> ProviderTransportFailure:
        return ProviderTransportFailure(
            failure_class=FailureClass.PERMANENT,
            code=MISSING_BASE_URL_CODE,
            message=(
                f"transport policy for route "
                f"{config.quota_identity.label()!r} has no base_url"
            ),
            retryable=False,
            raw_request=dict(payload),
        )

    def _missing_api_key_failure(
        self,
        payload: dict[str, Any],
    ) -> ProviderTransportFailure:
        return ProviderTransportFailure(
            failure_class=FailureClass.PERMANENT,
            code=MISSING_API_KEY_CODE,
            message=(
                f"environment variable {self._policy.api_key_env!r} is not set"
            ),
            retryable=False,
            raw_request=dict(payload),
        )

    def _request_url(self, config: ProviderCallConfig) -> str | None:
        base_url = self._policy.base_url
        if not base_url:
            return None
        return base_url.rstrip("/") + protocol_path(config)

    def _headers(self, config: ProviderCallConfig) -> dict[str, str] | None:
        api_key = self._api_key or os.environ.get(self._policy.api_key_env)
        if not api_key:
            return None
        if config.route.protocol is Protocol.ANTHROPIC_MESSAGES:
            return {
                ANTHROPIC_API_KEY_HEADER: api_key,
                ANTHROPIC_VERSION_HEADER: ANTHROPIC_VERSION,
            }
        return {AUTHORIZATION_HEADER: f"Bearer {api_key}"}

    def _httpx_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client


_RETRYABLE = frozenset({FailureClass.TRANSIENT, FailureClass.RATE_LIMITED})
