from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from dr_providers.core.failures import (
    FailureClass,
)
from dr_providers.modeling.route import Protocol
from dr_providers.outcomes.conformance import with_conformance_warnings
from dr_providers.outcomes.evidence import (
    ProviderInvocationEvidence,
    RawHttpRequest,
)
from dr_providers.outcomes.models import (
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
)
from dr_providers.translation.common import PARSE_ERROR_CODE
from dr_providers.translation.request import build_payload, protocol_path
from dr_providers.translation.response import parse_response
from dr_providers.transport.status import classify_status_code

if TYPE_CHECKING:
    from dr_providers.modeling.call import ProviderCallConfig
    from dr_providers.modeling.request import ProviderCallRequest
    from dr_providers.transport.policy import ProviderTransportPolicy

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_VERSION_HEADER = "anthropic-version"
ANTHROPIC_API_KEY_HEADER = "x-api-key"
AUTHORIZATION_HEADER = "Authorization"
MISSING_API_KEY_CODE = "missing_api_key"
MISSING_BASE_URL_CODE = "missing_base_url"
HTTP_STATUS_CODE_PREFIX = "http_status_"
TRANSPORT_ERROR_CODE = "transport_error"
INVALID_JSON_CODE = "invalid_response_json"
TIMEOUT_CODE = "timeout"
STALLED_RESPONSE_CODE = "stalled_response"

# Bound TCP/TLS setup independently of the response idle budget.
MAX_CONNECT_TIMEOUT_SECONDS = 30.0

# Give httpx idle failures time to win before the caller-visible watchdog.
ATTEMPT_DEADLINE_MARGIN_SECONDS = 5.0


def _operational_timeout_seconds(timeout_seconds: float) -> float:
    return min(timeout_seconds, threading.TIMEOUT_MAX)


def _operational_attempt_deadline_seconds(
    timeout_seconds: float,
) -> float:
    return _operational_timeout_seconds(
        timeout_seconds + ATTEMPT_DEADLINE_MARGIN_SECONDS
    )


def _httpx_timeout(idle_timeout_seconds: float) -> httpx.Timeout:
    """Use per-operation idle bounds, with a separately capped connect."""
    operational_timeout_seconds = _operational_timeout_seconds(
        idle_timeout_seconds
    )
    return httpx.Timeout(
        connect=min(MAX_CONNECT_TIMEOUT_SECONDS, operational_timeout_seconds),
        read=operational_timeout_seconds,
        write=operational_timeout_seconds,
        pool=operational_timeout_seconds,
    )


class HttpProvider:
    """Bound caller-visible waiting with one daemon worker per wire call.

    Owned clients are isolated per call and closed best-effort. Injected
    clients remain caller-owned, so their worker and socket may linger after
    a deadline result is returned.
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
        """Leave injected clients open; owned clients are closed per call."""

    def __enter__(self) -> HttpProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(
        self, request: ProviderCallRequest
    ) -> ProviderTransportOutcome:
        payload = build_payload(request)
        return self._run_pipeline(request, payload)

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        payload = build_payload(request)
        url = self._request_url(request.config)
        headers = self._headers(request.config)
        raw_request = RawHttpRequest.build(
            url=url or "<missing_base_url>",
            headers=headers or {},
            body=payload,
        )
        outcome = self._run_pipeline(request, payload)
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self._policy,
            raw_request=raw_request,
            outcome=outcome,
        )

    def _run_pipeline(
        self,
        request: ProviderCallRequest,
        payload: dict[str, Any],
    ) -> ProviderTransportOutcome:
        outcome = self._complete_with_retries(request, payload)
        if isinstance(outcome, ProviderTransportResponse):
            return with_conformance_warnings(request, outcome)
        return outcome

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
        return self._wire_attempt_within_deadline(
            request, url, headers, payload
        )

    def _wire_attempt_within_deadline(
        self,
        request: ProviderCallRequest,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> ProviderTransportOutcome:
        """Bound caller-visible waiting without joining a timed-out worker."""
        deadline = (
            self._policy.timeout_seconds + ATTEMPT_DEADLINE_MARGIN_SECONDS
        )
        operational_deadline = _operational_attempt_deadline_seconds(
            self._policy.timeout_seconds
        )
        outcome_box: list[ProviderTransportOutcome] = []
        error_box: list[BaseException] = []
        # Isolate owned pools; never close an injected client.
        if self._owns_client or self._client is None:
            call_client = httpx.Client()
        else:
            call_client = self._client
        done = threading.Event()

        def worker() -> None:
            try:
                outcome_box.append(
                    self._wire_call(
                        request, url, headers, payload, call_client
                    )
                )
            except BaseException as error:  # noqa: BLE001 -- box, never raise
                error_box.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        if not done.wait(timeout=operational_deadline):
            # Closing an owned client may unblock its worker; never join here.
            if self._owns_client:
                with contextlib.suppress(Exception):
                    call_client.close()
            return self._deadline_timeout_failure(url, payload, deadline)
        if self._owns_client:
            with contextlib.suppress(Exception):
                call_client.close()
        if error_box:
            raise error_box[0]
        return outcome_box[0]

    def _wire_call(
        self,
        request: ProviderCallRequest,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        client: httpx.Client,
    ) -> ProviderTransportOutcome:
        try:
            http_response = client.post(
                url,
                json=payload,
                headers=headers,
                timeout=_httpx_timeout(self._policy.idle_timeout_seconds),
                follow_redirects=False,
            )
        except httpx.TimeoutException as error:
            return self._httpx_timeout_failure(error, url, payload)
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

    def _httpx_timeout_failure(
        self,
        error: httpx.TimeoutException,
        url: str,
        payload: dict[str, Any],
    ) -> ProviderTransportFailure:
        """Classify read timeouts as idle stalls; keep other timeout phases."""
        is_idle_stall = isinstance(error, httpx.ReadTimeout)
        return ProviderTransportFailure(
            failure_class=FailureClass.TRANSIENT,
            code=STALLED_RESPONSE_CODE if is_idle_stall else TIMEOUT_CODE,
            message=f"{type(error).__name__}: {error}",
            retryable=True,
            raw_request=dict(payload),
            metadata={
                "url": url,
                "timeout_seconds": self._policy.timeout_seconds,
                "idle_timeout_seconds": self._policy.idle_timeout_seconds,
                "phase": type(error).__name__,
            },
        )

    def _deadline_timeout_failure(
        self,
        url: str,
        payload: dict[str, Any],
        deadline: float,
    ) -> ProviderTransportFailure:
        return ProviderTransportFailure(
            failure_class=FailureClass.TRANSIENT,
            code=STALLED_RESPONSE_CODE,
            message=(
                "provider attempt exceeded the per-attempt deadline "
                f"of {deadline:.1f}s (policy timeout_seconds="
                f"{self._policy.timeout_seconds:.1f}s); the response "
                "stalled without completing"
            ),
            retryable=True,
            raw_request=dict(payload),
            metadata={
                "url": url,
                "timeout_seconds": self._policy.timeout_seconds,
                "deadline_seconds": deadline,
            },
        )

    def _outcome_from_response(
        self,
        request: ProviderCallRequest,
        http_response: httpx.Response,
        url: str,
        payload: dict[str, Any],
    ) -> ProviderTransportOutcome:
        if not http_response.is_success:
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
        if not isinstance(body, Mapping):
            return ProviderTransportFailure(
                failure_class=FailureClass.PERMANENT,
                code=PARSE_ERROR_CODE,
                message="provider response JSON must be an object",
                retryable=False,
                raw_request=dict(payload),
                raw_response_body=body,
                status_code=http_response.status_code,
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


_RETRYABLE = frozenset({FailureClass.TRANSIENT, FailureClass.RATE_LIMITED})
