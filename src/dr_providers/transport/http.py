from __future__ import annotations

import json
import os
import threading
import traceback
from collections.abc import Callable, Mapping
from datetime import UTC
from email.utils import format_datetime, parsedate_to_datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import httpx

from dr_providers.core.failures import RecoverabilityClass
from dr_providers.modeling.route import Protocol
from dr_providers.outcomes.conformance import with_conformance_warnings
from dr_providers.outcomes.evidence import (
    MAX_RETRY_AFTER_DELTA_SECONDS,
    MAX_RETRY_AFTER_HEADER_BYTES,
    ProviderHttpRequestEvidence,
    ProviderInvocationEvidence,
    ProviderRetryAfterHint,
)
from dr_providers.outcomes.models import (
    INVALID_JSON_CODE,
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
    TransportTimeoutContainment,
)
from dr_providers.translation.common import PARSE_ERROR_CODE
from dr_providers.translation.request import build_payload, protocol_path
from dr_providers.translation.response import parse_response
from dr_providers.transport.httpx_errors import classify_httpx_error
from dr_providers.transport.status import classify_status_code

if TYPE_CHECKING:
    from dr_providers.modeling.call import ProviderCallConfig
    from dr_providers.modeling.request import ProviderCallRequest
    from dr_providers.transport.policy import ProviderTransportPolicy

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_VERSION_HEADER = "anthropic-version"
ANTHROPIC_API_KEY_HEADER = "x-api-key"
AUTHORIZATION_HEADER = "Authorization"
CONTENT_TYPE_HEADER = "Content-Type"
JSON_CONTENT_TYPE = "application/json"
MISSING_API_KEY_CODE = "missing_api_key"
MISSING_BASE_URL_CODE = "missing_base_url"
HTTP_STATUS_CODE_PREFIX = "http_status_"
REQUEST_TOO_LARGE_CODE = "request_body_too_large"
RESPONSE_TOO_LARGE_CODE = "response_body_too_large"

RESPONSE_STREAM_CHUNK_BYTES = 64 * 1024
SUCCESS_STATUS_FLOOR = 200
SUCCESS_STATUS_CEILING = 300


class _ProviderState(Enum):
    OPEN = auto()
    CLOSING = auto()
    CLOSED = auto()


def _operational_timeout_seconds(timeout_seconds: float) -> float:
    return min(timeout_seconds, threading.TIMEOUT_MAX)


def _httpx_timeout(policy: ProviderTransportPolicy) -> httpx.Timeout:
    """Use direct native phase timeouts for the synchronous HTTP operation."""
    operation_timeout = _operational_timeout_seconds(policy.timeout_seconds)
    connect_timeout = _operational_timeout_seconds(
        policy.connect_timeout_seconds
    )
    read_timeout = _operational_timeout_seconds(policy.idle_timeout_seconds)
    return httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=operation_timeout,
        pool=operation_timeout,
    )


def _exception_traceback(error: BaseException) -> str:
    return "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )


def _normalize_retry_after(value: str | None) -> ProviderRetryAfterHint | None:
    if (
        value is None
        or len(value.encode("utf-8")) > MAX_RETRY_AFTER_HEADER_BYTES
    ):
        return None
    normalized = value.strip()
    if normalized.isascii() and normalized.isdigit():
        seconds = int(normalized)
        if seconds <= MAX_RETRY_AFTER_DELTA_SECONDS:
            return ProviderRetryAfterHint(
                kind="delta_seconds",
                value=seconds,
            )
        return None
    try:
        parsed = parsedate_to_datetime(normalized)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return None
    return ProviderRetryAfterHint(
        kind="http_date",
        value=format_datetime(parsed.astimezone(UTC), usegmt=True),
    )


class HttpProvider:
    """Own and reuse one bounded synchronous HTTP client until closed."""

    def __init__(
        self,
        *,
        policy: ProviderTransportPolicy,
        api_key: str | None = None,
        _client_factory: Callable[..., httpx.Client] | None = None,
    ) -> None:
        self._policy = policy
        self._api_key = api_key
        limits = httpx.Limits(
            max_connections=policy.max_connections,
            max_keepalive_connections=policy.max_keepalive_connections,
        )
        client_factory = (
            httpx.Client if _client_factory is None else _client_factory
        )
        self._client = client_factory(
            limits=limits,
            follow_redirects=False,
        )
        self._condition = threading.Condition()
        self._state = _ProviderState.OPEN
        self._active_invocations = 0

    def close(self) -> None:
        """Stop admission, drain active invocations, and close exactly once."""
        with self._condition:
            if self._state is _ProviderState.CLOSED:
                return
            if self._state is _ProviderState.CLOSING:
                while self._state is _ProviderState.CLOSING:
                    self._condition.wait()
                return
            self._state = _ProviderState.CLOSING
            self._condition.notify_all()
            while self._active_invocations:
                self._condition.wait()

        try:
            self._client.close()
        finally:
            with self._condition:
                self._state = _ProviderState.CLOSED
                self._condition.notify_all()

    def __enter__(self) -> HttpProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self._begin_invocation()
        try:
            return self._invoke_admitted(request)
        finally:
            self._end_invocation()

    def _begin_invocation(self) -> None:
        with self._condition:
            if self._state is not _ProviderState.OPEN:
                msg = "HttpProvider is closing or closed"
                raise RuntimeError(msg)
            self._active_invocations += 1

    def _end_invocation(self) -> None:
        with self._condition:
            self._active_invocations -= 1
            if self._active_invocations == 0:
                self._condition.notify_all()

    def _invoke_admitted(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        if request.config.route.provider is not self._policy.provider_kind:
            msg = (
                "request route provider does not match transport policy: "
                f"{request.config.route.provider.value} != "
                f"{self._policy.provider_kind.value}"
            )
            raise ValueError(msg)
        payload = build_payload(request)
        encoded_payload = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        url = self._request_url(request.config)
        if url is None:
            return ProviderInvocationEvidence.build(
                request=request,
                policy=self._policy,
                http_request=None,
                outcome=self._missing_base_url_failure(request.config),
            )
        headers = self._headers(request.config)
        if headers is None:
            return ProviderInvocationEvidence.build(
                request=request,
                policy=self._policy,
                http_request=None,
                outcome=self._missing_api_key_failure(),
            )
        http_request = ProviderHttpRequestEvidence.build(
            url=url,
            headers=headers,
            body=payload,
            body_bytes=len(encoded_payload),
        )
        if len(encoded_payload) > self._policy.max_request_bytes:
            return ProviderInvocationEvidence.build(
                request=request,
                policy=self._policy,
                http_request=None,
                outcome=self._request_too_large_failure(len(encoded_payload)),
            )
        outcome, response_bytes, retry_after = self._wire_call(
            request,
            url,
            headers,
            encoded_payload,
        )
        if isinstance(outcome, ProviderTransportResponse):
            outcome = with_conformance_warnings(request, outcome)
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self._policy,
            http_request=http_request,
            outcome=outcome,
            response_bytes=response_bytes,
            retry_after=retry_after,
        )

    def _wire_call(
        self,
        request: ProviderCallRequest,
        url: str,
        headers: dict[str, str],
        encoded_payload: bytes,
    ) -> tuple[
        ProviderTransportOutcome,
        int | None,
        ProviderRetryAfterHint | None,
    ]:
        try:
            with self._client.stream(
                "POST",
                url,
                content=encoded_payload,
                headers=headers,
                timeout=_httpx_timeout(self._policy),
                follow_redirects=False,
            ) as http_response:
                retry_after = _normalize_retry_after(
                    http_response.headers.get("retry-after")
                )
                body, observed_bytes = self._read_response(http_response)
                if body is None:
                    return (
                        self._response_too_large_failure(observed_bytes),
                        observed_bytes,
                        retry_after,
                    )
                return (
                    self._outcome_from_response(
                        request,
                        http_response.status_code,
                        body,
                        url,
                    ),
                    observed_bytes,
                    retry_after,
                )
        except httpx.TimeoutException as error:
            return self._httpx_timeout_failure(error, url), None, None
        except httpx.HTTPError as error:
            recoverability, code = classify_httpx_error(error)
            return (
                ProviderTransportFailure(
                    recoverability=recoverability,
                    code=code,
                    message="provider transport error",
                    traceback=_exception_traceback(error),
                    metadata={
                        "url": url,
                        "exception_type": type(error).__name__,
                    },
                ),
                None,
                None,
            )

    def _read_response(
        self,
        http_response: httpx.Response,
    ) -> tuple[bytearray | None, int]:
        body = bytearray()
        observed_bytes = 0
        for chunk in http_response.iter_bytes(
            chunk_size=RESPONSE_STREAM_CHUNK_BYTES
        ):
            observed_bytes += len(chunk)
            if observed_bytes > self._policy.max_response_bytes:
                return None, observed_bytes
            body.extend(chunk)
        return body, observed_bytes

    def _httpx_timeout_failure(
        self,
        error: httpx.TimeoutException,
        url: str,
    ) -> ProviderTransportFailure:
        """Every native timeout is contained because the HTTP call returned."""
        is_idle_stall = isinstance(error, httpx.ReadTimeout)
        return ProviderTransportFailure(
            recoverability=RecoverabilityClass.TRANSIENT,
            code=STALLED_RESPONSE_CODE if is_idle_stall else TIMEOUT_CODE,
            message="provider transport timeout",
            traceback=_exception_traceback(error),
            containment=TransportTimeoutContainment.CONTAINED,
            metadata={
                "url": url,
                "exception_type": type(error).__name__,
                "timeout_seconds": self._policy.timeout_seconds,
                "connect_timeout_seconds": (
                    self._policy.connect_timeout_seconds
                ),
                "idle_timeout_seconds": self._policy.idle_timeout_seconds,
            },
        )

    def _outcome_from_response(
        self,
        request: ProviderCallRequest,
        status_code: int,
        response_bytes: bytearray,
        url: str,
    ) -> ProviderTransportOutcome:
        if not SUCCESS_STATUS_FLOOR <= status_code < SUCCESS_STATUS_CEILING:
            return self._http_status_failure(
                status_code,
                response_bytes,
                url,
            )
        try:
            body = json.loads(response_bytes)
        except ValueError:
            return ProviderTransportFailure(
                recoverability=RecoverabilityClass.PERMANENT,
                code=INVALID_JSON_CODE,
                message="provider response body is not valid JSON",
                response_body=response_bytes.decode(
                    "utf-8",
                    errors="replace",
                ),
                status_code=status_code,
                metadata={"url": url},
            )
        if not isinstance(body, Mapping):
            return ProviderTransportFailure(
                recoverability=RecoverabilityClass.PERMANENT,
                code=PARSE_ERROR_CODE,
                message="provider response JSON must be an object",
                response_body=body,
                status_code=status_code,
                metadata={"url": url},
            )
        return parse_response(body, config=request.config)

    def _http_status_failure(
        self,
        status_code: int,
        response_bytes: bytearray,
        url: str,
    ) -> ProviderTransportFailure:
        try:
            response_body: Any = json.loads(response_bytes)
        except ValueError:
            response_body = response_bytes.decode("utf-8", errors="replace")
        return ProviderTransportFailure(
            recoverability=classify_status_code(status_code),
            code=f"{HTTP_STATUS_CODE_PREFIX}{status_code}",
            message=f"provider returned HTTP status {status_code}",
            response_body=response_body,
            status_code=status_code,
            metadata={"url": url},
        )

    def _request_too_large_failure(
        self,
        observed_bytes: int,
    ) -> ProviderTransportFailure:
        limit = self._policy.max_request_bytes
        return ProviderTransportFailure(
            recoverability=RecoverabilityClass.RESOURCE_EXHAUSTION,
            code=REQUEST_TOO_LARGE_CODE,
            message=f"request body exceeds {limit} byte limit",
            metadata={
                "limit_bytes": limit,
                "observed_bytes": observed_bytes,
            },
        )

    def _response_too_large_failure(
        self,
        observed_bytes: int,
    ) -> ProviderTransportFailure:
        limit = self._policy.max_response_bytes
        return ProviderTransportFailure(
            recoverability=RecoverabilityClass.RESOURCE_EXHAUSTION,
            code=RESPONSE_TOO_LARGE_CODE,
            message=f"response body exceeds {limit} byte limit",
            metadata={
                "limit_bytes": limit,
                "observed_bytes": observed_bytes,
            },
        )

    def _missing_base_url_failure(
        self,
        config: ProviderCallConfig,
    ) -> ProviderTransportFailure:
        return ProviderTransportFailure(
            recoverability=RecoverabilityClass.PERMANENT,
            code=MISSING_BASE_URL_CODE,
            message=(
                "transport policy for route "
                f"{config.quota_identity.label()!r} has no base_url"
            ),
        )

    def _missing_api_key_failure(self) -> ProviderTransportFailure:
        return ProviderTransportFailure(
            recoverability=RecoverabilityClass.PERMANENT,
            code=MISSING_API_KEY_CODE,
            message=(
                f"environment variable {self._policy.api_key_env!r} is not set"
            ),
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
                CONTENT_TYPE_HEADER: JSON_CONTENT_TYPE,
            }
        return {
            AUTHORIZATION_HEADER: f"Bearer {api_key}",
            CONTENT_TYPE_HEADER: JSON_CONTENT_TYPE,
        }
