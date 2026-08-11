from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dr_http import (
    BoundedHttpClient,
    HttpClientConfig,
    WireFailure,
    WireRequest,
    WireResponse,
    is_dispatchable_url,
)

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
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
    TransportTimeoutContainment,
)
from dr_providers.translation.common import PARSE_ERROR_CODE
from dr_providers.translation.request import build_payload, protocol_path
from dr_providers.translation.response import parse_response
from dr_providers.transport.httpx_errors import (
    TIMEOUT_KINDS,
    classify_wire_failure_kind,
)
from dr_providers.transport.status import (
    classify_status_code,
    is_redirect_status,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from dr_http import ParsedRetryAfter

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
INVALID_BASE_URL_CODE = "invalid_base_url"
HTTP_STATUS_CODE_PREFIX = "http_status_"
REDIRECT_STATUS_CODE_PREFIX = "http_redirect_"
REQUEST_TOO_LARGE_CODE = "request_body_too_large"
RESPONSE_TOO_LARGE_CODE = "response_body_too_large"

SUCCESS_STATUS_FLOOR = 200
SUCCESS_STATUS_CEILING = 300
POST_METHOD = "POST"

TRANSPORT_TIMEOUT_MESSAGE = "provider transport timeout"
TRANSPORT_ERROR_MESSAGE = "provider transport error"
INVALID_BASE_URL_MESSAGE = (
    "transport policy base_url does not form a dispatchable http or https URL"
)


def _client_config(policy: ProviderTransportPolicy) -> HttpClientConfig:
    """Carry the policy's already-normalized sizing to the wire client.

    The policy clamps connect and idle timeouts to the general timeout
    before this point, because that normalization changes recorded policy
    identity and therefore belongs to the policy rather than the client.
    """
    return HttpClientConfig(
        timeout_seconds=policy.timeout_seconds,
        connect_timeout_seconds=policy.connect_timeout_seconds,
        idle_timeout_seconds=policy.idle_timeout_seconds,
        max_connections=policy.max_connections,
        max_keepalive_connections=policy.max_keepalive_connections,
        max_request_bytes=policy.max_request_bytes,
        max_response_bytes=policy.max_response_bytes,
    )


def _is_never_sent_failure(outcome: ProviderTransportOutcome) -> bool:
    """Report whether an outcome names a request that never reached the wire.

    A never-sent outcome carries no HTTP request evidence, so the request
    the transport had prepared is dropped rather than recorded as sent.
    """
    return (
        isinstance(outcome, ProviderTransportFailure)
        and outcome.code == INVALID_BASE_URL_CODE
    )


def _bounded_retry_after_hint(
    parsed: ParsedRetryAfter | None,
) -> ProviderRetryAfterHint | None:
    """Bound a parsed hint before it becomes retained evidence.

    Wire-boundary parsing is pure and uncapped, so the evidence caps are
    applied here: a delta or rendered date reaching past the retained
    bound names no hint this package is willing to store, and is dropped
    rather than recorded.
    """
    if parsed is None:
        return None
    if parsed.kind == "delta_seconds":
        seconds = int(parsed.value)
        if seconds > MAX_RETRY_AFTER_DELTA_SECONDS:
            return None
        return ProviderRetryAfterHint(kind="delta_seconds", value=seconds)
    rendered = str(parsed.value)
    if len(rendered.encode("utf-8")) > MAX_RETRY_AFTER_HEADER_BYTES:
        return None
    return ProviderRetryAfterHint(kind="http_date", value=rendered)


def _retry_after_hint(
    parsed: ParsedRetryAfter | None,
    raw_header_value: str | None,
) -> ProviderRetryAfterHint | None:
    """Drop an oversized header outright, then bound what it parsed to.

    A header longer than the retained bound is not credible input, so it
    yields no hint even when its prefix would have parsed.
    """
    if (
        raw_header_value is None
        or len(raw_header_value.encode("utf-8")) > MAX_RETRY_AFTER_HEADER_BYTES
    ):
        return None
    return _bounded_retry_after_hint(parsed)


class HttpProvider:
    """Turn one provider call into one bounded HTTP exchange and evidence.

    The wire capability, its lifecycle, and its resource bounds belong to
    an owned ``BoundedHttpClient``; this class owns everything the record
    keeps: endpoint and credential resolution, payload encoding, the
    mapping from wire results to persisted transport vocabulary, and
    invocation evidence.

    Each invocation is admitted through the client, so a close drains
    complete invocations rather than bare wire calls.
    """

    def __init__(
        self,
        *,
        policy: ProviderTransportPolicy,
        api_key: str | None = None,
        _client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._policy = policy
        self._api_key = api_key
        self._client = BoundedHttpClient(
            _client_config(policy),
            client_factory=_client_factory,
        )

    def close(self) -> None:
        """Drain admitted invocations and release the wire client once."""
        self._client.close()

    def __enter__(self) -> HttpProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        with self._client.admit():
            return self._invoke_admitted(request)

    def offload[ResultT](self, fn: Callable[[], ResultT]) -> Future[ResultT]:
        """Run ``fn`` on the client-owned executor until closing begins.

        The executor is sized from ``policy.max_connections``, the same
        value bounding the connection pool, so thread count and pool size
        cannot disagree.

        Offloaded work must not call ``close`` or ``offload`` on the
        provider that runs it: closing from inside offloaded work waits
        forever on that same work, and offloaded work that blocks on a
        nested offload starves once every worker is held that way.
        """
        return self._client.offload(fn)

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
        if not is_dispatchable_url(url):
            return ProviderInvocationEvidence.build(
                request=request,
                policy=self._policy,
                http_request=None,
                outcome=self._invalid_base_url_failure(url),
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
            http_request=(
                None if _is_never_sent_failure(outcome) else http_request
            ),
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
        """Dispatch one exchange and translate its wire result to evidence."""
        result = self._client.call(
            WireRequest(
                method=POST_METHOD,
                url=url,
                headers=headers,
                body=encoded_payload,
            )
        )
        if isinstance(result, WireFailure):
            return self._outcome_from_wire_failure(result, url)
        return self._outcome_from_wire_response(result, request, url)

    def _outcome_from_wire_response(
        self,
        response: WireResponse,
        request: ProviderCallRequest,
        url: str,
    ) -> tuple[
        ProviderTransportOutcome,
        int | None,
        ProviderRetryAfterHint | None,
    ]:
        retry_after = _retry_after_hint(
            response.retry_after,
            response.headers.get("retry-after"),
        )
        return (
            self._outcome_from_response(
                request,
                response.status_code,
                response.body,
                url,
            ),
            len(response.body),
            retry_after,
        )

    def _outcome_from_wire_failure(
        self,
        failure: WireFailure,
        url: str,
    ) -> tuple[
        ProviderTransportOutcome,
        int | None,
        ProviderRetryAfterHint | None,
    ]:
        """Record one wire failure in this package's own vocabulary.

        Byte-bound refusals are this provider's own sizing decision, so
        they report the configured limit and the bytes observed rather
        than an exception the wire raised.
        """
        recoverability, code = classify_wire_failure_kind(failure.kind)
        if code == RESPONSE_TOO_LARGE_CODE:
            return (
                self._response_too_large_failure(failure.observed_bytes or 0),
                failure.observed_bytes,
                _bounded_retry_after_hint(failure.retry_after),
            )
        if failure.kind in TIMEOUT_KINDS:
            return self._timeout_failure(code, failure, url), None, None
        if code == INVALID_BASE_URL_CODE:
            return (
                ProviderTransportFailure(
                    recoverability=recoverability,
                    code=code,
                    message=INVALID_BASE_URL_MESSAGE,
                    traceback=failure.traceback,
                    metadata={
                        "url": url,
                        "exception_type": failure.exception_type,
                    },
                ),
                None,
                None,
            )
        return (
            ProviderTransportFailure(
                recoverability=recoverability,
                code=code,
                message=TRANSPORT_ERROR_MESSAGE,
                traceback=failure.traceback,
                metadata={
                    "url": url,
                    "exception_type": failure.exception_type,
                },
            ),
            None,
            None,
        )

    def _timeout_failure(
        self,
        code: str,
        failure: WireFailure,
        url: str,
    ) -> ProviderTransportFailure:
        """Every native timeout is contained because the HTTP call returned.

        Pool starvation is local contention for this provider's own
        connection pool, so it carries its own code rather than reporting
        provider slowness.
        """
        return ProviderTransportFailure(
            recoverability=RecoverabilityClass.TRANSIENT,
            code=code,
            message=TRANSPORT_TIMEOUT_MESSAGE,
            traceback=failure.traceback,
            containment=TransportTimeoutContainment.CONTAINED,
            metadata={
                "url": url,
                "exception_type": failure.exception_type,
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
        response_bytes: bytes,
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
        response_bytes: bytes,
        url: str,
    ) -> ProviderTransportFailure:
        try:
            response_body: Any = json.loads(response_bytes)
        except ValueError:
            response_body = response_bytes.decode("utf-8", errors="replace")
        if is_redirect_status(status_code):
            # Redirects are not followed, so a redirecting base_url is a
            # transport misconfiguration rather than a provider rejection.
            return ProviderTransportFailure(
                recoverability=RecoverabilityClass.PERMANENT,
                code=f"{REDIRECT_STATUS_CODE_PREFIX}{status_code}",
                message=f"provider redirected with HTTP status {status_code}",
                response_body=response_body,
                status_code=status_code,
                metadata={"url": url},
            )
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

    def _invalid_base_url_failure(self, url: str) -> ProviderTransportFailure:
        return ProviderTransportFailure(
            recoverability=RecoverabilityClass.PERMANENT,
            code=INVALID_BASE_URL_CODE,
            message=INVALID_BASE_URL_MESSAGE,
            metadata={"url": url},
        )

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
