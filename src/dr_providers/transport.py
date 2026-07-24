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

import contextlib
import os
import threading
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
TIMEOUT_CODE = "timeout"
STALLED_RESPONSE_CODE = "stalled_response"

# A stalled TCP/TLS handshake must fail fast: a wedged connect should
# never consume the full read budget. The read phase is bounded by the
# IDLE timeout (per inter-byte gap), not the absolute cap, so a legitimate
# long stream making steady progress is never killed for running long; the
# per-invocation deadline (below) is the absolute wall-clock backstop.
MAX_CONNECT_TIMEOUT_SECONDS = 30.0

# Small fixed margin added on top of ``timeout_seconds`` (the absolute cap)
# for the overall per-invocation deadline. The idle-read timeout is the
# primary stall detector and returns a typed failure first for a genuine
# stall; the hard deadline (absolute cap) only fires when a response defeats
# the idle timer entirely — e.g. a dribble sending one byte per idle window.
INVOCATION_DEADLINE_MARGIN_SECONDS = 5.0


def _httpx_timeout(idle_timeout_seconds: float) -> httpx.Timeout:
    """Progress-aware timeout discipline: idle-bounded read, capped connect.

    The READ phase is bounded by ``idle_timeout_seconds`` -- httpx's read
    timeout is per-read-operation (per inter-byte gap), which is exactly a
    PROGRESS/IDLE bound: a stream that keeps delivering bytes resets it and
    runs as long as it makes progress, while a stream that goes silent longer
    than the idle window fails ``stalled_response`` promptly. This is why a
    legitimate long streaming response (steady tokens for minutes) is no
    longer killed by the flat ``timeout_seconds``. Connect is capped so a
    wedged handshake fails fast; write/pool take the idle budget too. The
    absolute ``timeout_seconds`` cap is enforced separately as the
    per-invocation deadline (the dribble backstop).
    """
    return httpx.Timeout(
        connect=min(MAX_CONNECT_TIMEOUT_SECONDS, idle_timeout_seconds),
        read=idle_timeout_seconds,
        write=idle_timeout_seconds,
        pool=idle_timeout_seconds,
    )


class HttpProvider:
    """Single-shot provider calls over raw httpx.

    The Provider Transport Policy supplies credentials env var, base URL,
    timeout, and native retry count. An explicit ``api_key`` overrides
    the env lookup (for tests). An injected client is left open on close.

    Client lifecycle and deadline isolation
    ---------------------------------------
    Each wire call runs on its own short-lived daemon thread under a hard
    per-invocation wall-clock deadline. On a deadline breach the transport
    returns the typed timeout/stalled outcome immediately WITHOUT joining
    the worker, so a wedged socket read can never make the call hang.

    * OWNED client (``client is None``): every wire call gets its OWN
      ``httpx.Client`` (its own connection pool). A deadline breach closes
      only that call's client -- unblocking only that call's worker -- so a
      timed-out request can never tear down the pool or fail a concurrent
      healthy call on the same provider. Each per-call client is closed as
      its call completes, so ``close()`` has no owned state to release.
    * INJECTED (caller-owned) client: the transport must not close a client
      it does not own, so a deadline breach cannot forcibly unblock the
      wedged ``client.post`` -- that worker stays blocked until httpx's own
      idle/read timeout fires (bounded, but not instant). RESIDUAL LIMITATION:
      with a caller-owned SYNC httpx client there is no clean cancellation, so
      one leaked worker thread + socket may linger for up to the idle timeout
      per timed-out call. It is a DAEMON thread, so it never blocks interpreter
      exit and never corrupts other calls; the deadline still returns the
      typed outcome on schedule. Inject a per-call client, or accept the
      idle-bounded lag, if this matters.
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
        """Owned wire clients are per-call and already closed per call; an
        injected client is caller-owned and left open. Kept for the
        context-manager contract."""

    def __enter__(self) -> HttpProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(
        self, request: ProviderCallRequest
    ) -> ProviderTransportOutcome:
        """Return the typed no-throw outcome for one request."""
        payload = build_payload(request)
        return self._run_pipeline(request, payload)

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
        """Shared complete/invoke pipeline: retries plus conformance."""
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
        return self._wire_call_within_deadline(request, url, headers, payload)

    def _wire_call_within_deadline(
        self,
        request: ProviderCallRequest,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> ProviderTransportOutcome:
        """Run the blocking wire call under a hard wall-clock deadline.

        httpx's own timeout discipline (``_httpx_timeout``) is the primary
        bound and returns a typed failure for a normal stall. This
        deadline is the belt-and-suspenders backstop: it bounds the entire
        invocation — including response streaming/JSON decode — so a
        stalled response that defeats the per-read timeout (a trickling
        edge that keeps resetting the per-read timer) can never exceed the
        budget. On breach the typed timeout failure is returned; nothing
        hangs and nothing raises.

        The call runs on its OWN short-lived daemon thread — never a shared
        executor — so a timed-out worker can never block a concurrent call
        or interpreter exit. For an owned client the worker gets a fresh
        per-call client; a deadline breach closes ONLY that client, so only
        the timed-out call is unblocked and no other in-flight call is
        disturbed. For an injected client the worker cannot be forcibly
        unblocked (see the class docstring's residual limitation).
        """
        deadline = (
            self._policy.timeout_seconds + INVOCATION_DEADLINE_MARGIN_SECONDS
        )
        outcome_box: list[ProviderTransportOutcome] = []
        error_box: list[BaseException] = []
        # A per-call client for the owned case so an interrupt tears down only
        # this call's pool; the injected client is shared and left untouched.
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
        if not done.wait(timeout=deadline):
            # The worker is wedged in a socket read past the deadline. Close
            # this call's OWNED client to unblock it (best-effort), then return
            # the typed failure WITHOUT joining the leaked daemon worker.
            if self._owns_client:
                with contextlib.suppress(Exception):
                    call_client.close()
            return self._deadline_timeout_failure(url, payload, deadline)
        # Completed within the deadline: close the owned per-call client and
        # surface the outcome (or re-raise an unexpected programming error).
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
        """Typed failure for an httpx-enforced connect/read/write timeout.

        A ReadTimeout is an IDLE stall: no bytes arrived within
        ``idle_timeout_seconds`` (the progress/idle bound), so it is classified
        ``stalled_response``. A connect/write/pool timeout keeps the generic
        ``timeout`` code.
        """
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
        """Typed failure for the hard per-invocation deadline breach."""
        return ProviderTransportFailure(
            failure_class=FailureClass.TRANSIENT,
            code=STALLED_RESPONSE_CODE,
            message=(
                "provider invocation exceeded the per-invocation deadline "
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


_RETRYABLE = frozenset({FailureClass.TRANSIENT, FailureClass.RATE_LIMITED})
