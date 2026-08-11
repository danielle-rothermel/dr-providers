from __future__ import annotations

import httpx

from dr_providers.core.failures import RecoverabilityClass
from dr_providers.outcomes.models import (
    POOL_TIMEOUT_CODE,
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
)
from dr_providers.transport.status import classify_status_code

TRANSPORT_ERROR_CODE = "transport_error"
TRANSPORT_PROTOCOL_ERROR_CODE = "transport_protocol_error"
REMOTE_PROTOCOL_ERROR_CODE = "transport_remote_protocol_error"
HTTP_STATUS_CODE_PREFIX = "http_status_"


def timeout_code(error: httpx.TimeoutException) -> str:
    """Name the timeout phase so local contention stays distinguishable.

    Every caller that classifies a timeout uses this one mapping, so the
    wire path and the exported classifier cannot disagree about which
    phase a timeout names.
    """
    if isinstance(error, httpx.PoolTimeout):
        return POOL_TIMEOUT_CODE
    if isinstance(error, httpx.ReadTimeout):
        return STALLED_RESPONSE_CODE
    return TIMEOUT_CODE


def classify_httpx_error(
    error: httpx.HTTPError,
) -> tuple[RecoverabilityClass, str]:
    """Classify a non-timeout httpx wire error into recoverability and code.

    Timeouts are handled here only as a guard: the wire path classifies them
    earlier with containment evidence this function cannot supply.
    """
    if isinstance(error, httpx.TimeoutException):
        return RecoverabilityClass.TRANSIENT, timeout_code(error)
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return (
            classify_status_code(status_code),
            f"{HTTP_STATUS_CODE_PREFIX}{status_code}",
        )
    if isinstance(error, httpx.RemoteProtocolError):
        # A server that disconnects mid-exchange, classically a dropped
        # keepalive connection, is retryable in a way a local protocol
        # violation is not.
        return RecoverabilityClass.TRANSIENT, REMOTE_PROTOCOL_ERROR_CODE
    if isinstance(
        error,
        (
            httpx.ProtocolError,
            httpx.DecodingError,
            httpx.UnsupportedProtocol,
            httpx.TooManyRedirects,
        ),
    ):
        return RecoverabilityClass.PERMANENT, TRANSPORT_PROTOCOL_ERROR_CODE
    if isinstance(error, httpx.NetworkError | httpx.ProxyError):
        return RecoverabilityClass.TRANSIENT, TRANSPORT_ERROR_CODE
    return RecoverabilityClass.UNKNOWN, TRANSPORT_ERROR_CODE
