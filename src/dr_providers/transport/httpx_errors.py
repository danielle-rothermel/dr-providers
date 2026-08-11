from __future__ import annotations

import httpx

from dr_providers.core.failures import RecoverabilityClass
from dr_providers.outcomes.models import STALLED_RESPONSE_CODE, TIMEOUT_CODE
from dr_providers.transport.status import classify_status_code

TRANSPORT_ERROR_CODE = "transport_error"
TRANSPORT_PROTOCOL_ERROR_CODE = "transport_protocol_error"
REMOTE_PROTOCOL_ERROR_CODE = "transport_remote_protocol_error"
HTTP_STATUS_CODE_PREFIX = "http_status_"


def classify_httpx_error(
    error: httpx.HTTPError,
) -> tuple[RecoverabilityClass, str]:
    """Classify a non-timeout httpx wire error into recoverability and code.

    Timeouts are handled here only as a guard: the wire path classifies them
    earlier with containment evidence this function cannot supply.
    """
    if isinstance(error, httpx.TimeoutException):
        code = (
            STALLED_RESPONSE_CODE
            if isinstance(error, httpx.ReadTimeout)
            else TIMEOUT_CODE
        )
        return RecoverabilityClass.TRANSIENT, code
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
