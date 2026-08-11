from __future__ import annotations

import httpx

from dr_providers.core.failures import RecoverabilityClass
from dr_providers.transport.status import classify_status_code

TRANSPORT_ERROR_CODE = "transport_error"
TRANSPORT_PROTOCOL_ERROR_CODE = "transport_protocol_error"
HTTP_STATUS_CODE_PREFIX = "http_status_"


def classify_httpx_error(
    error: httpx.HTTPError,
) -> tuple[RecoverabilityClass, str]:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return (
            classify_status_code(status_code),
            f"{HTTP_STATUS_CODE_PREFIX}{status_code}",
        )
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
