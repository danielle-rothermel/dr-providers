import httpx
import pytest

from dr_providers import RecoverabilityClass
from dr_providers.transport.httpx_errors import (
    TRANSPORT_ERROR_CODE,
    TRANSPORT_PROTOCOL_ERROR_CODE,
    classify_httpx_error,
)


@pytest.mark.parametrize(
    ("error", "expected_recoverability", "expected_code"),
    [
        (
            httpx.ConnectError("down"),
            RecoverabilityClass.TRANSIENT,
            TRANSPORT_ERROR_CODE,
        ),
        (
            httpx.ProxyError("proxy down"),
            RecoverabilityClass.TRANSIENT,
            TRANSPORT_ERROR_CODE,
        ),
        (
            httpx.LocalProtocolError("bad framing"),
            RecoverabilityClass.PERMANENT,
            TRANSPORT_PROTOCOL_ERROR_CODE,
        ),
        (
            httpx.RemoteProtocolError("bad framing"),
            RecoverabilityClass.PERMANENT,
            TRANSPORT_PROTOCOL_ERROR_CODE,
        ),
        (
            httpx.DecodingError("bad encoding"),
            RecoverabilityClass.PERMANENT,
            TRANSPORT_PROTOCOL_ERROR_CODE,
        ),
        (
            httpx.UnsupportedProtocol("bad scheme"),
            RecoverabilityClass.PERMANENT,
            TRANSPORT_PROTOCOL_ERROR_CODE,
        ),
        (
            httpx.TooManyRedirects("too many"),
            RecoverabilityClass.PERMANENT,
            TRANSPORT_PROTOCOL_ERROR_CODE,
        ),
    ],
)
def test_classify_httpx_error(
    error: httpx.HTTPError,
    expected_recoverability: RecoverabilityClass,
    expected_code: str,
) -> None:
    assert classify_httpx_error(error) == (
        expected_recoverability,
        expected_code,
    )


def test_classify_httpx_error_uses_status_code_for_http_status_error() -> None:
    error = httpx.HTTPStatusError(
        "rate limited",
        request=httpx.Request("POST", "https://example.test"),
        response=httpx.Response(429),
    )

    assert classify_httpx_error(error) == (
        RecoverabilityClass.RATE_LIMITED,
        "http_status_429",
    )
