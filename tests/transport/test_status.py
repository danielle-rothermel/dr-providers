import pytest

from dr_providers import RecoverabilityClass, classify_status_code
from dr_providers.transport.status import is_redirect_status


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, RecoverabilityClass.RATE_LIMITED),
        (500, RecoverabilityClass.TRANSIENT),
        (503, RecoverabilityClass.TRANSIENT),
        (408, RecoverabilityClass.TRANSIENT),
        (409, RecoverabilityClass.TRANSIENT),
        (425, RecoverabilityClass.TRANSIENT),
        (400, RecoverabilityClass.PERMANENT),
        (401, RecoverabilityClass.PERMANENT),
        (402, RecoverabilityClass.PERMANENT),
        (404, RecoverabilityClass.PERMANENT),
        (413, RecoverabilityClass.RESOURCE_EXHAUSTION),
    ],
)
def test_classify_status_code(
    status: int,
    expected: RecoverabilityClass,
) -> None:
    assert classify_status_code(status) is expected


def test_payload_too_large_matches_the_local_request_bound() -> None:
    """413 is the server detecting what max_request_bytes detects locally."""
    assert classify_status_code(413) is RecoverabilityClass.RESOURCE_EXHAUSTION


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_statuses_are_redirects(status: int) -> None:
    assert is_redirect_status(status)


@pytest.mark.parametrize("status", [200, 299, 400, 404, 500])
def test_non_redirect_statuses_are_not_redirects(status: int) -> None:
    assert not is_redirect_status(status)


@pytest.mark.parametrize("status", [300, 304, 305, 306, 399])
def test_non_redirecting_3xx_statuses_are_not_redirects(status: int) -> None:
    """304 answers the request rather than naming a new location."""
    assert not is_redirect_status(status)
    assert classify_status_code(status) is RecoverabilityClass.PERMANENT
