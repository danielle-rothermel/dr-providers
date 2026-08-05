"""HTTP status classification tests."""

import pytest

from dr_providers import FailureClass, classify_status_code


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, FailureClass.RATE_LIMITED),
        (500, FailureClass.TRANSIENT),
        (503, FailureClass.TRANSIENT),
        (408, FailureClass.TRANSIENT),
        (409, FailureClass.TRANSIENT),
        (425, FailureClass.TRANSIENT),
        (400, FailureClass.PERMANENT),
        (401, FailureClass.PERMANENT),
        (404, FailureClass.PERMANENT),
    ],
)
def test_classify_status_code(
    status: int,
    expected: FailureClass,
) -> None:
    assert classify_status_code(status) is expected
