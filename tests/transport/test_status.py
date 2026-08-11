import pytest

from dr_providers import RecoverabilityClass, classify_status_code


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
    ],
)
def test_classify_status_code(
    status: int,
    expected: RecoverabilityClass,
) -> None:
    assert classify_status_code(status) is expected
