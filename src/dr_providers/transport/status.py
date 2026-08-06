from dr_providers.core.failures import FailureClass

RATE_LIMIT_STATUS = 429
TRANSIENT_STATUS_CODES = frozenset({408, 409, 425})
SERVER_ERROR_FLOOR = 500


def classify_status_code(status_code: int) -> FailureClass:
    if status_code == RATE_LIMIT_STATUS:
        return FailureClass.RATE_LIMITED
    if (
        status_code >= SERVER_ERROR_FLOOR
        or status_code in TRANSIENT_STATUS_CODES
    ):
        return FailureClass.TRANSIENT
    return FailureClass.PERMANENT
