from dr_providers.core.failures import RecoverabilityClass

RATE_LIMIT_STATUS = 429
BUDGET_EXHAUSTED_STATUS = 402
TRANSIENT_STATUS_CODES = frozenset({408, 409, 425})
SERVER_ERROR_FLOOR = 500


def classify_status_code(status_code: int) -> RecoverabilityClass:
    if status_code == RATE_LIMIT_STATUS:
        return RecoverabilityClass.RATE_LIMITED
    if status_code == BUDGET_EXHAUSTED_STATUS:
        return RecoverabilityClass.PERMANENT
    if (
        status_code >= SERVER_ERROR_FLOOR
        or status_code in TRANSIENT_STATUS_CODES
    ):
        return RecoverabilityClass.TRANSIENT
    return RecoverabilityClass.PERMANENT
