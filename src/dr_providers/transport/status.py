from dr_providers.core.failures import RecoverabilityClass

RATE_LIMIT_STATUS = 429
BUDGET_EXHAUSTED_STATUS = 402
PAYLOAD_TOO_LARGE_STATUS = 413
TRANSIENT_STATUS_CODES = frozenset({408, 409, 425})
REDIRECT_STATUS_FLOOR = 300
REDIRECT_STATUS_CEILING = 400
SERVER_ERROR_FLOOR = 500


def classify_status_code(status_code: int) -> RecoverabilityClass:
    if status_code == RATE_LIMIT_STATUS:
        return RecoverabilityClass.RATE_LIMITED
    if status_code == BUDGET_EXHAUSTED_STATUS:
        return RecoverabilityClass.PERMANENT
    if status_code == PAYLOAD_TOO_LARGE_STATUS:
        # The server-side detection of the condition the local
        # max_request_bytes check already reports as resource exhaustion.
        return RecoverabilityClass.RESOURCE_EXHAUSTION
    if (
        status_code >= SERVER_ERROR_FLOOR
        or status_code in TRANSIENT_STATUS_CODES
    ):
        return RecoverabilityClass.TRANSIENT
    return RecoverabilityClass.PERMANENT


def is_redirect_status(status_code: int) -> bool:
    """Report whether a status redirects rather than answering the request."""
    return REDIRECT_STATUS_FLOOR <= status_code < REDIRECT_STATUS_CEILING
