from dr_providers.core.failures import RecoverabilityClass

RATE_LIMIT_STATUS = 429
BUDGET_EXHAUSTED_STATUS = 402
PAYLOAD_TOO_LARGE_STATUS = 413
TRANSIENT_STATUS_CODES = frozenset({408, 409, 425})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
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
    """Report whether a status redirects rather than answering the request.

    Only the statuses that name a new location redirect. The rest of the
    3xx range, such as 304 Not Modified, answers the request the client
    made and is classified like any other status.
    """
    return status_code in REDIRECT_STATUS_CODES
