"""Map wire failure kinds onto persisted transport failure vocabulary.

``dr_wire.WireFailureKind`` is an in-process description of one wire
condition and is never persisted. The codes below are persisted: they
land in ``ProviderTransportFailure.code`` inside stored invocation
evidence, so each one is a recorded-data format pinned in
``tests/outcomes/test_persisted_literals.py``.

The mapping is exhaustive over ``WireFailureKind`` so a new wire
condition cannot silently acquire a code that already means something
else in recorded data.

The mapping is not injective, and the collapses below are recorded
obligations rather than accidents of the current table:

``CONNECT_ERROR`` and ``NETWORK_ERROR`` both map to the single literal
``"transport_error"`` with ``TRANSIENT`` recoverability. Both name an
exchange that never completed against a reachable peer, and recorded
data does not distinguish them; a consumer that needs the distinction
reads ``metadata.exception_type``, not the code.

``UNKNOWN`` maps to that same ``"transport_error"`` literal with
``UNKNOWN`` recoverability. The code is shared deliberately, so an
unrecognized wire condition is recorded as a transport failure whose
recoverability, not whose code, states that it was not classified.

Because the code alone does not identify the kind that produced it,
splitting either collapse later is a recorded-data format change, not a
refinement.
"""

from __future__ import annotations

from dr_wire import WireFailureKind

from dr_providers.core.failures import RecoverabilityClass
from dr_providers.outcomes.models import (
    POOL_TIMEOUT_CODE,
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
)

TRANSPORT_ERROR_CODE = "transport_error"
TRANSPORT_PROTOCOL_ERROR_CODE = "transport_protocol_error"
REMOTE_PROTOCOL_ERROR_CODE = "transport_remote_protocol_error"
HTTP_STATUS_CODE_PREFIX = "http_status_"
INVALID_BASE_URL_CODE = "invalid_base_url"
REQUEST_TOO_LARGE_CODE = "request_body_too_large"
RESPONSE_TOO_LARGE_CODE = "response_body_too_large"

WIRE_FAILURE_CLASSIFICATION: dict[
    WireFailureKind, tuple[RecoverabilityClass, str]
] = {
    # Every native timeout is transient; the phase keeps its own code so
    # local pool contention stays distinguishable from provider slowness.
    WireFailureKind.TIMEOUT: (RecoverabilityClass.TRANSIENT, TIMEOUT_CODE),
    WireFailureKind.STALLED_RESPONSE: (
        RecoverabilityClass.TRANSIENT,
        STALLED_RESPONSE_CODE,
    ),
    WireFailureKind.POOL_TIMEOUT: (
        RecoverabilityClass.TRANSIENT,
        POOL_TIMEOUT_CODE,
    ),
    WireFailureKind.INVALID_URL: (
        RecoverabilityClass.PERMANENT,
        INVALID_BASE_URL_CODE,
    ),
    # A server that disconnects mid-exchange, classically a dropped
    # keepalive connection, is retryable in a way a local protocol
    # violation is not.
    WireFailureKind.REMOTE_PROTOCOL_ERROR: (
        RecoverabilityClass.TRANSIENT,
        REMOTE_PROTOCOL_ERROR_CODE,
    ),
    WireFailureKind.LOCAL_PROTOCOL_ERROR: (
        RecoverabilityClass.PERMANENT,
        TRANSPORT_PROTOCOL_ERROR_CODE,
    ),
    # Connect and wider network failures share one recorded code: both
    # name an exchange that never completed against a reachable peer.
    WireFailureKind.CONNECT_ERROR: (
        RecoverabilityClass.TRANSIENT,
        TRANSPORT_ERROR_CODE,
    ),
    WireFailureKind.NETWORK_ERROR: (
        RecoverabilityClass.TRANSIENT,
        TRANSPORT_ERROR_CODE,
    ),
    WireFailureKind.REQUEST_TOO_LARGE: (
        RecoverabilityClass.RESOURCE_EXHAUSTION,
        REQUEST_TOO_LARGE_CODE,
    ),
    WireFailureKind.RESPONSE_TOO_LARGE: (
        RecoverabilityClass.RESOURCE_EXHAUSTION,
        RESPONSE_TOO_LARGE_CODE,
    ),
    WireFailureKind.UNKNOWN: (
        RecoverabilityClass.UNKNOWN,
        TRANSPORT_ERROR_CODE,
    ),
}


def classify_wire_failure_kind(
    kind: WireFailureKind,
) -> tuple[RecoverabilityClass, str]:
    """Classify one wire failure kind into recoverability and code.

    Every member of the closed kind enum is mapped, so an unmapped kind
    is a defect in this boundary rather than a wire condition and raises
    instead of being recorded under a borrowed code.
    """
    classification = WIRE_FAILURE_CLASSIFICATION.get(kind)
    if classification is None:
        msg = f"unmapped wire failure kind: {kind}"
        raise ValueError(msg)
    return classification
