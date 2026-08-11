"""Pin the wire-kind boundary mapping onto persisted vocabulary.

The expected pairs are hand-written rather than derived from the mapping
under test, so a silent edit to a recorded code or recoverability fails
here instead of restating whatever the code currently says.
"""

from __future__ import annotations

import pytest
from dr_http import WireFailureKind

from dr_providers import RecoverabilityClass
from dr_providers.transport.httpx_errors import (
    WIRE_FAILURE_CLASSIFICATION,
    classify_wire_failure_kind,
)

EXPECTED_CLASSIFICATION: dict[WireFailureKind, tuple[str, str]] = {
    WireFailureKind.TIMEOUT: ("transient", "timeout"),
    WireFailureKind.STALLED_RESPONSE: ("transient", "stalled_response"),
    WireFailureKind.POOL_TIMEOUT: ("transient", "pool_timeout"),
    WireFailureKind.INVALID_URL: ("permanent", "invalid_base_url"),
    WireFailureKind.REMOTE_PROTOCOL_ERROR: (
        "transient",
        "transport_remote_protocol_error",
    ),
    WireFailureKind.LOCAL_PROTOCOL_ERROR: (
        "permanent",
        "transport_protocol_error",
    ),
    WireFailureKind.CONNECT_ERROR: ("transient", "transport_error"),
    WireFailureKind.NETWORK_ERROR: ("transient", "transport_error"),
    WireFailureKind.REQUEST_TOO_LARGE: (
        "resource_exhaustion",
        "request_body_too_large",
    ),
    WireFailureKind.RESPONSE_TOO_LARGE: (
        "resource_exhaustion",
        "response_body_too_large",
    ),
    WireFailureKind.UNKNOWN: ("unknown", "transport_error"),
}


@pytest.mark.parametrize(
    ("kind", "expected"),
    sorted(EXPECTED_CLASSIFICATION.items(), key=lambda item: item[0].name),
    ids=lambda value: value.name if isinstance(value, WireFailureKind) else "",
)
def test_wire_failure_kind_classification_is_pinned(
    kind: WireFailureKind,
    expected: tuple[str, str],
) -> None:
    recoverability, code = classify_wire_failure_kind(kind)

    assert (recoverability.value, code) == expected


def test_every_wire_failure_kind_is_mapped() -> None:
    """An unmapped kind must fail here, not borrow a code at runtime."""
    assert set(WIRE_FAILURE_CLASSIFICATION) == set(WireFailureKind)
    assert set(EXPECTED_CLASSIFICATION) == set(WireFailureKind)


def test_connect_and_network_failures_share_one_recorded_code() -> None:
    """The wire client splits a bucket this package records as one code."""
    connect = classify_wire_failure_kind(WireFailureKind.CONNECT_ERROR)
    network = classify_wire_failure_kind(WireFailureKind.NETWORK_ERROR)

    assert connect == network
    assert connect == (RecoverabilityClass.TRANSIENT, "transport_error")


def test_unmapped_kind_raises_rather_than_borrowing_a_code() -> None:
    unmapped: WireFailureKind = "not-a-kind"  # ty: ignore[invalid-assignment]

    with pytest.raises(ValueError, match="unmapped wire failure kind"):
        classify_wire_failure_kind(unmapped)
