"""Pin the wire-kind boundary mapping onto persisted vocabulary.

The expected pairs are hand-written rather than derived from the mapping
under test, so a silent edit to a recorded code or recoverability fails
here instead of restating whatever the code currently says.
"""

from __future__ import annotations

import pytest
from _policy import (
    TEST_MAX_REQUEST_BYTES,
    TEST_MAX_RESPONSE_BYTES,
    make_transport_policy,
)
from dr_wire import WireFailure, WireFailureKind

from dr_providers import ProviderTransportFailure, RecoverabilityClass
from dr_providers.transport.http import HttpProvider
from dr_providers.transport.wire_failures import (
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


URL = "https://example.test/v1/chat/completions"
OBSERVED_BYTES = 4096

TRANSPORT_ERROR_METADATA_KEYS = {"url", "exception_type"}
TIMEOUT_METADATA_KEYS = {
    "url",
    "exception_type",
    "timeout_seconds",
    "connect_timeout_seconds",
    "idle_timeout_seconds",
}
BYTE_BOUND_METADATA_KEYS = {"limit_bytes", "observed_bytes"}

EXPECTED_EVIDENCE_SHAPE: dict[WireFailureKind, tuple[str, set[str]]] = {
    WireFailureKind.TIMEOUT: (
        "provider transport timeout",
        TIMEOUT_METADATA_KEYS,
    ),
    WireFailureKind.STALLED_RESPONSE: (
        "provider transport timeout",
        TIMEOUT_METADATA_KEYS,
    ),
    WireFailureKind.POOL_TIMEOUT: (
        "provider transport timeout",
        TIMEOUT_METADATA_KEYS,
    ),
    WireFailureKind.INVALID_URL: (
        "transport policy base_url does not form a dispatchable http or "
        "https URL",
        TRANSPORT_ERROR_METADATA_KEYS,
    ),
    WireFailureKind.CONNECT_ERROR: (
        "provider transport error",
        TRANSPORT_ERROR_METADATA_KEYS,
    ),
    WireFailureKind.NETWORK_ERROR: (
        "provider transport error",
        TRANSPORT_ERROR_METADATA_KEYS,
    ),
    WireFailureKind.REMOTE_PROTOCOL_ERROR: (
        "provider transport error",
        TRANSPORT_ERROR_METADATA_KEYS,
    ),
    WireFailureKind.LOCAL_PROTOCOL_ERROR: (
        "provider transport error",
        TRANSPORT_ERROR_METADATA_KEYS,
    ),
    WireFailureKind.UNKNOWN: (
        "provider transport error",
        TRANSPORT_ERROR_METADATA_KEYS,
    ),
    WireFailureKind.REQUEST_TOO_LARGE: (
        f"request body exceeds {TEST_MAX_REQUEST_BYTES} byte limit",
        BYTE_BOUND_METADATA_KEYS,
    ),
    WireFailureKind.RESPONSE_TOO_LARGE: (
        f"response body exceeds {TEST_MAX_RESPONSE_BYTES} byte limit",
        BYTE_BOUND_METADATA_KEYS,
    ),
}


def _wire_failure(kind: WireFailureKind) -> WireFailure:
    """Build the wire failure the client reports for one kind.

    Byte-bound refusals are detected by the client rather than caught,
    so they carry no exception detail and do carry ``observed_bytes``.
    """
    if kind in {
        WireFailureKind.REQUEST_TOO_LARGE,
        WireFailureKind.RESPONSE_TOO_LARGE,
    }:
        return WireFailure(
            kind=kind,
            exception_type="",
            message="",
            traceback="",
            observed_bytes=OBSERVED_BYTES,
        )
    return WireFailure(
        kind=kind,
        exception_type="ExampleWireError",
        message="wire detail",
        traceback="Traceback (most recent call last):\n",
    )


@pytest.mark.parametrize(
    "kind",
    sorted(WireFailureKind, key=lambda kind: kind.name),
    ids=lambda kind: kind.name,
)
def test_wire_failure_evidence_shape_is_pinned_per_kind(
    kind: WireFailureKind,
) -> None:
    """Each kind's recorded message and metadata keys are pinned.

    Recoverability and code alone would not catch a branch recording the
    wrong summary or dropping the metadata a consumer reads instead of
    the collapsed code.
    """
    expected_message, expected_metadata_keys = EXPECTED_EVIDENCE_SHAPE[kind]
    provider = HttpProvider(
        policy=make_transport_policy(base_url="https://example.test/v1"),
        api_key="test-key",
        _client_factory=lambda **_kwargs: None,
    )

    result = provider._outcome_from_wire_failure(
        _wire_failure(kind),
        URL,
    )
    outcome = result.outcome

    expected_recoverability, expected_code = EXPECTED_CLASSIFICATION[kind]
    assert isinstance(outcome, ProviderTransportFailure)
    assert outcome.recoverability.value == expected_recoverability
    assert outcome.code == expected_code
    assert outcome.message == expected_message
    assert set(outcome.metadata) == expected_metadata_keys
    if expected_metadata_keys is BYTE_BOUND_METADATA_KEYS:
        assert outcome.metadata["observed_bytes"] == OBSERVED_BYTES


def test_every_wire_failure_kind_has_a_pinned_evidence_shape() -> None:
    """A new kind must fail here rather than fall through a branch."""
    assert set(EXPECTED_EVIDENCE_SHAPE) == set(WireFailureKind)
