from __future__ import annotations

import pytest

from dr_providers import (
    LocalBackendFailure,
    RecoverabilityClass,
    failure_record,
)
from dr_providers.local.backend import encode_pair


def test_encode_pair_moves_whitespace_and_prefixes_empty_context() -> None:
    def encode(text: str) -> list[int]:
        return [ord(char) for char in text]

    assert encode_pair(
        context="a \t", continuation="b", encode=encode, prefix_token=9
    ) == ([97], [32, 9, 98])
    assert encode_pair(
        context="", continuation="b", encode=encode, prefix_token=9
    ) == ([9], [98])
    assert encode_pair(
        context=" \t", continuation="b", encode=encode, prefix_token=9
    ) == ([9], [32, 9, 98])


def test_encode_pair_slices_joint_encoding_at_context_length() -> None:
    encodings = {"a": [1], "bc": [2, 3], "abc": [4, 5]}
    assert encode_pair(
        context="a",
        continuation="bc",
        encode=encodings.__getitem__,
        prefix_token=9,
    ) == ([1], [5])


def test_encode_pair_rejects_zero_tokens_and_missing_prefix() -> None:
    with pytest.raises(ValueError, match="continuation tokens"):
        encode_pair(
            context="a",
            continuation="b",
            encode={"a": [1], "ab": [2]}.__getitem__,
            prefix_token=9,
        )
    with pytest.raises(ValueError, match="BOS or EOS"):
        encode_pair(
            context="",
            continuation="b",
            encode=lambda _: [1],
            prefix_token=None,
        )


def test_local_failure_rejects_nonlocal_code_and_wrong_recoverability() -> (
    None
):
    with pytest.raises(ValueError, match="local code"):
        LocalBackendFailure(
            failure_record(
                code="unknown",
                recoverability=RecoverabilityClass.PERMANENT,
                message="failure",
            )
        )
    with pytest.raises(ValueError, match="recoverability mismatch"):
        LocalBackendFailure(
            failure_record(
                code="local_out_of_memory",
                recoverability=RecoverabilityClass.PERMANENT,
                message="failure",
            )
        )
