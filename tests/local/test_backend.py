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


@pytest.mark.parametrize("context", ["a", ""])
def test_encode_pair_records_empty_tokenized_continuation(
    context: str,
) -> None:
    with pytest.raises(LocalBackendFailure) as raised:
        encode_pair(
            context=context,
            continuation="b",
            encode={"a": [1], "ab": [2], "b": []}.__getitem__,
            prefix_token=9,
        )
    assert raised.value.failure.code == "local_empty_continuation"
    assert raised.value.failure.recoverability is RecoverabilityClass.PERMANENT


def test_encode_pair_rejects_invalid_direct_usage() -> None:
    with pytest.raises(ValueError, match="nonempty continuation"):
        encode_pair(
            context="a ", continuation="", encode=lambda _: [1], prefix_token=9
        )
    with pytest.raises(ValueError, match="context tokens"):
        encode_pair(
            context="a", continuation="b", encode=lambda _: [], prefix_token=9
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


@pytest.mark.parametrize(
    ("context", "expected_context", "expected_targets"),
    [
        ("a", [94, 97], [98]),
        ("", [94], [98]),
        (" ", [94], [32, 98]),
        ("^a", [94, 97], [98]),
    ],
)
def test_encode_pair_bos_only_prefixes_context_once(
    context: str, expected_context: list[int], expected_targets: list[int]
) -> None:
    context_ids, targets = encode_pair(
        context=context,
        continuation="b",
        encode=lambda text: [ord(char) for char in text],
        prefix_token=94,
        add_bos_token=True,
    )
    assert context_ids == expected_context
    assert targets == expected_targets
