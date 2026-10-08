from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dr_providers import (
    ContinuationScore,
    ProviderScoreResponse,
    ProviderTransportFailure,
    ProviderTransportResponse,
    RecoverabilityClass,
    is_failure,
    is_response,
    is_score_response,
)


@pytest.mark.parametrize(
    "override",
    [
        {"log_likelihood": 1.0},
        {"log_likelihood": 5e-324},
        {"log_likelihood": float("nan")},
        {"log_likelihood": float("inf")},
        {"log_likelihood": float("-inf")},
        {"log_likelihood": "-1.0"},
        {"log_likelihood": True},
        {"token_count": 0},
        {"token_count": True},
        {"char_count": 0},
        {"char_count": "1"},
        {"token_logprobs": (1.0,)},
        {"token_logprobs": (5e-324,)},
        {"token_logprobs": (float("nan"),)},
        {"token_logprobs": (float("inf"),)},
        {"token_logprobs": (float("-inf"),)},
        {"token_logprobs": ("-1.0",)},
        {"token_logprobs": (True,)},
        {"token_logprobs": ()},
        {"token_logprobs": (-1.0, -2.0)},
    ],
)
def test_continuation_score_rejects_invalid_values(
    override: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ContinuationScore.model_validate(
            {
                "log_likelihood": -1.0,
                "token_count": 1,
                "char_count": 1,
                **override,
            }
        )


def test_score_accepts_zero_log_probabilities() -> None:
    score = ContinuationScore(
        log_likelihood=0.0,
        token_count=1,
        char_count=1,
        token_logprobs=(0.0,),
    )
    assert score.log_likelihood == 0.0
    assert score.token_logprobs == (0.0,)
    assert (
        ContinuationScore.model_validate_json(score.model_dump_json()) == score
    )


def test_score_does_not_require_exact_floating_point_sum() -> None:
    score = ContinuationScore(
        log_likelihood=-1.0,
        token_count=1,
        char_count=1,
        token_logprobs=(-1.001,),
    )
    assert score.token_logprobs == (-1.001,)
    assert score.log_likelihood == -1.0


def test_score_response_requires_nonempty_scores() -> None:
    with pytest.raises(ValidationError):
        ProviderScoreResponse(scores=())


def test_transport_outcome_guards_distinguish_all_three_variants() -> None:
    generate = ProviderTransportResponse(text="a")
    score = ProviderScoreResponse(
        scores=(
            ContinuationScore(
                log_likelihood=-1.0, token_count=1, char_count=1
            ),
        )
    )
    failure = ProviderTransportFailure(
        recoverability=RecoverabilityClass.PERMANENT, message="failure"
    )
    assert (
        is_response(generate),
        is_score_response(generate),
        is_failure(generate),
    ) == (True, False, False)
    assert (
        is_response(score),
        is_score_response(score),
        is_failure(score),
    ) == (False, True, False)
    assert (
        is_response(failure),
        is_score_response(failure),
        is_failure(failure),
    ) == (False, False, True)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("local_out_of_memory", RecoverabilityClass.RESOURCE_EXHAUSTION),
        ("local_device_unavailable", RecoverabilityClass.PERMANENT),
        ("local_model_not_found", RecoverabilityClass.PERMANENT),
        ("local_sequence_too_long", RecoverabilityClass.PERMANENT),
    ],
)
def test_local_failure_codes_require_fixed_recoverability(
    code: str, expected: RecoverabilityClass
) -> None:
    for recoverability in RecoverabilityClass:
        if recoverability is expected:
            failure = ProviderTransportFailure(
                code=code,
                recoverability=recoverability,
                message="local failure",
            )
            assert (
                ProviderTransportFailure.model_validate_json(
                    failure.model_dump_json()
                )
                == failure
            )
        else:
            with pytest.raises(
                ValueError, match=r"requires .* recoverability"
            ):
                ProviderTransportFailure(
                    code=code,
                    recoverability=recoverability,
                    message="local failure",
                )
