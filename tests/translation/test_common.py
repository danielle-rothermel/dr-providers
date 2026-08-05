"""Shared response-translation helper tests."""

from typing import Any

import pytest

from dr_providers import TokenUsage, token_usage_from_body


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                }
            },
            TokenUsage(
                prompt_tokens=3,
                completion_tokens=5,
                total_tokens=8,
                reasoning_tokens=2,
            ),
        ),
        (
            {
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 6,
                    "output_tokens_details": {"reasoning_tokens": 3},
                }
            },
            TokenUsage(
                prompt_tokens=4,
                completion_tokens=6,
                total_tokens=10,
                reasoning_tokens=3,
            ),
        ),
        (
            {"usage": {"input_tokens": 7, "output_tokens": 11}},
            TokenUsage(
                prompt_tokens=7,
                completion_tokens=11,
                total_tokens=18,
            ),
        ),
    ],
    ids=["chat_completions", "responses", "anthropic_messages"],
)
def test_token_usage_provider_shapes(
    body: dict[str, Any], expected: TokenUsage
) -> None:
    assert token_usage_from_body(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"usage": {}},
        {"usage": []},
        {
            "usage": {
                "prompt_tokens": True,
                "completion_tokens": "5",
                "total_tokens": 8.0,
                "completion_tokens_details": {"reasoning_tokens": False},
            }
        },
    ],
    ids=["absent", "empty", "not_mapping", "wrong_types"],
)
def test_token_usage_ignores_absent_or_non_integer_fields(
    body: dict[str, Any],
) -> None:
    assert token_usage_from_body(body) is None
