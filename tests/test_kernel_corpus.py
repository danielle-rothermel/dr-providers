"""Corpus-backed parser regression tests for the kernel.

Every kernel parser change is checked against
``data/kernel-corpus/responses.jsonl`` — real response shapes (grown
from whetstone-ai's boundary fixtures) with expected parses. Add
entries when new shapes are observed; never edit expected values to
match a parser change without a recorded decision.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from dr_providers import (
    PermanentProviderError,
    openai_chat_config,
    openai_responses_config,
    parse_response,
)

CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "kernel-corpus"
    / "responses.jsonl"
)


def corpus_entries() -> list[dict[str, Any]]:
    entries = []
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        entries.extend(json.loads(line) for line in handle if line.strip())
    return entries


@pytest.mark.parametrize(
    "entry",
    corpus_entries(),
    ids=[entry["name"] for entry in corpus_entries()],
)
def test_corpus_entry_parses_to_ground_truth(entry: dict[str, Any]) -> None:
    model = entry["config_model"]
    if entry["endpoint_kind"] == "chat_completions":
        config = openai_chat_config(model=model)
    else:
        config = openai_responses_config(model=model)

    expected_failure = entry.get("expected_failure")
    if expected_failure is not None:
        with pytest.raises(PermanentProviderError) as exc_info:
            parse_response(entry["body"], config=config)
        failure = exc_info.value.failure
        assert failure.code == expected_failure["code"]
        assert failure.retryable is False
        assert failure.metadata["diagnostics"] == _expected_diagnostics(entry)
        return

    response = parse_response(entry["body"], config=config)
    expected = entry["expected"]

    assert response.text == expected["text"]
    assert response.finish_reason == expected["finish_reason"]
    assert response.model == expected["model"]
    assert response.response_id == expected["response_id"]
    if expected["usage"] is None:
        assert response.usage is None
    else:
        assert response.usage is not None
        assert response.usage.model_dump() == expected["usage"]
    if expected["cost"] is None:
        assert response.cost is None
    else:
        assert response.cost is not None
        assert response.cost.total_cost == expected["cost"]
    assert response.provider_metadata == entry["body"]
    if entry["endpoint_kind"] == "responses":
        assert response.diagnostics is not None
        assert response.diagnostics.model_dump() == _expected_diagnostics(
            entry
        )
    else:
        assert response.diagnostics is None


def _expected_diagnostics(entry: dict[str, Any]) -> dict[str, Any]:
    expected = dict(entry["expected_diagnostics"])
    response_id = entry["body"].get("id")
    expected["response_id_hash"] = (
        sha256(response_id.encode()).hexdigest()[:16]
        if isinstance(response_id, str)
        else None
    )
    return expected
