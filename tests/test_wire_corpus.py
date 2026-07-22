"""Offline parser regression tests over live-captured wire bodies.

``tests/live/test_live_matrix.py`` writes each successful live call's
raw response body to
``data/wire-corpus/<provider>_<protocol>.json``. This test re-parses
every captured body with :func:`parse_response` so a kernel parser
regression is caught without touching the network. The corpus starts
empty (no live run has happened yet on this machine) and the test skips
cleanly in that case so the offline suite stays green.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dr_providers import (
    ProviderCallConfig,
    ProviderKind,
    ProviderTransportResponse,
    parse_response,
)
from dr_providers.config import (
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)
from dr_providers.route import Protocol

WIRE_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "wire-corpus"


def wire_corpus_files() -> list[Path]:
    if not WIRE_CORPUS_DIR.is_dir():
        return []
    return sorted(WIRE_CORPUS_DIR.glob("*.json"))


def _config_for_stem(stem: str) -> ProviderCallConfig:
    """Reconstruct a minimal config from a ``<provider>_<protocol>`` stem.

    The corpus filename is the ground truth for which parser branch to
    exercise; the model name is irrelevant to parsing.
    """
    provider_kind, _, protocol = stem.partition("_")
    provider = ProviderKind(provider_kind)
    if protocol == Protocol.RESPONSES.value:
        return openai_responses_config(model="wire-corpus-replay")
    if protocol == Protocol.ANTHROPIC_MESSAGES.value:
        return anthropic_messages_config(model="wire-corpus-replay")
    factories = {
        ProviderKind.OPENROUTER: openrouter_chat_config,
        ProviderKind.OPENAI: openai_chat_config,
        ProviderKind.GEMINI: gemini_chat_config,
    }
    return factories[provider](model="wire-corpus-replay")


@pytest.mark.skipif(
    not wire_corpus_files(),
    reason="data/wire-corpus/ is empty; run `uv run pytest -m live` first",
)
@pytest.mark.parametrize(
    "corpus_file",
    wire_corpus_files(),
    ids=[path.stem for path in wire_corpus_files()],
)
def test_wire_corpus_entry_parses(corpus_file: Path) -> None:
    body: dict[str, Any] = json.loads(corpus_file.read_text())
    config = _config_for_stem(corpus_file.stem)

    outcome = parse_response(body, config=config)

    assert isinstance(outcome, ProviderTransportResponse)
    assert outcome.text.strip()
    assert outcome.raw_body == body
