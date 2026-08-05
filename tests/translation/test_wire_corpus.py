from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dr_providers import (
    GenerationControls,
    ProviderCallConfig,
    ProviderKind,
    ProviderTransportResponse,
    parse_response,
)
from dr_providers.modeling.presets import (
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)
from dr_providers.modeling.route import Protocol

WIRE_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "wire-corpus"


def wire_corpus_files() -> list[Path]:
    if not WIRE_CORPUS_DIR.is_dir():
        return []
    return sorted(WIRE_CORPUS_DIR.glob("*.json"))


def _config_for_stem(stem: str) -> ProviderCallConfig:
    """Use the corpus filename as parser ground truth; ignore its model."""
    provider_kind, _, protocol = stem.partition("_")
    provider = ProviderKind(provider_kind)
    if protocol == Protocol.RESPONSES.value:
        return openai_responses_config(model="wire-corpus-replay")
    if protocol == Protocol.ANTHROPIC_MESSAGES.value:
        # Anthropic materialization requires a token limit; parsing ignores it.
        return anthropic_messages_config(
            model="wire-corpus-replay",
            controls=GenerationControls(token_limit=1),
        )
    factories = {
        ProviderKind.OPENROUTER: openrouter_chat_config,
        ProviderKind.OPENAI: openai_chat_config,
        ProviderKind.GEMINI: gemini_chat_config,
    }
    return factories[provider](model="wire-corpus-replay")


@pytest.mark.skipif(
    not wire_corpus_files(),
    reason=("data/wire-corpus/ is empty; use scripts/capture_live_corpus.py"),
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
    assert outcome.response_body == body
