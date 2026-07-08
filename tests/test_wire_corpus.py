"""Offline parser regression tests over live-captured wire bodies.

``tests/live/test_live_matrix.py`` writes each successful live call's
raw response body to
``data/wire-corpus/<provider_kind>_<endpoint_kind>.json``. This test
re-parses every captured body with :func:`parse_response` so a kernel
parser regression is caught without touching the network. The corpus
starts empty (no live run has happened yet on this machine) and the
test skips cleanly in that case so the offline suite stays green.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dr_providers import EndpointKind, ProviderKind, parse_response
from dr_providers.config import ProviderConfig, TokenLimitParameter

WIRE_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "wire-corpus"

TOKEN_LIMIT_PARAMETER_BY_PROVIDER: dict[ProviderKind, TokenLimitParameter] = {
    ProviderKind.OPENROUTER: TokenLimitParameter.MAX_COMPLETION_TOKENS,
    ProviderKind.OPENAI: TokenLimitParameter.MAX_COMPLETION_TOKENS,
    ProviderKind.GEMINI: TokenLimitParameter.MAX_COMPLETION_TOKENS,
}


def wire_corpus_files() -> list[Path]:
    if not WIRE_CORPUS_DIR.is_dir():
        return []
    return sorted(WIRE_CORPUS_DIR.glob("*.json"))


def _config_for_stem(stem: str) -> ProviderConfig:
    """Reconstruct a minimal config from a ``<provider>_<endpoint>`` stem.

    The corpus filename is the ground truth for which parser branch to
    exercise; the model name is irrelevant to parsing.
    """
    provider_kind, _, endpoint_kind = stem.partition("_")
    provider = ProviderKind(provider_kind)
    endpoint = EndpointKind(endpoint_kind)
    token_limit_parameter = (
        TokenLimitParameter.MAX_OUTPUT_TOKENS
        if endpoint is EndpointKind.RESPONSES
        else TOKEN_LIMIT_PARAMETER_BY_PROVIDER[provider]
    )
    return ProviderConfig(
        provider_kind=provider,
        endpoint_kind=endpoint,
        model="wire-corpus-replay",
        api_key_env="UNUSED_API_KEY",
        token_limit_parameter=token_limit_parameter,
    )


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

    response = parse_response(body, config=config)

    assert response.text.strip()
    assert response.provider_metadata == body
