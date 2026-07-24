"""Live provider verification matrix.

Marked ``live``; excluded from the default run via ``addopts = "-m
'not live'"``. Each case skips (not fails) when its API key env var
is unset, so ``uv run pytest -m live`` is safe to run without every
provider's key configured.

Every successful call writes its raw response body to
``data/wire-corpus/<provider>_<protocol>.json`` (pretty JSON,
overwritten each run). ``tests/test_wire_corpus.py`` re-parses those
bodies offline so a kernel parser regression is caught without
touching the network.

Temperature is set to 0.0 only for the openrouter and gemini cases.
All five presets declare ``temperature`` in ``supported_controls``,
so the kernel is willing to transport it everywhere, but the
``gpt-5-mini`` reasoning model used for the openai cases rejects it
at the wire level ("Unsupported parameter: 'temperature' is not
supported with this model") even though the config layer has no way
to know that ahead of time. Omitting it for openai keeps this matrix
green without papering over that provider-side restriction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dr_providers import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    GenerationControls,
    HttpProvider,
    MessageRole,
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    ReasoningEffort,
    Transcript,
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)

pytestmark = pytest.mark.live

WIRE_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "wire-corpus"

PROMPT = "Say hello in one word."
TOKEN_LIMIT = 2048

OPENROUTER_MODEL_ENV = "DR_LIVE_OPENROUTER_MODEL"
OPENAI_MODEL_ENV = "DR_LIVE_OPENAI_MODEL"
GEMINI_MODEL_ENV = "DR_LIVE_GEMINI_MODEL"

ANTHROPIC_MODEL_ENV = "DR_LIVE_ANTHROPIC_MODEL"

OPENROUTER_MODEL_DEFAULT = "openai/gpt-5-mini"
OPENAI_MODEL_DEFAULT = "gpt-5-mini"
GEMINI_MODEL_DEFAULT = "gemini-2.5-flash"
ANTHROPIC_MODEL_DEFAULT = "claude-haiku-4-5"


def _model(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


LIVE_CASES = [
    pytest.param(
        "OPENROUTER_API_KEY",
        lambda: openrouter_chat_config(
            model=_model(OPENROUTER_MODEL_ENV, OPENROUTER_MODEL_DEFAULT)
        ),
        True,
        id="openrouter_chat_completions",
    ),
    pytest.param(
        "OPENAI_API_KEY",
        lambda: openai_chat_config(
            model=_model(OPENAI_MODEL_ENV, OPENAI_MODEL_DEFAULT)
        ),
        False,
        id="openai_chat_completions",
    ),
    pytest.param(
        "OPENAI_API_KEY",
        lambda: openai_responses_config(
            model=_model(OPENAI_MODEL_ENV, OPENAI_MODEL_DEFAULT)
        ),
        False,
        id="openai_responses",
    ),
    pytest.param(
        "GEMINI_API_KEY",
        lambda: gemini_chat_config(
            model=_model(GEMINI_MODEL_ENV, GEMINI_MODEL_DEFAULT)
        ),
        True,
        id="gemini_chat_completions",
    ),
    pytest.param(
        "ANTHROPIC_API_KEY",
        lambda: anthropic_messages_config(
            model=_model(ANTHROPIC_MODEL_ENV, ANTHROPIC_MODEL_DEFAULT)
        ),
        True,
        id="anthropic_messages",
    ),
]


def _config_with_controls(
    config_factory, *, set_temperature: bool
) -> ProviderCallConfig:
    base = config_factory()
    return base.definition.materialize(
        controls=GenerationControls(
            temperature=0.0 if set_temperature else None,
            token_limit=TOKEN_LIMIT,
            reasoning=ReasoningEffort.LOW,
        )
    )


@pytest.mark.parametrize(
    ("api_key_env", "config_factory", "set_temperature"), LIVE_CASES
)
def test_live_matrix(
    api_key_env: str,
    config_factory,
    set_temperature: bool,  # noqa: FBT001
) -> None:
    if not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} is not set")

    config = _config_with_controls(
        config_factory, set_temperature=set_temperature
    )
    request = ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=PROMPT),)
        ),
    )
    kind = config.route.provider
    policy = ProviderTransportPolicy(
        api_key_env=str(DEFAULT_API_KEY_ENVS[kind]),
        base_url=str(DEFAULT_BASE_URLS[kind]),
    )

    with HttpProvider(policy=policy) as provider:
        outcome = provider.complete(request)

    assert isinstance(outcome, ProviderTransportResponse)
    assert outcome.text.strip()
    assert outcome.usage is not None

    _write_corpus_entry(config, outcome.raw_body)


def _write_corpus_entry(
    config: ProviderCallConfig, body: dict[str, object]
) -> None:
    WIRE_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    provider = config.route.provider.value
    protocol = config.route.protocol.value
    file_name = f"{provider}_{protocol}.json"
    (WIRE_CORPUS_DIR / file_name).write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n"
    )
