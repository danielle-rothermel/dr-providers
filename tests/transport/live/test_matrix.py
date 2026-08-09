from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import pytest

from dr_providers import (
    GenerationControls,
    HttpProvider,
    MessageRole,
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
    ReasoningEffort,
    Transcript,
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    policy_for,
)
from dr_providers.lifecycle import (
    AcceptAllSemanticResponseClassifier,
    ProviderCallOutcomeKind,
    ProviderCallState,
    StandardProviderCallRetryPolicy,
    run_local_provider_call,
)
from scripts.live_matrix_support import (
    CAPTURE_DIR_ENV,
    LIVE_CASES,
    LiveCase,
    require_external_capture_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    ConfigFactory = Callable[[GenerationControls], ProviderCallConfig]

pytestmark = pytest.mark.live

PROMPT = "Say hello in one word."
TOKEN_LIMIT = 2048

OPENROUTER_MODEL_ENV = "DR_LIVE_OPENROUTER_MODEL"
OPENAI_MODEL_ENV = "DR_LIVE_OPENAI_MODEL"
GEMINI_MODEL_ENV = "DR_LIVE_GEMINI_MODEL"

ANTHROPIC_MODEL_ENV = "DR_LIVE_ANTHROPIC_MODEL"

OPENROUTER_MODEL_DEFAULT = "openai/gpt-5-mini"
OPENAI_MODEL_DEFAULT = "gpt-5-mini"
GEMINI_MODEL_DEFAULT = "gemini-2.5-flash"
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-6"


def _model(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


CONFIG_FACTORIES: dict[str, ConfigFactory] = {
    "openrouter_chat_completions": lambda controls: openrouter_chat_config(
        model=_model(OPENROUTER_MODEL_ENV, OPENROUTER_MODEL_DEFAULT),
        controls=controls,
    ),
    "openai_chat_completions": lambda controls: openai_chat_config(
        model=_model(OPENAI_MODEL_ENV, OPENAI_MODEL_DEFAULT),
        controls=controls,
    ),
    "openai_responses": lambda controls: openai_responses_config(
        model=_model(OPENAI_MODEL_ENV, OPENAI_MODEL_DEFAULT),
        controls=controls,
    ),
    "gemini_chat_completions": lambda controls: gemini_chat_config(
        model=_model(GEMINI_MODEL_ENV, GEMINI_MODEL_DEFAULT),
        controls=controls,
    ),
    "anthropic_messages": lambda controls: anthropic_messages_config(
        model=_model(ANTHROPIC_MODEL_ENV, ANTHROPIC_MODEL_DEFAULT),
        controls=controls,
    ),
}

TEMPERATURE_CASES = {
    "openrouter_chat_completions",
    "gemini_chat_completions",
    "anthropic_messages",
}

LIVE_PARAMETERS = [
    pytest.param(
        case,
        CONFIG_FACTORIES[case.case_id],
        case.case_id in TEMPERATURE_CASES,
        id=case.case_id,
    )
    for case in LIVE_CASES
]


def _config_with_controls(
    config_factory, *, set_temperature: bool
) -> ProviderCallConfig:
    controls = GenerationControls(
        temperature=0.0 if set_temperature else None,
        token_limit=TOKEN_LIMIT,
        reasoning=ReasoningEffort.LOW,
    )
    return config_factory(controls)


@pytest.mark.parametrize(
    ("case", "config_factory", "set_temperature"), LIVE_PARAMETERS
)
def test_live_matrix(
    case: LiveCase,
    config_factory,
    set_temperature: bool,  # noqa: FBT001
) -> None:
    if not os.environ.get(case.credential_env):
        pytest.fail(f"{case.credential_env} is not set")

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
    policy = policy_for(
        kind,
        max_connections=1,
        max_keepalive_connections=1,
    )
    classifier = AcceptAllSemanticResponseClassifier()
    state = ProviderCallState.initial(
        request=request,
        retry_policy=StandardProviderCallRetryPolicy(),
        classifier_identifier=classifier.identifier,
    )

    with HttpProvider(policy=policy) as provider:
        result = run_local_provider_call(
            provider=provider,
            state=state,
            classifier=classifier,
            cancellation=Event(),
        )

    assert result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    response = result.completed_invocations[-1].observation.evidence.response
    assert response is not None
    assert response.text.strip()
    assert response.usage is not None

    _stage_capture(case, response.response_body)


def _stage_capture(case: LiveCase, body: dict[str, object]) -> None:
    capture_dir_value = os.environ.get(CAPTURE_DIR_ENV)
    if capture_dir_value is None:
        return
    capture_dir = require_external_capture_dir(Path(capture_dir_value))
    capture_dir.mkdir(parents=True, exist_ok=True)
    destination = capture_dir / case.corpus_file
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
