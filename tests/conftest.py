from __future__ import annotations

import pytest

from dr_providers import (
    ControlConstraints,
    ModelRoute,
    Protocol,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderCallKind,
    ProviderKind,
    ProviderScoreRequest,
    TokenLimitParameter,
)


@pytest.fixture
def score_config() -> ProviderCallConfig:
    return ProviderCallDefinition(
        definition_id="test.score",
        supported_kinds=frozenset({ProviderCallKind.SCORE}),
        route=ModelRoute(
            provider=ProviderKind.OPENAI,
            protocol=Protocol.CHAT_COMPLETIONS,
            model="m",
        ),
        constraints=ControlConstraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
        ),
    ).materialize()


@pytest.fixture
def score_request(score_config: ProviderCallConfig) -> ProviderScoreRequest:
    return ProviderScoreRequest(
        config=score_config, context="context", continuations=("answer",)
    )
