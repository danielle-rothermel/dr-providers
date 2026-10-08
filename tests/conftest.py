from __future__ import annotations

import pytest

from dr_providers import (
    ControlConstraints,
    LocalExecutionEvidence,
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


@pytest.fixture
def local_config() -> ProviderCallConfig:
    from dr_providers import LocalDevice, LocalDtype, huggingface_config

    return huggingface_config(
        model="example/model",
        revision="0123456789abcdef0123456789abcdef01234567",
        device=LocalDevice.CUDA,
        dtype=LocalDtype.FLOAT32,
        batch_size=1,
        max_sequence_length=1024,
    )


@pytest.fixture
def local_execution() -> LocalExecutionEvidence:
    from dr_providers import (
        Float32MatmulPrecision,
        LocalDevice,
        LocalDtype,
        LocalExecutionEvidence,
        Quantization,
    )

    return LocalExecutionEvidence(
        device=LocalDevice.CUDA,
        device_name="NVIDIA A100",
        dtype=LocalDtype.FLOAT32,
        float32_matmul_precision=Float32MatmulPrecision.HIGHEST,
        quantization=Quantization.NONE,
        revision_commit="0123456789abcdef0123456789abcdef01234567",
        torch_version="2.8.0",
        transformers_version="4.55.0",
        chat_template_applied=False,
        add_bos_token=True,
    )
