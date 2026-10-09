from __future__ import annotations

import math
import sys

import pytest

from dr_providers import (
    LocalDevice,
    LocalDtype,
    LocalModelProvider,
    PromptRendering,
    ProviderCallConfig,
    ProviderScoreRequest,
    ProviderScoreResponse,
    TransformersBackend,
    huggingface_config,
)

pytestmark = pytest.mark.local_model

GPT2 = (
    "hf-internal-testing/tiny-random-gpt2",
    "71034c5d8bde858ff824298bdedc65515b97d2b9",
)
OLMO = (
    "hf-internal-testing/tiny-random-OlmoForCausalLM",
    "22d63f3f78574f9d7061b9d1f97ba31c1bc26200",
)
MODELS = pytest.mark.parametrize(
    ("model", "revision"), [GPT2, OLMO], ids=["gpt2", "olmo"]
)
CONTEXT = "Question: What is the capital of France?\nAnswer:"
CONTINUATIONS = (" Paris", " Lyon, Marseille, and Toulouse")
TOLERANCE = 1e-3


def _config(
    model: str, revision: str, device: LocalDevice
) -> ProviderCallConfig:
    return huggingface_config(
        model=model,
        revision=revision,
        device=device,
        dtype=LocalDtype.FLOAT32,
        batch_size=2,
        max_sequence_length=128,
        prompt_rendering=PromptRendering.FLAT_TEXT,
    )


def _score(config: ProviderCallConfig) -> ProviderScoreResponse:
    with LocalModelProvider(config=config) as provider:
        evidence = provider.invoke(
            ProviderScoreRequest(
                config=config,
                context=CONTEXT,
                continuations=CONTINUATIONS,
                token_logprobs=True,
            )
        )
    response = evidence.score_response
    assert response is not None, evidence.failure
    assert response.warnings == ()
    spec = config.definition.local
    assert spec is not None
    assert evidence.local_execution is not None
    assert evidence.local_execution.revision_commit == spec.revision
    return response


@MODELS
def test_native_architecture_loads_without_remote_code(
    model: str, revision: str
) -> None:
    pytest.importorskip("torch")
    backend = TransformersBackend(_config(model, revision, LocalDevice.CPU))
    try:
        loaded, _ = backend._loaded()
        assert type(loaded).__module__.startswith("transformers.models.")
        assert "hf_olmo" not in sys.modules
    finally:
        backend.close()


@MODELS
def test_mps_scores_match_cpu(model: str, revision: str) -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    cpu = _score(_config(model, revision, LocalDevice.CPU)).scores
    mps = _score(_config(model, revision, LocalDevice.MPS)).scores
    for cpu_score, mps_score in zip(cpu, mps, strict=True):
        assert math.isfinite(mps_score.log_likelihood)
        assert mps_score.token_count == cpu_score.token_count
        assert mps_score.log_likelihood == pytest.approx(
            cpu_score.log_likelihood, abs=TOLERANCE
        )
        assert mps_score.token_logprobs == pytest.approx(
            cpu_score.token_logprobs, abs=TOLERANCE
        )
    cpu_gap = cpu[0].log_likelihood - cpu[1].log_likelihood
    if abs(cpu_gap) > 2 * TOLERANCE:
        mps_gap = mps[0].log_likelihood - mps[1].log_likelihood
        assert (mps_gap > 0) == (cpu_gap > 0)
