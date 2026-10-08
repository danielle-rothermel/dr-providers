"""Real CPU execution, opt-in and collected without optional dependencies."""

from __future__ import annotations

import math
import re
from threading import Event
from typing import TYPE_CHECKING

import pytest

from dr_providers import (
    AcceptAllSemanticResponseClassifier,
    GenerationControls,
    LocalBackendFailure,
    LocalDevice,
    LocalDtype,
    LocalModelProvider,
    MessageRole,
    PromptMessage,
    PromptRendering,
    ProviderCallConfig,
    ProviderCallOutcomeKind,
    ProviderCallState,
    ProviderInvocationOutcome,
    ProviderScoreRequest,
    ProviderStopReason,
    RecoverabilityClass,
    StandardProviderCallRetryPolicy,
    Transcript,
    TransformersBackend,
    huggingface_config,
    run_local_provider_call,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

MODEL = "hf-internal-testing/tiny-random-gpt2"
REVISION = "71034c5d8bde858ff824298bdedc65515b97d2b9"
pytestmark = pytest.mark.local_model


def _config(
    *,
    maximum: int = 512,
    rendering: PromptRendering = PromptRendering.FLAT_TEXT,
) -> ProviderCallConfig:
    return huggingface_config(
        model=MODEL,
        revision=REVISION,
        device=LocalDevice.CPU,
        dtype=LocalDtype.FLOAT32,
        batch_size=2,
        max_sequence_length=maximum,
        prompt_rendering=rendering,
    )


@pytest.fixture(scope="module")
def backend() -> Iterator[TransformersBackend]:
    with_dependencies = pytest.importorskip("torch")
    # A tiny model should not pay large-host thread-pool overhead for each row.
    previous = with_dependencies.get_num_threads()
    with_dependencies.set_num_threads(1)
    loaded = TransformersBackend(_config())
    try:
        yield loaded
    finally:
        loaded.close()
        with_dependencies.set_num_threads(previous)


def test_load_and_execution_record(backend: TransformersBackend) -> None:
    import torch
    import transformers

    execution = backend.execution()
    assert execution.device is LocalDevice.CPU
    assert execution.dtype is LocalDtype.FLOAT32
    assert execution.add_bos_token is False
    assert execution.device_name is None
    assert execution.revision_commit == REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", execution.revision_commit)
    assert execution.torch_version == str(torch.__version__)
    assert execution.transformers_version == transformers.__version__


@pytest.mark.parametrize(
    "context", ["The answer is", "The answer is ", "", " \t"]
)
def test_batched_scores_match_independent_unbatched_reference(
    backend: TransformersBackend,
    context: str,
) -> None:
    import torch

    model, tokenizer = backend._loaded()
    continuations = (" yes", " no", " maybe")
    scores = backend.score(
        context=context, continuations=continuations, token_logprobs=True
    )
    for continuation, score in zip(continuations, scores, strict=True):
        # Independent encoding and indexing, without production encode_pair.
        trailing = len(context) - len(context.rstrip())
        adjusted_context = context[:-trailing] if trailing else context
        adjusted_continuation = (
            context[-trailing:] + continuation if trailing else continuation
        )
        if adjusted_context:
            context_ids = tokenizer.encode(
                adjusted_context, add_special_tokens=False
            )
            whole = tokenizer.encode(
                adjusted_context + adjusted_continuation,
                add_special_tokens=False,
            )
            targets = whole[len(context_ids) :]
        else:
            context_ids = [
                tokenizer.bos_token_id
                if tokenizer.bos_token_id is not None
                else tokenizer.eos_token_id
            ]
            targets = tokenizer.encode(
                adjusted_continuation, add_special_tokens=False
            )
        tokens = context_ids + targets
        with torch.inference_mode():
            logits = (
                model(torch.tensor([tokens[:-1]]), use_cache=False)
                .logits[0]
                .float()
            )
        expected = [
            float(
                torch.log_softmax(
                    logits[len(context_ids) - 1 + index], dim=-1
                )[target].item()
            )
            for index, target in enumerate(targets)
        ]
        assert score.log_likelihood == pytest.approx(
            math.fsum(expected), abs=1e-5
        )
        assert score.token_count == len(targets)
        assert score.input_tokens == len(tokens) - 1
        assert score.token_logprobs == pytest.approx(expected, abs=1e-5)
        assert score.token_logprobs is not None
        assert sum(score.token_logprobs) == pytest.approx(
            score.log_likelihood, abs=1e-5
        )
    assert backend.execution().chat_template_applied is False


def test_generate_fixed_defaults_seed_and_prompt_usage(
    backend: TransformersBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = backend._loaded()
    transcript = Transcript(
        messages=(PromptMessage(role=MessageRole.USER, content="Hello"),)
    )
    # Model defaults must not activate beams, penalties, or minimum generation.
    monkeypatch.setattr(model.generation_config, "num_beams", 3)
    monkeypatch.setattr(model.generation_config, "repetition_penalty", 2.0)
    monkeypatch.setattr(model.generation_config, "min_new_tokens", 20)
    result = backend.generate(
        transcript=transcript,
        rendering=PromptRendering.FLAT_TEXT,
        controls=GenerationControls(token_limit=4),
    )
    assert result.text
    assert result.completion_tokens == 4
    assert result.stop_reason is ProviderStopReason.LENGTH
    assert result.prompt_tokens == len(
        tokenizer.encode("Hello", add_special_tokens=False)
    )
    controls = GenerationControls(token_limit=6, temperature=1.0, seed=42)
    first = backend.generate(
        transcript=transcript,
        rendering=PromptRendering.FLAT_TEXT,
        controls=controls,
    )
    second = backend.generate(
        transcript=transcript,
        rendering=PromptRendering.FLAT_TEXT,
        controls=controls,
    )
    assert first == second
    assert model.generation_config.top_k == 0
    assert model.generation_config.top_p == 1.0
    assert model.generation_config.num_beams != 3
    assert model.generation_config.repetition_penalty != 2.0
    assert model.generation_config.min_new_tokens != 20


def test_sequence_limits_and_empty_generation(
    backend: TransformersBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        backend,
        "_spec",
        backend._spec.model_copy(update={"max_sequence_length": 2}),
    )
    with pytest.raises(LocalBackendFailure) as raised:
        backend.score(
            context="this context is too long",
            continuations=(" answer",),
            token_logprobs=False,
        )
    assert raised.value.failure.code == "local_sequence_too_long"
    with pytest.raises(LocalBackendFailure) as raised:
        backend.generate(
            transcript=Transcript(
                messages=(
                    PromptMessage(
                        role=MessageRole.USER, content="long prompt"
                    ),
                )
            ),
            rendering=PromptRendering.FLAT_TEXT,
            controls=GenerationControls(),
        )
    assert raised.value.failure.code == "local_sequence_too_long"
    with pytest.raises(ValueError, match="nonempty tokenized prompt"):
        backend.generate(
            transcript=Transcript(messages=()),
            rendering=PromptRendering.FLAT_TEXT,
            controls=GenerationControls(),
        )


def test_missing_chat_template_is_a_load_failure() -> None:
    with pytest.raises(LocalBackendFailure) as raised:
        TransformersBackend(_config(rendering=PromptRendering.ROLE_MESSAGES))
    assert raised.value.failure.code == "local_chat_template_missing"


def test_nonfinite_and_oom_are_expected_failures(
    backend: TransformersBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    model, _ = backend._loaded()
    original = model.forward

    def nonfinite(*args: object, **kwargs: object) -> object:
        output = original(*args, **kwargs)
        output.logits.fill_(float("nan"))
        return output

    monkeypatch.setattr(model, "forward", nonfinite)
    with pytest.raises(LocalBackendFailure) as raised:
        backend.score(
            context="ctx", continuations=(" answer",), token_logprobs=False
        )
    assert raised.value.failure.code == "local_non_finite_score"
    released = []
    monkeypatch.setattr(
        backend, "release_memory", lambda: released.append(True)
    )

    def oom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise torch.cuda.OutOfMemoryError("controlled allocation failure")

    monkeypatch.setattr(model, "forward", oom)
    with pytest.raises(LocalBackendFailure) as raised:
        backend.score(
            context="ctx", continuations=(" answer",), token_logprobs=False
        )
    assert raised.value.failure.code == "local_out_of_memory"
    assert released == [True]


def test_changed_precision_is_refused(backend: TransformersBackend) -> None:
    import torch

    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with pytest.raises(ValueError, match="precision changed"):
            backend.score(
                context="ctx", continuations=(" answer",), token_logprobs=False
            )
    finally:
        torch.set_float32_matmul_precision(previous)


def test_real_provider_lifecycle_and_memory_release() -> None:
    config = _config()
    classifier = AcceptAllSemanticResponseClassifier()
    with LocalModelProvider(config=config) as provider:
        result = run_local_provider_call(
            provider=provider,
            cancellation=Event(),
            classifier=classifier,
            state=ProviderCallState.initial(
                request=ProviderScoreRequest(
                    config=config, context="ctx", continuations=(" answer",)
                ),
                retry_policy=StandardProviderCallRetryPolicy(),
                classifier_identifier=classifier.identifier,
            ),
        )
        provider.release_memory()
    provider.close()
    assert result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    evidence = result.completed_invocations[0].observation.evidence
    assert evidence.local_execution is not None
    assert evidence.local_execution.revision_commit == REVISION
    assert evidence.wall_time_seconds is not None
    assert evidence.wall_time_seconds > 0


def test_boundary_merge_records_failure_and_next_request_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    classifier = AcceptAllSemanticResponseClassifier()
    backend = TransformersBackend(config)
    with LocalModelProvider(config=config, backend=backend) as provider:
        model, tokenizer = backend._loaded()

        def no_forward(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError(
                "empty continuation must fail before inference"
            )

        with monkeypatch.context() as patch:
            # First candidate is scoreable; the second merges into its context.
            patch.setattr(
                tokenizer,
                "encode",
                lambda text, **_kwargs: {"a": [1], "abc": [1, 3], "ab": [2]}[
                    text
                ],
            )
            patch.setattr(model, "forward", no_forward)
            failed = run_local_provider_call(
                provider=provider,
                cancellation=Event(),
                classifier=classifier,
                state=ProviderCallState.initial(
                    request=ProviderScoreRequest(
                        config=config, context="a", continuations=("bc", "b")
                    ),
                    retry_policy=StandardProviderCallRetryPolicy(),
                    classifier_identifier=classifier.identifier,
                ),
            )
        assert len(failed.completed_invocations) == 1
        assert (
            failed.outcome.kind is ProviderCallOutcomeKind.INVOCATION_OUTCOME
        )
        assert failed.outcome.invocation_outcome is (
            ProviderInvocationOutcome.PERMANENT_PROVIDER_OR_TRANSPORT_FAILURE
        )
        evidence = failed.completed_invocations[0].observation.evidence
        assert evidence.score_response is None
        assert evidence.local_execution is not None
        assert evidence.failure is not None
        assert evidence.failure.code == "local_empty_continuation"
        assert evidence.failure.recoverability is RecoverabilityClass.PERMANENT
        assert evidence.failure.traceback is None
        accepted = run_local_provider_call(
            provider=provider,
            cancellation=Event(),
            classifier=classifier,
            state=ProviderCallState.initial(
                request=ProviderScoreRequest(
                    config=config, context="ctx", continuations=(" answer",)
                ),
                retry_policy=StandardProviderCallRetryPolicy(),
                classifier_identifier=classifier.identifier,
            ),
        )
        assert accepted.outcome.kind is ProviderCallOutcomeKind.ACCEPTED


def test_backend_close_is_idempotent() -> None:
    backend = TransformersBackend(_config())
    backend.release_memory()
    backend.close()
    backend.close()
    with pytest.raises(ValueError, match="closed"):
        backend.score(
            context="ctx", continuations=(" answer",), token_logprobs=False
        )


def test_role_rendering_evidence_is_per_invocation(
    backend: TransformersBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer = backend._loaded()
    monkeypatch.setattr(backend, "_rendering", PromptRendering.ROLE_MESSAGES)
    monkeypatch.setattr(
        tokenizer,
        "chat_template",
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %} assistant:{% endif %}",
    )
    backend.generate(
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content="Hello"),)
        ),
        rendering=PromptRendering.ROLE_MESSAGES,
        controls=GenerationControls(token_limit=2),
    )
    assert backend.execution().chat_template_applied is True
    backend.score(
        context="ctx", continuations=(" answer",), token_logprobs=False
    )
    assert backend.execution().chat_template_applied is False


@pytest.mark.parametrize("device", [LocalDevice.CUDA, LocalDevice.MPS])
def test_unavailable_device_checked_before_loading(
    device: LocalDevice,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    data = _config().definition.model_dump()
    data["local"]["device"] = device
    config = type(_config().definition).model_validate(data).materialize()
    with pytest.raises(LocalBackendFailure) as raised:
        TransformersBackend(config)
    assert raised.value.failure.code == "local_device_unavailable"


def test_missing_local_path_is_a_load_failure() -> None:
    data = _config().definition.model_dump()
    data["route"]["model"] = "/nonexistent/dr-providers-test-model"
    config = type(_config().definition).model_validate(data).materialize()
    with pytest.raises(LocalBackendFailure) as raised:
        TransformersBackend(config)
    assert raised.value.failure.code == "local_model_not_found"


def test_eos_stop_and_unset_seed(
    backend: TransformersBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    model, _ = backend._loaded()

    def no_reseed(_: int) -> None:
        raise AssertionError("unset seed must not reseed")

    def eos_output(**kwargs: object) -> object:
        del kwargs
        # This pinned model uses EOS 98; its tokenizer EOS is separately 0.
        return torch.tensor([[1, 98]])

    monkeypatch.setattr(torch, "manual_seed", no_reseed)
    monkeypatch.setattr(model, "generate", eos_output)
    result = backend.generate(
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content="a"),)
        ),
        rendering=PromptRendering.FLAT_TEXT,
        controls=GenerationControls(token_limit=2),
    )
    assert result.stop_reason is ProviderStopReason.STOP
    assert result.completion_tokens == 1
    assert result.text == "�"
