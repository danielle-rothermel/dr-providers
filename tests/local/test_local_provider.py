from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from dr_providers import (
    AcceptAllSemanticResponseClassifier,
    CustomProviderCallRetryPolicy,
    FakeLocalBackend,
    GenerationControls,
    GenerationOutput,
    LocalBackend,
    LocalBackendFailure,
    LocalExecutionEvidence,
    LocalModelProvider,
    MessageRole,
    OffloadingProvider,
    PromptMessage,
    Provider,
    ProviderCallConfig,
    ProviderCallKind,
    ProviderCallOutcomeKind,
    ProviderCallState,
    ProviderGenerateRequest,
    ProviderInvocationOutcome,
    ProviderScoreRequest,
    ProviderStopReason,
    RecoverabilityClass,
    ScoredContinuation,
    StandardProviderCallRetryPolicy,
    Transcript,
    failure_record,
    openai_chat_config,
    run_local_provider_call,
    run_local_provider_call_async,
)
from dr_providers.outcomes.models import LOCAL_FAILURE_RECOVERABILITY

WATCHDOG = 5
GENERATED = GenerationOutput("answer", 3, 2, ProviderStopReason.LENGTH)


def _request(config: ProviderCallConfig) -> ProviderGenerateRequest:
    return ProviderGenerateRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content="hi"),)
        ),
    )


def test_generate_response_and_wall_time_identity(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(execution=local_execution, results=[GENERATED])
    with LocalModelProvider(config=local_config, backend=backend) as provider:
        request = _request(local_config)
        evidence = provider.invoke(request)
        assert evidence.kind is ProviderCallKind.GENERATE
        assert evidence.local_execution == local_execution
        assert evidence.policy_identity is None
        assert evidence.http_request is None
        assert evidence.wall_time_seconds is not None
        assert evidence.wall_time_seconds >= 0
        response = evidence.response
        assert response is not None
        assert response.text == "answer"
        assert response.usage is not None
        assert response.usage.model_dump(exclude_none=True) == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        assert response.cost is not None
        assert response.cost.total_cost == 0.0
        assert response.model == "example/model"
        assert response.stop_reason is ProviderStopReason.LENGTH
        assert provider.invoke(request).identity_hash == evidence.identity_hash
        assert (
            evidence.model_copy(
                update={"wall_time_seconds": 42.0}
            ).identity_hash
            == evidence.identity_hash
        )
        assert len(backend.calls) == 2
        provider.release_memory()
        assert backend.release_memory_calls == 1
    provider.close()
    assert backend.close_calls == 1


def test_score_preserves_order_original_chars_detail_and_input_usage(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(
        execution=local_execution,
        results=[
            (
                ScoredContinuation(-1.0, 2, (-0.4, -0.6), 5),
                ScoredContinuation(-2.0, 1, (-2.0,), 4),
            )
        ],
    )
    with LocalModelProvider(config=local_config, backend=backend) as provider:
        evidence = provider.invoke(
            ProviderScoreRequest(
                config=local_config,
                context="a ",
                continuations=("bc", "d"),
                token_logprobs=True,
            )
        )
    assert evidence.kind is ProviderCallKind.SCORE
    response = evidence.score_response
    assert response is not None
    assert [score.char_count for score in response.scores] == [2, 1]
    assert [score.log_likelihood for score in response.scores] == [-1.0, -2.0]
    assert response.scores[0].token_logprobs == (-0.4, -0.6)
    assert response.usage is not None
    assert response.usage.model_dump(exclude_none=True) == {
        "prompt_tokens": 9,
        "total_tokens": 9,
    }
    assert response.model == "example/model"
    assert response.cost is not None
    assert response.cost.total_cost == 0.0


@pytest.mark.parametrize(
    "code",
    [
        "local_out_of_memory",
        "local_device_unavailable",
        "local_model_not_found",
        "local_sequence_too_long",
        "local_chat_template_missing",
        "local_non_finite_score",
    ],
)
def test_all_local_failures_reach_lifecycle(
    code: str,
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(
        execution=local_execution,
        results=[
            LocalBackendFailure(
                failure_record(
                    code=code,
                    recoverability=LOCAL_FAILURE_RECOVERABILITY[code],
                    message="expected failure",
                    metadata={"reason": "test"},
                )
            ),
        ],
    )
    classifier = AcceptAllSemanticResponseClassifier()
    with LocalModelProvider(config=local_config, backend=backend) as provider:
        result = run_local_provider_call(
            provider=provider,
            cancellation=threading.Event(),
            classifier=classifier,
            state=ProviderCallState.initial(
                request=_request(local_config),
                retry_policy=StandardProviderCallRetryPolicy(),
                classifier_identifier=classifier.identifier,
            ),
        )
    observation = result.completed_invocations[0].observation
    assert observation.outcome is (
        ProviderInvocationOutcome.RESOURCE_EXHAUSTION
        if code == "local_out_of_memory"
        else ProviderInvocationOutcome.PERMANENT_PROVIDER_OR_TRANSPORT_FAILURE
    )
    failure = observation.evidence.failure
    assert failure is not None
    assert failure.code == code
    assert failure.traceback is None
    assert failure.metadata == {"reason": "test"}


def test_wrong_definition_tools_and_closed_provider_reject_before_backend(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(execution=local_execution, results=[GENERATED])
    provider = LocalModelProvider(config=local_config, backend=backend)
    data = local_config.definition.model_dump()
    data["definition_id"] = "other"
    other = type(local_config.definition).model_validate(data).materialize()
    with pytest.raises(ValueError, match="definition differs"):
        provider.invoke(_request(other))
    tool_request = ProviderGenerateRequest(
        config=local_config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.TOOL, content="hi"),)
        ),
    )
    with pytest.raises(ValueError, match="support tools"):
        provider.invoke(tool_request)
    provider.close()
    with pytest.raises(ValueError, match="closed"):
        provider.invoke(_request(local_config))
    with pytest.raises(ValueError, match="closed"):
        provider.offload(lambda: 1)
    assert backend.calls == []


@pytest.mark.parametrize(
    "controls",
    [
        GenerationControls(temperature=-1.0),
        GenerationControls(top_p=0.0),
        GenerationControls(top_p=1.1),
        GenerationControls(token_limit=0),
        GenerationControls(seed=2**64),
    ],
)
def test_local_control_ranges_reject_before_backend(
    controls: GenerationControls,
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(execution=local_execution, results=[GENERATED])
    with (
        LocalModelProvider(config=local_config, backend=backend) as provider,
        pytest.raises(ValueError, match="local"),
    ):
        provider.invoke(
            _request(local_config.definition.materialize(controls=controls))
        )
    assert backend.calls == []


def test_invalid_local_definition_is_refused(
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(execution=local_execution, results=[GENERATED])
    with pytest.raises(ValueError, match="HuggingFace"):
        LocalModelProvider(
            config=openai_chat_config(model="m"), backend=backend
        )


def test_unexpected_backend_exception_propagates(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    class BrokenBackend(FakeLocalBackend):
        def generate(self, **kwargs: object) -> GenerationOutput:
            del kwargs
            raise RuntimeError("unexpected defect")

    backend = BrokenBackend(execution=local_execution, results=[GENERATED])
    with (
        LocalModelProvider(config=local_config, backend=backend) as provider,
        pytest.raises(RuntimeError, match="unexpected defect"),
    ):
        provider.invoke(_request(local_config))


def test_protocol_assignments(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend: LocalBackend = FakeLocalBackend(
        execution=local_execution, results=[GENERATED]
    )
    with LocalModelProvider(config=local_config, backend=backend) as provider:
        synchronous: Provider = provider
        offloading: OffloadingProvider = provider
        assert synchronous is offloading


def test_single_worker_and_async_retry(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(
        execution=local_execution,
        results=[
            LocalBackendFailure(
                failure_record(
                    code="local_out_of_memory",
                    recoverability=RecoverabilityClass.RESOURCE_EXHAUSTION,
                    message="memory pressure",
                )
            ),
            GENERATED,
        ],
    )
    classifier = AcceptAllSemanticResponseClassifier()
    state = ProviderCallState.initial(
        request=_request(local_config),
        classifier_identifier=classifier.identifier,
        retry_policy=CustomProviderCallRetryPolicy(
            maximum_invocations=2,
            eligible_outcomes=frozenset(
                {ProviderInvocationOutcome.RESOURCE_EXHAUSTION}
            ),
            declared_delays_seconds=(0.0,),
        ),
    )
    with LocalModelProvider(config=local_config, backend=backend) as provider:
        worker_ids = [provider.offload(threading.get_ident) for _ in range(3)]
        assert len({future.result(WATCHDOG) for future in worker_ids}) == 1
        assert worker_ids[0].result() != threading.get_ident()
        result = asyncio.run(
            run_local_provider_call_async(
                provider=provider,
                state=state,
                classifier=classifier,
                cancellation=threading.Event(),
            )
        )
    assert result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED
    assert len(result.completed_invocations) == 2
    assert len(backend.calls) == 2


def test_close_drains_admitted_work_and_refuses_new_admission(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(execution=local_execution, results=[GENERATED])
    provider = LocalModelProvider(config=local_config, backend=backend)
    entered, release = threading.Event(), threading.Event()

    def work() -> None:
        entered.set()
        assert release.wait(WATCHDOG)
        provider.invoke(_request(local_config))

    future = provider.offload(work)
    assert entered.wait(WATCHDOG)
    with ThreadPoolExecutor(max_workers=2) as closers:
        first = closers.submit(provider.close)
        # Wait for the actual closing transition, not a scheduling delay.
        with provider._lifetime:
            assert provider._lifetime.wait_for(
                lambda: provider._closing, WATCHDOG
            )
        second = closers.submit(provider.close)
        with pytest.raises(ValueError, match="closing"):
            provider.offload(lambda: None)
        with pytest.raises(ValueError, match="closing"):
            provider.invoke(_request(local_config))
        assert backend.close_calls == 0
        release.set()
        future.result(WATCHDOG)
        first.result(WATCHDOG)
        second.result(WATCHDOG)
    assert len(backend.calls) == 1
    assert backend.close_calls == 1


@pytest.mark.parametrize(
    "scores",
    [
        (),
        (ScoredContinuation(-1.0, 1, None, 2),),
        (ScoredContinuation(-1.0, 0, (), 2),),
    ],
)
def test_backend_score_contract_defects_raise(
    scores: tuple[ScoredContinuation, ...],
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(execution=local_execution, results=[scores])
    with (
        LocalModelProvider(config=local_config, backend=backend) as provider,
        pytest.raises(
            ValueError, match=r"score count|token_logprobs|token_count"
        ),
    ):
        provider.invoke(
            ProviderScoreRequest(
                config=local_config,
                context="ctx",
                continuations=("a",),
                token_logprobs=True,
            )
        )


def test_fake_records_and_repeats_last_result(
    local_config: ProviderCallConfig,
    local_execution: LocalExecutionEvidence,
) -> None:
    backend = FakeLocalBackend(
        execution=local_execution,
        results=[GENERATED, replace(GENERATED, text="last")],
    )
    with LocalModelProvider(config=local_config, backend=backend) as provider:
        texts = [
            provider.invoke(_request(local_config)).response for _ in range(3)
        ]
    assert [response.text for response in texts if response is not None] == [
        "answer",
        "last",
        "last",
    ]
    assert len(backend.calls) == 3
