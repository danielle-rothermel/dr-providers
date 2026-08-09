import pytest

from dr_providers.core.failures import FailureClass
from dr_providers.outcomes.models import (
    CostInfo,
    ProviderTransportFailure,
    ProviderTransportWarning,
    TokenUsage,
)
from dr_providers.surfaces.serve.query import ServeProviderKind
from dr_providers.surfaces.serve.variance import (
    ModelVariance,
    VarianceRecord,
    run_variance,
)
from dr_providers.surfaces.testing.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
)

PROMPT = "Say hello."


def test_run_variance_preserves_exact_records_and_summaries() -> None:
    failure = ProviderTransportFailure(
        failure_class=FailureClass.RATE_LIMITED,
        code="rate_limited",
        message="scripted",
    )
    warning = ProviderTransportWarning(
        code="provider_notice",
        message="first sample warning",
    )
    provider = ScriptedProvider(
        [
            ScriptedOutcome(
                text="alpha",
                finish_reason="stop",
                usage=TokenUsage(completion_tokens=3),
                cost=CostInfo(total_cost=0.01),
                warnings=(warning,),
            ),
            ScriptedOutcome(
                text="longer",
                finish_reason="length",
                usage=TokenUsage(completion_tokens=7),
                cost=CostInfo(total_cost=0.02),
            ),
            ScriptedOutcome(failure=failure),
        ]
    )

    report = run_variance(
        PROMPT,
        models=["model-a", "model-b"],
        samples=2,
        provider_kind=ServeProviderKind.OPENROUTER,
        provider=provider,
    )

    assert report.prompt == PROMPT
    assert report.samples_per_model == 2
    assert report.models == ("model-a", "model-b")
    assert [request.config.route.model for request in provider.requests] == [
        "model-a",
        "model-a",
        "model-b",
        "model-b",
    ]
    assert [payload["model"] for payload in provider.payloads] == [
        "model-a",
        "model-a",
        "model-b",
        "model-b",
    ]
    assert report.records == (
        VarianceRecord(
            model="model-a",
            sample_index=0,
            ok=True,
            text="alpha",
            finish_reason="stop",
            completion_tokens=3,
            total_cost=0.01,
            warning_codes=("provider_notice",),
        ),
        VarianceRecord(
            model="model-a",
            sample_index=1,
            ok=True,
            text="longer",
            finish_reason="length",
            completion_tokens=7,
            total_cost=0.02,
        ),
        VarianceRecord(
            model="model-b",
            sample_index=0,
            ok=False,
            failure_code="rate_limited",
        ),
        VarianceRecord(
            model="model-b",
            sample_index=1,
            ok=False,
            failure_code="rate_limited",
        ),
    )
    assert report.per_model == (
        ModelVariance(
            model="model-a",
            samples=2,
            failures=0,
            distinct_outputs=2,
            mean_length=5.5,
            min_length=5,
            max_length=6,
        ),
        ModelVariance(
            model="model-b",
            samples=2,
            failures=2,
            distinct_outputs=0,
            mean_length=None,
            min_length=None,
            max_length=None,
        ),
    )


def test_run_variance_does_not_treat_blank_response_as_accepted() -> None:
    report = run_variance(
        PROMPT,
        models=["model-a"],
        samples=1,
        provider_kind=ServeProviderKind.OPENROUTER,
        provider=ScriptedProvider([ScriptedOutcome(text="   ")]),
    )

    assert report.records == (
        VarianceRecord(
            model="model-a",
            sample_index=0,
            ok=False,
            failure_code="blank_response",
        ),
    )


@pytest.mark.parametrize(
    ("models", "samples", "message"),
    [
        pytest.param(["model"], 0, "samples must be >= 1", id="samples"),
        pytest.param([], 1, "at least one model", id="models"),
    ],
)
def test_run_variance_validates_inputs(
    models: list[str], samples: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_variance(
            PROMPT,
            models=models,
            samples=samples,
            provider_kind=ServeProviderKind.OPENROUTER,
            provider=ScriptedProvider(),
        )
