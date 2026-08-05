import pytest

from dr_providers.core.failures import FailureClass
from dr_providers.outcomes.models import ProviderTransportFailure
from dr_providers.surfaces.serve.query import (
    ServeProviderKind,
)
from dr_providers.surfaces.serve.variance import (
    run_variance,
)
from dr_providers.surfaces.testing.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
)

PROMPT = "Say hello."


def test_run_variance_reports_dispersion_per_model() -> None:
    provider = ScriptedProvider(
        [
            ScriptedOutcome(text="alpha"),
            ScriptedOutcome(text="beta"),
            ScriptedOutcome(text="beta"),
        ]
    )
    report = run_variance(
        PROMPT,
        models=["model-a"],
        samples=3,
        provider_kind=ServeProviderKind.OPENROUTER,
        provider=provider,
    )

    assert report.samples_per_model == 3
    assert len(report.records) == 3
    model_report = report.per_model[0]
    assert model_report.samples == 3
    assert model_report.failures == 0
    assert model_report.distinct_outputs == 2
    assert model_report.min_length == 4
    assert model_report.max_length == 5


def test_run_variance_counts_failures() -> None:
    failure = ProviderTransportFailure(
        failure_class=FailureClass.TRANSIENT,
        code="rate_limited",
        message="scripted",
        retryable=True,
    )
    provider = ScriptedProvider(
        [ScriptedOutcome(text="fine"), ScriptedOutcome(failure=failure)]
    )
    report = run_variance(
        PROMPT,
        models=["model-a"],
        samples=2,
        provider_kind=ServeProviderKind.OPENROUTER,
        provider=provider,
    )

    assert report.per_model[0].failures == 1
    failed = [record for record in report.records if not record.ok]
    assert failed[0].failure_code == "rate_limited"


def test_run_variance_validates_inputs() -> None:
    provider = ScriptedProvider()
    with pytest.raises(ValueError, match="samples"):
        run_variance(
            PROMPT,
            models=["m"],
            samples=0,
            provider_kind=ServeProviderKind.OPENROUTER,
            provider=provider,
        )
    with pytest.raises(ValueError, match="model"):
        run_variance(
            PROMPT,
            models=[],
            samples=1,
            provider_kind=ServeProviderKind.OPENROUTER,
            provider=provider,
        )
