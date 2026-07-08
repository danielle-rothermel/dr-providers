import pytest

from dr_providers.kernel.config import MessageRole, PromptMessage
from dr_providers.kernel.failures import FailureClass, failure_record
from dr_providers.kernel.fixture import FixtureOutcome, FixtureProvider
from dr_providers.kernel.response import TokenUsage
from dr_providers.serve.runner import (
    QuerySpec,
    ServeProviderKind,
    run_query,
    run_variance,
)

PROMPT = "Say hello."


def make_spec(**overrides: object) -> QuerySpec:
    defaults: dict[str, object] = {
        "provider_kind": ServeProviderKind.OPENROUTER,
        "model": "test/model",
        "messages": (PromptMessage(role=MessageRole.USER, content=PROMPT),),
    }
    defaults.update(overrides)
    return QuerySpec.model_validate(defaults)


def test_run_query_returns_payload_and_response() -> None:
    provider = FixtureProvider([FixtureOutcome(text="hello")])
    result = run_query(make_spec(), provider)

    assert result.ok
    assert result.endpoint_path == "/chat/completions"
    assert result.payload["model"] == "test/model"
    assert result.payload["messages"] == [{"role": "user", "content": PROMPT}]
    assert result.response is not None
    assert result.response.text == "hello"


def test_run_query_applies_conformance_warnings() -> None:
    provider = FixtureProvider(
        [
            FixtureOutcome(
                text="over budget",
                usage=TokenUsage(completion_tokens=99),
            )
        ]
    )
    result = run_query(make_spec(token_limit=10), provider)

    assert result.response is not None
    codes = [warning.code for warning in result.response.warnings]
    assert "token_limit_exceeded" in codes


def test_run_query_does_not_duplicate_warnings() -> None:
    provider = FixtureProvider(
        [
            FixtureOutcome(
                text="over budget",
                usage=TokenUsage(completion_tokens=99),
            )
        ]
    )
    result = run_query(make_spec(token_limit=10), provider)

    assert result.response is not None
    codes = [warning.code for warning in result.response.warnings]
    assert codes.count("token_limit_exceeded") == 1


def test_run_query_surfaces_failure_records() -> None:
    failure = failure_record(
        failure_class=FailureClass.PERMANENT,
        code="fixture_down",
        message="scripted failure",
    )
    provider = FixtureProvider([FixtureOutcome(failure=failure)])
    result = run_query(make_spec(), provider)

    assert not result.ok
    assert result.failure is not None
    assert result.failure.code == "fixture_down"
    assert result.payload["model"] == "test/model"


def test_run_variance_reports_dispersion_per_model() -> None:
    provider = FixtureProvider(
        [
            FixtureOutcome(text="alpha"),
            FixtureOutcome(text="beta"),
            FixtureOutcome(text="beta"),
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
    failure = failure_record(
        failure_class=FailureClass.TRANSIENT,
        code="rate_limited",
        message="scripted",
    )
    provider = FixtureProvider(
        [FixtureOutcome(text="fine"), FixtureOutcome(failure=failure)]
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
    provider = FixtureProvider()
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
