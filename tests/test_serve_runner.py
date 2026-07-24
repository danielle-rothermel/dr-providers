import pytest

from dr_providers.failures import FailureClass
from dr_providers.outcome import ProviderTransportFailure, TokenUsage
from dr_providers.scripted import ScriptedOutcome, ScriptedProvider
from dr_providers.serve.runner import (
    QuerySpec,
    ServeProviderKind,
    run_query,
    run_variance,
)
from dr_providers.transcript import MessageRole, PromptMessage

PROMPT = "Say hello."


def make_spec(**overrides: object) -> QuerySpec:
    defaults: dict[str, object] = {
        "provider_kind": ServeProviderKind.OPENROUTER,
        "model": "test/model",
        "messages": (PromptMessage(role=MessageRole.USER, content=PROMPT),),
    }
    defaults.update(overrides)
    return QuerySpec.model_validate(defaults)


def test_run_query_anthropic_kind_supplies_default_token_limit() -> None:
    from dr_providers.serve.runner import (
        DEFAULT_ANTHROPIC_TOKEN_LIMIT,
        build_request,
    )

    provider = ScriptedProvider([ScriptedOutcome(text="hi")])
    spec = make_spec(
        provider_kind=ServeProviderKind.ANTHROPIC, model="claude-test"
    )
    # anthropic's preset REQUIRES a token limit; build_request supplies the
    # default when the spec omits one, so materialization succeeds.
    request = build_request(spec)
    assert request.config.route.provider.value == "anthropic"
    assert request.config.controls.token_limit == DEFAULT_ANTHROPIC_TOKEN_LIMIT

    result = run_query(spec, provider)
    assert result.ok
    assert result.endpoint_path == "/messages"
    assert result.response is not None
    assert result.response.text == "hi"


def test_run_query_anthropic_kind_honors_explicit_token_limit() -> None:
    provider = ScriptedProvider([ScriptedOutcome(text="hi")])
    spec = make_spec(
        provider_kind=ServeProviderKind.ANTHROPIC,
        model="claude-test",
        token_limit=256,
    )
    result = run_query(spec, provider)
    assert result.ok
    from dr_providers.serve.runner import build_request

    assert build_request(spec).config.controls.token_limit == 256


def test_run_query_returns_payload_and_response() -> None:
    provider = ScriptedProvider([ScriptedOutcome(text="hello")])
    result = run_query(make_spec(), provider)

    assert result.ok
    assert result.endpoint_path == "/chat/completions"
    assert result.payload["model"] == "test/model"
    assert result.payload["messages"] == [{"role": "user", "content": PROMPT}]
    assert result.response is not None
    assert result.response.text == "hello"


def test_run_query_applies_conformance_warnings() -> None:
    provider = ScriptedProvider(
        [
            ScriptedOutcome(
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
    provider = ScriptedProvider(
        [
            ScriptedOutcome(
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
    failure = ProviderTransportFailure(
        failure_class=FailureClass.PERMANENT,
        code="scripted_down",
        message="scripted failure",
        retryable=False,
    )
    provider = ScriptedProvider([ScriptedOutcome(failure=failure)])
    result = run_query(make_spec(), provider)

    assert not result.ok
    assert result.failure is not None
    assert result.failure.code == "scripted_down"
    assert result.payload["model"] == "test/model"


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
