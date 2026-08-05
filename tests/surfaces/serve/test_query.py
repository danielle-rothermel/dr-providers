import pytest

from dr_providers.core.failures import FailureClass
from dr_providers.modeling.controls import ReasoningEffort
from dr_providers.modeling.route import Protocol, ProviderKind
from dr_providers.modeling.transcript import MessageRole, PromptMessage
from dr_providers.outcomes.models import ProviderTransportFailure
from dr_providers.surfaces.serve.query import (
    DEFAULT_ANTHROPIC_TOKEN_LIMIT,
    QuerySpec,
    ServeProviderKind,
    build_request,
    run_query,
)
from dr_providers.surfaces.testing.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
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


@pytest.mark.parametrize(
    (
        "serve_kind",
        "expected_provider",
        "expected_protocol",
        "expected_endpoint",
        "expected_token_limit",
    ),
    [
        pytest.param(
            ServeProviderKind.OPENROUTER,
            ProviderKind.OPENROUTER,
            Protocol.CHAT_COMPLETIONS,
            "/chat/completions",
            None,
            id="openrouter-chat",
        ),
        pytest.param(
            ServeProviderKind.OPENAI,
            ProviderKind.OPENAI,
            Protocol.CHAT_COMPLETIONS,
            "/chat/completions",
            None,
            id="openai-chat",
        ),
        pytest.param(
            ServeProviderKind.OPENAI_RESPONSES,
            ProviderKind.OPENAI,
            Protocol.RESPONSES,
            "/responses",
            None,
            id="openai-responses",
        ),
        pytest.param(
            ServeProviderKind.GEMINI,
            ProviderKind.GEMINI,
            Protocol.CHAT_COMPLETIONS,
            "/chat/completions",
            None,
            id="gemini-chat",
        ),
        pytest.param(
            ServeProviderKind.ANTHROPIC,
            ProviderKind.ANTHROPIC,
            Protocol.ANTHROPIC_MESSAGES,
            "/messages",
            DEFAULT_ANTHROPIC_TOKEN_LIMIT,
            id="anthropic-messages",
        ),
    ],
)
def test_build_request_maps_every_serve_provider(
    serve_kind: ServeProviderKind,
    expected_provider: ProviderKind,
    expected_protocol: Protocol,
    expected_endpoint: str,
    expected_token_limit: int | None,
) -> None:
    request = build_request(make_spec(provider_kind=serve_kind))

    assert request.config.route.provider is expected_provider
    assert request.config.route.protocol is expected_protocol
    assert request.config.controls.token_limit == expected_token_limit
    result = run_query(
        make_spec(provider_kind=serve_kind),
        ScriptedProvider([ScriptedOutcome(text="hi")]),
    )
    assert result.endpoint_path == expected_endpoint


def test_build_request_transfers_every_query_spec_field() -> None:
    messages = (
        PromptMessage(role=MessageRole.SYSTEM, content="Be exact."),
        PromptMessage(role=MessageRole.USER, content="Answer this."),
    )
    spec = QuerySpec(
        provider_kind=ServeProviderKind.OPENROUTER,
        model="full-field-model",
        messages=messages,
        temperature=0.25,
        top_p=0.75,
        token_limit=128,
        reasoning=ReasoningEffort.HIGH,
        extra_body={"provider": {"order": ["first", "second"]}},
    )

    request = build_request(spec)

    assert request.config.route.provider is ProviderKind.OPENROUTER
    assert request.config.route.model == "full-field-model"
    assert request.transcript.messages == messages
    assert request.config.controls.temperature == 0.25
    assert request.config.controls.top_p == 0.75
    assert request.config.controls.token_limit == 128
    assert request.config.controls.reasoning is ReasoningEffort.HIGH
    assert request.config.extensions.model_dump(mode="json") == {
        "extra_body": {"provider": {"order": ["first", "second"]}}
    }


def test_build_request_anthropic_honors_explicit_token_limit() -> None:
    spec = make_spec(
        provider_kind=ServeProviderKind.ANTHROPIC,
        model="claude-test",
        token_limit=256,
    )

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
