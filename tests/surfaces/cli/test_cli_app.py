from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click import unstyle
from typer.testing import CliRunner

from dr_providers.modeling.controls import ReasoningEffort
from dr_providers.modeling.route import Protocol, ProviderKind
from dr_providers.surfaces.cli import app as cli
from dr_providers.surfaces.testing.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
)
from dr_providers.transport.policy import ProviderTransportPolicy

if TYPE_CHECKING:
    from dr_providers.modeling.request import ProviderCallRequest

runner = CliRunner()


class ScriptedHttpProvider:
    def __init__(self, scripted: ScriptedProvider) -> None:
        self._scripted = scripted
        self.kwargs: dict[str, object] = {}

    def __call__(
        self, *_args: object, **kwargs: object
    ) -> ScriptedHttpProvider:
        self.kwargs = kwargs
        return self

    def __enter__(self) -> ScriptedProvider:
        return self._scripted

    def __exit__(self, *exc: object) -> None:
        return None

    @property
    def requests(self) -> list[ProviderCallRequest]:
        return self._scripted.requests


def patch_http_provider(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[ScriptedOutcome] | None = None,
) -> ScriptedHttpProvider:
    scripted = ScriptedProvider(outcomes)
    stub = ScriptedHttpProvider(scripted)
    monkeypatch.setattr(cli, "HttpProvider", stub)
    return stub


def test_query_prints_response_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_http_provider(
        monkeypatch,
        [ScriptedOutcome(text="hello from scripted", finish_reason="stop")],
    )

    result = runner.invoke(
        cli.app,
        [
            "--provider",
            "openrouter",
            "--model",
            "test/model",
            "-m",
            "Say hello.",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == "hello from scripted\n"
    assert result.stderr == ("model: test/model\nfinish_reason: stop\n")


@pytest.mark.parametrize(
    (
        "provider_flag",
        "expected_provider",
        "expected_protocol",
        "expected_token_limit",
    ),
    [
        pytest.param(
            "openrouter",
            ProviderKind.OPENROUTER,
            Protocol.CHAT_COMPLETIONS,
            None,
            id="openrouter-chat",
        ),
        pytest.param(
            "openai",
            ProviderKind.OPENAI,
            Protocol.CHAT_COMPLETIONS,
            None,
            id="openai-chat",
        ),
        pytest.param(
            "openai-responses",
            ProviderKind.OPENAI,
            Protocol.RESPONSES,
            None,
            id="openai-responses",
        ),
        pytest.param(
            "gemini",
            ProviderKind.GEMINI,
            Protocol.CHAT_COMPLETIONS,
            None,
            id="gemini-chat",
        ),
        pytest.param(
            "anthropic",
            ProviderKind.ANTHROPIC,
            Protocol.ANTHROPIC_MESSAGES,
            cli.DEFAULT_ANTHROPIC_TOKEN_LIMIT,
            id="anthropic-messages",
        ),
    ],
)
def test_provider_flags_select_request_route(
    monkeypatch: pytest.MonkeyPatch,
    provider_flag: str,
    expected_provider: ProviderKind,
    expected_protocol: Protocol,
    expected_token_limit: int | None,
) -> None:
    scripted = patch_http_provider(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "--provider",
            provider_flag,
            "--model",
            "test-model",
            "-m",
            "hello",
        ],
    )

    assert result.exit_code == 0
    request = scripted.requests[0]
    assert request.config.route.provider is expected_provider
    assert request.config.route.protocol is expected_protocol
    assert request.config.controls.token_limit == expected_token_limit
    policy = scripted.kwargs["policy"]
    assert isinstance(policy, ProviderTransportPolicy)
    assert policy.provider_kind is expected_provider
    assert policy.max_connections == 1
    assert policy.max_keepalive_connections == 1


def test_query_flags_build_full_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = patch_http_provider(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "--provider",
            "openai-responses",
            "--model",
            "gpt-test",
            "-m",
            "Say hello.",
            "--system",
            "Be terse.",
            "--effort",
            "high",
            "--temperature",
            "0.5",
            "--top-p",
            "0.9",
            "--token-limit",
            "128",
        ],
    )

    assert result.exit_code == 0
    request = scripted.requests[0]
    assert request.config.route.provider is ProviderKind.OPENAI
    assert request.config.route.protocol is Protocol.RESPONSES
    assert request.config.route.model == "gpt-test"
    assert request.config.controls.reasoning is ReasoningEffort.HIGH
    assert request.config.controls.temperature == 0.5
    assert request.config.controls.top_p == 0.9
    assert request.config.controls.token_limit == 128
    assert [message.role.value for message in request.transcript.messages] == [
        "system",
        "user",
    ]
    assert [message.content for message in request.transcript.messages] == [
        "Be terse.",
        "Say hello.",
    ]


@pytest.mark.parametrize(
    ("arguments", "invalid_value"),
    [
        pytest.param(
            [
                "--provider",
                "not-a-provider",
                "--model",
                "test/model",
                "-m",
                "hi",
            ],
            "not-a-provider",
            id="provider",
        ),
        pytest.param(
            [
                "--provider",
                "openrouter",
                "--model",
                "test/model",
                "-m",
                "hi",
                "--effort",
                "extreme",
            ],
            "extreme",
            id="reasoning-effort",
        ),
    ],
)
def test_invalid_choice_exits_with_clear_diagnostic(
    arguments: list[str], invalid_value: str
) -> None:
    result = runner.invoke(cli.app, arguments)

    assert result.exit_code != 0
    assert invalid_value in result.output


def test_query_failure_prints_stderr_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_providers.core.failures import FailureClass
    from dr_providers.outcomes.models import ProviderTransportFailure

    failure = ProviderTransportFailure(
        failure_class=FailureClass.PERMANENT,
        code="boom_code",
        message="boom message",
    )
    patch_http_provider(monkeypatch, [ScriptedOutcome(failure=failure)])

    result = runner.invoke(
        cli.app,
        [
            "--provider",
            "openrouter",
            "--model",
            "test/model",
            "-m",
            "hi",
        ],
    )

    assert result.exit_code == 1
    assert result.stderr == "failure: boom_code: boom message\n"


def test_removed_retries_flag_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = patch_http_provider(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "--provider",
            "openrouter",
            "--model",
            "test/model",
            "-m",
            "hi",
            "--retries",
            "3",
        ],
    )

    assert result.exit_code == 2
    assert "--retries" in unstyle(result.output)
    assert stub.requests == []
