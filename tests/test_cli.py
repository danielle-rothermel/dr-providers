from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from dr_providers import ProviderTransportPolicy, cli
from dr_providers.controls import ReasoningEffort
from dr_providers.scripted import ScriptedOutcome, ScriptedProvider

if TYPE_CHECKING:
    from dr_providers.request import ProviderCallRequest

runner = CliRunner()


class ScriptedHttpProvider:
    """Wraps ScriptedProvider as a context manager, standing in for
    HttpProvider so tests never touch the network.

    Records the constructor kwargs (notably ``policy``) so tests can assert the
    transport policy the CLI built from its flags."""

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


def test_query_happy_path_prints_text(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http_provider(
        monkeypatch, [ScriptedOutcome(text="hello from scripted")]
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
    assert "hello from scripted" in result.stdout


def test_query_prints_metadata_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_http_provider(
        monkeypatch, [ScriptedOutcome(text="hi", finish_reason="stop")]
    )

    result = runner.invoke(
        cli.app,
        [
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "-m",
            "Hi",
        ],
    )

    assert result.exit_code == 0
    assert "model: gpt-test" in result.stderr
    assert "finish_reason: stop" in result.stderr


def test_effort_and_provider_flags_build_expected_request(
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
    assert len(scripted.requests) == 1
    request: ProviderCallRequest = scripted.requests[0]
    assert request.config.route.provider.value == "openai"
    assert request.config.route.protocol.value == "responses"
    assert request.config.route.model == "gpt-test"
    assert request.config.controls.reasoning == ReasoningEffort.HIGH
    assert request.config.controls.temperature == 0.5
    assert request.config.controls.top_p == 0.9
    assert request.config.controls.token_limit == 128
    roles = [m.role.value for m in request.transcript.messages]
    contents = [m.content for m in request.transcript.messages]
    assert roles == ["system", "user"]
    assert contents == ["Be terse.", "Say hello."]


def test_gemini_provider_flag_builds_gemini_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = patch_http_provider(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "--provider",
            "gemini",
            "--model",
            "gemini-test",
            "-m",
            "hi",
        ],
    )

    assert result.exit_code == 0
    assert scripted.requests[0].config.route.provider.value == "gemini"


def test_bad_provider_value_exits_with_clear_message() -> None:
    result = runner.invoke(
        cli.app,
        [
            "--provider",
            "not-a-provider",
            "--model",
            "test/model",
            "-m",
            "hi",
        ],
    )

    assert result.exit_code != 0
    assert "not-a-provider" in result.stdout or "not-a-provider" in str(
        result.output
    )


def test_bad_effort_value_exits_with_clear_message() -> None:
    result = runner.invoke(
        cli.app,
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
    )

    assert result.exit_code != 0
    assert "extreme" in result.output


def test_query_failure_path_prints_stderr_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_providers.failures import FailureClass
    from dr_providers.outcome import ProviderTransportFailure

    failure = ProviderTransportFailure(
        failure_class=FailureClass.PERMANENT,
        code="boom_code",
        message="boom message",
        retryable=False,
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
    assert "failure: boom_code: boom message" in result.stderr


def test_retries_flag_and_provider_map_to_expected_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_providers.policy import ApiKeyEnv, ProviderBaseUrl

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

    assert result.exit_code == 0
    policy = stub.kwargs["policy"]
    assert isinstance(policy, ProviderTransportPolicy)
    # policy_for(OPENROUTER) derives env/base_url from the DEFAULT_* maps and
    # --retries wires straight to native_retry_count.
    assert policy.api_key_env == str(ApiKeyEnv.OPENROUTER)
    assert policy.base_url == str(ProviderBaseUrl.OPENROUTER)
    assert policy.native_retry_count == 3


def test_anthropic_provider_maps_policy_and_supplies_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_providers.policy import ApiKeyEnv, ProviderBaseUrl

    stub = patch_http_provider(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "--provider",
            "anthropic",
            "--model",
            "claude-test",
            "-m",
            "hi",
        ],
    )

    assert result.exit_code == 0
    request = stub.requests[0]
    assert request.config.route.provider.value == "anthropic"
    # the anthropic preset requires a token limit; the CLI supplies its default
    # when --token-limit is omitted so the call materializes.
    assert (
        request.config.controls.token_limit
        == cli.DEFAULT_ANTHROPIC_TOKEN_LIMIT
    )
    policy = stub.kwargs["policy"]
    assert isinstance(policy, ProviderTransportPolicy)
    assert policy.api_key_env == str(ApiKeyEnv.ANTHROPIC)
    assert policy.base_url == str(ProviderBaseUrl.ANTHROPIC)


def test_entry_main_invokes_typer_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_providers import _entry

    # With typer installed, main() resolves and runs the typer app. Drive it
    # through argv so the shim's app() dispatch is exercised end to end; --help
    # exits zero via SystemExit.
    monkeypatch.setattr(sys, "argv", ["dr-providers", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        _entry.main()
    assert excinfo.value.code == 0


def test_entry_main_missing_cli_extra_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dr_providers import _entry

    # Simulate the [cli] extra (typer) being absent. Setting sys.modules[name]
    # to None makes Python raise ImportError on any import of that name, so
    # re-importing dr_providers.cli fails at its top-level ``import typer`` --
    # exactly the runtime condition the shim guards against. Drop the cached
    # cli module first so the import is actually re-executed.
    monkeypatch.delitem(sys.modules, "dr_providers.cli", raising=False)
    monkeypatch.setitem(sys.modules, "typer", None)

    with pytest.raises(SystemExit) as excinfo:
        _entry.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "dr-providers[cli]" in captured.err
