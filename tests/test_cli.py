from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from dr_providers import cli
from dr_providers.controls import ReasoningEffort
from dr_providers.scripted import ScriptedOutcome, ScriptedProvider

if TYPE_CHECKING:
    import pytest

    from dr_providers.request import ProviderCallRequest

runner = CliRunner()


class ScriptedHttpProvider:
    """Wraps ScriptedProvider as a context manager, standing in for
    HttpProvider so tests never touch the network."""

    def __init__(self, scripted: ScriptedProvider) -> None:
        self._scripted = scripted

    def __call__(
        self, *_args: object, **_kwargs: object
    ) -> ScriptedHttpProvider:
        return self

    def __enter__(self) -> ScriptedProvider:
        return self._scripted

    def __exit__(self, *exc: object) -> None:
        return None


def patch_http_provider(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[ScriptedOutcome] | None = None,
) -> ScriptedProvider:
    scripted = ScriptedProvider(outcomes)
    monkeypatch.setattr(cli, "HttpProvider", ScriptedHttpProvider(scripted))
    return scripted


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
