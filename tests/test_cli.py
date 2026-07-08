from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from dr_providers import cli
from dr_providers.config import ReasoningEffort
from dr_providers.fixture import FixtureOutcome, FixtureProvider

if TYPE_CHECKING:
    import pytest

    from dr_providers.request import LlmRequest

runner = CliRunner()


class FixtureHttpProvider:
    """Wraps FixtureProvider as a context manager, standing in for
    HttpProvider so tests never touch the network."""

    def __init__(self, fixture: FixtureProvider) -> None:
        self._fixture = fixture

    def __call__(
        self, *_args: object, **_kwargs: object
    ) -> FixtureHttpProvider:
        return self

    def __enter__(self) -> FixtureProvider:
        return self._fixture

    def __exit__(self, *exc: object) -> None:
        return None


def patch_http_provider(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[FixtureOutcome] | None = None,
) -> FixtureProvider:
    fixture = FixtureProvider(outcomes)
    monkeypatch.setattr(cli, "HttpProvider", FixtureHttpProvider(fixture))
    return fixture


def test_query_happy_path_prints_text(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http_provider(
        monkeypatch, [FixtureOutcome(text="hello from fixture")]
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
    assert "hello from fixture" in result.stdout


def test_query_prints_metadata_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_http_provider(
        monkeypatch, [FixtureOutcome(text="hi", finish_reason="stop")]
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
    fixture = patch_http_provider(monkeypatch)

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
    assert len(fixture.requests) == 1
    request: LlmRequest = fixture.requests[0]
    assert request.provider_config.provider_kind.value == "openai"
    assert request.provider_config.endpoint_kind.value == "responses"
    assert request.provider_config.model == "gpt-test"
    assert request.reasoning == ReasoningEffort.HIGH
    assert request.temperature == 0.5
    assert request.top_p == 0.9
    assert request.token_limit == 128
    roles = [m.role.value for m in request.messages]
    contents = [m.content for m in request.messages]
    assert roles == ["system", "user"]
    assert contents == ["Be terse.", "Say hello."]


def test_gemini_provider_flag_builds_gemini_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = patch_http_provider(monkeypatch)

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
    assert fixture.requests[0].provider_config.provider_kind.value == "gemini"


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
