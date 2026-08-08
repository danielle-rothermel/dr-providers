from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import test_matrix

from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    TokenUsage,
    openai_chat_config,
)
from dr_providers.surfaces.testing.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
)
from scripts import capture_live_corpus, run_live_matrix
from scripts.live_matrix_support import (
    CAPTURE_DIR_ENV,
    LIVE_CASES,
    ROOT,
    credential_values,
    mapped_provider_environment,
    require_external_capture_dir,
    select_cases,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class _SuccessfulProvider:
    def __init__(self, **_kwargs: object) -> None:
        self._provider = ScriptedProvider(
            [
                ScriptedOutcome(
                    text="hello",
                    response_body={"id": "response-1"},
                    usage=TokenUsage(total_tokens=1),
                )
            ]
        )

    def __enter__(self) -> _SuccessfulProvider:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        return self._provider.invoke(request)


def test_live_verification_does_not_write_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir = ROOT / "data" / "wire-corpus"
    before = {path: path.read_bytes() for path in corpus_dir.glob("*.json")}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(test_matrix, "HttpProvider", _SuccessfulProvider)

    test_matrix.test_live_matrix(
        LIVE_CASES[1],
        lambda controls: openai_chat_config(model="test", controls=controls),
        set_temperature=False,
    )

    after = {path: path.read_bytes() for path in corpus_dir.glob("*.json")}
    assert after == before


def test_explicit_capture_stages_outside_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture_dir = tmp_path / "capture"
    monkeypatch.setenv(CAPTURE_DIR_ENV, str(capture_dir))

    test_matrix._stage_capture(LIVE_CASES[1], {"id": "response-1"})

    assert json.loads(
        (capture_dir / "openai_chat_completions.json").read_text()
    ) == {"id": "response-1"}


def test_capture_staging_rejects_repository_paths() -> None:
    with pytest.raises(
        ValueError, match="staging must be outside the repository"
    ):
        require_external_capture_dir(ROOT / "data" / "wire-corpus")


def test_openai_provider_selection_maps_dotfiles_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    child_environments: list[Mapping[str, str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        child_environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_live_matrix.subprocess, "run", fake_run)

    result = run_live_matrix.run_selected_cases(
        ["--provider", "openai"],
        environment={"MARIMO_OPENAI_API_KEY": "private-openai-key"},
    )

    assert result == 0
    assert len(commands) == 1
    assert child_environments[0]["OPENAI_API_KEY"] == "private-openai-key"
    assert "private-openai-key" not in commands[0]
    assert commands[0][-2:] == [
        LIVE_CASES[1].pytest_node,
        LIVE_CASES[2].pytest_node,
    ]


def test_selected_case_missing_credential_is_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("pytest must not run without the selected credential")

    monkeypatch.setattr(run_live_matrix.subprocess, "run", unexpected_run)

    result = run_live_matrix.run_selected_cases(
        ["--case", "gemini_chat_completions"], environment={}
    )

    assert result == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_case_contract_preserves_five_case_breadth() -> None:
    assert [case.case_id for case in select_cases()] == [
        "openrouter_chat_completions",
        "openai_chat_completions",
        "openai_responses",
        "gemini_chat_completions",
        "anthropic_messages",
    ]


def _synthetic_capture_bodies() -> dict[str, dict[str, object]]:
    chat_body: dict[str, object] = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }
    return {
        "openrouter_chat_completions.json": {
            **chat_body,
            "metadata": {
                "authorization": "Bearer private-token",
                "echo": "private-token",
                "url": (
                    "https://user:password@example.test/v1"
                    "?api_key=unlisted-token"
                    "#access_token=fragment-secret&state=public"
                ),
            },
        },
        "openai_chat_completions.json": chat_body,
        "openai_responses.json": {
            "id": "response-1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "gemini_chat_completions.json": chat_body,
        "anthropic_anthropic_messages.json": {
            "id": "message-1",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


def _write_synthetic_capture(staging_dir: Path) -> None:
    staging_dir.mkdir()
    for file_name, body in _synthetic_capture_bodies().items():
        (staging_dir / file_name).write_text(json.dumps(body))


def test_complete_capture_is_redacted_validated_and_promoted(
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / "staging"
    corpus_dir = tmp_path / "curated"
    _write_synthetic_capture(staging_dir)

    capture_live_corpus.promote_capture(
        staging_dir,
        corpus_dir,
        secrets=("private-token",),
    )

    promoted = {
        path.name: json.loads(path.read_text())
        for path in corpus_dir.glob("*.json")
    }
    assert set(promoted) == {case.corpus_file for case in LIVE_CASES}
    metadata = promoted["openrouter_chat_completions.json"]["metadata"]
    assert metadata == {
        "authorization": "[REDACTED]",
        "echo": "[REDACTED]",
        "url": (
            "https://redacted@example.test/v1?api_key=%5BREDACTED%5D"
            "#[REDACTED]"
        ),
    }


def test_all_live_credentials_are_collected_and_redacted() -> None:
    environment = {
        "OPENROUTER_API_KEY": "secret-openrouter",
        "GEMINI_API_KEY": "secret-gemini",
        "MARIMO_OPENAI_API_KEY": "secret-openai",
        "OPENCODE_ANTHROPIC_API_KEY": "secret-anthropic",
    }

    mapped = mapped_provider_environment(environment)
    secrets = credential_values(mapped)

    assert set(secrets) == set(environment.values())
    assert capture_live_corpus.redact_capture(
        {"echoes": list(secrets)}, secrets
    ) == {"echoes": ["[REDACTED]"] * len(secrets)}


def test_overlapping_live_credentials_redact_longest_first() -> None:
    environment = {
        "OPENROUTER_API_KEY": "shared-prefix",
        "GEMINI_API_KEY": "shared-prefix-suffix",
    }

    secrets = credential_values(environment)

    assert (
        capture_live_corpus.redact_capture(
            "echo: shared-prefix-suffix", secrets
        )
        == "echo: [REDACTED]"
    )


def test_url_redaction_preserves_opaque_fragment() -> None:
    url = "https://example.test/v1#section-2"

    assert capture_live_corpus.redact_capture(url, ()) == url


def test_url_redaction_replaces_sensitive_routed_fragment() -> None:
    url = (
        "https://example.test/v1#/callback/unchanged"
        "?route=/foo/bar&access_token=fragment-secret&state=public"
    )

    assert capture_live_corpus.redact_capture(url, ()) == (
        "https://example.test/v1#[REDACTED]"
    )


def test_promote_cli_reexecs_under_mise_and_redacts_mapped_secret(
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / "staging"
    corpus_dir = tmp_path / "curated"
    _write_synthetic_capture(staging_dir)
    configured_credential = "configured-mise-value"
    openrouter_path = staging_dir / "openrouter_chat_completions.json"
    openrouter_body = json.loads(openrouter_path.read_text())
    openrouter_body["metadata"]["echo"] = configured_credential
    openrouter_path.write_text(json.dumps(openrouter_body))

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mise = fake_bin / "mise"
    fake_mise.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "os.environ['MARIMO_OPENAI_API_KEY'] = "
        "os.environ.pop('FAKE_MISE_SECRET')\n"
        "os.execvp(sys.argv[3], sys.argv[3:])\n"
    )
    fake_mise.chmod(0o755)

    environment = os.environ.copy()
    credential_names = {case.credential_env for case in LIVE_CASES} | {
        "MARIMO_OPENAI_API_KEY",
        "OPENCODE_ANTHROPIC_API_KEY",
    }
    for name in credential_names:
        environment.pop(name, None)
    environment.pop("DR_PROVIDERS_LIVE_UNDER_MISE", None)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["FAKE_MISE_SECRET"] = configured_credential

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts" / "capture_live_corpus.py"),
            "promote",
            str(staging_dir),
            "--corpus-dir",
            str(corpus_dir),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    promoted = (corpus_dir / "openrouter_chat_completions.json").read_text()
    assert configured_credential not in promoted
    assert json.loads(promoted)["metadata"]["echo"] == "[REDACTED]"


def test_documented_promotion_revalidates_raw_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging_dir = tmp_path / "staging"
    corpus_dir = tmp_path / "curated"
    _write_synthetic_capture(staging_dir)
    first_validated = capture_live_corpus.prepare_capture(staging_dir)
    (first_validated / "openai_chat_completions.json").write_text(
        '{"tampered": true}\n'
    )
    monkeypatch.setenv("DR_PROVIDERS_LIVE_UNDER_MISE", "1")

    result = capture_live_corpus.main(
        [
            "promote",
            str(staging_dir),
            "--corpus-dir",
            str(corpus_dir),
        ]
    )

    assert result == 0
    promoted = json.loads(
        (corpus_dir / "openai_chat_completions.json").read_text()
    )
    assert (
        promoted == _synthetic_capture_bodies()["openai_chat_completions.json"]
    )


def test_incomplete_capture_cannot_change_curated_corpus(
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / "staging"
    corpus_dir = tmp_path / "curated"
    corpus_dir.mkdir()
    existing = corpus_dir / "openai_chat_completions.json"
    existing.write_text('{"existing": true}\n')
    _write_synthetic_capture(staging_dir)
    (staging_dir / "openai_responses.json").unlink()

    with pytest.raises(
        capture_live_corpus.CaptureValidationError,
        match="capture set is not complete",
    ):
        capture_live_corpus.prepare_capture(staging_dir)

    assert existing.read_text() == '{"existing": true}\n'


def test_failed_install_restores_complete_prior_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging_dir = tmp_path / "staging"
    corpus_dir = tmp_path / "curated"
    _write_synthetic_capture(staging_dir)
    validated_dir = capture_live_corpus.prepare_capture(staging_dir)
    corpus_dir.mkdir()
    prior_generation = {
        case.corpus_file: (
            f'{{"generation": "prior", "file": "{case.corpus_file}"}}\n'
        ).encode()
        for case in LIVE_CASES
    }
    for file_name, contents in prior_generation.items():
        (corpus_dir / file_name).write_bytes(contents)

    original_replace = Path.replace
    promotion_count = 0

    def fail_third_promotion(source: Path, target: Path) -> Path:
        nonlocal promotion_count
        if source.name.endswith(".promotion"):
            promotion_count += 1
            if promotion_count == 3:
                raise OSError("injected promotion failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_third_promotion)

    with pytest.raises(OSError, match="injected promotion failure"):
        capture_live_corpus._install_validated_capture(
            validated_dir, corpus_dir
        )

    assert promotion_count == 3
    assert {
        path.name: path.read_bytes() for path in corpus_dir.iterdir()
    } == prior_generation


def test_capture_executable_help_works_from_external_cwd(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts" / "capture_live_corpus.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Capture, validate, redact" in result.stdout


def test_both_dotfiles_credential_aliases_are_mapped() -> None:
    environment = mapped_provider_environment(
        {
            "MARIMO_OPENAI_API_KEY": "openai-secret",
            "OPENCODE_ANTHROPIC_API_KEY": "anthropic-secret",
        }
    )

    assert environment["OPENAI_API_KEY"] == "openai-secret"
    assert environment["ANTHROPIC_API_KEY"] == "anthropic-secret"
