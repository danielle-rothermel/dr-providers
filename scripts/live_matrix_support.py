"""Shared contract for live-matrix execution and corpus capture."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
LIVE_TEST = ROOT / "tests" / "transport" / "live" / "test_matrix.py"
CAPTURE_DIR_ENV = "DR_PROVIDERS_LIVE_CAPTURE_DIR"


@verify(UNIQUE)
class LiveProvider(StrEnum):
    """Provider selectors accepted by the canonical live runner."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True, slots=True)
class LiveCase:
    case_id: str
    provider: LiveProvider
    credential_env: str
    corpus_file: str

    @property
    def pytest_node(self) -> str:
        return f"{LIVE_TEST}::test_live_matrix[{self.case_id}]"


LIVE_CASES = (
    LiveCase(
        "openrouter_chat_completions",
        LiveProvider.OPENROUTER,
        "OPENROUTER_API_KEY",
        "openrouter_chat_completions.json",
    ),
    LiveCase(
        "openai_chat_completions",
        LiveProvider.OPENAI,
        "OPENAI_API_KEY",
        "openai_chat_completions.json",
    ),
    LiveCase(
        "openai_responses",
        LiveProvider.OPENAI,
        "OPENAI_API_KEY",
        "openai_responses.json",
    ),
    LiveCase(
        "gemini_chat_completions",
        LiveProvider.GEMINI,
        "GEMINI_API_KEY",
        "gemini_chat_completions.json",
    ),
    LiveCase(
        "anthropic_messages",
        LiveProvider.ANTHROPIC,
        "ANTHROPIC_API_KEY",
        "anthropic_anthropic_messages.json",
    ),
)

CREDENTIAL_ALIASES = {
    "OPENAI_API_KEY": "MARIMO_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY": "OPENCODE_ANTHROPIC_API_KEY",
}


def select_cases(
    *, providers: Sequence[str] = (), case_ids: Sequence[str] = ()
) -> tuple[LiveCase, ...]:
    """Resolve an explicit provider or case selection."""
    if providers and case_ids:
        raise ValueError("provider and case selectors cannot be combined")
    if providers:
        selected_providers = {LiveProvider(value) for value in providers}
        return tuple(
            case for case in LIVE_CASES if case.provider in selected_providers
        )
    if case_ids:
        by_id = {case.case_id: case for case in LIVE_CASES}
        unknown = sorted(set(case_ids) - by_id.keys())
        if unknown:
            raise ValueError(f"unknown live case(s): {', '.join(unknown)}")
        selected_ids = set(case_ids)
        return tuple(
            case for case in LIVE_CASES if case.case_id in selected_ids
        )
    return LIVE_CASES


def mapped_provider_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return a child environment with dotfiles credential aliases mapped."""
    environment = dict(source)
    for target, alias in CREDENTIAL_ALIASES.items():
        if value := source.get(alias):
            environment[target] = value
    return environment


def missing_credentials(
    cases: Sequence[LiveCase], environment: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                case.credential_env
                for case in cases
                if not environment.get(case.credential_env)
            }
        )
    )


def credential_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    """Return configured credential values for redaction, never display."""
    names = set(CREDENTIAL_ALIASES) | set(CREDENTIAL_ALIASES.values())
    return tuple(value for name in names if (value := environment.get(name)))


def require_external_capture_dir(path: Path) -> Path:
    """Reject capture paths inside the repository, including curated data."""
    resolved = path.expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("live capture staging must be outside the repository")
    return resolved


def under_mise() -> bool:
    return os.environ.get("DR_PROVIDERS_LIVE_UNDER_MISE") == "1"
