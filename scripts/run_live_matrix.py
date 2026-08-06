#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.live_matrix_support import (  # noqa: E402
    LIVE_CASES,
    LiveProvider,
    mapped_provider_environment,
    missing_credentials,
    select_cases,
    under_mise,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected live-provider cases under the repository's "
            "mise environment."
        )
    )
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument(
        "--provider",
        action="append",
        choices=[provider.value for provider in LiveProvider],
        default=[],
        help="Run every case for this provider; repeatable.",
    )
    selectors.add_argument(
        "--case",
        action="append",
        choices=[case.case_id for case in LIVE_CASES],
        default=[],
        help="Run this exact matrix case; repeatable.",
    )
    return parser


def _reexec_under_mise(argv: Sequence[str]) -> int:
    command = [
        "mise",
        "exec",
        "--",
        "env",
        "DR_PROVIDERS_LIVE_UNDER_MISE=1",
        "uv",
        "run",
        "python",
        str(ROOT / "scripts" / "run_live_matrix.py"),
        *argv,
    ]
    try:
        return subprocess.run(  # noqa: S603
            command, cwd=ROOT, check=False
        ).returncode
    except FileNotFoundError:
        print("ERROR: mise is not installed or not on PATH.", file=sys.stderr)
        return 2


def run_selected_cases(
    argv: Sequence[str], *, environment: dict[str, str] | None = None
) -> int:
    args = _parser().parse_args(argv)
    cases = select_cases(providers=args.provider, case_ids=args.case)
    source_environment = os.environ if environment is None else environment
    child_environment = mapped_provider_environment(source_environment)
    missing = missing_credentials(cases, child_environment)
    if missing:
        detail = ", ".join(missing)
        print(
            f"ERROR: missing credentials for selected cases: {detail}",
            file=sys.stderr,
        )
        return 2

    command = [
        "uv",
        "run",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-m",
        "live",
        *(case.pytest_node for case in cases),
    ]
    return subprocess.run(  # noqa: S603
        command,
        cwd=ROOT,
        env=child_environment,
        check=False,
    ).returncode


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if not under_mise():
        return _reexec_under_mise(arguments)
    return run_selected_cases(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
