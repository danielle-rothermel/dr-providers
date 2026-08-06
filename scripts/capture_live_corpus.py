#!/usr/bin/env -S uv run python

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import (
    parse_qsl,
    unquote_plus,
    urlencode,
    urlsplit,
    urlunsplit,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from dr_providers import (  # noqa: E402
    GenerationControls,
    ProviderTransportResponse,
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    parse_response,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

from scripts.live_matrix_support import (  # noqa: E402
    CAPTURE_DIR_ENV,
    LIVE_CASES,
    LiveCase,
    credential_values,
    mapped_provider_environment,
    missing_credentials,
    require_external_capture_dir,
    under_mise,
)
from scripts.run_live_matrix import run_selected_cases  # noqa: E402

REDACTED = "[REDACTED]"
VALIDATED_DIR_PREFIX = "validated-"
SENSITIVE_KEY_NAMES = {
    "accesstoken",
    "apikey",
    "authorization",
    "clientsecret",
    "cookie",
    "proxyauthorization",
    "refreshtoken",
    "setcookie",
    "xapikey",
}


class CaptureValidationError(ValueError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture, validate, redact, and deliberately promote "
            "live wire evidence."
        )
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    capture = subcommands.add_parser(
        "capture", help="Run and validate a complete five-case live capture."
    )
    capture.add_argument(
        "--staging-dir",
        type=Path,
        help="Empty external directory for raw and validated captures.",
    )
    capture.add_argument(
        "--promote",
        action="store_true",
        help="Deliberately update data/wire-corpus after validation.",
    )

    promote = subcommands.add_parser(
        "promote", help="Validate and promote an existing complete capture."
    )
    promote.add_argument("staging_dir", type=Path)
    promote.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "data" / "wire-corpus",
        help=argparse.SUPPRESS,
    )
    return parser


def _normal_key(value: str) -> str:
    return "".join(
        character for character in value.lower() if character.isalnum()
    )


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return value

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"redacted@{netloc.rsplit('@', maxsplit=1)[1]}"

    query = urlencode(
        [
            (
                key,
                REDACTED if _normal_key(key) in SENSITIVE_KEY_NAMES else item,
            )
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    fragment_keys = (
        part.partition("=")[0] for part in re.split(r"[?&]", parsed.fragment)
    )
    fragment = (
        REDACTED
        if any(
            _normal_key(unquote_plus(key)) in SENSITIVE_KEY_NAMES
            for key in fragment_keys
        )
        else parsed.fragment
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, fragment))


def redact_capture(value: Any, secrets: Sequence[str]) -> Any:
    """Redact credential fields and configured secret values."""
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED
            if _normal_key(str(key)) in SENSITIVE_KEY_NAMES
            else redact_capture(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_capture(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            redacted = redacted.replace(secret, REDACTED)
        return _redact_url(redacted)
    return value


def _config_for_case(case: LiveCase):
    if case.case_id == "openai_responses":
        return openai_responses_config(model="live-capture-validation")
    if case.case_id == "anthropic_messages":
        return anthropic_messages_config(
            model="live-capture-validation",
            controls=GenerationControls(token_limit=1),
        )
    factories = {
        "openrouter_chat_completions": openrouter_chat_config,
        "openai_chat_completions": openai_chat_config,
        "gemini_chat_completions": gemini_chat_config,
    }
    return factories[case.case_id](model="live-capture-validation")


def _load_capture(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        msg = f"could not read capture {path.name}: {error}"
        raise CaptureValidationError(msg) from error
    if not isinstance(loaded, dict):
        msg = f"capture {path.name} must contain a JSON object"
        raise CaptureValidationError(msg)
    return loaded


def prepare_capture(staging_dir: Path, *, secrets: Sequence[str] = ()) -> Path:
    staging_dir = require_external_capture_dir(staging_dir)
    expected = {case.corpus_file: case for case in LIVE_CASES}
    present = {path.name for path in staging_dir.glob("*.json")}
    missing = sorted(expected.keys() - present)
    unknown = sorted(present - expected.keys())
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unexpected: {', '.join(unknown)}")
        msg = f"capture set is not complete ({'; '.join(details)})"
        raise CaptureValidationError(msg)

    prepared: dict[str, dict[str, Any]] = {}
    for file_name, case in expected.items():
        body = redact_capture(_load_capture(staging_dir / file_name), secrets)
        if not isinstance(body, dict):
            msg = f"capture {file_name} did not remain a JSON object"
            raise CaptureValidationError(msg)
        serialized = json.dumps(body, sort_keys=True)
        if any(secret in serialized for secret in secrets):
            msg = f"capture {file_name} retained configured credential data"
            raise CaptureValidationError(msg)
        outcome = parse_response(body, config=_config_for_case(case))
        if not isinstance(outcome, ProviderTransportResponse):
            msg = (
                f"capture {file_name} does not parse as a successful response"
            )
            raise CaptureValidationError(msg)
        if not outcome.text.strip() or outcome.usage is None:
            msg = f"capture {file_name} lacks live verification evidence"
            raise CaptureValidationError(msg)
        prepared[file_name] = body

    validated_dir = Path(
        tempfile.mkdtemp(prefix=VALIDATED_DIR_PREFIX, dir=staging_dir)
    )
    for file_name, body in prepared.items():
        (validated_dir / file_name).write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n"
        )
    return validated_dir


def _install_validated_capture(validated_dir: Path, corpus_dir: Path) -> None:
    expected_names = {case.corpus_file for case in LIVE_CASES}
    present = {path.name for path in validated_dir.glob("*.json")}
    if present != expected_names:
        msg = "validated capture is not the complete five-case set"
        raise CaptureValidationError(msg)

    corpus_dir.mkdir(parents=True, exist_ok=True)
    unknown = sorted(
        path.name
        for path in corpus_dir.glob("*.json")
        if path.name not in expected_names
    )
    if unknown:
        msg = f"curated corpus has unexpected files: {', '.join(unknown)}"
        raise CaptureValidationError(msg)

    installations = [
        (
            corpus_dir / f".{file_name}.promotion",
            corpus_dir / file_name,
            corpus_dir / f".{file_name}.rollback",
            (corpus_dir / file_name).exists(),
        )
        for file_name in sorted(expected_names)
    ]
    installation_complete = False
    replacements_started = False
    rollback_complete = False
    replaced: list[tuple[Path, Path, bool]] = []
    try:
        for temporary, destination, backup, existed in installations:
            temporary.write_bytes(
                (validated_dir / destination.name).read_bytes()
            )
            if existed:
                backup.write_bytes(destination.read_bytes())

        replacements_started = True
        try:
            for temporary, destination, backup, existed in installations:
                temporary.replace(destination)
                replaced.append((destination, backup, existed))
        except BaseException:
            for destination, backup, existed in reversed(replaced):
                if existed:
                    backup.replace(destination)
                else:
                    destination.unlink(missing_ok=True)
            rollback_complete = True
            raise
        installation_complete = True
    finally:
        for temporary, _, backup, _ in installations:
            temporary.unlink(missing_ok=True)
            if (
                installation_complete
                or rollback_complete
                or not replacements_started
            ):
                backup.unlink(missing_ok=True)


def promote_capture(
    staging_dir: Path,
    corpus_dir: Path,
    *,
    secrets: Sequence[str] = (),
) -> Path:
    validated_dir = prepare_capture(staging_dir, secrets=secrets)
    _install_validated_capture(validated_dir, corpus_dir)
    return validated_dir


def _reexec_capture_under_mise(argv: Sequence[str]) -> int:
    command = [
        "mise",
        "exec",
        "--",
        "env",
        "DR_PROVIDERS_LIVE_UNDER_MISE=1",
        "uv",
        "run",
        "python",
        str(ROOT / "scripts" / "capture_live_corpus.py"),
        *argv,
    ]
    try:
        return subprocess.run(  # noqa: S603
            command, cwd=ROOT, check=False
        ).returncode
    except FileNotFoundError:
        print("ERROR: mise is not installed or not on PATH.", file=sys.stderr)
        return 2


def _capture(staging_dir: Path | None, *, promote: bool) -> int:
    environment = mapped_provider_environment(os.environ)
    missing = missing_credentials(LIVE_CASES, environment)
    if missing:
        detail = ", ".join(missing)
        print(
            f"ERROR: missing credentials for complete capture: {detail}",
            file=sys.stderr,
        )
        return 2

    if staging_dir is None:
        staging_dir = Path(
            tempfile.mkdtemp(prefix="dr-providers-live-capture-")
        )
    staging_dir = require_external_capture_dir(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    if any(staging_dir.iterdir()):
        print(
            f"ERROR: staging directory is not empty: {staging_dir}",
            file=sys.stderr,
        )
        return 2

    environment[CAPTURE_DIR_ENV] = str(staging_dir)
    result = run_selected_cases((), environment=environment)
    if result != 0:
        print(
            "ERROR: live capture failed; partial staging retained at "
            f"{staging_dir}",
            file=sys.stderr,
        )
        return result

    try:
        secrets = credential_values(environment)
        if promote:
            validated = promote_capture(
                staging_dir,
                ROOT / "data" / "wire-corpus",
                secrets=secrets,
            )
        else:
            validated = prepare_capture(staging_dir, secrets=secrets)
    except CaptureValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Validated capture: {validated}")
    if promote:
        print(f"Promoted curated corpus: {ROOT / 'data' / 'wire-corpus'}")
    else:
        print(
            "Promotion is deliberate; run: "
            f"scripts/capture_live_corpus.py promote {staging_dir}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    if not under_mise():
        return _reexec_capture_under_mise(arguments)
    if args.command == "capture":
        return _capture(args.staging_dir, promote=args.promote)

    environment = mapped_provider_environment(os.environ)
    try:
        promote_capture(
            args.staging_dir,
            args.corpus_dir,
            secrets=credential_values(environment),
        )
    except CaptureValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Promoted curated corpus: {args.corpus_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
