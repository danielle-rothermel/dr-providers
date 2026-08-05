from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from dr_providers.surfaces.serve import cli

runner = CliRunner()

EXPECTED_ROUTES = {"/health", "/build_payload", "/query", "/variance"}


def test_openapi_command_emits_the_public_route_set() -> None:
    result = runner.invoke(cli.app, ["openapi"])

    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert set(schema["paths"]) == EXPECTED_ROUTES


def test_serve_command_wires_localhost_port_and_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = object()
    observed: dict[str, object] = {}

    def fake_run(app: object, *, host: str, port: int, log_level: str) -> None:
        observed.update(
            app=app,
            host=host,
            port=port,
            log_level=log_level,
        )

    monkeypatch.setattr(cli, "create_app", lambda: application)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--port", "9999"])

    assert result.exit_code == 0
    assert result.stdout == (
        "dr-providers serve listening on http://127.0.0.1:9999\n"
    )
    assert observed == {
        "app": application,
        "host": "127.0.0.1",
        "port": 9999,
        "log_level": "info",
    }


def test_serve_module_openapi_smoke_from_external_cwd(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dr_providers.surfaces.serve",
            "openapi",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    schema = json.loads(result.stdout)
    assert set(schema["paths"]) == EXPECTED_ROUTES
    assert result.stderr == ""
