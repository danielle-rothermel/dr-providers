from __future__ import annotations

import builtins
import sys
from importlib.metadata import distribution
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType


def test_installed_entry_point_resolves_and_invokes_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_providers.surfaces.cli import entry

    entry_points = [
        item
        for item in distribution("dr-providers").entry_points
        if item.group == "console_scripts" and item.name == "dr-providers"
    ]
    assert len(entry_points) == 1
    installed_entry = entry_points[0]
    assert installed_entry.value == "dr_providers.surfaces.cli.entry:main"
    assert installed_entry.load() is entry.main

    monkeypatch.setattr(sys, "argv", ["dr-providers", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        installed_entry.load()()
    assert excinfo.value.code == 0


def test_entry_main_missing_cli_extra_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dr_providers.surfaces.cli import entry

    monkeypatch.delitem(
        sys.modules, "dr_providers.surfaces.cli.app", raising=False
    )
    monkeypatch.setitem(sys.modules, "typer", None)

    with pytest.raises(SystemExit) as excinfo:
        entry.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "dr-providers[cli]" in captured.err


def test_entry_main_preserves_unrelated_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dr_providers.surfaces.cli import entry

    original_import = builtins.__import__

    def fail_app_import(
        name: str,
        globalns: Mapping[str, object] | None = None,
        localns: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "dr_providers.surfaces.cli.app":
            raise ModuleNotFoundError(
                "missing internal dependency",
                name="dr_providers.synthetic_internal",
            )
        return original_import(name, globalns, localns, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_app_import)

    with pytest.raises(
        ModuleNotFoundError, match="missing internal dependency"
    ):
        entry.main()

    assert "dr-providers[cli]" not in capsys.readouterr().err
