"""Package-layout boundary tests."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "dr_providers"
TEST_ROOT = REPO_ROOT / "tests"

FUNCTIONAL_AREAS = {
    "core",
    "modeling",
    "outcomes",
    "surfaces",
    "translation",
    "transport",
}
APPROVED_TEST_AREAS = FUNCTIONAL_AREAS | {"data"}
SURFACE_AREAS = {"cli", "serve", "testing"}


def _directories_with_python(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and any(path.rglob("*.py"))
    }


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_source_tree_exposes_only_functional_areas() -> None:
    assert _directories_with_python(PACKAGE_ROOT) == FUNCTIONAL_AREAS
    assert {path.name for path in PACKAGE_ROOT.glob("*.py")} == {"__init__.py"}
    assert _directories_with_python(PACKAGE_ROOT / "surfaces") == SURFACE_AREAS


def test_test_tree_uses_only_approved_top_level_areas() -> None:
    assert _directories_with_python(TEST_ROOT) == APPROVED_TEST_AREAS


def test_inner_areas_do_not_depend_on_surfaces() -> None:
    for area in FUNCTIONAL_AREAS - {"surfaces"}:
        for path in (PACKAGE_ROOT / area).rglob("*.py"):
            surface_imports = {
                imported
                for imported in _absolute_imports(path)
                if imported == "dr_providers.surfaces"
                or imported.startswith("dr_providers.surfaces.")
            }
            assert not surface_imports, (
                f"{path.relative_to(REPO_ROOT)} imports {surface_imports}"
            )
