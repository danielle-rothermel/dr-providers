#!/usr/bin/env bash

set -euo pipefail

repository_root="$({
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
    pwd -P
})"
cd -- "${repository_root}"

uv sync --locked --all-extras
uv run --locked --all-extras ruff format --check .
uv run --locked --all-extras ruff check .
uv run --locked --all-extras ty check
uv run --locked --all-extras pytest
uvx tombi@1.2.5 lint --offline .defs/terms.toml
uv run --locked --all-extras python scripts/check_defs.py
