#!/usr/bin/env bash

set -euo pipefail

repository_root="$({
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
    pwd -P
})"
cd -- "${repository_root}"

uv sync --locked --extra cli
uv run --locked --extra cli ruff format --check .
uv run --locked --extra cli ruff check .
uv run --locked --extra cli ty check
uv run --locked --extra cli pytest
uvx tombi@1.2.5 lint --offline .defs/terms.toml
uv run --locked --extra cli python scripts/check_defs.py
