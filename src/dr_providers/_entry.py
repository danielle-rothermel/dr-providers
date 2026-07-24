"""Console-script entry point for the ``dr-providers`` CLI.

The console script is installed unconditionally by pip, but the typer CLI
lives in the optional ``[cli]`` extra. ``dr_providers.cli`` imports typer at
module top, so this shim must NOT import typer (or the cli module) at module
level. It defers the import into ``main`` and turns the ImportError raised
when the extra is absent into a clear one-line hint plus a nonzero exit,
rather than an opaque traceback.
"""

from __future__ import annotations

import sys

_MISSING_CLI_HINT = (
    "dr-providers: the CLI requires the cli extra: "
    "pip install 'dr-providers[cli]'"
)


def main() -> None:
    """Invoke the typer CLI, or exit nonzero with a hint if it is missing."""
    try:
        from dr_providers.cli import app  # noqa: PLC0415 -- deferred import
    except ImportError:
        print(_MISSING_CLI_HINT, file=sys.stderr)
        raise SystemExit(1) from None
    app()


if __name__ == "__main__":
    main()
