from __future__ import annotations

import sys

_MISSING_CLI_HINT = (
    "dr-providers: the CLI requires the cli extra: "
    "pip install 'dr-providers[cli]'"
)


def main() -> None:
    try:
        from dr_providers.surfaces.cli.app import (  # noqa: PLC0415
            app,
        )
    except ImportError:
        print(_MISSING_CLI_HINT, file=sys.stderr)
        raise SystemExit(1) from None
    app()


if __name__ == "__main__":
    main()
