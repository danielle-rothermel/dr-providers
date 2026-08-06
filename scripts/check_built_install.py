import argparse
from importlib.metadata import version
from pathlib import Path

import dr_providers
from dr_providers.surfaces.serve.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_version")
    expected_version = parser.parse_args().expected_version

    if version("dr-providers") != expected_version:
        raise RuntimeError("installed distribution version does not match")
    if dr_providers.__version__ != expected_version:
        raise RuntimeError("public package version does not match")
    if not Path(dr_providers.__file__).with_name("py.typed").is_file():
        raise RuntimeError("built distribution does not contain py.typed")
    create_app()


if __name__ == "__main__":
    main()
