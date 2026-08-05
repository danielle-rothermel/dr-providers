import json

import typer
import uvicorn

from dr_providers.surfaces.serve.app import create_app

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 8322
PORT_OPTION = typer.Option(DEFAULT_PORT, help="Localhost port")

app = typer.Typer(help="dr-providers serve facade", no_args_is_help=True)


@app.command()
def serve(port: int = PORT_OPTION) -> None:
    """Run the query facade on localhost."""
    typer.echo(f"dr-providers serve listening on http://{LOCALHOST}:{port}")
    uvicorn.run(create_app(), host=LOCALHOST, port=port, log_level="info")


@app.command()
def openapi() -> None:
    """Print the OpenAPI schema."""
    typer.echo(json.dumps(create_app().openapi(), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
