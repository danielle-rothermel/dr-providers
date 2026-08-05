from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import typer

from dr_providers.modeling.controls import GenerationControls, ReasoningEffort
from dr_providers.modeling.presets import FACTORY_BY_KIND, ProviderFactoryKind
from dr_providers.modeling.request import ProviderCallRequest
from dr_providers.modeling.transcript import (
    MessageRole,
    PromptMessage,
    Transcript,
)
from dr_providers.outcomes.models import ProviderTransportResponse
from dr_providers.transport.http import HttpProvider
from dr_providers.transport.policy import policy_for

DEFAULT_RETRIES = 0

# Anthropic requires max_tokens; the CLI defaults it when omitted.
DEFAULT_ANTHROPIC_TOKEN_LIMIT = 4096


class ProviderChoice(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai-responses"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


_CHOICE_TO_FACTORY_KIND: dict[ProviderChoice, ProviderFactoryKind] = {
    ProviderChoice.OPENROUTER: ProviderFactoryKind.OPENROUTER,
    ProviderChoice.OPENAI: ProviderFactoryKind.OPENAI,
    ProviderChoice.OPENAI_RESPONSES: ProviderFactoryKind.OPENAI_RESPONSES,
    ProviderChoice.GEMINI: ProviderFactoryKind.GEMINI,
    ProviderChoice.ANTHROPIC: ProviderFactoryKind.ANTHROPIC,
}

PROVIDER_OPTION = typer.Option("--provider", help="Provider to call.")
MODEL_OPTION = typer.Option("--model", help="Model name.")
MESSAGE_OPTION = typer.Option("-m", "--message", help="User message content.")
SYSTEM_OPTION = typer.Option(
    "--system", help="Optional system message content."
)
EFFORT_OPTION = typer.Option("--effort", help="Reasoning effort level.")
TEMPERATURE_OPTION = typer.Option(
    "--temperature", "--temp", help="Sampling temperature."
)
TOP_P_OPTION = typer.Option("--top-p", help="Nucleus sampling top-p.")
TOKEN_LIMIT_OPTION = typer.Option(
    "--token-limit",
    help=(
        "Max output/completion tokens. Required by the anthropic preset; "
        f"if omitted for --provider anthropic, defaults to "
        f"{DEFAULT_ANTHROPIC_TOKEN_LIMIT}."
    ),
)
RETRIES_OPTION = typer.Option(
    "--retries",
    help="Native transport retry count (defaults to zero).",
)

app = typer.Typer(help="dr-providers CLI: one-shot provider calls.")


@app.command()
def query(  # noqa: PLR0913
    provider: Annotated[ProviderChoice, PROVIDER_OPTION],
    model: Annotated[str, MODEL_OPTION],
    message: Annotated[str, MESSAGE_OPTION],
    system: Annotated[str | None, SYSTEM_OPTION] = None,
    effort: Annotated[ReasoningEffort | None, EFFORT_OPTION] = None,
    temperature: Annotated[float | None, TEMPERATURE_OPTION] = None,
    top_p: Annotated[float | None, TOP_P_OPTION] = None,
    token_limit: Annotated[int | None, TOKEN_LIMIT_OPTION] = None,
    retries: Annotated[int, RETRIES_OPTION] = DEFAULT_RETRIES,
) -> None:
    """Run a single-shot provider query and print the response text."""
    if token_limit is None and provider is ProviderChoice.ANTHROPIC:
        token_limit = DEFAULT_ANTHROPIC_TOKEN_LIMIT
    factory = FACTORY_BY_KIND[_CHOICE_TO_FACTORY_KIND[provider]]
    config = factory(
        model=model,
        controls=GenerationControls(
            temperature=temperature,
            top_p=top_p,
            token_limit=token_limit,
            reasoning=effort,
        ),
    )
    messages: list[PromptMessage] = []
    if system is not None:
        messages.append(PromptMessage(role=MessageRole.SYSTEM, content=system))
    messages.append(PromptMessage(role=MessageRole.USER, content=message))
    request = ProviderCallRequest(
        config=config,
        transcript=Transcript(messages=tuple(messages)),
    )

    policy = policy_for(config.route.provider, native_retry_count=retries)
    with HttpProvider(policy=policy) as http_provider:
        outcome = http_provider.complete(request)

    if not isinstance(outcome, ProviderTransportResponse):
        typer.echo(f"failure: {outcome.code}: {outcome.message}", err=True)
        raise typer.Exit(code=1)

    typer.echo(outcome.text)
    typer.echo(f"model: {outcome.model}", err=True)
    typer.echo(f"finish_reason: {outcome.finish_reason}", err=True)
    if outcome.usage is not None:
        typer.echo(f"usage: {outcome.usage.model_dump()}", err=True)
    for warning in outcome.warnings:
        typer.echo(f"warning: {warning.code}", err=True)


if __name__ == "__main__":
    app()
