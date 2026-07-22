"""CLI for one-shot provider calls over the kernel.

Thin: build a ``ProviderCallRequest`` from flags, run it through
``HttpProvider``, and print the transport outcome. Not wired into
``dr_providers``'s pure import surface — this module (and its ``[cli]``
extra) is only imported when running the CLI, so it is free to import
``HttpProvider`` (and therefore httpx) at module level.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

import typer

from dr_providers.config import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    ProviderCallConfig,
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)
from dr_providers.controls import GenerationControls, ReasoningEffort
from dr_providers.outcome import ProviderTransportResponse
from dr_providers.policy import ProviderTransportPolicy
from dr_providers.request import ProviderCallRequest
from dr_providers.transcript import MessageRole, PromptMessage, Transcript
from dr_providers.transport import HttpProvider

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_RETRIES = 0


class ProviderChoice(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai-responses"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


CONFIG_FACTORIES: dict[ProviderChoice, Callable[..., ProviderCallConfig]] = {
    ProviderChoice.OPENROUTER: openrouter_chat_config,
    ProviderChoice.OPENAI: openai_chat_config,
    ProviderChoice.OPENAI_RESPONSES: openai_responses_config,
    ProviderChoice.GEMINI: gemini_chat_config,
    ProviderChoice.ANTHROPIC: anthropic_messages_config,
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
    "--token-limit", help="Max output/completion tokens."
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
    config = CONFIG_FACTORIES[provider](
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

    kind = config.route.provider
    policy = ProviderTransportPolicy(
        api_key_env=str(DEFAULT_API_KEY_ENVS[kind]),
        base_url=str(DEFAULT_BASE_URLS[kind]),
        native_retry_count=retries,
    )
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
