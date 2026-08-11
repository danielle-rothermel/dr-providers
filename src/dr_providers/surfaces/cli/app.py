from __future__ import annotations

from enum import StrEnum
from threading import Event
from typing import Annotated

import typer

from dr_providers.core.failures import ControlValidationError
from dr_providers.lifecycle import (
    AcceptAllSemanticResponseClassifier,
    ProviderCallOutcomeKind,
    ProviderCallState,
    StandardProviderCallRetryPolicy,
    run_local_provider_call,
)
from dr_providers.modeling.controls import GenerationControls, ReasoningEffort
from dr_providers.modeling.presets import FACTORY_BY_KIND, ProviderFactoryKind
from dr_providers.modeling.request import ProviderCallRequest
from dr_providers.modeling.transcript import (
    MessageRole,
    PromptMessage,
    Transcript,
)
from dr_providers.transport.http import HttpProvider
from dr_providers.transport.policy import policy_for

CLI_TIMEOUT_SECONDS = 120.0
CLI_IDLE_TIMEOUT_SECONDS = 90.0
CLI_MAX_REQUEST_BYTES = 1024 * 1024
CLI_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


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
        "Max output/completion tokens. Required when --provider is anthropic."
    ),
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
) -> None:
    """Run a single-shot provider query and print the response text."""
    factory = FACTORY_BY_KIND[_CHOICE_TO_FACTORY_KIND[provider]]
    try:
        config = factory(
            model=model,
            controls=GenerationControls(
                temperature=temperature,
                top_p=top_p,
                token_limit=token_limit,
                reasoning=effort,
            ),
        )
    except ControlValidationError as exc:
        failure = exc.failure
        typer.echo(f"failure: {failure.code}: {failure.message}", err=True)
        raise typer.Exit(code=1) from exc
    messages: list[PromptMessage] = []
    if system is not None:
        messages.append(PromptMessage(role=MessageRole.SYSTEM, content=system))
    messages.append(PromptMessage(role=MessageRole.USER, content=message))
    request = ProviderCallRequest(
        config=config,
        transcript=Transcript(messages=tuple(messages)),
    )

    classifier = AcceptAllSemanticResponseClassifier()
    state = ProviderCallState.initial(
        request=request,
        retry_policy=StandardProviderCallRetryPolicy(),
        classifier_identifier=classifier.identifier,
    )
    transport_policy = policy_for(
        config.route.provider,
        timeout_seconds=CLI_TIMEOUT_SECONDS,
        idle_timeout_seconds=CLI_IDLE_TIMEOUT_SECONDS,
        max_connections=1,
        max_keepalive_connections=1,
        max_request_bytes=CLI_MAX_REQUEST_BYTES,
        max_response_bytes=CLI_MAX_RESPONSE_BYTES,
    )
    with HttpProvider(policy=transport_policy) as http_provider:
        result = run_local_provider_call(
            provider=http_provider,
            state=state,
            classifier=classifier,
            cancellation=Event(),
        )

    final_evidence = result.completed_invocations[-1].observation.evidence
    if result.outcome.kind is not ProviderCallOutcomeKind.ACCEPTED:
        failure = final_evidence.failure
        if failure is not None:
            typer.echo(f"failure: {failure.code}: {failure.message}", err=True)
        else:
            assert result.outcome.invocation_outcome is not None
            typer.echo(
                f"failure: {result.outcome.invocation_outcome.value}",
                err=True,
            )
        raise typer.Exit(code=1)

    response = final_evidence.response
    assert response is not None
    typer.echo(response.text)
    typer.echo(f"model: {response.model}", err=True)
    typer.echo(f"stop_reason: {response.stop_reason}", err=True)
    if response.usage is not None:
        typer.echo(f"usage: {response.usage.model_dump()}", err=True)
    for warning in response.warnings:
        typer.echo(f"warning: {warning.code}", err=True)


if __name__ == "__main__":
    app()
