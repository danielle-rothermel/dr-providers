#!/usr/bin/env python3
"""Query OpenRouter with fixed gpt-5-nano settings at lowest reasoning effort."""

import typer

from dr_providers.config import ReasoningSpec
from dr_providers.names import EffortLevel, MessageRole, ProviderName
from dr_providers.query import LlmRequest, Message, OpenRouterProvider

MODEL = "openai/gpt-5-nano"
REASONING = ReasoningSpec(effort=EffortLevel.LOW)
MAX_TOKENS = 512
PROMPT = "Say hello in one short sentence."


def main(
    message: str = typer.Option(
        PROMPT,
        "--message",
        "-m",
        help="User message to send.",
    ),
) -> None:
    request = LlmRequest(
        provider=ProviderName.OPENROUTER,
        model=MODEL,
        messages=[Message(role=MessageRole.USER, content=message)],
        max_tokens=MAX_TOKENS,
        reasoning=REASONING,
    )
    with OpenRouterProvider() as provider:
        response = provider.generate(request)
    typer.echo(response.text)
    typer.echo(f"({response.latency_ms} ms)", err=True)


if __name__ == "__main__":
    typer.run(main)
