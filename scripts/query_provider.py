#!/usr/bin/env python3
"""Query OpenRouter with model, reasoning, sampling, and token options."""

import typer

from dr_providers.config import ReasoningSpec, SamplingControls
from dr_providers.names import EffortLevel, ProviderName
from dr_providers.query import OpenRouterProvider
from dr_providers.query.response import LlmResponse
from dr_providers.query.transport import execute_query


def _parse_effort(value: str) -> EffortLevel:
    normalized = value.lower()
    try:
        return EffortLevel(normalized)
    except ValueError:
        allowed = ", ".join(level.value for level in EffortLevel)
        raise typer.BadParameter(
            f"Invalid effort {value!r}. Expected one of: {allowed}"
        ) from None


def _build_reasoning(
    *,
    effort: str | None,
    reasoning_enabled: bool | None,
) -> ReasoningSpec | None:
    if effort is not None and reasoning_enabled is not None:
        raise typer.BadParameter(
            "Use --effort for effort-style models or "
            "--reasoning-enabled/--reasoning-disabled for toggle-style "
            "models, not both."
        )
    if effort is not None:
        return ReasoningSpec(effort=_parse_effort(effort))
    if reasoning_enabled is not None:
        return ReasoningSpec(enabled=reasoning_enabled)
    return None


def _build_sampling(
    *,
    temperature: float | None,
    top_p: float | None,
) -> SamplingControls | None:
    if temperature is None and top_p is None:
        return None
    return SamplingControls(temperature=temperature, top_p=top_p)


def _query_provider(  # noqa: PLR0913
    *,
    effort: str | None,
    reasoning_enabled: bool | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    model: str,
    message: str,
) -> LlmResponse:
    reasoning = _build_reasoning(
        effort=effort,
        reasoning_enabled=reasoning_enabled,
    )
    sampling = _build_sampling(temperature=temperature, top_p=top_p)
    with OpenRouterProvider() as provider:
        return execute_query(
            provider=provider,
            provider_name=ProviderName.OPENROUTER,
            model=model,
            prompt=message,
            reasoning=reasoning,
            sampling=sampling,
            max_tokens=max_tokens,
        )


def main(  # noqa: PLR0913
    model: str = typer.Option(..., "--model", help="OpenRouter model id."),
    message: str = typer.Option(
        ...,
        "--message",
        "-m",
        help="User message to send.",
    ),
    effort: str | None = typer.Option(
        None,
        "--effort",
        "-e",
        help="Reasoning effort (low, medium, high) for effort-style models.",
    ),
    reasoning_enabled: bool | None = typer.Option(  # noqa: FBT001
        None,
        "--reasoning-enabled/--reasoning-disabled",
        help="Enable or disable reasoning for toggle-style models.",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Completion token limit. Omit to use provider defaults.",
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        "--temp",
        help="Sampling temperature.",
    ),
    top_p: float | None = typer.Option(
        None,
        "--top-p",
        help="Sampling top-p.",
    ),
) -> None:
    response = _query_provider(
        model=model,
        message=message,
        effort=effort,
        reasoning_enabled=reasoning_enabled,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    typer.echo(response.text)
    typer.echo(f"({response.latency_ms} ms)", err=True)


if __name__ == "__main__":
    typer.run(main)
