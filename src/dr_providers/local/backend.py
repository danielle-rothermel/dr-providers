"""Torch-free boundary between local provider bookkeeping and model work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dr_providers.core.failures import ProviderFailureError
from dr_providers.outcomes.models import LOCAL_FAILURE_RECOVERABILITY

if TYPE_CHECKING:
    from collections.abc import Callable

    from dr_providers.core.failures import ProviderFailure
    from dr_providers.modeling.controls import (
        GenerationControls,
        PromptRendering,
    )
    from dr_providers.modeling.transcript import Transcript
    from dr_providers.outcomes.evidence import LocalExecutionEvidence
    from dr_providers.outcomes.models import ProviderStopReason


class LocalBackendFailure(ProviderFailureError):  # noqa: N818 -- public name
    """An expected local failure, translated into invocation evidence."""

    def __init__(self, failure: ProviderFailure) -> None:
        if failure.code not in LOCAL_FAILURE_RECOVERABILITY:
            raise ValueError("local backend failure requires a local code")
        if (
            failure.recoverability
            is not LOCAL_FAILURE_RECOVERABILITY[failure.code]
        ):
            raise ValueError("local backend failure recoverability mismatch")
        super().__init__(failure)


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    stop_reason: ProviderStopReason


@dataclass(frozen=True, slots=True)
class ScoredContinuation:
    log_likelihood: float
    token_count: int
    token_logprobs: tuple[float, ...] | None
    # Unpadded forward-input length, including the context/prefix and excluding
    # the final target token. Recomputed context is counted for every row.
    input_tokens: int


class LocalBackend(Protocol):
    def execution(self) -> LocalExecutionEvidence: ...

    def generate(
        self,
        *,
        transcript: Transcript,
        rendering: PromptRendering,
        controls: GenerationControls,
    ) -> GenerationOutput: ...

    def score(
        self,
        *,
        context: str,
        continuations: tuple[str, ...],
        token_logprobs: bool,
    ) -> tuple[ScoredContinuation, ...]: ...

    def release_memory(self) -> None: ...

    def close(self) -> None: ...


def encode_pair(
    *,
    context: str,
    continuation: str,
    encode: Callable[[str], list[int]],
    prefix_token: int | None,
) -> tuple[list[int], list[int]]:
    """lm-eval-derived causal encoding, with an empty-after-rstrip prefix.

    ``encode`` must disable automatic special-token insertion. This function
    deliberately slices by encoded context length, even at a token merge.
    """
    trailing = len(context) - len(context.rstrip())
    if trailing:
        continuation = context[-trailing:] + continuation
        context = context[:-trailing]
    if not context:
        if prefix_token is None:
            raise ValueError("empty-context scoring requires a BOS or EOS id")
        context_ids = [prefix_token]
        continuation_ids = encode(continuation)
    else:
        context_ids = encode(context)
        continuation_ids = encode(context + continuation)[len(context_ids) :]
    if not context_ids or not continuation_ids:
        raise ValueError("scoring requires context and continuation tokens")
    return context_ids, continuation_ids
