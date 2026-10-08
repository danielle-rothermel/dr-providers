"""Network- and torch-free scripted local backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_providers.local.backend import (
    GenerationOutput,
    LocalBackendFailure,
    ScoredContinuation,
)

if TYPE_CHECKING:
    from dr_providers.modeling.controls import (
        GenerationControls,
        PromptRendering,
    )
    from dr_providers.modeling.transcript import Transcript
    from dr_providers.outcomes.evidence import LocalExecutionEvidence


@dataclass(frozen=True, slots=True)
class GenerateCall:
    transcript: Transcript
    rendering: PromptRendering
    controls: GenerationControls


@dataclass(frozen=True, slots=True)
class ScoreCall:
    context: str
    continuations: tuple[str, ...]
    token_logprobs: bool


class FakeLocalBackend:
    """Consume scripted results in order, then repeat the final result."""

    def __init__(
        self,
        *,
        execution: LocalExecutionEvidence,
        results: list[
            GenerationOutput
            | tuple[ScoredContinuation, ...]
            | LocalBackendFailure
        ],
    ) -> None:
        if not results:
            raise ValueError("fake local backend requires at least one result")
        self._execution = execution
        self._results = list(results)
        self.calls: list[GenerateCall | ScoreCall] = []
        self.release_memory_calls = 0
        self.close_calls = 0

    def execution(self) -> LocalExecutionEvidence:
        return self._execution

    def _next(
        self,
    ) -> GenerationOutput | tuple[ScoredContinuation, ...]:
        result = self._results[
            min(len(self.calls) - 1, len(self._results) - 1)
        ]
        if isinstance(result, LocalBackendFailure):
            raise result
        return result

    def generate(
        self,
        *,
        transcript: Transcript,
        rendering: PromptRendering,
        controls: GenerationControls,
    ) -> GenerationOutput:
        self.calls.append(GenerateCall(transcript, rendering, controls))
        result = self._next()
        if not isinstance(result, GenerationOutput):
            raise TypeError("fake generate requires a GenerationOutput")
        return result

    def score(
        self,
        *,
        context: str,
        continuations: tuple[str, ...],
        token_logprobs: bool,
    ) -> tuple[ScoredContinuation, ...]:
        self.calls.append(ScoreCall(context, continuations, token_logprobs))
        result = self._next()
        if not isinstance(result, tuple):
            raise TypeError("fake score requires scored continuations")
        return result

    def release_memory(self) -> None:
        self.release_memory_calls += 1

    def close(self) -> None:
        self.close_calls += 1
