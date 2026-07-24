"""Serve-side query and variance machinery over the kernel.

Pure library logic: declarative query specs resolve to kernel
requests, run against any ``Provider`` (Scripted or Http), and come
back as structured results with conformance warnings applied. The
variance runner fans one prompt across models x samples and reports
output dispersion — the same records the playground downloads as
JSONL.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dr_providers.provider import Provider

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
)

from dr_providers._factories import FACTORY_BY_KIND, ProviderFactoryKind
from dr_providers.controls import GenerationControls, ReasoningEffort
from dr_providers.outcome import (
    ProviderTransportFailure,
    ProviderTransportResponse,
)
from dr_providers.request import (
    ProviderCallRequest,
    build_payload,
    protocol_path,
)
from dr_providers.transcript import MessageRole, PromptMessage, Transcript


class ServeProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    # Serve API spelling is snake_case; the CLI uses "openai-responses".
    # Both map to the same shared factory registry (see _factories.py).
    OPENAI_RESPONSES = "openai_responses"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


# The serve kind → the canonical shared factory kind. The serve spelling of
# the OpenAI Responses member already matches the canonical snake_case value.
_KIND_TO_FACTORY_KIND: dict[ServeProviderKind, ProviderFactoryKind] = {
    ServeProviderKind.OPENROUTER: ProviderFactoryKind.OPENROUTER,
    ServeProviderKind.OPENAI: ProviderFactoryKind.OPENAI,
    ServeProviderKind.OPENAI_RESPONSES: ProviderFactoryKind.OPENAI_RESPONSES,
    ServeProviderKind.GEMINI: ProviderFactoryKind.GEMINI,
    ServeProviderKind.ANTHROPIC: ProviderFactoryKind.ANTHROPIC,
}

# The anthropic preset requires a token limit; serve supplies this default when
# a spec targeting anthropic omits one (see ``build_request``).
DEFAULT_ANTHROPIC_TOKEN_LIMIT = 4096


class QuerySpec(BaseModel):
    """Declarative single query: provider kind + model + transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_kind: ServeProviderKind
    model: StrictStr
    messages: tuple[PromptMessage, ...]
    temperature: float | None = None
    top_p: float | None = None
    token_limit: StrictInt | None = None
    reasoning: ReasoningEffort | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """One provider call: wire payload plus response or failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint_path: StrictStr
    payload: dict[str, Any]
    response: ProviderTransportResponse | None = None
    failure: ProviderTransportFailure | None = None

    @property
    def ok(self) -> bool:
        return self.response is not None


class VarianceRecord(BaseModel):
    """One sampled call in a variance run (JSONL row)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: StrictStr
    sample_index: StrictInt
    ok: StrictBool
    text: StrictStr = ""
    finish_reason: StrictStr | None = None
    completion_tokens: StrictInt | None = None
    total_cost: float | None = None
    warning_codes: tuple[StrictStr, ...] = ()
    failure_code: StrictStr | None = None


class ModelVariance(BaseModel):
    """Dispersion summary for one model's samples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: StrictStr
    samples: StrictInt
    failures: StrictInt
    distinct_outputs: StrictInt
    mean_length: float | None = None
    min_length: StrictInt | None = None
    max_length: StrictInt | None = None


class VarianceReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: StrictStr
    samples_per_model: StrictInt
    models: tuple[StrictStr, ...]
    per_model: tuple[ModelVariance, ...]
    records: tuple[VarianceRecord, ...]


def build_request(spec: QuerySpec) -> ProviderCallRequest:
    from dr_providers.controls import ProviderBodyExtensions  # noqa: PLC0415

    # Anthropic's Messages preset REQUIRES a token limit; supply a sensible
    # default when serving an anthropic spec that omits one so the call is
    # well-formed rather than raising ControlValidationError.
    token_limit = spec.token_limit
    if (
        token_limit is None
        and spec.provider_kind is ServeProviderKind.ANTHROPIC
    ):
        token_limit = DEFAULT_ANTHROPIC_TOKEN_LIMIT
    factory = FACTORY_BY_KIND[_KIND_TO_FACTORY_KIND[spec.provider_kind]]
    config = factory(
        model=spec.model,
        controls=GenerationControls(
            temperature=spec.temperature,
            top_p=spec.top_p,
            token_limit=token_limit,
            reasoning=spec.reasoning,
        ),
        extensions=ProviderBodyExtensions(extra_body=dict(spec.extra_body)),
    )
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(messages=spec.messages),
    )


def run_query(spec: QuerySpec, provider: Provider) -> QueryResult:
    request = build_request(spec)
    payload = build_payload(request)
    endpoint_path = protocol_path(request.config)
    outcome = provider.complete(request)
    if isinstance(outcome, ProviderTransportResponse):
        return QueryResult(
            endpoint_path=endpoint_path,
            payload=payload,
            response=outcome,
        )
    return QueryResult(
        endpoint_path=endpoint_path,
        payload=payload,
        failure=outcome,
    )


def _variance_record(
    model: str,
    sample_index: int,
    result: QueryResult,
) -> VarianceRecord:
    if result.response is None:
        return VarianceRecord(
            model=model,
            sample_index=sample_index,
            ok=False,
            failure_code=result.failure.code if result.failure else None,
        )
    response = result.response
    usage = response.usage
    return VarianceRecord(
        model=model,
        sample_index=sample_index,
        ok=True,
        text=response.text,
        finish_reason=response.finish_reason,
        completion_tokens=usage.completion_tokens if usage else None,
        total_cost=response.cost.total_cost if response.cost else None,
        warning_codes=tuple(warning.code for warning in response.warnings),
    )


def _model_variance(
    model: str,
    records: Sequence[VarianceRecord],
) -> ModelVariance:
    ok_lengths = [len(record.text) for record in records if record.ok]
    distinct = len({record.text for record in records if record.ok})
    return ModelVariance(
        model=model,
        samples=len(records),
        failures=sum(1 for record in records if not record.ok),
        distinct_outputs=distinct,
        mean_length=(
            sum(ok_lengths) / len(ok_lengths) if ok_lengths else None
        ),
        min_length=min(ok_lengths) if ok_lengths else None,
        max_length=max(ok_lengths) if ok_lengths else None,
    )


def run_variance(  # noqa: PLR0913
    prompt: str,
    *,
    models: Sequence[str],
    samples: int,
    provider_kind: ServeProviderKind,
    provider: Provider,
    temperature: float | None = None,
) -> VarianceReport:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not models:
        raise ValueError("at least one model is required")

    messages = (PromptMessage(role=MessageRole.USER, content=prompt),)
    records: list[VarianceRecord] = []
    per_model: list[ModelVariance] = []
    for model in models:
        spec = QuerySpec(
            provider_kind=provider_kind,
            model=model,
            messages=messages,
            temperature=temperature,
        )
        model_records = [
            _variance_record(model, index, run_query(spec, provider))
            for index in range(samples)
        ]
        records.extend(model_records)
        per_model.append(_model_variance(model, model_records))

    return VarianceReport(
        prompt=prompt,
        samples_per_model=samples,
        models=tuple(models),
        per_model=tuple(per_model),
        records=tuple(records),
    )
