"""Serve-side query and variance machinery over the kernel.

Pure library logic: declarative query specs resolve to kernel
requests, run against any ``Provider`` (Fixture or Http), and come
back as structured results with conformance warnings applied. The
variance runner fans one prompt across models x samples and reports
output dispersion — the same records the playground downloads as
JSONL.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dr_providers.provider import Provider

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
)

from dr_providers.config import (
    MessageRole,
    PromptMessage,
    ProviderConfig,
    ReasoningEffort,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)
from dr_providers.failures import (
    ProviderFailure,
    ProviderFailureError,
)
from dr_providers.request import (
    ENDPOINT_PATHS,
    LlmRequest,
    build_payload,
)
from dr_providers.response import LlmResponse  # noqa: TC001


class ServeProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    GEMINI = "gemini"


CONFIG_FACTORIES: dict[ServeProviderKind, Callable[..., ProviderConfig]] = {
    ServeProviderKind.OPENROUTER: openrouter_chat_config,
    ServeProviderKind.OPENAI: openai_chat_config,
    ServeProviderKind.OPENAI_RESPONSES: openai_responses_config,
    ServeProviderKind.GEMINI: gemini_chat_config,
}


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
    response: LlmResponse | None = None
    failure: ProviderFailure | None = None

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


def build_request(spec: QuerySpec) -> LlmRequest:
    config = CONFIG_FACTORIES[spec.provider_kind](model=spec.model)
    return LlmRequest(
        provider_config=config,
        messages=spec.messages,
        temperature=spec.temperature,
        top_p=spec.top_p,
        token_limit=spec.token_limit,
        reasoning=spec.reasoning,
        extra_body=spec.extra_body,
    )


def run_query(spec: QuerySpec, provider: Provider) -> QueryResult:
    request = build_request(spec)
    payload = build_payload(request)
    endpoint_path = ENDPOINT_PATHS[request.provider_config.endpoint_kind]
    try:
        response = provider.complete(request)
    except ProviderFailureError as error:
        return QueryResult(
            endpoint_path=endpoint_path,
            payload=payload,
            failure=error.failure,
        )
    return QueryResult(
        endpoint_path=endpoint_path,
        payload=payload,
        response=response,
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
