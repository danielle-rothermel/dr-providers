from __future__ import annotations

from enum import StrEnum
from threading import Event
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dr_providers.core.provider import Provider
    from dr_providers.outcomes.models import (
        ProviderTransportFailure,
        ProviderTransportResponse,
    )

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
)

from dr_providers.lifecycle import (
    AcceptAllSemanticResponseClassifier,
    ProviderCallOutcomeKind,
    ProviderCallResult,
    ProviderCallState,
    StandardProviderCallRetryPolicy,
    run_local_provider_call,
)
from dr_providers.modeling.controls import (
    GenerationControls,
    ProviderBodyExtensions,
    ReasoningEffort,
)
from dr_providers.modeling.presets import FACTORY_BY_KIND, ProviderFactoryKind
from dr_providers.modeling.request import ProviderCallRequest
from dr_providers.modeling.transcript import PromptMessage, Transcript
from dr_providers.translation.request import build_payload, protocol_path


class ServeProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


_KIND_TO_FACTORY_KIND: dict[ServeProviderKind, ProviderFactoryKind] = {
    ServeProviderKind.OPENROUTER: ProviderFactoryKind.OPENROUTER,
    ServeProviderKind.OPENAI: ProviderFactoryKind.OPENAI,
    ServeProviderKind.OPENAI_RESPONSES: ProviderFactoryKind.OPENAI_RESPONSES,
    ServeProviderKind.GEMINI: ProviderFactoryKind.GEMINI,
    ServeProviderKind.ANTHROPIC: ProviderFactoryKind.ANTHROPIC,
}

# Anthropic requires max_tokens; serve defaults it when omitted.
DEFAULT_ANTHROPIC_TOKEN_LIMIT = 4096


class QuerySpec(BaseModel):
    """Declarative single query: provider kind + model + transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_kind: ServeProviderKind
    model: StrictStr
    messages: tuple[PromptMessage, ...]
    temperature: float | None = Field(
        default=None,
        allow_inf_nan=False,
        strict=True,
    )
    top_p: float | None = Field(
        default=None,
        allow_inf_nan=False,
        strict=True,
    )
    token_limit: StrictInt | None = None
    reasoning: ReasoningEffort | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """One provider call with its complete lifecycle result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint_path: StrictStr
    payload: dict[str, Any]
    provider_call_result: ProviderCallResult

    @property
    def ok(self) -> bool:
        return (
            self.provider_call_result.outcome.kind
            is ProviderCallOutcomeKind.ACCEPTED
        )

    @property
    def response(self) -> ProviderTransportResponse | None:
        records = self.provider_call_result.completed_invocations
        if not records:
            return None
        return records[-1].observation.evidence.response

    @property
    def failure(self) -> ProviderTransportFailure | None:
        records = self.provider_call_result.completed_invocations
        if not records:
            return None
        return records[-1].observation.evidence.failure


def build_request(spec: QuerySpec) -> ProviderCallRequest:
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
    classifier = AcceptAllSemanticResponseClassifier()
    state = ProviderCallState.initial(
        request=request,
        retry_policy=StandardProviderCallRetryPolicy(),
        classifier_identifier=classifier.identifier,
    )
    provider_call_result = run_local_provider_call(
        provider=provider,
        state=state,
        classifier=classifier,
        cancellation=Event(),
    )
    return QueryResult(
        endpoint_path=endpoint_path,
        payload=payload,
        provider_call_result=provider_call_result,
    )
