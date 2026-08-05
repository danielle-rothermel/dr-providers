"""Serve-side query machinery over the kernel.

Pure library logic: declarative query specs resolve to kernel requests,
run against any ``Provider`` (Scripted or Http), and return structured
results with conformance warnings applied.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dr_providers.core.provider import Provider

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
)

from dr_providers.modeling.controls import (
    GenerationControls,
    ProviderBodyExtensions,
    ReasoningEffort,
)
from dr_providers.modeling.presets import FACTORY_BY_KIND, ProviderFactoryKind
from dr_providers.modeling.request import ProviderCallRequest
from dr_providers.modeling.transcript import PromptMessage, Transcript
from dr_providers.outcomes.models import (
    ProviderTransportFailure,
    ProviderTransportResponse,
)
from dr_providers.translation.request import build_payload, protocol_path


class ServeProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    # Serve API spelling is snake_case; the CLI uses "openai-responses".
    # Both map to the same shared preset registry.
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


def build_request(spec: QuerySpec) -> ProviderCallRequest:
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
