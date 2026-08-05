import contextlib
import os
from collections.abc import Iterator
from enum import StrEnum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_providers.core.failures import (
    ControlValidationError,
    FailureClass,
)
from dr_providers.core.provider import Provider
from dr_providers.outcomes.models import (
    CostInfo,
    ProviderTransportFailure,
    TokenUsage,
)
from dr_providers.surfaces.serve.query import (
    QueryResult,
    QuerySpec,
    ServeProviderKind,
    build_request,
    run_query,
)
from dr_providers.surfaces.serve.variance import (
    VarianceReport,
    run_variance,
)
from dr_providers.surfaces.testing.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
)
from dr_providers.translation.request import build_payload, protocol_path
from dr_providers.transport.http import MISSING_API_KEY_CODE, HttpProvider
from dr_providers.transport.policy import (
    DEFAULT_API_KEY_ENVS,
    policy_for,
)

SERVE_TITLE = "dr-providers serve"
SERVE_VERSION = "0.2.1"
LOCALHOST_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?"
MAX_VARIANCE_SAMPLES = 25
MAX_VARIANCE_MODELS = 8


class ProviderChoiceKind(StrEnum):
    SCRIPTED = "scripted"
    LIVE = "live"


class ProviderChoice(BaseModel):
    """Which implementation executes the call: scripted or live HTTP."""

    model_config = ConfigDict(extra="forbid")

    kind: ProviderChoiceKind = ProviderChoiceKind.SCRIPTED
    scripted_outcomes: list["ScriptedOutcomeSpec"] = Field(
        default_factory=list
    )


class ScriptedOutcomeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr = ""
    finish_reason: StrictStr | None = "stop"
    completion_tokens: StrictInt | None = None
    total_cost: float | None = None
    failure_code: StrictStr | None = None
    failure_message: StrictStr | None = None

    def to_outcome(self) -> ScriptedOutcome:
        failure: ProviderTransportFailure | None = None
        if self.failure_code is not None:
            failure = ProviderTransportFailure(
                failure_class=FailureClass.PERMANENT,
                code=self.failure_code,
                message=self.failure_message or self.failure_code,
                retryable=False,
            )
        usage = (
            TokenUsage(completion_tokens=self.completion_tokens)
            if self.completion_tokens is not None
            else None
        )
        cost = (
            CostInfo(total_cost=self.total_cost)
            if self.total_cost is not None
            else None
        )
        return ScriptedOutcome(
            text=self.text,
            finish_reason=self.finish_reason,
            usage=usage,
            cost=cost,
            failure=failure,
        )


class BuildPayloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: QuerySpec


class BuildPayloadResponse(BaseModel):
    endpoint_path: str
    payload: dict[str, object]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: QuerySpec
    provider: ProviderChoice = Field(default_factory=ProviderChoice)


class VarianceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr
    models: list[StrictStr]
    samples: StrictInt = 3
    provider_kind: ServeProviderKind = ServeProviderKind.OPENROUTER
    temperature: float | None = None
    provider: ProviderChoice = Field(default_factory=ProviderChoice)


class HealthResponse(BaseModel):
    status: str
    version: str


@contextlib.contextmanager
def resolve_provider(
    choice: ProviderChoice, spec: QuerySpec
) -> Iterator[Provider]:
    if choice.kind is ProviderChoiceKind.SCRIPTED:
        outcomes = [
            outcome.to_outcome() for outcome in choice.scripted_outcomes
        ]
        yield ScriptedProvider(outcomes or None)
        return
    kind = build_request(spec).config.route.provider
    api_key_env = str(DEFAULT_API_KEY_ENVS[kind])
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise HTTPException(
            status_code=424,
            detail=(
                f"{MISSING_API_KEY_CODE}: set {api_key_env} "
                "to run live queries"
            ),
        )
    policy = policy_for(kind)
    with HttpProvider(policy=policy, api_key=api_key) as provider:
        yield provider


def create_app() -> FastAPI:
    app = FastAPI(title=SERVE_TITLE, version=SERVE_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=LOCALHOST_ORIGIN_REGEX,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=SERVE_VERSION)

    @app.post("/build_payload", response_model=BuildPayloadResponse)
    def build_payload_endpoint(
        request: BuildPayloadRequest,
    ) -> BuildPayloadResponse:
        try:
            call_request = build_request(request.spec)
            return BuildPayloadResponse(
                endpoint_path=protocol_path(call_request.config),
                payload=build_payload(call_request),
            )
        except ControlValidationError as error:
            raise HTTPException(
                status_code=422,
                detail=error.failure.model_dump(mode="json"),
            ) from error

    @app.post("/query", response_model=QueryResult)
    def query(request: QueryRequest) -> QueryResult:
        try:
            with resolve_provider(request.provider, request.spec) as provider:
                return run_query(request.spec, provider)
        except ControlValidationError as error:
            raise HTTPException(
                status_code=422,
                detail=error.failure.model_dump(mode="json"),
            ) from error

    @app.post("/variance", response_model=VarianceReport)
    def variance(request: VarianceRequest) -> VarianceReport:
        if request.samples < 1 or request.samples > MAX_VARIANCE_SAMPLES:
            raise HTTPException(
                status_code=422,
                detail=f"samples must be 1..{MAX_VARIANCE_SAMPLES}",
            )
        if not request.models or len(request.models) > MAX_VARIANCE_MODELS:
            raise HTTPException(
                status_code=422,
                detail=f"models must be 1..{MAX_VARIANCE_MODELS}",
            )
        first_spec = QuerySpec(
            provider_kind=request.provider_kind,
            model=request.models[0],
            messages=(),
        )
        with resolve_provider(request.provider, first_spec) as provider:
            return run_variance(
                request.prompt,
                models=request.models,
                samples=request.samples,
                provider_kind=request.provider_kind,
                provider=provider,
                temperature=request.temperature,
            )

    return app
