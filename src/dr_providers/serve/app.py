"""FastAPI facade over query/build_payload/variance ([serve] extra).

Providers: ``fixture`` (scripted outcomes, no network — what the
playground e2e uses) or ``live`` (raw-httpx transport; requires the
provider's API key env var and is never exercised by tests).
"""

import os
from enum import StrEnum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_providers.kernel.failures import (
    FailureClass,
    ProviderFailure,
    UnsupportedControlError,
    failure_record,
)
from dr_providers.kernel.fixture import FixtureOutcome, FixtureProvider
from dr_providers.kernel.provider import Provider
from dr_providers.kernel.request import ENDPOINT_PATHS, build_payload
from dr_providers.kernel.response import CostInfo, TokenUsage
from dr_providers.kernel.transport import HttpProvider
from dr_providers.serve.runner import (
    QueryResult,
    QuerySpec,
    ServeProviderKind,
    VarianceReport,
    build_request,
    run_query,
    run_variance,
)

SERVE_TITLE = "dr-providers serve"
SERVE_VERSION = "0.1.0"
LOCALHOST_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?"
MAX_VARIANCE_SAMPLES = 25
MAX_VARIANCE_MODELS = 8
MISSING_API_KEY_CODE = "missing_api_key"


class ProviderChoiceKind(StrEnum):
    FIXTURE = "fixture"
    LIVE = "live"


class ProviderChoice(BaseModel):
    """Which provider executes the call: scripted fixture or live."""

    model_config = ConfigDict(extra="forbid")

    kind: ProviderChoiceKind = ProviderChoiceKind.FIXTURE
    fixture_outcomes: list["FixtureOutcomeSpec"] = Field(default_factory=list)


class FixtureOutcomeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr = ""
    finish_reason: StrictStr | None = "stop"
    completion_tokens: StrictInt | None = None
    total_cost: float | None = None
    failure_code: StrictStr | None = None
    failure_message: StrictStr | None = None

    def to_outcome(self) -> FixtureOutcome:
        failure: ProviderFailure | None = None
        if self.failure_code is not None:
            failure = failure_record(
                failure_class=FailureClass.PERMANENT,
                code=self.failure_code,
                message=self.failure_message or self.failure_code,
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
        return FixtureOutcome(
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


def resolve_provider(choice: ProviderChoice, spec: QuerySpec) -> Provider:
    if choice.kind is ProviderChoiceKind.FIXTURE:
        outcomes = [
            outcome.to_outcome() for outcome in choice.fixture_outcomes
        ]
        return FixtureProvider(outcomes or None)
    config = build_request(spec).provider_config
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise HTTPException(
            status_code=424,
            detail=(
                f"{MISSING_API_KEY_CODE}: set {config.api_key_env} "
                "to run live queries"
            ),
        )
    return HttpProvider(api_key=api_key)


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
            llm_request = build_request(request.spec)
            return BuildPayloadResponse(
                endpoint_path=ENDPOINT_PATHS[
                    llm_request.provider_config.endpoint_kind
                ],
                payload=build_payload(llm_request),
            )
        except UnsupportedControlError as error:
            raise HTTPException(
                status_code=422,
                detail=error.failure.model_dump(mode="json"),
            ) from error

    @app.post("/query", response_model=QueryResult)
    def query(request: QueryRequest) -> QueryResult:
        provider = resolve_provider(request.provider, request.spec)
        try:
            return run_query(request.spec, provider)
        except UnsupportedControlError as error:
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
        provider = resolve_provider(request.provider, first_spec)
        return run_variance(
            request.prompt,
            models=request.models,
            samples=request.samples,
            provider_kind=request.provider_kind,
            provider=provider,
            temperature=request.temperature,
        )

    return app
