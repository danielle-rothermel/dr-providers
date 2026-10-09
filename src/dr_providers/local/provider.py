from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import TYPE_CHECKING

from dr_providers.local.backend import LocalBackendFailure
from dr_providers.modeling.controls import (
    ProviderCallKind,
    TokenLimitParameter,
)
from dr_providers.modeling.route import ProviderKind
from dr_providers.modeling.transcript import MessageRole
from dr_providers.outcomes.conformance import with_conformance_warnings
from dr_providers.outcomes.evidence import ProviderInvocationEvidence
from dr_providers.outcomes.models import (
    ContinuationScore,
    CostInfo,
    ProviderScoreResponse,
    ProviderTransportFailure,
    ProviderTransportResponse,
    TokenUsage,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from dr_providers.local.backend import LocalBackend
    from dr_providers.modeling.call import ProviderCallConfig
    from dr_providers.modeling.controls import GenerationControls
    from dr_providers.modeling.request import ProviderCallRequest
    from dr_providers.outcomes.models import ProviderTransportOutcome


def validate_generation_controls(controls: GenerationControls) -> None:
    """Local ranges belong here rather than changing HTTP control contracts."""
    if controls.temperature is not None and controls.temperature < 0:
        raise ValueError("local temperature must be nonnegative")
    if controls.top_p is not None and not 0 < controls.top_p <= 1:
        raise ValueError("local top_p must be in (0, 1]")
    if controls.token_limit is not None and controls.token_limit < 1:
        raise ValueError("local token_limit must be positive")
    if controls.seed is not None and not -(2**63) <= controls.seed < 2**64:
        raise ValueError("local seed is outside torch's supported range")


class LocalModelProvider:
    """Own one loaded definition and serialize its model work.

    Closing refuses new work, drains admitted offloads, and then releases the
    backend. Direct invocations and cache release share the same model lock.
    """

    def __init__(
        self,
        *,
        config: ProviderCallConfig,
        backend: LocalBackend | None = None,
    ) -> None:
        if (
            config.route.provider is not ProviderKind.HUGGINGFACE
            or config.definition.local is None
        ):
            raise ValueError(
                "local provider requires a HuggingFace definition"
            )
        if (
            config.definition.constraints.token_limit_parameter
            is not TokenLimitParameter.MAX_NEW_TOKENS
        ):
            raise ValueError("local token limit must map to max_new_tokens")
        validate_generation_controls(config.controls)
        if backend is None:
            from dr_providers.local.transformers import (  # noqa: PLC0415
                TransformersBackend,
            )

            backend = TransformersBackend(config)
        self._backend = backend
        self._definition_hash = config.definition.identity_hash
        self._executor: ThreadPoolExecutor | None = None
        self._lifetime = threading.Condition()
        self._model_lock = threading.RLock()
        self._offloaded = threading.local()
        self._closing = False
        self._closed = False

    def _validate_open(self) -> None:
        with self._lifetime:
            if self._closed or (
                self._closing and not getattr(self._offloaded, "active", False)
            ):
                raise ValueError("local provider is closed or closing")

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        with self._model_lock:
            self._validate_open()
            if (
                request.config.definition.identity_hash
                != self._definition_hash
            ):
                raise ValueError(
                    "request definition differs from loaded model"
                )
            if request.kind == ProviderCallKind.GENERATE:
                if any(
                    message.role is MessageRole.TOOL
                    for message in request.transcript.messages
                ):
                    raise ValueError("local generation does not support tools")
                validate_generation_controls(request.config.controls)
            start = perf_counter()
            try:
                outcome = self._invoke_backend(request)
            except LocalBackendFailure as error:
                failure = error.failure
                outcome = ProviderTransportFailure(
                    code=failure.code,
                    recoverability=failure.recoverability,
                    message=failure.message,
                    metadata=failure.metadata,
                )
            elapsed = perf_counter() - start
            return ProviderInvocationEvidence.build(
                request=request,
                policy=None,
                http_request=None,
                outcome=outcome,
                local_execution=self._backend.execution(),
                wall_time_seconds=elapsed,
            )

    def _invoke_backend(
        self, request: ProviderCallRequest
    ) -> ProviderTransportOutcome:
        if request.kind == ProviderCallKind.SCORE:
            results = self._backend.score(
                context=request.context,
                continuations=request.continuations,
                token_logprobs=request.token_logprobs,
            )
            if len(results) != len(request.continuations):
                raise ValueError("score count must equal continuation count")
            scores = tuple(
                ContinuationScore(
                    log_likelihood=result.log_likelihood,
                    token_count=result.token_count,
                    char_count=len(continuation),
                    token_logprobs=result.token_logprobs,
                )
                for result, continuation in zip(
                    results, request.continuations, strict=True
                )
            )
            processed = sum(result.input_tokens for result in results)
            return with_conformance_warnings(
                request,
                ProviderScoreResponse(
                    scores=scores,
                    usage=TokenUsage(
                        prompt_tokens=processed, total_tokens=processed
                    ),
                    cost=CostInfo(total_cost=0.0),
                    model=request.config.route.model,
                ),
            )
        result = self._backend.generate(
            transcript=request.transcript,
            rendering=request.config.definition.prompt_rendering,
            controls=request.config.controls,
        )
        return with_conformance_warnings(
            request,
            ProviderTransportResponse(
                text=result.text,
                usage=TokenUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens
                    + result.completion_tokens,
                ),
                cost=CostInfo(total_cost=0.0),
                stop_reason=result.stop_reason,
                model=request.config.route.model,
            ),
        )

    def offload[ResultT](self, fn: Callable[[], ResultT]) -> Future[ResultT]:
        with self._lifetime:
            if self._closing or self._closed:
                raise ValueError("local provider is closed or closing")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="local-model-provider"
                )

            def run() -> ResultT:
                self._offloaded.active = True
                try:
                    return fn()
                finally:
                    self._offloaded.active = False

            return self._executor.submit(run)

    def release_memory(self) -> None:
        with self._model_lock:
            self._validate_open()
            self._backend.release_memory()

    def close(self) -> None:
        if getattr(self._offloaded, "active", False):
            raise ValueError("cannot close local provider from its own worker")
        with self._lifetime:
            if self._closing:
                self._lifetime.wait_for(lambda: self._closed)
                return
            if self._closed:
                return
            self._closing = True
            self._lifetime.notify_all()
            executor = self._executor
        try:
            if executor is not None:
                executor.shutdown(wait=True)
            with self._model_lock:
                self._backend.close()
        finally:
            with self._lifetime:
                self._closed = True
                self._lifetime.notify_all()

    def __enter__(self) -> LocalModelProvider:
        self._validate_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
