from __future__ import annotations

from typing import Any

import httpx
import pytest

from dr_providers import (
    ApiKeyEnv,
    ContinuationScore,
    FakeLocalBackend,
    GenerationControls,
    GenerationOutput,
    LocalDevice,
    LocalDtype,
    LocalExecutionEvidence,
    LocalModelProvider,
    MessageRole,
    PromptMessage,
    PromptRendering,
    ProviderBaseUrl,
    ProviderCallConfig,
    ProviderGenerateRequest,
    ProviderKind,
    ProviderScoreRequest,
    ProviderScoreResponse,
    ProviderStopReason,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    ReasoningEffort,
    ScoredContinuation,
    ScriptedOutcome,
    ScriptedProvider,
    Transcript,
    conformance_warnings,
    huggingface_config,
    openai_chat_config,
)
from dr_providers.outcomes.conformance import (
    MODEL_SUBSTITUTION_CODE,
    REASONING_NOT_OBSERVED_CODE,
    REVISION_NOT_COMMIT_SHA_CODE,
)
from dr_providers.transport.http import HttpProvider

MESSAGES = (PromptMessage(role=MessageRole.USER, content="write add"),)
CHAT_BODY_OK: dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "m",
    "choices": [
        {
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}
OPENAI_POLICY = ProviderTransportPolicy(
    provider_kind=ProviderKind.OPENAI,
    api_key_env=str(ApiKeyEnv.OPENAI),
    base_url=str(ProviderBaseUrl.OPENAI),
    timeout_seconds=120.0,
    connect_timeout_seconds=30.0,
    idle_timeout_seconds=90.0,
    max_connections=10,
    max_keepalive_connections=5,
    max_request_bytes=1024 * 1024,
    max_response_bytes=8 * 1024 * 1024,
)


def request_for(
    config: ProviderCallConfig, messages=MESSAGES
) -> ProviderGenerateRequest:
    return ProviderGenerateRequest(
        config=config, transcript=Transcript(messages=messages)
    )


def openai_request(**control_overrides: Any) -> ProviderGenerateRequest:
    controls = GenerationControls(**control_overrides)
    return request_for(openai_chat_config(model="m", controls=controls))


def mock_provider(
    handler: Any,
    *,
    policy: ProviderTransportPolicy = OPENAI_POLICY,
) -> HttpProvider:
    return HttpProvider(
        policy=policy,
        _client_factory=lambda **_kwargs: httpx.Client(
            transport=httpx.MockTransport(handler)
        ),
        api_key="test-key",
    )


class TestConformance:
    def test_reasoning_not_observed_warning(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        outcome = provider.invoke(
            openai_request(reasoning=ReasoningEffort.LOW)
        ).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        codes = [w.code for w in outcome.warnings]
        assert REASONING_NOT_OBSERVED_CODE in codes

    def test_model_substitution_warning(self) -> None:
        body = dict(CHAT_BODY_OK)
        body["model"] = "m-other"
        provider = mock_provider(lambda _req: httpx.Response(200, json=body))
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        codes = [w.code for w in outcome.warnings]
        assert MODEL_SUBSTITUTION_CODE in codes

    def test_clean_response_has_no_warnings(self) -> None:
        provider = mock_provider(
            lambda _req: httpx.Response(200, json=CHAT_BODY_OK)
        )
        outcome = provider.invoke(openai_request()).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.warnings == ()


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def local_config_with(revision: str) -> ProviderCallConfig:
    return huggingface_config(
        model="example/model",
        revision=revision,
        device=LocalDevice.CUDA,
        dtype=LocalDtype.FLOAT32,
        batch_size=1,
        max_sequence_length=1024,
        prompt_rendering=PromptRendering.FLAT_TEXT,
    )


def local_outcomes(
    config: ProviderCallConfig, execution: LocalExecutionEvidence
) -> tuple[ProviderTransportResponse, ProviderScoreResponse]:
    backend = FakeLocalBackend(
        execution=execution,
        results=[
            GenerationOutput("answer", 3, 2, ProviderStopReason.LENGTH),
            (ScoredContinuation(-1.0, 1, None, 2),),
        ],
    )
    with LocalModelProvider(config=config, backend=backend) as provider:
        generated = provider.invoke(request_for(config)).response
        scored = provider.invoke(
            ProviderScoreRequest(
                config=config, context="ctx", continuations=(" a",)
            )
        ).score_response
    assert generated is not None
    assert scored is not None
    return generated, scored


class TestRevisionConformance:
    @pytest.mark.parametrize("revision", ["main", "step1000-seed0", "0123456"])
    def test_local_non_sha_revision_warns_on_both_kinds(
        self, revision: str, local_execution: LocalExecutionEvidence
    ) -> None:
        for response in local_outcomes(
            local_config_with(revision), local_execution
        ):
            warnings = [
                warning
                for warning in response.warnings
                if warning.code == REVISION_NOT_COMMIT_SHA_CODE
            ]
            assert len(warnings) == 1
            assert warnings[0].metadata == {"revision": revision}

    def test_local_commit_sha_has_no_warnings(
        self, local_execution: LocalExecutionEvidence
    ) -> None:
        for response in local_outcomes(
            local_config_with(COMMIT), local_execution
        ):
            assert response.warnings == ()

    def test_scripted_score_applies_revision_warning(self) -> None:
        config = local_config_with("main")
        request = ProviderScoreRequest(
            config=config, context="ctx", continuations=(" a",)
        )
        score = ContinuationScore(
            log_likelihood=-1.0, token_count=1, char_count=2
        )
        with ScriptedProvider([ScriptedOutcome(scores=(score,))]) as provider:
            response = provider.invoke(request).score_response
        assert response is not None
        assert [w.code for w in response.warnings] == [
            REVISION_NOT_COMMIT_SHA_CODE
        ]

    def test_http_routes_have_no_revision_warning(self) -> None:
        response = ProviderTransportResponse(text="ok", model="m")
        assert conformance_warnings(openai_request(), response) == ()
