from dr_providers import (
    LlmRequest,
    Message,
    MessageRole,
    ProviderName,
    ReasoningSpec,
    SamplingControls,
)
from dr_providers.names import EffortLevel


def test_llm_request_stores_request_fields() -> None:
    request = LlmRequest(
        provider=ProviderName.OPENROUTER,
        model="test-model",
        messages=[Message(role=MessageRole.USER, content="hi")],
        max_tokens=100,
        sampling=SamplingControls(temperature=0.5, top_p=0.9),
        reasoning=ReasoningSpec(effort=EffortLevel.LOW),
        metadata={"idempotency_key": "fixed-key"},
    )

    assert request.provider == ProviderName.OPENROUTER
    assert request.model == "test-model"
    assert request.messages == [Message(role=MessageRole.USER, content="hi")]
    assert request.max_tokens == 100
    assert request.sampling == SamplingControls(temperature=0.5, top_p=0.9)
    assert request.reasoning == ReasoningSpec(effort=EffortLevel.LOW)
    assert request.metadata == {"idempotency_key": "fixed-key"}
