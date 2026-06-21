from dr_providers.config import LlmConfig, ReasoningSpec, SamplingControls
from dr_providers.names import MessageRole, ProviderName
from dr_providers.query.provider_config import (
    ProviderError,
    ProviderSemanticError,
    ProviderTransportError,
)
from dr_providers.query.request import (
    LlmRequest,
    Message,
    OpenAICompatRequest,
    RequestControls,
)
from dr_providers.query.response import LlmResponse
from dr_providers.query.transport import OpenRouterProvider

__all__ = [
    "LlmConfig",
    "LlmRequest",
    "LlmResponse",
    "Message",
    "MessageRole",
    "OpenAICompatRequest",
    "OpenRouterProvider",
    "ProviderError",
    "ProviderName",
    "ProviderSemanticError",
    "ProviderTransportError",
    "ReasoningSpec",
    "RequestControls",
    "SamplingControls",
    "query",
]


def query(
    prompt: str,
    *,
    model: str = "openai/gpt-4o-mini",
    system: str | None = None,
    max_tokens: int | None = None,
) -> LlmResponse:
    messages: list[Message] = []
    if system is not None:
        messages.append(Message(role=MessageRole.SYSTEM, content=system))
    messages.append(Message(role=MessageRole.USER, content=prompt))
    request = LlmRequest(
        provider=ProviderName.OPENROUTER,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    with OpenRouterProvider() as provider:
        return provider.generate(request)
