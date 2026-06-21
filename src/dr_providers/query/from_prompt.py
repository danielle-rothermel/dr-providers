from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_providers.names import MessageRole, ProviderName
from dr_providers.query.request import LlmRequest, Message

if TYPE_CHECKING:
    from dr_providers.config import ReasoningSpec, SamplingControls
    from dr_providers.query.response import LlmResponse
    from dr_providers.query.transport import ApiProvider


def query_from_prompt(  # noqa: PLR0913
    provider: ApiProvider,
    provider_name: ProviderName,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningSpec | None = None,
    sampling: SamplingControls | None = None,
    metadata: dict[str, Any] | None = None,
) -> LlmResponse:
    messages: list[Message] = []
    if system is not None:
        messages.append(Message(role=MessageRole.SYSTEM, content=system))
    messages.append(Message(role=MessageRole.USER, content=prompt))
    request = LlmRequest(
        provider=provider_name,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        reasoning=reasoning,
        sampling=sampling,
        metadata=metadata or {},
    )
    return provider.generate(request)
