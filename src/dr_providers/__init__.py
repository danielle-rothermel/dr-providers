from importlib.metadata import version

from dr_providers.query import (
    ApiProvider,
    LlmRequest,
    LlmResponse,
    Message,
    MessageRole,
    OpenRouterProvider,
    ProviderError,
    ProviderName,
    ProviderSemanticError,
    ProviderTransportError,
    ReasoningSpec,
    SamplingControls,
)

PACKAGE_NAME = "dr-providers"

__all__ = [
    "ApiProvider",
    "LlmRequest",
    "LlmResponse",
    "Message",
    "MessageRole",
    "OpenRouterProvider",
    "ProviderError",
    "ProviderName",
    "ProviderSemanticError",
    "ProviderTransportError",
    "ReasoningSpec",
    "SamplingControls",
]

__version__ = version(PACKAGE_NAME)
