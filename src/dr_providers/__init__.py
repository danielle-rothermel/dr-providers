from importlib.metadata import version

from dr_providers.query import (
    ApiProvider,
    LlmConfig,
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
    RequestControls,
    SamplingControls,
)

PACKAGE_NAME = "dr-providers"

__all__ = [
    "ApiProvider",
    "LlmConfig",
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
    "RequestControls",
    "SamplingControls",
]

__version__ = version(PACKAGE_NAME)
