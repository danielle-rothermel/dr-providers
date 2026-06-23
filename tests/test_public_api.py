from importlib.metadata import version

import dr_providers
from dr_providers import LlmRequest, OpenRouterProvider


def test_public_api_exports() -> None:
    expected = {
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
    }
    assert set(dr_providers.__all__) == expected


def test_version() -> None:
    assert dr_providers.__version__ == version("dr-providers")


def test_top_level_imports() -> None:
    assert LlmRequest is not None
    assert OpenRouterProvider is not None
