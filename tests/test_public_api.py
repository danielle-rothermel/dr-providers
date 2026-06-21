import dr_providers
from dr_providers import LlmRequest, OpenRouterProvider


def test_public_api_exports() -> None:
    expected = {
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
    }
    assert set(dr_providers.__all__) == expected


def test_version() -> None:
    assert dr_providers.__version__ == "0.1.0"


def test_top_level_imports() -> None:
    assert LlmRequest is not None
    assert OpenRouterProvider is not None
