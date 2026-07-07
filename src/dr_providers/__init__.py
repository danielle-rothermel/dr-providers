"""dr-providers: LLM query kernel.

v0.2 kernel lives in ``dr_providers.kernel``. The 0.1.x ``query`` API
remains importable from the package root but loads lazily so that
importing kernel submodules with import-hygiene contracts (e.g. the
failure taxonomy) never pulls in httpx.
"""

from importlib.metadata import version
from typing import Any

PACKAGE_NAME = "dr-providers"

_QUERY_EXPORTS = frozenset(
    {
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
)

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


def __getattr__(name: str) -> Any:
    if name in _QUERY_EXPORTS:
        from dr_providers import query  # noqa: PLC0415 -- lazy by design

        return getattr(query, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
