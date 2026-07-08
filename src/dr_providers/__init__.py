"""dr-providers: typed LLM provider-call kernel.

The stable intersection of four provider-implementation lineages:
request, response, usage, warning, failure record, provider config,
transport. See whetstone-ai ``docs/composable/llm_provider.md``.

``HttpProvider`` / ``TransportPolicy`` load lazily so importing this
package's pure modules (failure taxonomy, config records, payloads)
never pulls in httpx — consumers with import-hygiene contracts rely
on this.
"""

from importlib.metadata import version
from typing import Any

from dr_providers.config import (
    ApiKeyEnv,
    EndpointKind,
    MessageRole,
    PromptMessage,
    ProviderBaseUrl,
    ProviderConfig,
    ProviderKind,
    ReasoningEffort,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)
from dr_providers.conformance import (
    conformance_warnings,
    with_conformance_warnings,
)
from dr_providers.failures import (
    FAILURE_ERROR_TYPES,
    RECOVERABLE_FAILURE_CLASSES,
    RETRYABLE_FAILURE_CLASSES,
    SANITIZE_KEYS,
    FailureClass,
    PermanentProviderError,
    ProviderFailure,
    ProviderFailureError,
    RateLimitedProviderError,
    ResourceExhaustionProviderError,
    TransientProviderError,
    UnknownProviderError,
    UnsupportedControlError,
    classify_status_code,
    failure_record,
    raise_failure,
    sanitize_kwargs,
)
from dr_providers.provider import Provider
from dr_providers.request import (
    LlmRequest,
    build_payload,
    endpoint_path,
)
from dr_providers.response import (
    CostInfo,
    LlmResponse,
    LlmWarning,
    TokenUsage,
    WarningSeverity,
    cost_from_body,
    parse_chat_completions_body,
    parse_response,
    parse_responses_body,
    token_usage_from_body,
)
from dr_providers.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
)

PACKAGE_NAME = "dr-providers"

__all__ = [
    "FAILURE_ERROR_TYPES",
    "RECOVERABLE_FAILURE_CLASSES",
    "RETRYABLE_FAILURE_CLASSES",
    "SANITIZE_KEYS",
    "ApiKeyEnv",
    "CostInfo",
    "EndpointKind",
    "FailureClass",
    "HttpProvider",
    "LlmRequest",
    "LlmResponse",
    "LlmWarning",
    "MessageRole",
    "PermanentProviderError",
    "PromptMessage",
    "Provider",
    "ProviderBaseUrl",
    "ProviderConfig",
    "ProviderFailure",
    "ProviderFailureError",
    "ProviderKind",
    "RateLimitedProviderError",
    "ReasoningEffort",
    "ReasoningRequestShape",
    "RequestControl",
    "ResourceExhaustionProviderError",
    "ScriptedOutcome",
    "ScriptedProvider",
    "TokenLimitParameter",
    "TokenUsage",
    "TransientProviderError",
    "TransportPolicy",
    "UnknownProviderError",
    "UnsupportedControlError",
    "WarningSeverity",
    "build_payload",
    "classify_status_code",
    "conformance_warnings",
    "cost_from_body",
    "endpoint_path",
    "failure_record",
    "gemini_chat_config",
    "openai_chat_config",
    "openai_responses_config",
    "openrouter_chat_config",
    "parse_chat_completions_body",
    "parse_response",
    "parse_responses_body",
    "raise_failure",
    "sanitize_kwargs",
    "token_usage_from_body",
    "with_conformance_warnings",
]

__version__ = version(PACKAGE_NAME)


_LAZY_TRANSPORT_EXPORTS = frozenset({"HttpProvider", "TransportPolicy"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_TRANSPORT_EXPORTS:
        from dr_providers import transport  # noqa: PLC0415 -- lazy

        return getattr(transport, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
