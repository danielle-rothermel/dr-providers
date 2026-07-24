"""dr-providers: typed LLM provider-call transport kernel.

Owns the Provider Call Definition -> Config identity, the Provider Call
Request identity, the Provider Transport Policy, the typed no-throw
Provider Transport Outcome, complete least-processed success/failure
evidence in a stable Provider Invocation Evidence artifact, native
retry-zero support, and the ``(provider, protocol, model)`` Provider
Quota Identity. Whetstone owns semantic acceptance, classification,
retry/backoff, checkpoints, results, and concurrency.

``HttpProvider`` loads lazily so importing this package's pure modules
(identity, config, payloads) never pulls in httpx.
"""

from importlib.metadata import version
from typing import Any

from dr_providers.config import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    PROVIDER_CALL_CONFIG_SCHEMA_VERSION,
    PROVIDER_CALL_DEFINITION_SCHEMA,
    PROVIDER_CALL_DEFINITION_SCHEMA_VERSION,
    ProviderCallConfig,
    ProviderCallDefinition,
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)
from dr_providers.conformance import (
    conformance_warnings,
    with_conformance_warnings,
)
from dr_providers.controls import (
    ControlConstraints,
    GenerationControls,
    ProviderBodyExtensions,
    ReasoningEffort,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
)
from dr_providers.evidence import (
    PROVIDER_INVOCATION_EVIDENCE_SCHEMA,
    PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION,
    ProviderInvocationEvidence,
    RawHttpRequest,
)
from dr_providers.failures import (
    FAILURE_ERROR_TYPES,
    RECOVERABLE_FAILURE_CLASSES,
    RETRYABLE_FAILURE_CLASSES,
    SANITIZE_KEYS,
    ControlValidationError,
    FailureClass,
    PermanentProviderError,
    ProviderFailure,
    ProviderFailureError,
    RateLimitedProviderError,
    ResourceExhaustionProviderError,
    TransientProviderError,
    UnknownProviderError,
    classify_status_code,
    failure_record,
    raise_failure,
    sanitize_headers,
    sanitize_kwargs,
)
from dr_providers.outcome import (
    CostInfo,
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
    ProviderTransportWarning,
    ResponsesDiagnostics,
    TokenUsage,
    WarningSeverity,
    is_failure,
    is_response,
)
from dr_providers.policy import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    ApiKeyEnv,
    ProviderBaseUrl,
    ProviderTransportPolicy,
    policy_for,
)
from dr_providers.provider import Provider
from dr_providers.request import (
    PROVIDER_CALL_REQUEST_SCHEMA,
    PROVIDER_CALL_REQUEST_SCHEMA_VERSION,
    ProviderCallRequest,
    build_payload,
    protocol_path,
)
from dr_providers.response import (
    cost_from_body,
    parse_anthropic_messages_body,
    parse_chat_completions_body,
    parse_response,
    parse_responses_body,
    token_usage_from_body,
)
from dr_providers.route import (
    ModelRoute,
    Protocol,
    ProviderKind,
    ProviderQuotaIdentity,
)
from dr_providers.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
)
from dr_providers.transcript import (
    MessageRole,
    PromptMessage,
    Transcript,
)

PACKAGE_NAME = "dr-providers"

__all__ = [
    "DEFAULT_API_KEY_ENVS",
    "DEFAULT_BASE_URLS",
    "DEFAULT_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "FAILURE_ERROR_TYPES",
    "PROVIDER_CALL_CONFIG_SCHEMA",
    "PROVIDER_CALL_CONFIG_SCHEMA_VERSION",
    "PROVIDER_CALL_DEFINITION_SCHEMA",
    "PROVIDER_CALL_DEFINITION_SCHEMA_VERSION",
    "PROVIDER_CALL_REQUEST_SCHEMA",
    "PROVIDER_CALL_REQUEST_SCHEMA_VERSION",
    "PROVIDER_INVOCATION_EVIDENCE_SCHEMA",
    "PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION",
    "RECOVERABLE_FAILURE_CLASSES",
    "RETRYABLE_FAILURE_CLASSES",
    "SANITIZE_KEYS",
    "ApiKeyEnv",
    "ControlConstraints",
    "ControlValidationError",
    "CostInfo",
    "FailureClass",
    "GenerationControls",
    "HttpProvider",
    "MessageRole",
    "ModelRoute",
    "PermanentProviderError",
    "PromptMessage",
    "Protocol",
    "Provider",
    "ProviderBaseUrl",
    "ProviderBodyExtensions",
    "ProviderCallConfig",
    "ProviderCallDefinition",
    "ProviderCallRequest",
    "ProviderFailure",
    "ProviderFailureError",
    "ProviderInvocationEvidence",
    "ProviderKind",
    "ProviderQuotaIdentity",
    "ProviderTransportFailure",
    "ProviderTransportOutcome",
    "ProviderTransportPolicy",
    "ProviderTransportResponse",
    "ProviderTransportWarning",
    "RateLimitedProviderError",
    "RawHttpRequest",
    "ReasoningEffort",
    "ReasoningRequestShape",
    "RequestControl",
    "ResourceExhaustionProviderError",
    "ResponsesDiagnostics",
    "ScriptedOutcome",
    "ScriptedProvider",
    "TokenLimitParameter",
    "TokenUsage",
    "Transcript",
    "TransientProviderError",
    "UnknownProviderError",
    "WarningSeverity",
    "anthropic_messages_config",
    "build_payload",
    "classify_status_code",
    "conformance_warnings",
    "cost_from_body",
    "failure_record",
    "gemini_chat_config",
    "is_failure",
    "is_response",
    "openai_chat_config",
    "openai_responses_config",
    "openrouter_chat_config",
    "parse_anthropic_messages_body",
    "parse_chat_completions_body",
    "parse_response",
    "parse_responses_body",
    "policy_for",
    "protocol_path",
    "raise_failure",
    "sanitize_headers",
    "sanitize_kwargs",
    "token_usage_from_body",
    "with_conformance_warnings",
]

__version__ = version(PACKAGE_NAME)


_LAZY_TRANSPORT_EXPORTS = frozenset({"HttpProvider"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_TRANSPORT_EXPORTS:
        from dr_providers import transport  # noqa: PLC0415 -- lazy

        return getattr(transport, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
