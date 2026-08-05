from importlib.metadata import version
from typing import Any

from dr_providers.core.failures import (
    FAILURE_ERROR_TYPES,
    RECOVERABLE_FAILURE_CLASSES,
    RETRYABLE_FAILURE_CLASSES,
    ControlValidationError,
    FailureClass,
    PermanentProviderError,
    ProviderFailure,
    ProviderFailureError,
    RateLimitedProviderError,
    ResourceExhaustionProviderError,
    TransientProviderError,
    UnknownProviderError,
    failure_record,
    raise_failure,
)
from dr_providers.core.provider import Provider
from dr_providers.modeling.call import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    PROVIDER_CALL_CONFIG_SCHEMA_VERSION,
    PROVIDER_CALL_DEFINITION_SCHEMA,
    PROVIDER_CALL_DEFINITION_SCHEMA_VERSION,
    ProviderCallConfig,
    ProviderCallDefinition,
)
from dr_providers.modeling.controls import (
    ControlConstraints,
    GenerationControls,
    ProviderBodyExtensions,
    ReasoningEffort,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
)
from dr_providers.modeling.presets import (
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)
from dr_providers.modeling.request import (
    PROVIDER_CALL_REQUEST_SCHEMA,
    PROVIDER_CALL_REQUEST_SCHEMA_VERSION,
    ProviderCallRequest,
)
from dr_providers.modeling.route import (
    ModelRoute,
    Protocol,
    ProviderKind,
    ProviderQuotaIdentity,
)
from dr_providers.modeling.transcript import (
    MessageRole,
    PromptMessage,
    Transcript,
)
from dr_providers.outcomes.conformance import (
    conformance_warnings,
    with_conformance_warnings,
)
from dr_providers.outcomes.evidence import (
    PROVIDER_INVOCATION_EVIDENCE_SCHEMA,
    PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION,
    SANITIZE_KEYS,
    ProviderInvocationEvidence,
    RawHttpRequest,
    sanitize_headers,
    sanitize_kwargs,
)
from dr_providers.outcomes.models import (
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
from dr_providers.surfaces.testing.scripted import (
    ScriptedOutcome,
    ScriptedProvider,
)
from dr_providers.translation.anthropic_messages import (
    parse_anthropic_messages_body,
)
from dr_providers.translation.chat_completions import (
    parse_chat_completions_body,
)
from dr_providers.translation.common import (
    cost_from_body,
    token_usage_from_body,
)
from dr_providers.translation.request import build_payload, protocol_path
from dr_providers.translation.response import parse_response
from dr_providers.translation.responses import parse_responses_body
from dr_providers.transport.policy import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    ApiKeyEnv,
    ProviderBaseUrl,
    ProviderTransportPolicy,
    policy_for,
)
from dr_providers.transport.status import classify_status_code

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
        from dr_providers.transport import http  # noqa: PLC0415 -- lazy

        return getattr(http, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
