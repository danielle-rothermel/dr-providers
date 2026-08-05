import subprocess
import sys
from importlib.metadata import version

import dr_providers

EXPECTED_ALL = {
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
}

PURE_MODULES = (
    "dr_providers.core.failures",
    "dr_providers.core.frozen",
    "dr_providers.core.provider",
    "dr_providers.modeling.call",
    "dr_providers.modeling.controls",
    "dr_providers.modeling.presets",
    "dr_providers.modeling.request",
    "dr_providers.modeling.route",
    "dr_providers.modeling.transcript",
    "dr_providers.outcomes.conformance",
    "dr_providers.outcomes.evidence",
    "dr_providers.outcomes.models",
    "dr_providers.translation.anthropic_messages",
    "dr_providers.translation.chat_completions",
    "dr_providers.translation.common",
    "dr_providers.translation.request",
    "dr_providers.translation.response",
    "dr_providers.translation.responses",
    "dr_providers.transport.policy",
    "dr_providers.transport.status",
    "dr_providers.surfaces.testing.scripted",
)


def test_public_api_exports() -> None:
    assert set(dr_providers.__all__) == EXPECTED_ALL
    for name in EXPECTED_ALL:
        assert getattr(dr_providers, name) is not None


def test_version() -> None:
    assert dr_providers.__version__ == version("dr-providers")


def test_import_root_does_not_load_httpx() -> None:
    code = "import sys, dr_providers; assert 'httpx' not in sys.modules"
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], check=True
    )


def test_import_pure_modules_does_not_load_httpx() -> None:
    imports = "; ".join(f"import {module}" for module in PURE_MODULES)
    code = f"import sys; {imports}; assert 'httpx' not in sys.modules"
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], check=True
    )


def test_api_key_env_names_follow_provider_convention() -> None:
    for member in dr_providers.ApiKeyEnv:
        assert member.value == f"{member.name}_API_KEY"
