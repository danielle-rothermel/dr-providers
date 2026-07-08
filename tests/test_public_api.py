import subprocess
import sys
from importlib.metadata import version

import dr_providers

EXPECTED_ALL = {
    "FAILURE_ERROR_TYPES",
    "RECOVERABLE_FAILURE_CLASSES",
    "RETRYABLE_FAILURE_CLASSES",
    "SANITIZE_KEYS",
    "ApiKeyEnv",
    "CostInfo",
    "EndpointKind",
    "FailureClass",
    "FixtureOutcome",
    "FixtureProvider",
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
}

PURE_MODULES = (
    "dr_providers.config",
    "dr_providers.failures",
    "dr_providers.request",
    "dr_providers.response",
    "dr_providers.conformance",
    "dr_providers.fixture",
    "dr_providers.provider",
)


def test_public_api_exports() -> None:
    assert set(dr_providers.__all__) == EXPECTED_ALL


def test_version() -> None:
    assert dr_providers.__version__ == version("dr-providers")


def test_top_level_imports() -> None:
    assert dr_providers.LlmRequest is not None
    assert dr_providers.HttpProvider is not None


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
