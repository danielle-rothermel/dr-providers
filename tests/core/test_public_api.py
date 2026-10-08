import subprocess
import sys

import dr_providers

PURE_MODULES = (
    "dr_providers.local.backend",
    "dr_providers.local.provider",
    "dr_providers.local.transformers",
    "dr_providers.surfaces.testing.fake_backend",
    "dr_providers.core.failures",
    "dr_providers.core.frozen",
    "dr_providers.core.provider",
    "dr_providers.lifecycle",
    "dr_providers.lifecycle.classifier",
    "dr_providers.lifecycle.driver",
    "dr_providers.lifecycle.models",
    "dr_providers.lifecycle.outcomes",
    "dr_providers.lifecycle.policy",
    "dr_providers.lifecycle.reducer",
    "dr_providers.modeling.call",
    "dr_providers.modeling.controls",
    "dr_providers.modeling.local",
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
    assert "provider_call_hash" in dr_providers.__all__
    assert "PromptRendering" in dr_providers.__all__
    for name in dr_providers.__all__:
        assert getattr(dr_providers, name) is not None


def test_import_root_does_not_load_httpx() -> None:
    code = "import sys, dr_providers; assert 'httpx' not in sys.modules"
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], check=True
    )


def test_import_pure_modules_does_not_load_httpx() -> None:
    imports = "; ".join(f"import {module}" for module in PURE_MODULES)
    code = (
        f"import sys; {imports}; "
        "assert {'httpx', 'torch', 'transformers', 'bitsandbytes'}"
        ".isdisjoint(sys.modules)"
    )
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], check=True
    )


def test_score_api_exports_and_removed_request_constants() -> None:
    expected = {
        "ProviderCallKind",
        "ProviderGenerateRequest",
        "ProviderScoreRequest",
        "ProviderCallRequest",
        "ContinuationScore",
        "ProviderScoreResponse",
        "is_score_response",
        "PROVIDER_GENERATE_REQUEST_SCHEMA",
        "PROVIDER_GENERATE_REQUEST_SCHEMA_VERSION",
        "PROVIDER_SCORE_REQUEST_SCHEMA",
        "PROVIDER_SCORE_REQUEST_SCHEMA_VERSION",
    }
    assert expected <= set(dr_providers.__all__)
    for name in (
        "PROVIDER_CALL_REQUEST_SCHEMA",
        "PROVIDER_CALL_REQUEST_SCHEMA_VERSION",
    ):
        assert name not in dr_providers.__all__
        assert not hasattr(dr_providers, name)


def test_local_public_surface_is_torch_free() -> None:
    expected = {
        "LocalModelProvider",
        "LocalBackend",
        "TransformersBackend",
        "LocalBackendFailure",
        "GenerationOutput",
        "ScoredContinuation",
        "FakeLocalBackend",
        "LocalModelSpec",
        "LocalDevice",
        "LocalDtype",
        "Float32MatmulPrecision",
        "Quantization",
        "ContinuationTokenization",
        "huggingface_config",
        "LocalExecutionEvidence",
    }
    assert expected <= set(dr_providers.__all__)
    assert {
        "HTTP_PROVIDER_KINDS",
        "WIRE_PROTOCOLS",
        "LOCAL_OUT_OF_MEMORY_CODE",
    }.isdisjoint(dr_providers.__all__)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, dr_providers; "
                "assert {'torch', 'transformers', 'bitsandbytes'}"
                ".isdisjoint(sys.modules)"
            ),
        ],
        check=True,
    )
