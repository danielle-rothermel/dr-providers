from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dr_providers import (
    ContinuationTokenization,
    ControlValidationError,
    Float32MatmulPrecision,
    GenerationControls,
    LocalDevice,
    LocalDtype,
    LocalModelSpec,
    PromptRendering,
    ProviderBodyExtensions,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderCallKind,
    ProviderGenerateRequest,
    ProviderScoreRequest,
    Quantization,
    ReasoningEffort,
    RecoverabilityClass,
    TokenLimitParameter,
    Transcript,
    Verbosity,
    huggingface_config,
    openai_chat_config,
)

EXPECTED_SPEC = {
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "device": "cuda",
    "dtype": "float32",
    "float32_matmul_precision": "highest",
    "quantization": "none",
    "batch_size": 1,
    "max_sequence_length": 1024,
    "continuation_tokenization": "separate_encode",
}


def test_local_spec_and_definition_payloads(
    local_config: ProviderCallConfig,
) -> None:
    spec = local_config.definition.local
    assert spec is not None
    assert spec.identity_payload() == EXPECTED_SPEC
    assert local_config.definition.identity_payload() == {
        "definition_id": "huggingface.transformers",
        "route": {
            "provider": "huggingface",
            "protocol": "transformers",
            "model": "example/model",
        },
        "local": EXPECTED_SPEC,
        "supported_kinds": ["generate", "score"],
        "prompt_rendering": "role_messages",
        "constraints": {
            "supported_controls": [
                "seed",
                "temperature",
                "token_limit",
                "top_p",
            ],
            "token_limit_parameter": "max_new_tokens",
            "reasoning_shape": "none",
        },
        "required_controls": [],
        "extension_keys": [],
    }
    assert spec == LocalModelSpec.model_validate_json(spec.model_dump_json())
    assert (
        local_config.definition.constraints.token_limit_parameter
        is TokenLimitParameter.MAX_NEW_TOKENS
    )
    assert local_config.definition.supported_kinds == {
        ProviderCallKind.GENERATE,
        ProviderCallKind.SCORE,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", ""),
        ("revision", 7),
        ("revision", None),
        ("device", "auto"),
        ("dtype", "float16"),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", True),
        ("batch_size", "1"),
        ("batch_size", 1.0),
        ("max_sequence_length", 0),
        ("max_sequence_length", False),
        ("max_sequence_length", "8"),
        ("max_sequence_length", 8.0),
        ("continuation_tokenization", "joint_encode"),
        ("float32_matmul_precision", "medium"),
        ("quantization", "int4"),
    ],
)
def test_local_spec_refuses_invalid_values(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        LocalModelSpec.model_validate({**EXPECTED_SPEC, field: value})


@pytest.mark.parametrize("field", list(EXPECTED_SPEC))
def test_all_local_spec_fields_are_required(field: str) -> None:
    data = {key: value for key, value in EXPECTED_SPEC.items() if key != field}
    with pytest.raises(ValidationError):
        LocalModelSpec.model_validate(data)


def test_local_models_are_frozen_and_refuse_extra_fields(
    local_config: ProviderCallConfig,
) -> None:
    with pytest.raises(ValidationError):
        LocalModelSpec.model_validate({**EXPECTED_SPEC, "extra": "x"})
    spec = local_config.definition.local
    assert spec is not None
    with pytest.raises(ValidationError):
        spec.batch_size = 2


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize(
    ("override", "code"),
    [
        (
            {"float32_matmul_precision": "high"},
            "matmul_precision_requires_cuda",
        ),
        ({"quantization": "bitsandbytes_int8"}, "quantization_requires_cuda"),
        ({"quantization": "bitsandbytes_nf4"}, "quantization_requires_cuda"),
    ],
)
def test_cuda_only_settings(
    device: str, override: dict[str, str], code: str
) -> None:
    with pytest.raises(ControlValidationError) as caught:
        LocalModelSpec.model_validate(
            {**EXPECTED_SPEC, "device": device, **override}
        )
    assert caught.value.failure.code == code
    assert caught.value.failure.recoverability is RecoverabilityClass.PERMANENT
    assert (
        LocalModelSpec.model_validate({**EXPECTED_SPEC, **override}).device
        is LocalDevice.CUDA
    )


@pytest.mark.parametrize(
    ("scenario", "code"),
    [
        ("missing", "local_spec_required"),
        ("http", "local_spec_forbidden"),
        ("extensions", "local_extensions_forbidden"),
        ("reasoning", "reasoning_protocol_unsupported"),
        ("verbosity", "verbosity_protocol_unsupported"),
    ],
)
def test_definition_local_validation_codes(
    local_config: ProviderCallConfig, scenario: str, code: str
) -> None:
    data = local_config.definition.model_dump(mode="python")
    if scenario == "missing":
        data["local"] = None
    elif scenario == "http":
        data["route"] = openai_chat_config(model="m").route
    elif scenario == "extensions":
        data["extension_keys"] = {"custom"}
    else:
        data["constraints"]["supported_controls"] = {scenario}
        data["constraints"]["reasoning_shape"] = "effort_field"
    with pytest.raises(ControlValidationError) as caught:
        ProviderCallDefinition.model_validate(data)
    assert caught.value.failure.code == code
    assert caught.value.failure.recoverability is RecoverabilityClass.PERMANENT


@pytest.mark.parametrize(
    "controls",
    [
        GenerationControls(reasoning=ReasoningEffort.LOW),
        GenerationControls(verbosity=Verbosity.HIGH),
    ],
)
def test_preset_refuses_unsupported_controls(
    local_config: ProviderCallConfig, controls: GenerationControls
) -> None:
    with pytest.raises(ControlValidationError) as caught:
        local_config.definition.materialize(controls=controls)
    assert caught.value.failure.code == "unsupported_control"


def test_preset_admits_generate_and_score(
    local_config: ProviderCallConfig,
) -> None:
    generate_config = local_config.definition.materialize(
        controls=GenerationControls(
            temperature=0.7,
            top_p=0.9,
            token_limit=8,
            seed=42,
        )
    )
    assert (
        ProviderGenerateRequest(
            config=generate_config, transcript=Transcript(messages=())
        ).kind
        is ProviderCallKind.GENERATE
    )
    assert (
        ProviderScoreRequest(
            config=local_config, context="", continuations=("answer",)
        ).kind
        is ProviderCallKind.SCORE
    )
    with pytest.raises(ControlValidationError):
        local_config.definition.materialize(
            extensions=ProviderBodyExtensions(extra_body={"custom": True})
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "other"),
        ("device", LocalDevice.CPU),
        ("device", LocalDevice.MPS),
        ("dtype", LocalDtype.BFLOAT16),
        ("float32_matmul_precision", Float32MatmulPrecision.HIGH),
        ("quantization", Quantization.BITSANDBYTES_INT8),
        ("quantization", Quantization.BITSANDBYTES_NF4),
        ("batch_size", 2),
        ("max_sequence_length", 2048),
    ],
)
def test_each_variable_local_dimension_changes_definition_identity(
    local_config: ProviderCallConfig, field: str, value: Any
) -> None:
    data = local_config.definition.model_dump(mode="python")
    data["local"][field] = value
    variant = ProviderCallDefinition.model_validate(data)
    assert variant.identity_hash != local_config.definition.identity_hash
    assert variant.identity_payload()["local"] == {
        **EXPECTED_SPEC,
        field: value,
    }


def test_singleton_tokenization_is_explicit_identity(
    local_config: ProviderCallConfig,
) -> None:
    assert list(ContinuationTokenization) == [
        ContinuationTokenization.SEPARATE_ENCODE
    ]
    assert (
        local_config.definition.identity_payload()["local"][
            "continuation_tokenization"
        ]
        == "separate_encode"
    )


def test_preset_preserves_model_and_rendering() -> None:
    config = huggingface_config(
        model="./models/../my-model",
        revision="local-version",
        device=LocalDevice.CPU,
        dtype=LocalDtype.BFLOAT16,
        batch_size=3,
        max_sequence_length=512,
        prompt_rendering=PromptRendering.FLAT_TEXT,
    )
    assert config.route.model == "./models/../my-model"
    assert config.definition.prompt_rendering is PromptRendering.FLAT_TEXT
    assert config.quota_identity == config.route.quota_identity


def test_local_golden_hashes(local_config: ProviderCallConfig) -> None:
    generate = ProviderGenerateRequest(
        config=local_config, transcript=Transcript(messages=())
    )
    score = ProviderScoreRequest(
        config=local_config, context="context", continuations=("answer",)
    )
    assert (
        local_config.definition.identity_hash
        == "5f53b843a13aae3301aacac26043250f7bee0f193f9173b9a5a23fee1312b33f"
    )
    assert (
        local_config.identity_hash
        == "1aa78e83e344dfddbf2c79d30801ebe85eb4010542771cbf0a6343f00cbaa94f"
    )
    assert (
        generate.identity_hash
        == "667b6a53d7af8e8c4c50df9adf94160f2ae4c75b1c6d33eec94916c62803de77"
    )
    assert (
        score.identity_hash
        == "831ff01046a6616124ef30de72b93726bf17ef5612ba76d9139888831a9912e1"
    )
