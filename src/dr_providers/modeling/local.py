"""Torch-free declarations of local model loading and scoring conditions."""

from __future__ import annotations

from enum import UNIQUE, StrEnum, verify
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from dr_providers.core.failures import (
    ControlValidationError,
    RecoverabilityClass,
    failure_record,
)


@verify(UNIQUE)
class LocalDevice(StrEnum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@verify(UNIQUE)
class LocalDtype(StrEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


@verify(UNIQUE)
class Float32MatmulPrecision(StrEnum):
    HIGHEST = "highest"
    HIGH = "high"


@verify(UNIQUE)
class Quantization(StrEnum):
    NONE = "none"
    BITSANDBYTES_INT8 = "bitsandbytes_int8"
    BITSANDBYTES_NF4 = "bitsandbytes_nf4"


@verify(UNIQUE)
class ContinuationTokenization(StrEnum):
    """Causal joint encoding split by context length after moving whitespace.

    Empty-after-rstrip contexts use BOS, falling back to EOS, as a prefix.
    Automatic special-token insertion is disabled.
    """

    LM_EVAL_ENCODE_PAIR = "lm_eval_encode_pair"


class LocalModelSpec(BaseModel):
    """Identity-bearing declared conditions, not proof of backend support.

    A revision selector or local path does not identify immutable contents.
    A backend must validate its supported models and hardware when loading.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision: StrictStr = Field(min_length=1)
    device: LocalDevice
    dtype: LocalDtype
    float32_matmul_precision: Float32MatmulPrecision
    quantization: Quantization
    batch_size: StrictInt = Field(ge=1)
    max_sequence_length: StrictInt = Field(ge=1)
    continuation_tokenization: ContinuationTokenization

    @model_validator(mode="after")
    def _validate_cuda_only_settings(self) -> LocalModelSpec:
        if self.device is not LocalDevice.CUDA:
            if (
                self.float32_matmul_precision
                is not Float32MatmulPrecision.HIGHEST
            ):
                raise ControlValidationError(
                    failure_record(
                        recoverability=RecoverabilityClass.PERMANENT,
                        code="matmul_precision_requires_cuda",
                        message="non-highest matmul precision requires CUDA",
                    )
                )
            if self.quantization is not Quantization.NONE:
                raise ControlValidationError(
                    failure_record(
                        recoverability=RecoverabilityClass.PERMANENT,
                        code="quantization_requires_cuda",
                        message="quantization requires CUDA",
                    )
                )
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "device": self.device.value,
            "dtype": self.dtype.value,
            "float32_matmul_precision": self.float32_matmul_precision.value,
            "quantization": self.quantization.value,
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "continuation_tokenization": self.continuation_tokenization.value,
        }
