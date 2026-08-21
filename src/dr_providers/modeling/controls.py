from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from dr_serialize import (
    Jsonable,
    StrictJsonError,
    canonical_sorted_values,
    validate_strict_json,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from dr_providers.core.failures import (
    ControlValidationError,
    RecoverabilityClass,
    failure_record,
)
from dr_providers.core.frozen import _deep_freeze, _FrozenMap, _thaw


class RequestControl(StrEnum):
    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    TOKEN_LIMIT = "token_limit"  # noqa: S105 -- knob name, not a secret
    REASONING = "reasoning"
    SEED = "seed"


CONTROL_ATTR: dict[RequestControl, str] = {
    control: control.value for control in RequestControl
}
assert set(CONTROL_ATTR) == set(RequestControl)


class TokenLimitParameter(StrEnum):
    MAX_TOKENS = "max_tokens"
    MAX_COMPLETION_TOKENS = "max_completion_tokens"
    MAX_OUTPUT_TOKENS = "max_output_tokens"


class ReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ReasoningRequestShape(StrEnum):
    NONE = "none"
    EFFORT_FIELD = "effort_field"
    REASONING_OBJECT = "reasoning_object"


class GenerationControls(BaseModel):
    """Only set controls participate in Config identity and the wire body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = Field(
        default=None,
        allow_inf_nan=False,
        strict=True,
    )
    top_p: float | None = Field(
        default=None,
        allow_inf_nan=False,
        strict=True,
    )
    token_limit: StrictInt | None = None
    reasoning: ReasoningEffort | None = None
    seed: StrictInt | None = None

    def identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.token_limit is not None:
            payload["token_limit"] = self.token_limit
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning.value
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


DEFAULT_SUPPORTED_CONTROLS: frozenset[RequestControl] = frozenset(
    {
        RequestControl.TEMPERATURE,
        RequestControl.TOP_P,
        RequestControl.TOKEN_LIMIT,
    }
)


class ControlConstraints(BaseModel):
    """Definition-owned wire mappings that participate in Config identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    supported_controls: frozenset[RequestControl] = DEFAULT_SUPPORTED_CONTROLS
    token_limit_parameter: TokenLimitParameter
    reasoning_shape: ReasoningRequestShape = ReasoningRequestShape.NONE

    @model_validator(mode="after")
    def _require_mapping_for_supported_reasoning(self) -> ControlConstraints:
        if (
            self.supports(RequestControl.REASONING)
            and self.reasoning_shape is ReasoningRequestShape.NONE
        ):
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="reasoning_mapping_missing",
                    message=(
                        "reasoning cannot be advertised as supported without "
                        "a wire mapping"
                    ),
                )
            )
        return self

    @field_serializer("supported_controls", when_used="json")
    def _serialize_supported_controls(
        self,
        value: frozenset[RequestControl],
    ) -> list[Jsonable]:
        return canonical_sorted_values(control.value for control in value)

    def supports(self, control: RequestControl) -> bool:
        return control in self.supported_controls

    def identity_payload(self) -> dict[str, Any]:
        return {
            "supported_controls": sorted(
                c.value for c in self.supported_controls
            ),
            "token_limit_parameter": self.token_limit_parameter.value,
            "reasoning_shape": self.reasoning_shape.value,
        }


class ProviderBodyExtensions(BaseModel):
    """Deeply frozen wire extensions that participate in Config identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extra_body: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("extra_body", mode="before")
    @classmethod
    def _reject_non_mapping(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            msg = "extra_body must be a mapping"
            raise TypeError(msg)
        return value

    @model_validator(mode="after")
    def _validate_and_freeze_extra_body(self) -> ProviderBodyExtensions:
        if isinstance(self.extra_body, _FrozenMap):
            return self
        # Pydantic converts Mapping fields to dicts; validate before freezing.
        try:
            validated = validate_strict_json(dict(self.extra_body))
        except StrictJsonError as error:
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="invalid_extension_json",
                    message=(
                        "extra_body must contain only strict finite JSON "
                        f"values: {error}"
                    ),
                    metadata=error.diagnostics(),
                ),
            ) from error
        object.__setattr__(self, "extra_body", _deep_freeze(validated))
        return self

    @field_serializer("extra_body")
    def _serialize_extra_body(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        return _thaw(value)

    def identity_payload(self) -> dict[str, Any]:
        return _thaw(self.extra_body)
