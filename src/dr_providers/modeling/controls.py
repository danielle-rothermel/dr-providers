"""Output-affecting generation controls and their wire mapping.

Generation controls shape the produced generation, so they are
identity-bearing fields of a Provider Call Config. How a control
serializes on the wire (token-limit parameter name, reasoning shape)
is declared by the Provider Call Definition, not by the request.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from dr_serialize import Jsonable, canonical_sorted_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from dr_providers.core.frozen import _deep_freeze, _thaw


class RequestControl(StrEnum):
    """Named output-affecting controls a Definition may declare."""

    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    TOKEN_LIMIT = "token_limit"  # noqa: S105 -- knob name, not a secret
    REASONING = "reasoning"


# The GenerationControls attribute name backing each control. Each control
# maps to exactly ``member.value``; this dict is the single source of truth
# for that correspondence and is asserted exhaustive over RequestControl.
CONTROL_ATTR: dict[RequestControl, str] = {
    control: control.value for control in RequestControl
}
assert set(CONTROL_ATTR) == set(RequestControl)


class TokenLimitParameter(StrEnum):
    MAX_TOKENS = "max_tokens"
    MAX_COMPLETION_TOKENS = "max_completion_tokens"
    MAX_OUTPUT_TOKENS = "max_output_tokens"


class ReasoningEffort(StrEnum):
    """Typed cross-provider reasoning level (see ADR 0001)."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ReasoningRequestShape(StrEnum):
    """How a config serializes reasoning effort on the wire."""

    NONE = "none"
    EFFORT_FIELD = "effort_field"
    REASONING_OBJECT = "reasoning_object"


class GenerationControls(BaseModel):
    """Assigned output-affecting generation controls (identity-bearing).

    Every set control participates in Config identity. Unset controls
    (``None``) are absent from both identity and the wire payload.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = None
    top_p: float | None = None
    token_limit: StrictInt | None = None
    reasoning: ReasoningEffort | None = None

    def identity_payload(self) -> dict[str, Any]:
        """Only set controls appear, so identity never depends on an
        absent knob."""
        payload: dict[str, Any] = {}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.token_limit is not None:
            payload["token_limit"] = self.token_limit
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning.value
        return payload


DEFAULT_SUPPORTED_CONTROLS: frozenset[RequestControl] = frozenset(
    RequestControl
)


class ControlConstraints(BaseModel):
    """Definition-declared constraints and wire mapping for controls.

    Declares which controls the route can transport, how the token
    limit and reasoning effort serialize on the wire, and whether an
    unsupported control set on a Config is dropped or rejected. These
    mapping choices are output-affecting, so they are part of Config
    identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    supported_controls: frozenset[RequestControl] = DEFAULT_SUPPORTED_CONTROLS
    token_limit_parameter: TokenLimitParameter
    reasoning_shape: ReasoningRequestShape = ReasoningRequestShape.NONE
    allow_unsupported_control_drop: bool = False

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
            "allow_unsupported_control_drop": (
                self.allow_unsupported_control_drop
            ),
        }


class ProviderBodyExtensions(BaseModel):
    """Output-affecting provider body extensions merged into the wire
    payload.

    ``extra_body`` is deeply immutable: nested mappings become read-only
    proxies and lists become tuples, so an extension set cannot be mutated
    after construction and thus can never drift from the cached identity
    hash of the owning Config.
    """

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
    def _freeze_extra_body(self) -> ProviderBodyExtensions:
        # Pydantic canonicalizes a ``Mapping`` field into a plain dict, so
        # deep-freeze after validation and reassign the read-only proxy: the
        # whole structure (including the top level) is then immutable and can
        # never drift from the owning Config's cached identity hash.
        object.__setattr__(self, "extra_body", _deep_freeze(self.extra_body))
        return self

    @field_serializer("extra_body")
    def _serialize_extra_body(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        return _thaw(value)

    def identity_payload(self) -> dict[str, Any]:
        return _thaw(self.extra_body)
