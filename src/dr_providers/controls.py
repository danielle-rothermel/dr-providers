"""Output-affecting generation controls and their wire mapping.

Generation controls shape the produced generation, so they are
identity-bearing fields of a Provider Call Config. How a control
serializes on the wire (token-limit parameter name, reasoning shape)
is declared by the Provider Call Definition, not by the request.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class RequestControl(StrEnum):
    """Named output-affecting controls a Definition may declare."""

    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    TOKEN_LIMIT = "token_limit"  # noqa: S105 -- knob name, not a secret
    REASONING = "reasoning"


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
    {
        RequestControl.TEMPERATURE,
        RequestControl.TOP_P,
        RequestControl.TOKEN_LIMIT,
        RequestControl.REASONING,
    }
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
    payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extra_body: dict[str, Any] = Field(default_factory=dict)

    def identity_payload(self) -> dict[str, Any]:
        return dict(self.extra_body)
