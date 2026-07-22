"""Provider Call Definition: versioned, variable-bearing owner of a Config.

A Provider Call Definition declares one Model Route, the output-affecting
generation controls and provider body extensions it exposes as Variables,
their constraints, and their identity effects. It materializes one or
more Provider Call Configs by assigning every required Variable. The
Definition is the sole owner of the Configs it materializes; its typed
reference and Identity Hash identify that owner.

Transport policy is not a Definition/Config concern and never appears
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, StrictStr

from dr_providers.controls import (
    ControlConstraints,
    GenerationControls,
    ProviderBodyExtensions,
    RequestControl,
)
from dr_providers.failures import (
    FailureClass,
    UnsupportedControlError,
    failure_record,
)

if TYPE_CHECKING:
    from dr_providers.config import ProviderCallConfig
from dr_providers.route import ModelRoute  # noqa: TC001 -- pydantic field

PROVIDER_CALL_DEFINITION_SCHEMA = "dr_providers.provider_call_definition"
PROVIDER_CALL_DEFINITION_SCHEMA_VERSION = 1

_CONTROL_ATTR: dict[RequestControl, str] = {
    RequestControl.TEMPERATURE: "temperature",
    RequestControl.TOP_P: "top_p",
    RequestControl.TOKEN_LIMIT: "token_limit",
    RequestControl.REASONING: "reasoning",
}


class ProviderCallDefinition(BaseModel):
    """Versioned variable-bearing description of a provider call's shape.

    ``route`` and ``constraints`` are fixed by the Definition. The
    controls named in ``required_controls`` are Variables that every
    materialized Config MUST assign; other supported controls are
    optional Variables. ``extension_keys`` names the provider body
    extension Variables the Definition exposes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PROVIDER_CALL_DEFINITION_SCHEMA_VERSION
    definition_id: StrictStr
    route: ModelRoute
    constraints: ControlConstraints
    required_controls: frozenset[RequestControl] = frozenset()
    extension_keys: frozenset[StrictStr] = frozenset()

    def identity_payload(self) -> dict[str, Any]:
        """Identity effects the Definition declares (its own identity)."""
        return {
            "schema_version": self.schema_version,
            "definition_id": self.definition_id,
            "route": self.route.identity_payload(),
            "constraints": self.constraints.identity_payload(),
            "required_controls": sorted(
                c.value for c in self.required_controls
            ),
            "extension_keys": sorted(self.extension_keys),
        }

    def materialize(
        self,
        *,
        controls: GenerationControls | None = None,
        extensions: ProviderBodyExtensions | None = None,
    ) -> ProviderCallConfig:
        """Assign Variables and return a complete validated Config.

        Rejects an assignment that sets an unsupported control (unless
        the Definition allows dropping it) or that leaves a required
        control unset, or that sets an undeclared extension key.
        """
        from dr_providers.config import ProviderCallConfig  # noqa: PLC0415

        controls = controls or GenerationControls()
        extensions = extensions or ProviderBodyExtensions()
        self._validate_controls(controls)
        self._validate_extensions(extensions)
        return ProviderCallConfig(
            definition=self,
            controls=controls,
            extensions=extensions,
        )

    def _validate_controls(self, controls: GenerationControls) -> None:
        for control, attr in _CONTROL_ATTR.items():
            is_set = getattr(controls, attr) is not None
            supported = self.constraints.supports(control)
            if is_set and not supported:
                if self.constraints.allow_unsupported_control_drop:
                    continue
                self._raise_unsupported(control)
            if control in self.required_controls and not is_set:
                self._raise_missing(control)

    def _validate_extensions(self, extensions: ProviderBodyExtensions) -> None:
        undeclared = set(extensions.extra_body) - set(self.extension_keys)
        if undeclared:
            raise UnsupportedControlError(
                failure_record(
                    failure_class=FailureClass.PERMANENT,
                    code="undeclared_extension",
                    message=(
                        f"config sets extension keys {sorted(undeclared)!r} "
                        f"not declared by definition "
                        f"{self.definition_id!r}"
                    ),
                    metadata={"undeclared_keys": sorted(undeclared)},
                )
            )

    def _raise_unsupported(self, control: RequestControl) -> None:
        raise UnsupportedControlError(
            failure_record(
                failure_class=FailureClass.PERMANENT,
                code="unsupported_control",
                message=(
                    f"config sets {control.value!r} but definition "
                    f"{self.definition_id!r} cannot transport it"
                ),
                metadata={
                    "provider": self.route.provider.value,
                    "protocol": self.route.protocol.value,
                    "control": control.value,
                },
            )
        )

    def _raise_missing(self, control: RequestControl) -> None:
        raise UnsupportedControlError(
            failure_record(
                failure_class=FailureClass.PERMANENT,
                code="missing_required_control",
                message=(
                    f"definition {self.definition_id!r} requires control "
                    f"{control.value!r} but the config leaves it unset"
                ),
                metadata={"control": control.value},
            )
        )
