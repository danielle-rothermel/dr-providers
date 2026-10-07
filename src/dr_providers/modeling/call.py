from __future__ import annotations

from functools import cached_property
from typing import Any

from dr_serialize import (
    IdentityDocument,
    Jsonable,
    build_identity_document,
    canonical_sorted_values,
    identity_document_hash,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    field_serializer,
    model_validator,
)

from dr_providers.core.failures import (
    ControlValidationError,
    RecoverabilityClass,
    failure_record,
)
from dr_providers.modeling.controls import (
    CONTROL_ATTR,
    ControlConstraints,
    GenerationControls,
    PromptRendering,
    ProviderBodyExtensions,
    ProviderCallKind,
    ReasoningEffort,
    RequestControl,
)
from dr_providers.modeling.route import (
    ModelRoute,
    Protocol,
    ProviderQuotaIdentity,
)

PROVIDER_CALL_DEFINITION_SCHEMA = "dr_providers.provider_call_definition"
PROVIDER_CALL_DEFINITION_SCHEMA_VERSION = 5

PROVIDER_CALL_CONFIG_SCHEMA = "dr_providers.provider_call_config"
PROVIDER_CALL_CONFIG_SCHEMA_VERSION = 2

# Controls whose single top-level wire key some protocols lack, mapped to
# those protocols and the failure code a definition advertising them raises.
# Responses nests verbosity under ``text``; this package maps only the
# top-level ``verbosity`` key, so Responses does not carry it.
_PROTOCOLS_WITHOUT_WIRE_KEY: dict[
    RequestControl, tuple[frozenset[Protocol], str]
] = {
    RequestControl.SEED: (
        frozenset({Protocol.RESPONSES, Protocol.ANTHROPIC_MESSAGES}),
        "seed_protocol_unsupported",
    ),
    RequestControl.VERBOSITY: (
        frozenset({Protocol.RESPONSES, Protocol.ANTHROPIC_MESSAGES}),
        "verbosity_protocol_unsupported",
    ),
}

_ANTHROPIC_REASONING_EFFORTS = frozenset(
    {
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
    }
)


class ProviderCallDefinition(BaseModel):
    """Identity includes fixed fields, supported kinds, and control variables.

    Score requests require empty ``required_controls``: config assignments
    must satisfy required controls, while scoring forbids set controls.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    definition_id: StrictStr
    route: ModelRoute
    supported_kinds: frozenset[ProviderCallKind]
    constraints: ControlConstraints
    prompt_rendering: PromptRendering = PromptRendering.ROLE_MESSAGES
    required_controls: frozenset[RequestControl] = frozenset()
    extension_keys: frozenset[StrictStr] = frozenset()

    @field_serializer("supported_kinds", when_used="json")
    def _serialize_supported_kinds(
        self, value: frozenset[ProviderCallKind]
    ) -> list[Jsonable]:
        return canonical_sorted_values(kind.value for kind in value)

    @model_validator(mode="after")
    def _require_supported_kind(self) -> ProviderCallDefinition:
        if not self.supported_kinds:
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="no_supported_kinds",
                    message="a definition must support at least one call kind",
                )
            )
        return self

    @field_serializer("required_controls", when_used="json")
    def _serialize_required_controls(
        self,
        value: frozenset[RequestControl],
    ) -> list[Jsonable]:
        return canonical_sorted_values(control.value for control in value)

    @field_serializer("extension_keys", when_used="json")
    def _serialize_extension_keys(
        self,
        value: frozenset[StrictStr],
    ) -> list[Jsonable]:
        return canonical_sorted_values(value)

    @model_validator(mode="after")
    def _required_subset_of_supported(self) -> ProviderCallDefinition:
        supported = self.constraints.supported_controls
        unsupported = self.required_controls - supported
        if unsupported:
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="required_control_unsupported",
                    message=(
                        f"definition {self.definition_id!r} requires controls "
                        f"{sorted(c.value for c in unsupported)!r} that its "
                        f"constraints do not support, so no Config could ever "
                        f"materialize"
                    ),
                    metadata={
                        "controls": sorted(c.value for c in unsupported),
                    },
                )
            )
        return self

    @model_validator(mode="after")
    def _advertised_controls_have_protocol_wire_key(
        self,
    ) -> ProviderCallDefinition:
        protocol = self.route.protocol
        for control, (protocols, code) in _PROTOCOLS_WITHOUT_WIRE_KEY.items():
            if self.constraints.supports(control) and protocol in protocols:
                raise ControlValidationError(
                    failure_record(
                        recoverability=RecoverabilityClass.PERMANENT,
                        code=code,
                        message=(
                            f"definition {self.definition_id!r} advertises "
                            f"{control.value} but protocol "
                            f"{protocol.value!r} has no {control.value} "
                            "wire key"
                        ),
                        metadata={
                            "protocol": protocol.value,
                            "control": control.value,
                            "definition_id": self.definition_id,
                        },
                    )
                )
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "route": self.route.identity_payload(),
            "supported_kinds": sorted(
                kind.value for kind in self.supported_kinds
            ),
            "prompt_rendering": self.prompt_rendering.value,
            "constraints": self.constraints.identity_payload(),
            "required_controls": sorted(
                c.value for c in self.required_controls
            ),
            "extension_keys": sorted(self.extension_keys),
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_CALL_DEFINITION_SCHEMA,
            schema_version=PROVIDER_CALL_DEFINITION_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())

    def materialize(
        self,
        *,
        controls: GenerationControls | None = None,
        extensions: ProviderBodyExtensions | None = None,
    ) -> ProviderCallConfig:
        return ProviderCallConfig(
            definition=self,
            controls=controls or GenerationControls(),
            extensions=extensions or ProviderBodyExtensions(),
        )


class ProviderCallConfig(BaseModel):
    """Validated assignment whose identity excludes transport policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    definition: ProviderCallDefinition
    controls: GenerationControls = GenerationControls()
    extensions: ProviderBodyExtensions = ProviderBodyExtensions()

    @model_validator(mode="after")
    def _validate_assignment(self) -> ProviderCallConfig:
        self._validate_controls()
        self._validate_extensions()
        return self

    def _validate_controls(self) -> None:
        constraints = self.definition.constraints
        for control, attr in CONTROL_ATTR.items():
            is_set = getattr(self.controls, attr) is not None
            supported = constraints.supports(control)
            if is_set and not supported:
                self._raise_unsupported(control)
            if control in self.definition.required_controls and not is_set:
                self._raise_missing(control)
        effort = self.controls.reasoning
        if (
            self.route.protocol is Protocol.ANTHROPIC_MESSAGES
            and effort is not None
            and constraints.supports(RequestControl.REASONING)
            and effort not in _ANTHROPIC_REASONING_EFFORTS
        ):
            allowed_efforts = sorted(
                allowed.value for allowed in _ANTHROPIC_REASONING_EFFORTS
            )
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="unmappable_reasoning_effort",
                    message=(
                        f"reasoning effort {effort.value!r} has no Anthropic "
                        f"Messages effort equivalent; use one of "
                        f"{allowed_efforts!r}"
                    ),
                    metadata={"effort": effort.value},
                )
            )

    def _validate_extensions(self) -> None:
        undeclared = set(self.extensions.extra_body) - set(
            self.definition.extension_keys
        )
        if undeclared:
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="undeclared_extension",
                    message=(
                        f"config sets extension keys {sorted(undeclared)!r} "
                        f"not declared by definition "
                        f"{self.definition.definition_id!r}"
                    ),
                    metadata={"undeclared_keys": sorted(undeclared)},
                )
            )
        reserved = set(self.extensions.extra_body) & self._reserved_wire_keys()
        if reserved:
            raise ControlValidationError(
                failure_record(
                    recoverability=RecoverabilityClass.PERMANENT,
                    code="reserved_extension_key",
                    message=(
                        f"config sets extension keys {sorted(reserved)!r} "
                        f"that would overwrite validated core wire fields "
                        f"on protocol {self.route.protocol.value!r}"
                    ),
                    metadata={"reserved_keys": sorted(reserved)},
                )
            )

    def _reserved_wire_keys(self) -> set[str]:
        """Reserve extension keys whose override would defeat identity."""
        constraints = self.definition.constraints
        return {
            "model",
            "messages",
            "input",
            "instructions",
            "system",
            "temperature",
            "top_p",
            "reasoning",
            "reasoning_effort",
            "output_config",
            "seed",
            "verbosity",
            constraints.token_limit_parameter.value,
        }

    @property
    def route(self) -> ModelRoute:
        return self.definition.route

    @property
    def quota_identity(self) -> ProviderQuotaIdentity:
        return self.definition.route.quota_identity

    def identity_payload(self) -> dict[str, Any]:
        """Include Definition hash and assignments; omit transport policy."""
        return {
            "definition_hash": self.definition.identity_hash,
            "controls": self.controls.identity_payload(),
            "extensions": self.extensions.identity_payload(),
        }

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_CALL_CONFIG_SCHEMA,
            schema_version=PROVIDER_CALL_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        return identity_document_hash(self.identity_document())

    def _raise_unsupported(self, control: RequestControl) -> None:
        raise ControlValidationError(
            failure_record(
                recoverability=RecoverabilityClass.PERMANENT,
                code="unsupported_control",
                message=(
                    f"config sets {control.value!r} but definition "
                    f"{self.definition.definition_id!r} cannot transport it"
                ),
                metadata={
                    "provider": self.route.provider.value,
                    "protocol": self.route.protocol.value,
                    "control": control.value,
                },
            )
        )

    def _raise_missing(self, control: RequestControl) -> None:
        raise ControlValidationError(
            failure_record(
                recoverability=RecoverabilityClass.PERMANENT,
                code="missing_required_control",
                message=(
                    f"definition {self.definition.definition_id!r} requires "
                    f"control {control.value!r} but the config leaves it unset"
                ),
                metadata={"control": control.value},
            )
        )
