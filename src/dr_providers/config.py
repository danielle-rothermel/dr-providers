"""Provider Call Definition and Config: identity-bearing call models.

A Provider Call Definition is the versioned, variable-bearing owner. It
declares one Model Route, the output-affecting generation controls and
provider body extensions it exposes as Variables, their constraints, and
their identity effects. It materializes one or more Provider Call Configs
by assigning every required Variable.

A Provider Call Config is a complete validated assignment carrying its
typed Definition reference and its full 64-char SHA-256 Identity Hash (via
dr-serialize). Its identity embeds the owning Definition's identity plus
the assigned controls and extensions. Transport policy is excluded from
every Definition/Config identity and never appears here.

Both models live in this module so materialization is a plain method with
no circular import; ``dr_providers.definition`` re-exports the Definition
for its historical import path.

Preset builders ship for OpenRouter, OpenAI, Gemini (OpenAI-compat), and
Anthropic Messages.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from dr_serialize import (
    IdentityDocument,
    build_identity_document,
    identity_document_hash,
)
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from dr_providers.controls import (
    CONTROL_ATTR,
    ControlConstraints,
    GenerationControls,
    ProviderBodyExtensions,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
)
from dr_providers.failures import (
    ControlValidationError,
    FailureClass,
    failure_record,
)
from dr_providers.route import (
    ModelRoute,
    Protocol,
    ProviderKind,
    ProviderQuotaIdentity,
)

PROVIDER_CALL_DEFINITION_SCHEMA = "dr_providers.provider_call_definition"
PROVIDER_CALL_DEFINITION_SCHEMA_VERSION = 1

PROVIDER_CALL_CONFIG_SCHEMA = "dr_providers.provider_call_config"
PROVIDER_CALL_CONFIG_SCHEMA_VERSION = 1


class ProviderCallDefinition(BaseModel):
    """Versioned variable-bearing description of a provider call's shape.

    ``route`` and ``constraints`` are fixed by the Definition. The
    controls named in ``required_controls`` are Variables that every
    materialized Config MUST assign; other supported controls are
    optional Variables. ``extension_keys`` names the provider body
    extension Variables the Definition exposes.

    The Definition is itself identified: its identity payload fully
    captures ``definition_id``, ``schema_version``, ``constraints``,
    ``required_controls`` and ``extension_keys`` so that a Config which
    embeds the Definition Identity Hash is bound to every declared
    variable and constraint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PROVIDER_CALL_DEFINITION_SCHEMA_VERSION
    definition_id: StrictStr
    route: ModelRoute
    constraints: ControlConstraints
    required_controls: frozenset[RequestControl] = frozenset()
    extension_keys: frozenset[StrictStr] = frozenset()

    @model_validator(mode="after")
    def _required_subset_of_supported(self) -> ProviderCallDefinition:
        supported = self.constraints.supported_controls
        unsupported = self.required_controls - supported
        if unsupported:
            raise ControlValidationError(
                failure_record(
                    failure_class=FailureClass.PERMANENT,
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

    def identity_document(self) -> IdentityDocument:
        return build_identity_document(
            schema=PROVIDER_CALL_DEFINITION_SCHEMA,
            schema_version=PROVIDER_CALL_DEFINITION_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @cached_property
    def identity_hash(self) -> str:
        """Full 64-char lowercase SHA-256 Definition Identity Hash."""
        return identity_document_hash(self.identity_document())

    def materialize(
        self,
        *,
        controls: GenerationControls | None = None,
        extensions: ProviderBodyExtensions | None = None,
    ) -> ProviderCallConfig:
        """Assign Variables and return a complete validated Config.

        Rejects an assignment that sets an unsupported control (unless
        the Definition allows dropping it) or that leaves a required
        control unset, or that sets an undeclared extension key. The
        invariants are enforced by ``ProviderCallConfig`` validation, so
        they hold however a Config is built.
        """
        return ProviderCallConfig(
            definition=self,
            controls=controls or GenerationControls(),
            extensions=extensions or ProviderBodyExtensions(),
        )


class ProviderCallConfig(BaseModel):
    """A Definition with every required Variable set, plus Identity Hash.

    Construct via :meth:`ProviderCallDefinition.materialize` or a preset
    builder; direct construction and deserialization are also validated,
    since the control/extension invariants live in model validation rather
    than only in ``materialize``.
    """

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
                if constraints.allow_unsupported_control_drop:
                    continue
                self._raise_unsupported(control)
            if control in self.definition.required_controls and not is_set:
                self._raise_missing(control)

    def _validate_extensions(self) -> None:
        undeclared = set(self.extensions.extra_body) - set(
            self.definition.extension_keys
        )
        if undeclared:
            raise ControlValidationError(
                failure_record(
                    failure_class=FailureClass.PERMANENT,
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
                    failure_class=FailureClass.PERMANENT,
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
        """Core wire-body keys an extension must never override.

        Extensions merge into the wire payload after the validated core
        fields; allowing them to overwrite ``model``/``messages``/a
        transported control would silently defeat identity, so they are
        rejected at validation time.
        """
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
            constraints.token_limit_parameter.value,
        }

    @property
    def route(self) -> ModelRoute:
        return self.definition.route

    @property
    def quota_identity(self) -> ProviderQuotaIdentity:
        return self.definition.route.quota_identity

    def identity_payload(self) -> dict[str, Any]:
        """Owning Definition identity + assigned controls/extensions.

        Embeds the Definition Identity Hash (which itself covers the Model
        Route, constraints, required controls, and declared extension keys)
        rather than a partial copy, so the Config identity fully determines
        what the undeclared-extension and required-control checks enforce.
        Transport policy is excluded.
        """
        return {
            "definition_identity_hash": self.definition.identity_hash,
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
        """Full 64-char lowercase SHA-256 Config Identity Hash."""
        return identity_document_hash(self.identity_document())

    def _raise_unsupported(self, control: RequestControl) -> None:
        raise ControlValidationError(
            failure_record(
                failure_class=FailureClass.PERMANENT,
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
                failure_class=FailureClass.PERMANENT,
                code="missing_required_control",
                message=(
                    f"definition {self.definition.definition_id!r} requires "
                    f"control {control.value!r} but the config leaves it unset"
                ),
                metadata={"control": control.value},
            )
        )


# --- Presets ---------------------------------------------------------------


def _chat_constraints(
    *,
    token_limit_parameter: TokenLimitParameter,
    reasoning_shape: ReasoningRequestShape,
) -> ControlConstraints:
    return ControlConstraints(
        token_limit_parameter=token_limit_parameter,
        reasoning_shape=reasoning_shape,
    )


def _config_from_route(  # noqa: PLR0913 -- explicit keyword-only builder
    *,
    definition_id: str,
    route: ModelRoute,
    constraints: ControlConstraints,
    controls: GenerationControls | None,
    extensions: ProviderBodyExtensions | None,
    required_controls: frozenset[RequestControl] = frozenset(),
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    # When ``extension_keys`` is given the caller declares the exact set of
    # extensions the Definition exposes and passed extensions are validated
    # strictly against it. When omitted, the declared set is derived from the
    # extensions actually passed; either way the declared set is captured in
    # the Definition identity, so the undeclared-extension check is never
    # vacuous.
    if extension_keys is None:
        declared_keys = (
            frozenset(extensions.extra_body) if extensions else frozenset()
        )
    else:
        declared_keys = frozenset(extension_keys)
    definition = ProviderCallDefinition(
        definition_id=definition_id,
        route=route,
        constraints=constraints,
        required_controls=required_controls,
        extension_keys=declared_keys,
    )
    return definition.materialize(controls=controls, extensions=extensions)


def openrouter_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    route = ModelRoute(
        provider=ProviderKind.OPENROUTER,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )
    return _config_from_route(
        definition_id="openrouter.chat_completions",
        route=route,
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
            reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        ),
        controls=controls,
        extensions=extensions,
        extension_keys=extension_keys,
    )


def openai_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    route = ModelRoute(
        provider=ProviderKind.OPENAI,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )
    return _config_from_route(
        definition_id="openai.chat_completions",
        route=route,
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
            reasoning_shape=ReasoningRequestShape.EFFORT_FIELD,
        ),
        controls=controls,
        extensions=extensions,
        extension_keys=extension_keys,
    )


def openai_responses_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    route = ModelRoute(
        provider=ProviderKind.OPENAI,
        protocol=Protocol.RESPONSES,
        model=model,
    )
    return _config_from_route(
        definition_id="openai.responses",
        route=route,
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_OUTPUT_TOKENS,
            reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        ),
        controls=controls,
        extensions=extensions,
        extension_keys=extension_keys,
    )


def gemini_chat_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    """Gemini via Google's OpenAI-compatible endpoint (AI Studio key)."""
    route = ModelRoute(
        provider=ProviderKind.GEMINI,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )
    return _config_from_route(
        definition_id="gemini.openai_compat",
        route=route,
        # The compat endpoint takes a flat reasoning_effort field, not an
        # OpenAI-style nested reasoning object; thinking budgets need the
        # native API (deferred until compat gaps require it).
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
            reasoning_shape=ReasoningRequestShape.EFFORT_FIELD,
        ),
        controls=controls,
        extensions=extensions,
        extension_keys=extension_keys,
    )


def anthropic_messages_config(
    *,
    model: str,
    controls: GenerationControls | None = None,
    extensions: ProviderBodyExtensions | None = None,
    extension_keys: frozenset[str] | None = None,
) -> ProviderCallConfig:
    """Anthropic Messages protocol over a (custom-capable) base URL.

    Anthropic requires ``max_tokens``, so ``TOKEN_LIMIT`` is a required
    control: a Config without a token limit cannot materialize.
    """
    route = ModelRoute(
        provider=ProviderKind.ANTHROPIC,
        protocol=Protocol.ANTHROPIC_MESSAGES,
        model=model,
    )
    return _config_from_route(
        definition_id="anthropic.messages",
        route=route,
        # Anthropic Messages requires max_tokens and serializes reasoning
        # as a native effort object.
        constraints=_chat_constraints(
            token_limit_parameter=TokenLimitParameter.MAX_TOKENS,
            reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
        ),
        controls=controls,
        extensions=extensions,
        required_controls=frozenset({RequestControl.TOKEN_LIMIT}),
        extension_keys=extension_keys,
    )
