from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dr_providers import (
    ControlConstraints,
    ControlValidationError,
    GenerationControls,
    Protocol,
    ProviderBodyExtensions,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderKind,
    ReasoningEffort,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)
from dr_providers.modeling.route import ModelRoute


class TestConfigPresets:
    def test_openrouter_chat(self) -> None:
        config = openrouter_chat_config(model="m")
        assert config.route.provider is ProviderKind.OPENROUTER
        assert config.route.protocol is Protocol.CHAT_COMPLETIONS
        param = config.definition.constraints.token_limit_parameter
        assert param is TokenLimitParameter.MAX_COMPLETION_TOKENS
        assert config.quota_identity.model_dump() == {
            "provider": "openrouter",
            "protocol": "chat_completions",
            "model": "m",
        }
        assert config.quota_identity.label() == (
            "openrouter:chat_completions:m"
        )

    def test_openai_responses(self) -> None:
        config = openai_responses_config(model="m")
        assert config.route.protocol is Protocol.RESPONSES
        param = config.definition.constraints.token_limit_parameter
        assert param.value == "max_output_tokens"

    def test_gemini_compat_preset(self) -> None:
        config = gemini_chat_config(model="gemini-2.5-flash")
        assert config.route.provider is ProviderKind.GEMINI
        assert config.route.protocol is Protocol.CHAT_COMPLETIONS

    def test_anthropic_messages_preset(self) -> None:
        config = anthropic_messages_config(
            model="claude", controls=GenerationControls(token_limit=64)
        )
        assert config.route.provider is ProviderKind.ANTHROPIC
        assert config.route.protocol is Protocol.ANTHROPIC_MESSAGES
        param = config.definition.constraints.token_limit_parameter
        assert param is TokenLimitParameter.MAX_TOKENS
        assert not config.definition.constraints.supports(RequestControl.SEED)

    def test_openai_compat_presets_advertise_seed(self) -> None:
        presets = (
            openrouter_chat_config(model="m"),
            openai_chat_config(model="m"),
            openai_responses_config(model="m"),
            gemini_chat_config(model="m"),
        )
        for config in presets:
            assert config.definition.constraints.supports(RequestControl.SEED)

    def test_anthropic_preset_rejects_seed(self) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            anthropic_messages_config(
                model="claude",
                controls=GenerationControls(token_limit=64, seed=7),
            )
        assert exc_info.value.failure.code == "unsupported_control"
        assert exc_info.value.failure.metadata["control"] == "seed"

    def test_anthropic_messages_requires_token_limit(self) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            anthropic_messages_config(model="claude")
        assert exc_info.value.failure.code == "missing_required_control"

    def test_anthropic_rejects_unmappable_reasoning_at_construction(
        self,
    ) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            anthropic_messages_config(
                model="claude",
                controls=GenerationControls(
                    token_limit=64,
                    reasoning=ReasoningEffort.XHIGH,
                ),
            )
        assert exc_info.value.failure.code == "unmappable_reasoning_effort"

    @pytest.mark.parametrize("field", ["temperature", "top_p"])
    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "positive-inf", "negative-inf"],
    )
    def test_generation_controls_reject_non_finite_values(
        self, field: str, value: float
    ) -> None:
        with pytest.raises(ValidationError):
            GenerationControls.model_validate({field: value})

    @pytest.mark.parametrize("field", ["temperature", "top_p"])
    @pytest.mark.parametrize(
        "value",
        ["0.5", True],
        ids=["numeric-string", "boolean"],
    )
    def test_generation_controls_reject_coercive_values(
        self, field: str, value: object
    ) -> None:
        with pytest.raises(ValidationError):
            GenerationControls.model_validate({field: value})


class TestDefinitionValidation:
    def _constrained_definition(
        self,
        supported: frozenset[RequestControl],
        *,
        required: frozenset[RequestControl] = frozenset(),
    ) -> ProviderCallDefinition:
        return ProviderCallDefinition(
            definition_id="test.chat",
            route=ModelRoute(
                provider=ProviderKind.OPENAI,
                protocol=Protocol.CHAT_COMPLETIONS,
                model="m",
            ),
            constraints=ControlConstraints(
                supported_controls=supported,
                token_limit_parameter=(
                    TokenLimitParameter.MAX_COMPLETION_TOKENS
                ),
            ),
            required_controls=required,
        )

    def test_unsupported_control_rejected_at_materialize(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT})
        )
        with pytest.raises(ControlValidationError) as exc_info:
            definition.materialize(
                controls=GenerationControls(temperature=0.5)
            )
        assert exc_info.value.failure.metadata["control"] == "temperature"

    def test_unsupported_control_always_refuses_construction(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT})
        )
        with pytest.raises(ControlValidationError) as exc_info:
            definition.materialize(
                controls=GenerationControls(temperature=0.5)
            )
        assert exc_info.value.failure.code == "unsupported_control"

    def test_unsupported_anthropic_reasoning_refuses_construction(
        self,
    ) -> None:
        definition = ProviderCallDefinition(
            definition_id="test.anthropic",
            route=ModelRoute(
                provider=ProviderKind.ANTHROPIC,
                protocol=Protocol.ANTHROPIC_MESSAGES,
                model="m",
            ),
            constraints=ControlConstraints(
                supported_controls=frozenset({RequestControl.TOKEN_LIMIT}),
                token_limit_parameter=TokenLimitParameter.MAX_TOKENS,
            ),
        )

        with pytest.raises(ControlValidationError) as exc_info:
            definition.materialize(
                controls=GenerationControls(reasoning=ReasoningEffort.NONE)
            )
        assert exc_info.value.failure.code == "unsupported_control"

    def test_default_constraints_do_not_advertise_reasoning(self) -> None:
        constraints = ControlConstraints(
            token_limit_parameter=TokenLimitParameter.MAX_OUTPUT_TOKENS
        )

        assert not constraints.supports(RequestControl.REASONING)

    def test_supported_reasoning_requires_wire_mapping(self) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            ControlConstraints(
                supported_controls=frozenset({RequestControl.REASONING}),
                token_limit_parameter=TokenLimitParameter.MAX_OUTPUT_TOKENS,
                reasoning_shape=ReasoningRequestShape.NONE,
            )

        assert exc_info.value.failure.code == "reasoning_mapping_missing"

    def test_required_control_must_be_assigned(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT}),
            required=frozenset({RequestControl.TOKEN_LIMIT}),
        )
        with pytest.raises(ControlValidationError) as exc_info:
            definition.materialize(controls=GenerationControls())
        assert exc_info.value.failure.code == "missing_required_control"

    def test_undeclared_extension_rejected(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT})
        )
        with pytest.raises(ControlValidationError) as exc_info:
            definition.materialize(
                extensions=ProviderBodyExtensions(extra_body={"nope": 1})
            )
        assert exc_info.value.failure.code == "undeclared_extension"

    def test_required_control_not_supported_rejected(self) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            ProviderCallDefinition(
                definition_id="bad",
                route=ModelRoute(
                    provider=ProviderKind.OPENAI,
                    protocol=Protocol.CHAT_COMPLETIONS,
                    model="m",
                ),
                constraints=ControlConstraints(
                    supported_controls=frozenset({RequestControl.TEMPERATURE}),
                    token_limit_parameter=(
                        TokenLimitParameter.MAX_COMPLETION_TOKENS
                    ),
                ),
                required_controls=frozenset({RequestControl.TOKEN_LIMIT}),
            )
        assert exc_info.value.failure.code == "required_control_unsupported"

    def test_reserved_extension_key_rejected(self) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            openai_chat_config(
                model="m",
                extensions=ProviderBodyExtensions(extra_body={"model": "x"}),
                extension_keys=frozenset({"model"}),
            )
        assert exc_info.value.failure.code == "reserved_extension_key"

    def test_seed_extra_body_is_reserved(self) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            openai_chat_config(
                model="m",
                extensions=ProviderBodyExtensions(extra_body={"seed": 7}),
                extension_keys=frozenset({"seed"}),
            )
        assert exc_info.value.failure.code == "reserved_extension_key"

    def test_required_seed_must_be_assigned(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.SEED}),
            required=frozenset({RequestControl.SEED}),
        )
        with pytest.raises(ControlValidationError) as exc_info:
            definition.materialize(controls=GenerationControls())
        assert exc_info.value.failure.code == "missing_required_control"
        assert exc_info.value.failure.metadata["control"] == "seed"

    def test_config_validates_on_direct_construction(self) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT}),
            required=frozenset({RequestControl.TOKEN_LIMIT}),
        )
        with pytest.raises(ControlValidationError):
            ProviderCallConfig(
                definition=definition, controls=GenerationControls()
            )

    def test_extra_body_is_deeply_immutable(self) -> None:
        extensions = ProviderBodyExtensions(
            extra_body={"nested": {"k": [1, 2]}}
        )
        with pytest.raises(TypeError):
            extensions.extra_body["nested"] = 1  # type: ignore[index]  # ty: ignore[invalid-assignment]
        with pytest.raises(AttributeError):
            extensions.extra_body["nested"]["k"].append(3)  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "positive-inf", "negative-inf"],
    )
    def test_extra_body_rejects_nested_non_finite_values(
        self, value: float
    ) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            ProviderBodyExtensions(extra_body={"nested": [value]})

        assert exc_info.value.failure.code == "invalid_extension_json"
        assert exc_info.value.failure.metadata == {
            "path": ["nested", 0],
            "detail": repr(value),
            "reason": "non-finite number",
            "type_name": "float",
        }

    @pytest.mark.parametrize(
        "value",
        [{1, 2}, (1, 2), object()],
        ids=["set", "tuple", "object"],
    )
    def test_extra_body_rejects_non_json_runtime_types(
        self, value: object
    ) -> None:
        with pytest.raises(ControlValidationError) as exc_info:
            ProviderBodyExtensions(extra_body={"value": value})

        assert exc_info.value.failure.code == "invalid_extension_json"
        assert exc_info.value.failure.metadata["path"] == ["value"]
        assert exc_info.value.failure.metadata["reason"] == "unsupported type"

    def test_extra_body_is_isolated_from_source_aliases(self) -> None:
        source: dict[str, Any] = {"nested": {"k": [1, 2]}}
        definition = ProviderCallDefinition(
            definition_id="test.extensions",
            route=ModelRoute(
                provider=ProviderKind.OPENAI,
                protocol=Protocol.CHAT_COMPLETIONS,
                model="m",
            ),
            constraints=ControlConstraints(
                token_limit_parameter=(
                    TokenLimitParameter.MAX_COMPLETION_TOKENS
                )
            ),
            extension_keys=frozenset({"nested"}),
        )
        config = definition.materialize(
            extensions=ProviderBodyExtensions(extra_body=source)
        )
        payload_before = config.extensions.identity_payload()
        hash_before = config.identity_hash

        source["nested"]["k"].append(3)

        assert config.extensions.identity_payload() == payload_before
        assert config.identity_hash == hash_before
