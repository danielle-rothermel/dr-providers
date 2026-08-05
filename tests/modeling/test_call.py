"""Provider-call modeling and validation tests."""

from __future__ import annotations

from typing import Any

import pytest

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

    def test_anthropic_messages_requires_token_limit(self) -> None:
        # Anthropic requires max_tokens, so TOKEN_LIMIT is a required
        # control: materializing without one is rejected.
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


class TestDefinitionValidation:
    def _constrained_definition(
        self,
        supported: frozenset[RequestControl],
        *,
        allow_drop: bool = False,
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
                allow_unsupported_control_drop=allow_drop,
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

    def test_unsupported_control_drop_opt_in_accepts_construction(
        self,
    ) -> None:
        definition = self._constrained_definition(
            frozenset({RequestControl.TOKEN_LIMIT}), allow_drop=True
        )
        config = definition.materialize(
            controls=GenerationControls(temperature=0.5)
        )
        assert config.controls.temperature == 0.5

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
        # A Definition that requires a control its constraints do not
        # support could never materialize; reject it at construction.
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
        # An extension declared with a reserved core wire key is rejected so
        # it cannot silently overwrite a validated field at build time.
        with pytest.raises(ControlValidationError) as exc_info:
            openai_chat_config(
                model="m",
                extensions=ProviderBodyExtensions(extra_body={"model": "x"}),
                extension_keys=frozenset({"model"}),
            )
        assert exc_info.value.failure.code == "reserved_extension_key"

    def test_config_validates_on_direct_construction(self) -> None:
        # The control/extension invariants live in model validation, so a
        # directly-constructed Config (bypassing materialize) is validated.
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
