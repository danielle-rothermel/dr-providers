from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dr_providers import (
    ApiKeyEnv,
    ControlConstraints,
    GenerationControls,
    MessageRole,
    ModelRoute,
    PromptMessage,
    PromptRendering,
    Protocol,
    ProviderBaseUrl,
    ProviderBodyExtensions,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderCallKind,
    ProviderGenerateRequest,
    ProviderKind,
    ProviderScoreRequest,
    ProviderTransportPolicy,
    ReasoningEffort,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
    Transcript,
    Verbosity,
    build_payload,
    openai_chat_config,
)
from dr_providers.modeling.call import (
    PROVIDER_CALL_DEFINITION_SCHEMA_VERSION,
)

TRANSCRIPT = Transcript(
    messages=(PromptMessage(role=MessageRole.USER, content="hi"),)
)


def _fixed_definition() -> ProviderCallDefinition:
    return ProviderCallDefinition(
        supported_kinds=frozenset({ProviderCallKind.GENERATE}),
        definition_id="openai.chat_completions",
        route=ModelRoute(
            provider=ProviderKind.OPENAI,
            protocol=Protocol.CHAT_COMPLETIONS,
            model="m",
        ),
        constraints=ControlConstraints(
            token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
        ),
        required_controls=frozenset({RequestControl.TOKEN_LIMIT}),
    )


def _fixed_config() -> ProviderCallConfig:
    return _fixed_definition().materialize(
        controls=GenerationControls(token_limit=64)
    )


def _fixed_request() -> ProviderGenerateRequest:
    return ProviderGenerateRequest(
        config=_fixed_config(), transcript=TRANSCRIPT
    )


# Regenerate pinned hashes only after an identity-contract decision.
GOLDEN_DEFINITION_HASH = (
    "41f14f577a2c36db23777a12f5ee936cba2588f0f697f85fa6eeb5c508ffff27"
)
GOLDEN_CONFIG_HASH = (
    "c9dd4c376d72980dc42e24925672602dc8da9a35bece4fcee08320527f1c02ed"
)
GOLDEN_REQUEST_HASH = (
    "f3fe0677a98004ec18fee3bed7d91df702b820f8382cea3ec16196329fef62d0"
)


class TestPinnedGoldenHashes:
    def test_definition_hash_is_pinned(self) -> None:
        assert _fixed_definition().identity_hash == GOLDEN_DEFINITION_HASH

    def test_config_hash_is_pinned(self) -> None:
        assert _fixed_config().identity_hash == GOLDEN_CONFIG_HASH

    def test_request_hash_is_pinned(self) -> None:
        assert _fixed_request().identity_hash == GOLDEN_REQUEST_HASH


class TestDefinitionSchemaVersionOwnership:
    def test_schema_version_exists_only_on_identity_document(self) -> None:
        definition = _fixed_definition()

        assert PROVIDER_CALL_DEFINITION_SCHEMA_VERSION == 5
        assert "schema_version" not in ProviderCallDefinition.model_fields
        properties = ProviderCallDefinition.model_json_schema()["properties"]
        assert "schema_version" not in properties
        assert "schema_version" not in definition.identity_payload()
        assert (
            definition.identity_document().schema_version
            == PROVIDER_CALL_DEFINITION_SCHEMA_VERSION
        )

    def test_explicit_schema_version_is_rejected(self) -> None:
        data = _fixed_definition().model_dump(mode="python")

        with pytest.raises(ValidationError):
            ProviderCallDefinition.model_validate(
                {
                    **data,
                    "schema_version": PROVIDER_CALL_DEFINITION_SCHEMA_VERSION,
                }
            )


def _definition_variant(
    path: tuple[str, ...], replacement: Any
) -> ProviderCallDefinition:
    data = _fixed_definition().model_dump(mode="python")
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    return ProviderCallDefinition.model_validate(data)


def _changed_payload_paths(
    left: object,
    right: object,
    prefix: tuple[str, ...] = (),
) -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        left_dict = cast("dict[str, object]", left)
        right_dict = cast("dict[str, object]", right)
        changed: set[str] = set()
        for key in left_dict.keys() | right_dict.keys():
            changed.update(
                _changed_payload_paths(
                    left_dict.get(key),
                    right_dict.get(key),
                    (*prefix, key),
                )
            )
        return changed
    if left != right:
        return {".".join(prefix)}
    return set()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("definition_id",), "openai.chat_completions.variant"),
        (("route", "provider"), ProviderKind.OPENROUTER),
        (("route", "protocol"), Protocol.RESPONSES),
        (("route", "model"), "other"),
        (
            ("constraints", "supported_controls"),
            frozenset(
                {RequestControl.TEMPERATURE, RequestControl.TOKEN_LIMIT}
            ),
        ),
        (
            ("constraints", "token_limit_parameter"),
            TokenLimitParameter.MAX_TOKENS,
        ),
        (
            ("constraints", "reasoning_shape"),
            ReasoningRequestShape.EFFORT_FIELD,
        ),
        (
            ("required_controls",),
            frozenset(
                {RequestControl.TEMPERATURE, RequestControl.TOKEN_LIMIT}
            ),
        ),
        (("extension_keys",), frozenset({"user"})),
        (("prompt_rendering",), PromptRendering.FLAT_TEXT),
        (("supported_kinds",), frozenset({"generate", "score"})),
    ],
    ids=(
        "definition-id",
        "provider",
        "protocol",
        "model",
        "supported-controls",
        "token-limit-parameter",
        "reasoning-shape",
        "required-controls",
        "extension-keys",
        "prompt-rendering",
        "supported-kinds",
    ),
)
def test_every_definition_dimension_changes_identity(
    path: tuple[str, ...], replacement: Any
) -> None:
    base = _fixed_definition()
    variant = _definition_variant(path, replacement)

    assert _changed_payload_paths(
        base.identity_payload(), variant.identity_payload()
    ) == {".".join(path)}
    assert variant.identity_hash != base.identity_hash


class TestDefinitionVersusConfig:
    def test_definition_identity_differs_from_config_identity(self) -> None:
        definition = _fixed_definition()
        config = definition.materialize(
            controls=GenerationControls(token_limit=64)
        )
        assert definition.identity_payload() != config.identity_payload()
        assert "controls" not in definition.identity_payload()
        assert "controls" in config.identity_payload()

    def test_config_embeds_definition_hash(self) -> None:
        definition = _fixed_definition()
        config = definition.materialize(
            controls=GenerationControls(token_limit=64)
        )
        payload = config.identity_payload()
        assert payload["definition_hash"] == definition.identity_hash

    def test_config_carries_typed_definition_reference(self) -> None:
        config = _fixed_definition().materialize(
            controls=GenerationControls(token_limit=64)
        )
        assert isinstance(config, ProviderCallConfig)
        assert isinstance(config.definition, ProviderCallDefinition)


class TestOutputAffectingControlsAreIdentity:
    def test_each_control_changes_config_identity(self) -> None:
        base = openai_chat_config(model="m")
        variants = [
            GenerationControls(temperature=0.5),
            GenerationControls(top_p=0.5),
            GenerationControls(token_limit=32),
            GenerationControls(reasoning=ReasoningEffort.LOW),
            GenerationControls(seed=7),
            GenerationControls(verbosity=Verbosity.LOW),
        ]
        hashes = {base.identity_hash}
        for controls in variants:
            hashes.add(
                openai_chat_config(model="m", controls=controls).identity_hash
            )
        assert len(hashes) == 7

    def test_body_extension_changes_config_identity(self) -> None:
        base = openai_chat_config(model="m")
        extended = openai_chat_config(
            model="m",
            extensions=ProviderBodyExtensions(extra_body={"user": "eval"}),
        )
        assert base.identity_hash != extended.identity_hash

    def test_model_route_changes_config_identity(self) -> None:
        a = openai_chat_config(model="m")
        b = openai_chat_config(model="other")
        assert a.identity_hash != b.identity_hash


class TestPolicyExclusion:
    def test_no_policy_keys_in_config_or_request_identity(self) -> None:
        config = openai_chat_config(
            model="m", controls=GenerationControls(token_limit=64)
        )
        request = ProviderGenerateRequest(config=config, transcript=TRANSCRIPT)
        policy = ProviderTransportPolicy(
            provider_kind=ProviderKind.OPENAI,
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=str(ProviderBaseUrl.OPENAI),
            timeout_seconds=5.0,
            connect_timeout_seconds=3.0,
            idle_timeout_seconds=3.0,
            max_connections=10,
            max_keepalive_connections=5,
            max_request_bytes=1024 * 1024,
            max_response_bytes=8 * 1024 * 1024,
        )
        policy_keys = set(policy.identity_payload())
        config_text = str(config.identity_payload())
        request_text = str(request.identity_payload())
        for key in policy_keys:
            assert key not in config_text
            assert key not in request_text
        assert policy.identity_payload()

    def test_request_fields_are_exactly_identity_bearing(self) -> None:
        assert set(ProviderGenerateRequest.model_fields) == {
            "kind",
            "config",
            "transcript",
        }


class TestRequestIdentity:
    def test_request_identity_is_config_ref_plus_transcript(self) -> None:
        config = openai_chat_config(
            model="m", controls=GenerationControls(token_limit=64)
        )
        request = ProviderGenerateRequest(config=config, transcript=TRANSCRIPT)
        payload = request.identity_payload()
        assert payload == {
            "kind": "generate",
            "config_hash": config.identity_hash,
            "transcript": [{"role": "user", "content": "hi"}],
        }
        assert payload["config_hash"] == config.identity_hash
        assert payload["transcript"] == [{"role": "user", "content": "hi"}]

    def test_request_hash_changes_with_transcript(self) -> None:
        config = openai_chat_config(model="m")
        a = ProviderGenerateRequest(config=config, transcript=TRANSCRIPT)
        b = ProviderGenerateRequest(
            config=config,
            transcript=Transcript(
                messages=(
                    PromptMessage(role=MessageRole.USER, content="other"),
                )
            ),
        )
        assert a.identity_hash != b.identity_hash

    def test_request_hash_changes_with_transcript_order(self) -> None:
        config = openai_chat_config(model="m")
        messages = (
            PromptMessage(role=MessageRole.USER, content="first"),
            PromptMessage(role=MessageRole.ASSISTANT, content="second"),
        )
        forward = ProviderGenerateRequest(
            config=config,
            transcript=Transcript(messages=messages),
        )
        reversed_order = ProviderGenerateRequest(
            config=config,
            transcript=Transcript(messages=tuple(reversed(messages))),
        )

        assert forward.identity_hash != reversed_order.identity_hash

    def test_request_hash_changes_with_config(self) -> None:
        transcript = TRANSCRIPT
        a = ProviderGenerateRequest(
            config=openai_chat_config(model="m"), transcript=transcript
        )
        b = ProviderGenerateRequest(
            config=openai_chat_config(model="other"), transcript=transcript
        )
        assert a.identity_hash != b.identity_hash


class TestPromptRenderingIdentity:
    def test_rendering_changes_identity_only_through_definition_reference(
        self,
    ) -> None:
        role = ProviderGenerateRequest(
            config=openai_chat_config(model="m"), transcript=TRANSCRIPT
        )
        flat = ProviderGenerateRequest(
            config=openai_chat_config(
                model="m", prompt_rendering=PromptRendering.FLAT_TEXT
            ),
            transcript=TRANSCRIPT,
        )

        assert (
            role.config.definition.identity_hash
            != flat.config.definition.identity_hash
        )
        assert _changed_payload_paths(
            role.config.identity_payload(), flat.config.identity_payload()
        ) == {"definition_hash"}
        assert _changed_payload_paths(
            role.identity_payload(), flat.identity_payload()
        ) == {"config_hash"}
        assert (
            role.identity_payload()["transcript"]
            == flat.identity_payload()["transcript"]
        )
        assert role.identity_hash != flat.identity_hash

    def test_default_and_explicit_role_messages_have_same_identity(
        self,
    ) -> None:
        default = openai_chat_config(model="m")
        explicit = openai_chat_config(
            model="m", prompt_rendering=PromptRendering.ROLE_MESSAGES
        )

        assert (
            default.definition.identity_payload()["prompt_rendering"]
            == "role_messages"
        )
        assert default.identity_hash == explicit.identity_hash

    def test_flat_text_keeps_role_distinctions_in_transcript_identity(
        self,
    ) -> None:
        config = openai_chat_config(
            model="m", prompt_rendering=PromptRendering.FLAT_TEXT
        )
        user = ProviderGenerateRequest(config=config, transcript=TRANSCRIPT)
        assistant = ProviderGenerateRequest(
            config=config,
            transcript=Transcript(
                messages=(
                    PromptMessage(role=MessageRole.ASSISTANT, content="hi"),
                )
            ),
        )
        before = assistant.identity_payload()

        assert build_payload(user) == build_payload(assistant)
        assert assistant.identity_payload() == before
        assert assistant.identity_payload()["transcript"] == [
            {"role": "assistant", "content": "hi"}
        ]
        assert user.identity_hash != assistant.identity_hash


if __name__ == "__main__":  # pragma: no cover -- golden-hash regeneration
    print("GOLDEN_DEFINITION_HASH =", _fixed_definition().identity_hash)
    print("GOLDEN_CONFIG_HASH =", _fixed_config().identity_hash)
    print("GOLDEN_REQUEST_HASH =", _fixed_request().identity_hash)


def test_score_request_identity_payload_and_hash_are_pinned(
    score_request: ProviderScoreRequest,
) -> None:
    assert score_request.identity_payload() == {
        "kind": "score",
        "config_hash": (
            "96094a19a792d17add4051bc45e575fd025e4e4c4036d9be22aaa9f736c603ad"
        ),
        "context": "context",
        "continuations": ["answer"],
        "token_logprobs": False,
    }
    assert score_request.identity_hash == (
        "6c7a5f6d7f9e89c6b1fce6ed42afc5b7e7985be84db6eecc1f375ef2cd5dc571"
    )
    assert set(ProviderScoreRequest.model_fields) == {
        "kind",
        "config",
        "context",
        "continuations",
        "token_logprobs",
    }


@pytest.mark.parametrize(
    "override",
    [
        {"context": "other context"},
        {"continuations": ("other answer",)},
        {"token_logprobs": True},
    ],
)
def test_each_score_request_input_changes_identity(
    score_request: ProviderScoreRequest, override: dict[str, Any]
) -> None:
    variant = ProviderScoreRequest.model_validate(
        {**score_request.model_dump(mode="python"), **override}
    )
    assert variant.identity_hash != score_request.identity_hash


def test_score_continuation_order_is_identity_bearing(
    score_request: ProviderScoreRequest,
) -> None:
    forward = ProviderScoreRequest(
        config=score_request.config, context="", continuations=("a", "b")
    )
    reverse = ProviderScoreRequest(
        config=score_request.config, context="", continuations=("b", "a")
    )
    assert forward.identity_hash != reverse.identity_hash
