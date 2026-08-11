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
    Protocol,
    ProviderBaseUrl,
    ProviderBodyExtensions,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderCallRequest,
    ProviderKind,
    ProviderTransportPolicy,
    ReasoningEffort,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
    Transcript,
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


def _fixed_request() -> ProviderCallRequest:
    return ProviderCallRequest(config=_fixed_config(), transcript=TRANSCRIPT)


# Regenerate pinned hashes only after an identity-contract decision.
GOLDEN_DEFINITION_HASH = (
    "00e72ae621c1e526567e89bb0b1dec4e11ef5a847e18b8732e2c4b173c2cd7ae"
)
GOLDEN_CONFIG_HASH = (
    "dfac8821edc52cdf9c60339d03f5714744f6051624c2aedfd9c674d2e148be0d"
)
GOLDEN_REQUEST_HASH = (
    "abc1fbbf03d898550e500fc8d29e46353a7ebfd88c90f48aed4d0535440ece2a"
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

        assert PROVIDER_CALL_DEFINITION_SCHEMA_VERSION == 2
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
        (("constraints", "allow_unsupported_control_drop"), True),
        (
            ("required_controls",),
            frozenset(
                {RequestControl.TEMPERATURE, RequestControl.TOKEN_LIMIT}
            ),
        ),
        (("extension_keys",), frozenset({"seed"})),
    ],
    ids=(
        "definition-id",
        "provider",
        "protocol",
        "model",
        "supported-controls",
        "token-limit-parameter",
        "reasoning-shape",
        "unsupported-drop-policy",
        "required-controls",
        "extension-keys",
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

    def test_config_embeds_definition_identity_hash(self) -> None:
        definition = _fixed_definition()
        config = definition.materialize(
            controls=GenerationControls(token_limit=64)
        )
        payload = config.identity_payload()
        assert payload["definition_identity_hash"] == definition.identity_hash

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
        ]
        hashes = {base.identity_hash}
        for controls in variants:
            hashes.add(
                openai_chat_config(model="m", controls=controls).identity_hash
            )
        assert len(hashes) == 5

    def test_body_extension_changes_config_identity(self) -> None:
        base = openai_chat_config(model="m")
        extended = openai_chat_config(
            model="m",
            extensions=ProviderBodyExtensions(extra_body={"seed": 7}),
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
        request = ProviderCallRequest(config=config, transcript=TRANSCRIPT)
        policy = ProviderTransportPolicy(
            provider_kind=ProviderKind.OPENAI,
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=str(ProviderBaseUrl.OPENAI),
            timeout_seconds=5.0,
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
        assert set(ProviderCallRequest.model_fields) == {
            "config",
            "transcript",
        }


class TestRequestIdentity:
    def test_request_identity_is_config_ref_plus_transcript(self) -> None:
        config = openai_chat_config(
            model="m", controls=GenerationControls(token_limit=64)
        )
        request = ProviderCallRequest(config=config, transcript=TRANSCRIPT)
        payload = request.identity_payload()
        assert set(payload) == {"config_identity_hash", "transcript"}
        assert payload["config_identity_hash"] == config.identity_hash
        assert payload["transcript"] == [{"role": "user", "content": "hi"}]

    def test_request_hash_changes_with_transcript(self) -> None:
        config = openai_chat_config(model="m")
        a = ProviderCallRequest(config=config, transcript=TRANSCRIPT)
        b = ProviderCallRequest(
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
        forward = ProviderCallRequest(
            config=config,
            transcript=Transcript(messages=messages),
        )
        reversed_order = ProviderCallRequest(
            config=config,
            transcript=Transcript(messages=tuple(reversed(messages))),
        )

        assert forward.identity_hash != reversed_order.identity_hash

    def test_request_hash_changes_with_config(self) -> None:
        transcript = TRANSCRIPT
        a = ProviderCallRequest(
            config=openai_chat_config(model="m"), transcript=transcript
        )
        b = ProviderCallRequest(
            config=openai_chat_config(model="other"), transcript=transcript
        )
        assert a.identity_hash != b.identity_hash


if __name__ == "__main__":  # pragma: no cover -- golden-hash regeneration
    print("GOLDEN_DEFINITION_HASH =", _fixed_definition().identity_hash)
    print("GOLDEN_CONFIG_HASH =", _fixed_config().identity_hash)
    print("GOLDEN_REQUEST_HASH =", _fixed_request().identity_hash)
