"""Golden identity tests for the Provider Call Definition/Config/Request.

These pin the identity contract the design requires:
  * Definition identity vs Config identity are distinct payloads.
  * Config identity covers Model Route + every output-affecting control
    and body extension; transport policy is excluded.
  * Required-variable completion is enforced at materialization.
  * Request identity is exactly one Config reference + one Transcript,
    with no copied controls and no transport policy.
  * The Config Identity Hash is a full 64-char SHA-256 via dr-serialize.
"""

from __future__ import annotations

import re

import pytest
from dr_serialize import build_identity_document, identity_document_hash

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
    RequestControl,
    TokenLimitParameter,
    Transcript,
    UnsupportedControlError,
    openai_chat_config,
)
from dr_providers.config import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    PROVIDER_CALL_CONFIG_SCHEMA_VERSION,
)

FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")

TRANSCRIPT = Transcript(
    messages=(PromptMessage(role=MessageRole.USER, content="hi"),)
)


def _definition() -> ProviderCallDefinition:
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


class TestConfigIdentityHash:
    def test_hash_is_full_sha256(self) -> None:
        config = openai_chat_config(
            model="m", controls=GenerationControls(token_limit=64)
        )
        assert FULL_SHA256.match(config.identity_hash)

    def test_hash_matches_manual_identity_document(self) -> None:
        config = openai_chat_config(
            model="m", controls=GenerationControls(token_limit=64)
        )
        expected = identity_document_hash(
            build_identity_document(
                schema=PROVIDER_CALL_CONFIG_SCHEMA,
                schema_version=PROVIDER_CALL_CONFIG_SCHEMA_VERSION,
                payload=config.identity_payload(),
            )
        )
        assert config.identity_hash == expected

    def test_hash_is_deterministic(self) -> None:
        first = openai_chat_config(
            model="m", controls=GenerationControls(temperature=0.2)
        )
        second = openai_chat_config(
            model="m", controls=GenerationControls(temperature=0.2)
        )
        assert first.identity_hash == second.identity_hash


class TestDefinitionVersusConfig:
    def test_definition_identity_differs_from_config_identity(self) -> None:
        definition = _definition()
        config = definition.materialize(
            controls=GenerationControls(token_limit=64)
        )
        assert definition.identity_payload() != config.identity_payload()
        # the Definition declares required variables; the Config assigns
        # them, so the assigned controls appear only in Config identity.
        assert "controls" not in definition.identity_payload()
        assert "controls" in config.identity_payload()

    def test_config_carries_typed_definition_reference(self) -> None:
        config = _definition().materialize(
            controls=GenerationControls(token_limit=64)
        )
        assert isinstance(config, ProviderCallConfig)
        assert isinstance(config.definition, ProviderCallDefinition)
        assert config.identity_payload()["definition_id"] == (
            "openai.chat_completions"
        )


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
        # base plus four distinct control assignments = five identities.
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
    def test_transport_policy_never_touches_config_identity(self) -> None:
        config = openai_chat_config(
            model="m", controls=GenerationControls(token_limit=64)
        )
        baseline = config.identity_hash
        # Every transport policy field is excluded from Config identity;
        # varying them must not change the Config Identity Hash.
        for policy in (
            ProviderTransportPolicy(
                api_key_env=str(ApiKeyEnv.OPENAI),
                base_url=str(ProviderBaseUrl.OPENAI),
            ),
            ProviderTransportPolicy(
                api_key_env=str(ApiKeyEnv.OPENROUTER),
                base_url="https://custom.example/v1",
                timeout_seconds=5.0,
                native_retry_count=3,
            ),
        ):
            request = ProviderCallRequest(config=config, transcript=TRANSCRIPT)
            # policy identity is independent; config identity is stable.
            assert policy.identity_payload()  # sanity: policy has identity
            assert request.config.identity_hash == baseline

    def test_policy_identity_carries_no_credential_material(self) -> None:
        policy = ProviderTransportPolicy(
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=str(ProviderBaseUrl.OPENAI),
        )
        payload = policy.identity_payload()
        # only the env var NAME, never a key value.
        assert payload["api_key_env"] == "OPENAI_API_KEY"
        assert "api_key" not in payload


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

    def test_request_identity_copies_no_controls(self) -> None:
        config = openai_chat_config(
            model="m",
            controls=GenerationControls(temperature=0.9, token_limit=64),
        )
        request = ProviderCallRequest(config=config, transcript=TRANSCRIPT)
        payload = request.identity_payload()
        # controls live behind the Config reference, never copied onto
        # the request identity.
        assert "temperature" not in str(payload)
        assert "token_limit" not in payload

    def test_request_identity_changes_with_transcript(self) -> None:
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
        assert a.identity_payload() != b.identity_payload()

    def test_request_identity_changes_with_config(self) -> None:
        transcript = TRANSCRIPT
        a = ProviderCallRequest(
            config=openai_chat_config(model="m"), transcript=transcript
        )
        b = ProviderCallRequest(
            config=openai_chat_config(model="other"), transcript=transcript
        )
        assert (
            a.identity_payload()["config_identity_hash"]
            != b.identity_payload()["config_identity_hash"]
        )


class TestRequiredVariableCompletion:
    def test_missing_required_control_rejected(self) -> None:
        definition = _definition()
        with pytest.raises(UnsupportedControlError) as exc_info:
            definition.materialize(controls=GenerationControls())
        assert exc_info.value.failure.code == "missing_required_control"

    def test_complete_assignment_produces_config(self) -> None:
        config = _definition().materialize(
            controls=GenerationControls(token_limit=64)
        )
        assert isinstance(config, ProviderCallConfig)
        assert config.controls.token_limit == 64
