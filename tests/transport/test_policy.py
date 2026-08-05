"""Provider transport-policy tests."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from dr_providers import (
    ApiKeyEnv,
    ProviderBaseUrl,
    ProviderKind,
    ProviderTransportPolicy,
    policy_for,
)


class TestPolicyFor:
    def test_derives_defaults_from_provider_kind(self) -> None:
        policy = policy_for(ProviderKind.ANTHROPIC)
        assert policy.api_key_env == ApiKeyEnv.ANTHROPIC.value
        assert policy.base_url == ProviderBaseUrl.ANTHROPIC.value

    def test_overrides_apply(self) -> None:
        policy = policy_for(
            ProviderKind.OPENAI,
            base_url="https://proxy.example/v1",
            api_key_env="CUSTOM_KEY_ENV",
            native_retry_count=2,
        )
        assert policy.base_url == "https://proxy.example/v1"
        assert policy.api_key_env == "CUSTOM_KEY_ENV"
        assert policy.native_retry_count == 2

    def test_idle_timeout_clamped_to_timeout(self) -> None:
        policy = policy_for(
            ProviderKind.OPENAI,
            timeout_seconds=10.0,
            idle_timeout_seconds=45.0,
        )
        assert policy.idle_timeout_seconds == 10.0

    @pytest.mark.parametrize(
        "timeout_seconds",
        [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
        ids=("zero", "negative", "nan", "positive-inf", "negative-inf"),
    )
    def test_invalid_timeout_rejected(self, timeout_seconds: float) -> None:
        with pytest.raises(ValidationError):
            policy_for(
                ProviderKind.OPENAI,
                timeout_seconds=timeout_seconds,
            )

    @pytest.mark.parametrize(
        "idle_timeout_seconds",
        [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
        ids=("zero", "negative", "nan", "positive-inf", "negative-inf"),
    )
    def test_invalid_idle_timeout_rejected(
        self, idle_timeout_seconds: float
    ) -> None:
        with pytest.raises(ValidationError):
            policy_for(
                ProviderKind.OPENAI,
                idle_timeout_seconds=idle_timeout_seconds,
            )

    @pytest.mark.parametrize(
        "invalid_value", [True, "1.0"], ids=("bool", "numeric-string")
    )
    def test_timeout_type_rejected_direct(self, invalid_value: object) -> None:
        with pytest.raises(ValidationError):
            ProviderTransportPolicy(
                api_key_env="OPENAI_API_KEY",
                timeout_seconds=cast("float", invalid_value),
            )

    @pytest.mark.parametrize(
        "invalid_value", [True, "1.0"], ids=("bool", "numeric-string")
    )
    def test_idle_timeout_type_rejected_direct(
        self, invalid_value: object
    ) -> None:
        with pytest.raises(ValidationError):
            ProviderTransportPolicy(
                api_key_env="OPENAI_API_KEY",
                idle_timeout_seconds=cast("float", invalid_value),
            )

    @pytest.mark.parametrize(
        "field_name", ["timeout_seconds", "idle_timeout_seconds"]
    )
    @pytest.mark.parametrize(
        "json_value", ["true", '"1.0"'], ids=("bool", "numeric-string")
    )
    def test_timeout_types_rejected_from_json(
        self, field_name: str, json_value: str
    ) -> None:
        payload = (
            f'{{"api_key_env":"OPENAI_API_KEY","{field_name}":{json_value}}}'
        )
        with pytest.raises(ValidationError):
            ProviderTransportPolicy.model_validate_json(payload)

    @pytest.mark.parametrize(
        "native_retry_count", [-1, True], ids=("negative", "bool")
    )
    def test_invalid_native_retry_count_rejected(
        self, native_retry_count: int
    ) -> None:
        with pytest.raises(ValidationError):
            policy_for(
                ProviderKind.OPENAI,
                native_retry_count=native_retry_count,
            )

    @pytest.mark.parametrize(
        "timeout_seconds",
        [1, 1.5, 1e300],
        ids=("int", "float", "large-float"),
    )
    def test_positive_finite_timeouts_accepted(
        self, timeout_seconds: int | float
    ) -> None:
        direct = ProviderTransportPolicy(
            api_key_env="OPENAI_API_KEY",
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=timeout_seconds,
        )
        from_json = ProviderTransportPolicy.model_validate_json(
            "{"
            '"api_key_env":"OPENAI_API_KEY",'
            f'"timeout_seconds":{timeout_seconds},'
            f'"idle_timeout_seconds":{timeout_seconds}'
            "}"
        )

        for policy in (direct, from_json):
            assert policy.timeout_seconds == timeout_seconds
            assert policy.idle_timeout_seconds == timeout_seconds

    def test_identity_carries_only_credential_environment_name(self) -> None:
        policy = ProviderTransportPolicy(
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=str(ProviderBaseUrl.OPENAI),
        )

        payload = policy.identity_payload()

        assert payload["api_key_env"] == "OPENAI_API_KEY"
        assert "api_key" not in payload
