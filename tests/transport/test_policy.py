from __future__ import annotations

import inspect
from typing import cast

import pytest
from _policy import (
    TEST_IDLE_TIMEOUT_SECONDS,
    TEST_MAX_CONNECTIONS,
    TEST_MAX_KEEPALIVE_CONNECTIONS,
    TEST_MAX_REQUEST_BYTES,
    TEST_MAX_RESPONSE_BYTES,
    TEST_TIMEOUT_SECONDS,
    make_transport_policy,
)
from pydantic import ValidationError

from dr_providers import (
    ApiKeyEnv,
    ProviderBaseUrl,
    ProviderKind,
    ProviderTransportPolicy,
    policy_for,
)


def _sized_policy_for(  # noqa: PLR0913 -- mirrors explicit policy_for sizing
    kind: ProviderKind,
    /,
    *,
    api_key_env: ApiKeyEnv | str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = TEST_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = TEST_IDLE_TIMEOUT_SECONDS,
    max_connections: int = TEST_MAX_CONNECTIONS,
    max_keepalive_connections: int = TEST_MAX_KEEPALIVE_CONNECTIONS,
    max_request_bytes: int = TEST_MAX_REQUEST_BYTES,
    max_response_bytes: int = TEST_MAX_RESPONSE_BYTES,
) -> ProviderTransportPolicy:
    kwargs: dict[str, ApiKeyEnv | str | None] = {}
    if api_key_env is not None:
        kwargs["api_key_env"] = api_key_env
    if base_url is not None:
        kwargs["base_url"] = base_url
    return policy_for(
        kind,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
        **kwargs,
    )


class TestPolicyFor:
    def test_provider_kind_is_required(self) -> None:
        with pytest.raises(ValidationError):
            ProviderTransportPolicy.model_validate(
                {"api_key_env": "OPENAI_API_KEY"}
            )

    def test_policy_for_requires_explicit_sizing(self) -> None:
        required = (
            "timeout_seconds",
            "idle_timeout_seconds",
            "max_connections",
            "max_keepalive_connections",
            "max_request_bytes",
            "max_response_bytes",
        )
        for name in required:
            parameter = inspect.signature(policy_for).parameters[name]
            assert parameter.default is inspect.Parameter.empty

    def test_resolves_provider_binding_from_kind(self) -> None:
        policy = _sized_policy_for(ProviderKind.ANTHROPIC)
        assert policy.provider_kind is ProviderKind.ANTHROPIC
        assert policy.api_key_env == ApiKeyEnv.ANTHROPIC.value
        assert policy.base_url == ProviderBaseUrl.ANTHROPIC.value
        assert policy.max_connections == TEST_MAX_CONNECTIONS
        assert (
            policy.max_keepalive_connections == TEST_MAX_KEEPALIVE_CONNECTIONS
        )
        assert policy.max_request_bytes == TEST_MAX_REQUEST_BYTES
        assert policy.max_response_bytes == TEST_MAX_RESPONSE_BYTES

    def test_overrides_apply(self) -> None:
        policy = _sized_policy_for(
            ProviderKind.OPENAI,
            base_url="https://proxy.example/v1",
            api_key_env="CUSTOM_KEY_ENV",
            max_connections=7,
            max_keepalive_connections=3,
            max_request_bytes=4096,
            max_response_bytes=8192,
        )
        assert policy.base_url == "https://proxy.example/v1"
        assert policy.api_key_env == "CUSTOM_KEY_ENV"
        assert policy.max_connections == 7
        assert policy.max_keepalive_connections == 3
        assert policy.max_request_bytes == 4096
        assert policy.max_response_bytes == 8192

    def test_idle_timeout_clamped_to_timeout(self) -> None:
        policy = _sized_policy_for(
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
            _sized_policy_for(
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
            _sized_policy_for(
                ProviderKind.OPENAI,
                idle_timeout_seconds=idle_timeout_seconds,
            )

    @pytest.mark.parametrize(
        "invalid_value", [True, "1.0"], ids=("bool", "numeric-string")
    )
    def test_timeout_type_rejected_direct(self, invalid_value: object) -> None:
        with pytest.raises(ValidationError):
            make_transport_policy(
                timeout_seconds=cast("float", invalid_value),
            )

    @pytest.mark.parametrize(
        "invalid_value", [True, "1.0"], ids=("bool", "numeric-string")
    )
    def test_idle_timeout_type_rejected_direct(
        self, invalid_value: object
    ) -> None:
        with pytest.raises(ValidationError):
            make_transport_policy(
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
            '{"provider_kind":"openai","api_key_env":"OPENAI_API_KEY",'
            f'"{field_name}":{json_value}}}'
        )
        with pytest.raises(ValidationError):
            ProviderTransportPolicy.model_validate_json(payload)

    @pytest.mark.parametrize(
        "field_name",
        [
            "max_connections",
            "max_keepalive_connections",
            "max_request_bytes",
            "max_response_bytes",
        ],
    )
    @pytest.mark.parametrize(
        "invalid_value",
        [0, -1, True, 1.5, "1"],
        ids=("zero", "negative", "bool", "float", "string"),
    )
    def test_invalid_integer_bound_rejected(
        self,
        field_name: str,
        invalid_value: object,
    ) -> None:
        with pytest.raises(ValidationError):
            ProviderTransportPolicy.model_validate(
                {
                    "provider_kind": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    field_name: invalid_value,
                }
            )

    def test_keepalive_limit_cannot_exceed_total_connections(self) -> None:
        with pytest.raises(
            ValidationError,
            match=(
                "max_keepalive_connections must not exceed max_connections"
            ),
        ):
            make_transport_policy(
                max_connections=2,
                max_keepalive_connections=3,
            )

    @pytest.mark.parametrize(
        "timeout_seconds",
        [1, 1.5, 1e300],
        ids=("int", "float", "large-float"),
    )
    def test_positive_finite_timeouts_accepted(
        self, timeout_seconds: int | float
    ) -> None:
        direct = make_transport_policy(
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=timeout_seconds,
            max_connections=1,
            max_keepalive_connections=1,
            max_request_bytes=1024,
            max_response_bytes=1024,
        )
        from_json = ProviderTransportPolicy.model_validate_json(
            "{"
            '"provider_kind":"openai",'
            '"api_key_env":"OPENAI_API_KEY",'
            f'"timeout_seconds":{timeout_seconds},'
            f'"idle_timeout_seconds":{timeout_seconds},'
            '"max_connections":1,'
            '"max_keepalive_connections":1,'
            '"max_request_bytes":1024,'
            '"max_response_bytes":1024'
            "}"
        )

        for policy in (direct, from_json):
            assert policy.timeout_seconds == timeout_seconds
            assert policy.idle_timeout_seconds == timeout_seconds

    def test_identity_carries_provider_and_credential_environment_name(
        self,
    ) -> None:
        policy = make_transport_policy(
            provider_kind=ProviderKind.OPENAI,
            api_key_env=str(ApiKeyEnv.OPENAI),
            base_url=str(ProviderBaseUrl.OPENAI),
        )

        payload = policy.identity_payload()

        assert payload == {
            "provider_kind": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "timeout_seconds": TEST_TIMEOUT_SECONDS,
            "idle_timeout_seconds": TEST_IDLE_TIMEOUT_SECONDS,
            "max_connections": TEST_MAX_CONNECTIONS,
            "max_keepalive_connections": TEST_MAX_KEEPALIVE_CONNECTIONS,
            "max_request_bytes": TEST_MAX_REQUEST_BYTES,
            "max_response_bytes": TEST_MAX_RESPONSE_BYTES,
        }
        assert "api_key" not in payload

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://user@example.test/v1",
            "https://user:secret@example.test/v1",
        ],
        ids=("username", "username-password"),
    )
    def test_base_url_userinfo_rejected(self, base_url: str) -> None:
        with pytest.raises(
            ValidationError, match="must not contain URL userinfo"
        ):
            make_transport_policy(base_url=base_url)

        with pytest.raises(
            ValidationError, match="must not contain URL userinfo"
        ):
            ProviderTransportPolicy.model_validate_json(
                '{"provider_kind":"openai",'
                f'"api_key_env":"OPENAI_API_KEY","base_url":"{base_url}"}}'
            )

    def test_transport_policy_requires_explicit_sizing(self) -> None:
        with pytest.raises(ValidationError):
            ProviderTransportPolicy.model_validate(
                {
                    "provider_kind": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                }
            )
