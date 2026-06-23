import pytest

from dr_providers.query.errors import ProviderSemanticError
from dr_providers.query.transport_config import ProviderConfig, resolve_api_key


def test_chat_path_gets_leading_slash() -> None:
    config = ProviderConfig(
        name="test",
        base_url="https://example.com",
        api_key_env="TEST_API_KEY",
        chat_path="chat/completions",
    )
    assert config.chat_path == "/chat/completions"


def test_resolve_api_key_from_config_field() -> None:
    config = ProviderConfig(
        name="test",
        base_url="https://example.com",
        api_key_env="TEST_API_KEY",
        api_key="inline-key",
    )
    assert resolve_api_key(config) == "inline-key"


def test_resolve_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "env-key")
    config = ProviderConfig(
        name="test",
        base_url="https://example.com",
        api_key_env="TEST_API_KEY",
    )
    assert resolve_api_key(config) == "env-key"


def test_resolve_api_key_missing_raises() -> None:
    config = ProviderConfig(
        name="test",
        base_url="https://example.com",
        api_key_env="MISSING_TEST_API_KEY",
    )
    with pytest.raises(ProviderSemanticError, match="Missing API key"):
        resolve_api_key(config, label="test-provider")
