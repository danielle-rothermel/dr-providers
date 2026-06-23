import os

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)

from dr_providers.query.errors import ProviderSemanticError


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    timeout_seconds: float = 120.0
    base_url: str
    api_key_env: str
    api_key: str | None = None
    chat_path: str | None = "/chat/completions"

    @field_validator("chat_path")
    @classmethod
    def _ensure_leading_slash(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith("/"):
            return f"/{v}"
        return v


def resolve_api_key(
    config: ProviderConfig, *, label: str | None = None
) -> str:
    key = config.api_key or os.getenv(config.api_key_env)
    if not key:
        provider_label = label or config.name
        raise ProviderSemanticError(
            f"Missing API key for {provider_label}. "
            f"Set {config.api_key_env} or pass config.api_key"
        )
    return key
