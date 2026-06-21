import os
import shutil
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ProviderError(Exception):
    pass


class ProviderTransportError(ProviderError):
    pass


class ProviderSemanticError(ProviderError):
    pass


class ReasoningWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    provider: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ProviderAvailabilityStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    available: bool
    missing_env_vars: tuple[str, ...] = Field(default_factory=tuple)
    missing_executables: tuple[str, ...] = Field(default_factory=tuple)
    supports_structured_output: bool = False


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    supports_structured_output: bool = True
    required_env_vars: list[str] = Field(default_factory=list)
    required_executables: list[str] = Field(default_factory=list)
    timeout_seconds: float = 120.0
    base_url: str
    api_key_env: str
    api_key: str | None = None

    @model_validator(mode="after")
    def _compute_api_env_requirements(self) -> Self:
        if not self.api_key and self.api_key_env not in self.required_env_vars:
            object.__setattr__(
                self,
                "required_env_vars",
                [*self.required_env_vars, self.api_key_env],
            )
        return self

    def missing_env_vars(self) -> tuple[str, ...]:
        return tuple(
            env_var
            for env_var in self.required_env_vars
            if not os.getenv(env_var)
        )

    def missing_executables(self) -> tuple[str, ...]:
        return tuple(
            executable
            for executable in self.required_executables
            if shutil.which(executable) is None
        )

    def availability_status(self) -> ProviderAvailabilityStatus:
        missing_env = self.missing_env_vars()
        missing_exec = self.missing_executables()
        return ProviderAvailabilityStatus(
            provider=self.name,
            available=not missing_env and not missing_exec,
            missing_env_vars=missing_env,
            missing_executables=missing_exec,
            supports_structured_output=self.supports_structured_output,
        )


class OpenAICompatConfig(ProviderConfig):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    chat_path: str = "/chat/completions"
    max_completion_token_model_families: tuple[Any, ...] = ()

    @field_validator("chat_path")
    @classmethod
    def _ensure_leading_slash(cls, v: str) -> str:
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
