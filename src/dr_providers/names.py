from __future__ import annotations

from enum import StrEnum
from typing import Self


class ProviderName(StrEnum):
    _api_base_url: str

    def __new__(cls, value: str, api_base_url: str) -> Self:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._api_base_url = api_base_url
        return obj

    OPENROUTER = "openrouter", "https://openrouter.ai/api/v1"

    def api_key_env(self) -> str:
        return f"{self.name}_API_KEY"

    @property
    def api_base_url(self) -> str:
        return self._api_base_url


class EffortLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
