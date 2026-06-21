from __future__ import annotations

from enum import StrEnum


class ProviderName(StrEnum):
    OPENROUTER = "openrouter"

    def api_key_env(self) -> str:
        return f"{self.name}_API_KEY"


class OpenRouterEffortLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OpenRouterControlRequestStyle(StrEnum):
    NONE = "none"
    ENABLED_FLAG = "enabled_flag"
    EFFORT = "effort"


class OpenRouterReasoningKey(StrEnum):
    REASONING = "reasoning"
    ENABLED = "enabled"
    EFFORT = "effort"


class OpenRouterUrls(StrEnum):
    API_BASE = "https://openrouter.ai/api/v1"
    MODELS_DOCS = "https://openrouter.ai/models"
