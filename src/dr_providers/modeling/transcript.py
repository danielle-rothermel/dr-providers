from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, StrictStr


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PromptMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: StrictStr

    def provider_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    def identity_payload(self) -> dict[str, str]:
        # Identity retains roles and content independently of prompt rendering.
        return self.provider_dict()


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[PromptMessage, ...]

    def identity_payload(self) -> list[dict[str, str]]:
        return [m.identity_payload() for m in self.messages]
