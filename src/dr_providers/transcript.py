"""Transcript: the ordered role-and-content provider-call input.

A Transcript is the single input carried by a Provider Call Request
alongside its Config reference — no controls, no transport policy.
"""

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
        # The wire shape sent in the request body.
        return {"role": self.role.value, "content": self.content}

    def identity_payload(self) -> dict[str, str]:
        # The identity shape is byte-identical to the wire shape today, but the
        # two names carry different intent (wire body vs Request identity) and
        # may diverge; identity delegates to the wire shape to stay in lockstep
        # until they must differ.
        return self.provider_dict()


class Transcript(BaseModel):
    """Ordered sequence of role-and-content messages sent as input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[PromptMessage, ...]

    def identity_payload(self) -> list[dict[str, str]]:
        return [m.identity_payload() for m in self.messages]
