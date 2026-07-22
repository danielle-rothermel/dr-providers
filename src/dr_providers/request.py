"""Provider Call Request and the pure ``build_payload``.

A Provider Call Request is immutable and identity-bearing: it references
exactly one Provider Call Config and carries exactly one Transcript. Its
identity is the Config Identity Hash plus the Transcript — no copied
controls and no transport policy. ``build_payload`` reads the referenced
Config's controls to construct the least-processed wire body for the
route's protocol (chat_completions, responses, or anthropic_messages).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from dr_providers.config import (  # noqa: TC001 -- pydantic field
    ProviderCallConfig,
)
from dr_providers.controls import (
    GenerationControls,
    ReasoningRequestShape,
    RequestControl,
)
from dr_providers.route import Protocol
from dr_providers.transcript import MessageRole, Transcript

CHAT_COMPLETIONS_PATH = "/chat/completions"
RESPONSES_PATH = "/responses"
ANTHROPIC_MESSAGES_PATH = "/messages"

PROTOCOL_PATHS: dict[Protocol, str] = {
    Protocol.CHAT_COMPLETIONS: CHAT_COMPLETIONS_PATH,
    Protocol.RESPONSES: RESPONSES_PATH,
    Protocol.ANTHROPIC_MESSAGES: ANTHROPIC_MESSAGES_PATH,
}


class ProviderCallRequest(BaseModel):
    """Immutable identity-bearing request: one Config + one Transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: ProviderCallConfig
    transcript: Transcript

    def identity_payload(self) -> dict[str, Any]:
        """Config reference (by Identity Hash) plus Transcript. No copied
        controls, no transport policy."""
        return {
            "config_identity_hash": self.config.identity_hash,
            "transcript": self.transcript.identity_payload(),
        }


def protocol_path(config: ProviderCallConfig) -> str:
    return PROTOCOL_PATHS[config.route.protocol]


def build_payload(request: ProviderCallRequest) -> dict[str, Any]:
    """Pure least-processed wire-payload construction.

    Chat completions: ``model`` + ``messages`` + controls.
    Responses: ``model`` + ``instructions``/``input`` + controls, with a
    leading system message lifted into ``instructions``.
    Anthropic messages: ``model`` + ``system`` + ``messages`` + controls.
    """
    config = request.config
    protocol = config.route.protocol
    messages = request.transcript.messages
    if protocol is Protocol.CHAT_COMPLETIONS:
        kwargs: dict[str, Any] = {
            "model": config.route.model,
            "messages": [m.provider_dict() for m in messages],
        }
    elif protocol is Protocol.RESPONSES:
        instructions, input_messages = _responses_input_messages(messages)
        kwargs = {"model": config.route.model, "input": input_messages}
        if instructions is not None:
            kwargs["instructions"] = instructions
    else:
        system, chat_messages = _anthropic_input_messages(messages)
        kwargs = {
            "model": config.route.model,
            "messages": chat_messages,
        }
        if system is not None:
            kwargs["system"] = system
    _set_controls(kwargs, config)
    return kwargs


def _set_controls(kwargs: dict[str, Any], config: ProviderCallConfig) -> None:
    # The Config was validated against its Definition at materialization,
    # so every set control is transportable; build_payload never raises.
    constraints = config.definition.constraints
    controls = config.controls
    if controls.temperature is not None and constraints.supports(
        RequestControl.TEMPERATURE
    ):
        kwargs["temperature"] = controls.temperature
    if controls.top_p is not None and constraints.supports(
        RequestControl.TOP_P
    ):
        kwargs["top_p"] = controls.top_p
    if controls.token_limit is not None and constraints.supports(
        RequestControl.TOKEN_LIMIT
    ):
        kwargs[constraints.token_limit_parameter.value] = controls.token_limit
    _set_reasoning(kwargs, config, controls)
    kwargs.update(config.extensions.extra_body)


def _set_reasoning(
    kwargs: dict[str, Any],
    config: ProviderCallConfig,
    controls: GenerationControls,
) -> None:
    constraints = config.definition.constraints
    effort = controls.reasoning
    if effort is None or not constraints.supports(RequestControl.REASONING):
        return
    shape = constraints.reasoning_shape
    if shape is ReasoningRequestShape.EFFORT_FIELD:
        kwargs["reasoning_effort"] = effort.value
    elif shape is ReasoningRequestShape.REASONING_OBJECT:
        kwargs["reasoning"] = {"effort": effort.value}


def _responses_input_messages(
    messages: tuple[Any, ...],
) -> tuple[str | None, list[dict[str, Any]]]:
    dicts = [m.provider_dict() for m in messages]
    if dicts and dicts[0].get("role") == MessageRole.SYSTEM.value:
        return dicts[0].get("content"), dicts[1:]
    return None, dicts


def _anthropic_input_messages(
    messages: tuple[Any, ...],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Lift a leading system message into the top-level ``system`` field,
    which the Anthropic Messages protocol requires separate from the
    conversational ``messages`` array."""
    dicts = [m.provider_dict() for m in messages]
    if dicts and dicts[0].get("role") == MessageRole.SYSTEM.value:
        return dicts[0].get("content"), dicts[1:]
    return None, dicts
