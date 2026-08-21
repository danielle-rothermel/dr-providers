from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_providers.core.frozen import _thaw
from dr_providers.modeling.controls import (
    GenerationControls,
    ReasoningRequestShape,
)
from dr_providers.modeling.route import Protocol
from dr_providers.modeling.transcript import MessageRole

if TYPE_CHECKING:
    from dr_providers.modeling.call import ProviderCallConfig
    from dr_providers.modeling.request import ProviderCallRequest
    from dr_providers.modeling.transcript import PromptMessage

CHAT_COMPLETIONS_PATH = "/chat/completions"
RESPONSES_PATH = "/responses"
ANTHROPIC_MESSAGES_PATH = "/messages"

PROTOCOL_PATHS: dict[Protocol, str] = {
    Protocol.CHAT_COMPLETIONS: CHAT_COMPLETIONS_PATH,
    Protocol.RESPONSES: RESPONSES_PATH,
    Protocol.ANTHROPIC_MESSAGES: ANTHROPIC_MESSAGES_PATH,
}


def protocol_path(config: ProviderCallConfig) -> str:
    return PROTOCOL_PATHS[config.route.protocol]


def build_payload(request: ProviderCallRequest) -> dict[str, Any]:
    config = request.config
    protocol = config.route.protocol
    messages = request.transcript.messages
    if protocol is Protocol.CHAT_COMPLETIONS:
        payload: dict[str, Any] = {
            "model": config.route.model,
            "messages": [message.provider_dict() for message in messages],
        }
    elif protocol is Protocol.RESPONSES:
        instructions, input_messages = _input_messages(messages)
        payload = {"model": config.route.model, "input": input_messages}
        if instructions is not None:
            payload["instructions"] = instructions
    else:
        system, chat_messages = _input_messages(messages)
        payload = {"model": config.route.model, "messages": chat_messages}
        if system is not None:
            payload["system"] = system
    _set_controls(payload, config)
    return payload


def _set_controls(payload: dict[str, Any], config: ProviderCallConfig) -> None:
    constraints = config.definition.constraints
    controls = config.controls
    if controls.temperature is not None:
        payload["temperature"] = controls.temperature
    if controls.top_p is not None:
        payload["top_p"] = controls.top_p
    if controls.token_limit is not None:
        payload[constraints.token_limit_parameter.value] = controls.token_limit
    if controls.seed is not None:
        payload["seed"] = controls.seed
    _set_reasoning(payload, config, controls)
    # Thaw for JSON encoding; Config validation prevents core-field overrides.
    payload.update(_thaw(config.extensions.extra_body))


def _set_reasoning(
    payload: dict[str, Any],
    config: ProviderCallConfig,
    controls: GenerationControls,
) -> None:
    constraints = config.definition.constraints
    effort = controls.reasoning
    if effort is None:
        return
    if config.route.protocol is Protocol.ANTHROPIC_MESSAGES:
        # Anthropic nests effort under output_config, not reasoning.
        payload["output_config"] = {"effort": effort.value}
        return
    shape = constraints.reasoning_shape
    if shape is ReasoningRequestShape.EFFORT_FIELD:
        payload["reasoning_effort"] = effort.value
    elif shape is ReasoningRequestShape.REASONING_OBJECT:
        payload["reasoning"] = {"effort": effort.value}


def _input_messages(
    messages: tuple[PromptMessage, ...],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Responses and Anthropic carry a leading system message separately."""
    dicts = [message.provider_dict() for message in messages]
    if dicts and dicts[0].get("role") == MessageRole.SYSTEM.value:
        return dicts[0].get("content"), dicts[1:]
    return None, dicts
