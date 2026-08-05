"""Translate a validated Provider Call Request into its wire request."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_providers.core.frozen import _thaw
from dr_providers.modeling.controls import (
    GenerationControls,
    ReasoningRequestShape,
    RequestControl,
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
    # The Config was validated against its Definition at materialization,
    # so every set control is transportable; build_payload never raises.
    constraints = config.definition.constraints
    controls = config.controls
    if controls.temperature is not None and constraints.supports(
        RequestControl.TEMPERATURE
    ):
        payload["temperature"] = controls.temperature
    if controls.top_p is not None and constraints.supports(
        RequestControl.TOP_P
    ):
        payload["top_p"] = controls.top_p
    if controls.token_limit is not None and constraints.supports(
        RequestControl.TOKEN_LIMIT
    ):
        payload[constraints.token_limit_parameter.value] = controls.token_limit
    _set_reasoning(payload, config, controls)
    # Reserved core keys are rejected when the Config is validated, so an
    # extension can never overwrite a validated core wire field here. Thaw
    # so nested frozen mappings stay JSON-serializable on the wire.
    payload.update(_thaw(config.extensions.extra_body))


def _set_reasoning(
    payload: dict[str, Any],
    config: ProviderCallConfig,
    controls: GenerationControls,
) -> None:
    constraints = config.definition.constraints
    effort = controls.reasoning
    if effort is None or not constraints.supports(RequestControl.REASONING):
        return
    if config.route.protocol is Protocol.ANTHROPIC_MESSAGES:
        # The Anthropic Messages API rejects a top-level {"reasoning": ...};
        # it takes {"output_config": {"effort": ...}} with a restricted set
        # of effort levels, already validated on the Config.
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
    """Lift a leading system message into a separate top-level field.

    Both the Responses (``instructions``) and Anthropic Messages
    (``system``) protocols carry a leading system message separately from
    the conversational array, so the split is identical for both.
    """
    dicts = [message.provider_dict() for message in messages]
    if dicts and dicts[0].get("role") == MessageRole.SYSTEM.value:
        return dicts[0].get("content"), dicts[1:]
    return None, dicts
