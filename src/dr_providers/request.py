"""Request model and the public, pure ``build_payload``.

Structure is validated before send: frozen models, ``extra="forbid"``,
no silent defaults — every knob is ``None`` (never serialized) or
explicit. Setting a knob the config cannot transport raises
:class:`UnsupportedControlError` unless the config explicitly opts into
dropping it (``allow_unsupported_control_drop``, for models that reject
a knob their provider otherwise supports).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_providers.config import (
    EndpointKind,
    MessageRole,
    PromptMessage,
    ProviderConfig,
    ReasoningEffort,
    ReasoningRequestShape,
    RequestControl,
)
from dr_providers.failures import (
    FailureClass,
    UnsupportedControlError,
    failure_record,
)

CHAT_COMPLETIONS_PATH = "/chat/completions"
RESPONSES_PATH = "/responses"

ENDPOINT_PATHS: dict[EndpointKind, str] = {
    EndpointKind.CHAT_COMPLETIONS: CHAT_COMPLETIONS_PATH,
    EndpointKind.RESPONSES: RESPONSES_PATH,
}


class LlmRequest(BaseModel):
    """Single-shot request: full transcript plus explicit knobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_config: ProviderConfig
    messages: tuple[PromptMessage, ...]
    temperature: float | None = None
    top_p: float | None = None
    token_limit: StrictInt | None = None
    reasoning: ReasoningEffort | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: StrictStr | None = None


def _set_controls(kwargs: dict[str, Any], request: LlmRequest) -> None:
    config = request.provider_config
    if request.temperature is not None:
        if config.supports(RequestControl.TEMPERATURE):
            kwargs["temperature"] = request.temperature
        elif not config.allow_unsupported_control_drop:
            _raise_unsupported(config, RequestControl.TEMPERATURE)
    if request.top_p is not None:
        if config.supports(RequestControl.TOP_P):
            kwargs["top_p"] = request.top_p
        elif not config.allow_unsupported_control_drop:
            _raise_unsupported(config, RequestControl.TOP_P)
    if request.token_limit is not None:
        if config.supports(RequestControl.TOKEN_LIMIT):
            kwargs[config.token_limit_parameter.value] = request.token_limit
        elif not config.allow_unsupported_control_drop:
            _raise_unsupported(config, RequestControl.TOKEN_LIMIT)

    _set_reasoning(kwargs, request)
    # extra_body rides inline on the wire payload (raw httpx: the body
    # is the payload; the SDK-era extra_body indirection is flattened).
    merged_extra_body = dict(config.extra_body)
    merged_extra_body.update(request.extra_body)
    kwargs.update(merged_extra_body)


def _set_reasoning(kwargs: dict[str, Any], request: LlmRequest) -> None:
    config = request.provider_config
    effort = request.reasoning
    if effort is None:
        return
    if not config.supports(RequestControl.REASONING) or (
        config.reasoning_shape is ReasoningRequestShape.NONE
    ):
        if not config.allow_unsupported_control_drop:
            _raise_unsupported(config, RequestControl.REASONING)
        return
    if config.reasoning_shape is ReasoningRequestShape.EFFORT_FIELD:
        kwargs["reasoning_effort"] = effort.value
    elif config.reasoning_shape is ReasoningRequestShape.REASONING_OBJECT:
        kwargs["reasoning"] = {"effort": effort.value}


def _raise_unsupported(
    config: ProviderConfig, control: RequestControl
) -> None:
    failure = failure_record(
        failure_class=FailureClass.PERMANENT,
        code="unsupported_control",
        message=(
            f"request sets {control.value!r} but provider config "
            f"{config.throttle_identity!r} cannot transport it"
        ),
        metadata={
            "provider_kind": config.provider_kind.value,
            "endpoint_kind": config.endpoint_kind.value,
            "control": control.value,
        },
    )
    raise UnsupportedControlError(failure)


def build_payload(request: LlmRequest) -> dict[str, Any]:
    """Pure wire-payload construction.

    Chat completions: ``model`` + ``messages`` + knobs.
    Responses: ``model`` + ``instructions``/``input`` + knobs, with a
    leading system message lifted into ``instructions``.
    """
    config = request.provider_config
    if config.endpoint_kind is EndpointKind.CHAT_COMPLETIONS:
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [m.provider_dict() for m in request.messages],
        }
    else:
        instructions, input_messages = _responses_input_messages(
            request.messages
        )
        kwargs = {
            "model": config.model,
            "input": input_messages,
        }
        if instructions is not None:
            kwargs["instructions"] = instructions
    _set_controls(kwargs, request)
    return kwargs


def endpoint_path(config: ProviderConfig) -> str:
    return ENDPOINT_PATHS[config.endpoint_kind]


def _responses_input_messages(
    messages: tuple[PromptMessage, ...],
) -> tuple[str | None, list[dict[str, Any]]]:
    dicts = [m.provider_dict() for m in messages]
    if dicts and dicts[0].get("role") == MessageRole.SYSTEM.value:
        return dicts[0].get("content"), dicts[1:]
    return None, dicts
