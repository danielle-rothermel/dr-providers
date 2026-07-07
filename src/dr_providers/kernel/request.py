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

from dr_providers.kernel.config import (
    EndpointKind,
    MessageRole,
    PromptMessage,
    ProviderConfig,
    ReasoningRequestShape,
    RequestControl,
)
from dr_providers.kernel.failures import (
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
    token_limit: StrictInt | None = None
    reasoning: dict[str, Any] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: StrictStr | None = None


def _set_controls(kwargs: dict[str, Any], request: LlmRequest) -> None:
    config = request.provider_config
    if request.temperature is not None:
        if config.supports(RequestControl.TEMPERATURE):
            kwargs["temperature"] = request.temperature
        elif not config.allow_unsupported_control_drop:
            _raise_unsupported(config, RequestControl.TEMPERATURE)
    if request.token_limit is not None:
        if config.supports(RequestControl.TOKEN_LIMIT):
            kwargs[config.token_limit_parameter.value] = request.token_limit
        elif not config.allow_unsupported_control_drop:
            _raise_unsupported(config, RequestControl.TOKEN_LIMIT)

    merged_extra_body = dict(config.extra_body)
    merged_extra_body.update(request.extra_body)
    if request.reasoning:
        if not config.supports(RequestControl.REASONING) or (
            config.reasoning_shape is ReasoningRequestShape.NONE
        ):
            if not config.allow_unsupported_control_drop:
                _raise_unsupported(config, RequestControl.REASONING)
        elif config.reasoning_shape is ReasoningRequestShape.TOP_LEVEL:
            kwargs["reasoning"] = dict(request.reasoning)
        elif config.reasoning_shape is ReasoningRequestShape.EXTRA_BODY:
            merged_extra_body["reasoning"] = dict(request.reasoning)
    # extra_body rides inline on the wire payload (raw httpx: the body
    # is the payload; the SDK-era extra_body indirection is flattened).
    kwargs.update(merged_extra_body)


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
    """Pure wire-payload construction; the payload rides on the response.

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
