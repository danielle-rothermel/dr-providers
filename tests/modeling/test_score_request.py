from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from dr_providers import (
    ControlValidationError,
    GenerationControls,
    ProviderBodyExtensions,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderCallKind,
    ProviderCallRequest,
    ProviderGenerateRequest,
    ProviderScoreRequest,
    RecoverabilityClass,
    Transcript,
    anthropic_messages_config,
    gemini_chat_config,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
)


@pytest.mark.parametrize(
    "config",
    [
        openai_chat_config(model="m"),
        openai_responses_config(model="m"),
        openrouter_chat_config(model="m"),
        gemini_chat_config(model="m"),
        anthropic_messages_config(
            model="m", controls=GenerationControls(token_limit=1)
        ),
    ],
)
def test_presets_refuse_score(config: ProviderCallConfig) -> None:
    with pytest.raises(ControlValidationError) as exc:
        ProviderScoreRequest(config=config, context="", continuations=("a",))
    assert exc.value.failure.code == "unsupported_call_kind"
    assert exc.value.failure.recoverability is RecoverabilityClass.PERMANENT
    assert exc.value.failure.metadata == {
        "kind": "score",
        "definition_id": config.definition.definition_id,
    }


def test_score_only_definition_refuses_generate(
    score_config: ProviderCallConfig,
) -> None:
    with pytest.raises(ControlValidationError) as exc:
        ProviderGenerateRequest(
            config=score_config, transcript=Transcript(messages=())
        )
    assert exc.value.failure.code == "unsupported_call_kind"
    assert exc.value.failure.metadata == {
        "kind": "generate",
        "definition_id": "test.score",
    }
    assert exc.value.failure.recoverability is RecoverabilityClass.PERMANENT


def test_definition_requires_nonempty_supported_kinds(
    score_config: ProviderCallConfig,
) -> None:
    data = score_config.definition.model_dump(mode="python")
    with pytest.raises(ControlValidationError) as exc:
        ProviderCallDefinition.model_validate({**data, "supported_kinds": []})
    assert exc.value.failure.code == "no_supported_kinds"
    assert exc.value.failure.recoverability is RecoverabilityClass.PERMANENT
    data.pop("supported_kinds")
    with pytest.raises(ValidationError, match="supported_kinds"):
        ProviderCallDefinition.model_validate(data)


@pytest.mark.parametrize(
    "override",
    [
        {"continuations": ()},
        {"continuations": ("",)},
        {"continuations": (1,)},
        {"context": 1},
        {"token_logprobs": 1},
        {"kind": "generate"},
    ],
)
def test_score_request_rejects_invalid_fields(
    score_request: ProviderScoreRequest, override: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        ProviderScoreRequest.model_validate(
            {**score_request.model_dump(mode="python"), **override}
        )


def test_score_request_allows_empty_context_and_duplicates(
    score_config: ProviderCallConfig,
) -> None:
    request = ProviderScoreRequest(
        config=score_config, context="", continuations=("a", "a")
    )
    assert request.context == ""
    assert request.continuations == ("a", "a")


@pytest.mark.parametrize(
    "controls",
    [GenerationControls(temperature=0.0), GenerationControls(seed=0)],
)
def test_score_refuses_set_controls(
    score_config: ProviderCallConfig, controls: GenerationControls
) -> None:
    # The zero value is set, not an unset control.
    data = score_config.definition.model_dump(mode="python")
    data["constraints"]["supported_controls"] = ["temperature", "seed"]
    config = ProviderCallDefinition.model_validate(data).materialize(
        controls=controls
    )
    with pytest.raises(ControlValidationError) as exc:
        ProviderScoreRequest(config=config, context="", continuations=("a",))
    assert exc.value.failure.code == "score_request_rejects_controls"
    assert exc.value.failure.metadata == {
        "controls": [
            "temperature" if controls.temperature is not None else "seed"
        ]
    }
    assert exc.value.failure.recoverability is RecoverabilityClass.PERMANENT


def test_score_refuses_extensions(score_config: ProviderCallConfig) -> None:
    data = score_config.definition.model_dump(mode="python")
    definition = ProviderCallDefinition.model_validate(
        {**data, "extension_keys": ["user"]}
    )
    config = definition.materialize(
        extensions=ProviderBodyExtensions(extra_body={"user": "test"})
    )
    with pytest.raises(ControlValidationError) as exc:
        ProviderScoreRequest(config=config, context="", continuations=("a",))
    assert exc.value.failure.code == "score_request_rejects_extensions"
    assert exc.value.failure.recoverability is RecoverabilityClass.PERMANENT


def test_request_union_round_trips_both_kinds(
    score_request: ProviderScoreRequest,
) -> None:
    adapter = TypeAdapter(ProviderCallRequest)
    for request in (
        ProviderGenerateRequest(
            config=openai_chat_config(model="m"),
            transcript=Transcript(messages=()),
        ),
        score_request,
    ):
        restored = adapter.validate_python(request.model_dump(mode="json"))
        assert type(restored) is type(request)
        assert restored == request
        assert adapter.validate_json(request.model_dump_json()) == request
    with pytest.raises(ValidationError, match="union_tag_not_found"):
        adapter.validate_python({"config": {}, "context": ""})


def test_supported_kinds_serialization_is_sorted(
    score_config: ProviderCallConfig,
) -> None:
    data = score_config.definition.model_dump(mode="python")
    definition = ProviderCallDefinition.model_validate(
        {
            **data,
            "supported_kinds": [
                ProviderCallKind.SCORE,
                ProviderCallKind.GENERATE,
            ],
        }
    )
    assert definition.model_dump(mode="json")["supported_kinds"] == [
        "generate",
        "score",
    ]
    assert definition.identity_payload()["supported_kinds"] == [
        "generate",
        "score",
    ]
