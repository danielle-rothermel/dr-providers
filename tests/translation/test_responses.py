import json

import pytest

from dr_providers import (
    ProviderTransportFailure,
    ProviderTransportResponse,
    openai_responses_config,
    parse_responses_body,
)


def test_responses_body_reports_response_id() -> None:
    body = {
        "id": "resp-1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }

    response = parse_responses_body(
        body, config=openai_responses_config(model="m")
    )

    assert isinstance(response, ProviderTransportResponse)
    assert response.response_id == "resp-1"
    assert response.finish_reason == "stop"
    assert response.diagnostics is not None
    assert response.diagnostics.output_text_len == 2


def test_responses_output_parts_fallback() -> None:
    body = {
        "id": "resp-2",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "part one "},
                    {"type": "output_text", "text": "part two"},
                ],
            }
        ],
    }

    response = parse_responses_body(
        body, config=openai_responses_config(model="m")
    )

    assert isinstance(response, ProviderTransportResponse)
    assert response.text == "part one part two"


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        (
            {
                "id": "resp-private-refusal",
                "status": "completed",
                "prompt": "PRIVATE_PROMPT",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "PRIVATE_REFUSAL",
                            }
                        ],
                    }
                ],
            },
            "response_refusal",
        ),
        (
            {
                "id": "resp-private-output",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "arguments": "PRIVATE_OUTPUT",
                    }
                ],
            },
            "response_no_text",
        ),
        (
            {
                "id": "resp-private-malformed",
                "prompt": "PRIVATE_PROMPT",
                "output": {"text": "PRIVATE_OUTPUT"},
            },
            "response_parse_error",
        ),
    ],
    ids=["refusal", "tool_only", "malformed"],
)
def test_responses_failure_metadata_is_content_free(
    body: dict, expected_code: str
) -> None:
    failure = parse_responses_body(
        body, config=openai_responses_config(model="m")
    )

    assert isinstance(failure, ProviderTransportFailure)
    assert failure.code == expected_code
    serialized_metadata = json.dumps(failure.metadata)
    for private_value in (
        "PRIVATE_PROMPT",
        "PRIVATE_OUTPUT",
        "PRIVATE_REFUSAL",
        body["id"],
    ):
        assert private_value not in serialized_metadata
    assert len(failure.metadata["diagnostics"]["response_id_hash"]) == 16
    assert failure.raw_response_body == body
