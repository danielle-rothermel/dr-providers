import pytest
from pydantic import ValidationError

from dr_providers import (
    FailureClass,
    ProviderTransportFailure,
    ProviderTransportResponse,
    is_failure,
    is_response,
)


class TestOutcomeGuards:
    def test_is_response_and_is_failure_narrow(self) -> None:
        response = ProviderTransportResponse(text="hi")
        failure = ProviderTransportFailure(
            failure_class=FailureClass.PERMANENT,
            code="x",
            message="m",
        )
        assert is_response(response) is True
        assert is_failure(response) is False
        assert is_failure(failure) is True
        assert is_response(failure) is False


def test_removed_persisted_field_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ProviderTransportResponse.model_validate(
            {"text": "hi", "raw_body": {}}
        )
    for removed_field in ("retryable", "request_body", "raw_request"):
        with pytest.raises(ValidationError):
            ProviderTransportFailure.model_validate(
                {
                    "failure_class": FailureClass.PERMANENT,
                    "message": "bad",
                    removed_field: False,
                }
            )
