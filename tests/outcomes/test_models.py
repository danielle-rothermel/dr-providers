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
            retryable=False,
        )
        assert is_response(response) is True
        assert is_failure(response) is False
        assert is_failure(failure) is True
        assert is_response(failure) is False
