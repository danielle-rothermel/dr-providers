"""Shared failure-vocabulary tests."""

from dr_providers import (
    FailureClass,
    failure_record,
)


class TestFailures:
    def test_failure_record_retryable_follows_class(self) -> None:
        rate_limited = failure_record(
            failure_class=FailureClass.RATE_LIMITED, message="slow down"
        )
        assert rate_limited.retryable is True
        permanent = failure_record(
            failure_class=FailureClass.PERMANENT, message="bad key"
        )
        assert permanent.retryable is False
