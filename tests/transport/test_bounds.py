from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from dr_providers import (
    FailureClass,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    Transcript,
    openai_chat_config,
)
from dr_providers.transport.http import (
    MAX_FAILURE_MESSAGE_CHARS,
    MAX_RETRY_AFTER_HEADER_BYTES,
    REQUEST_TOO_LARGE_CODE,
    RESPONSE_TOO_LARGE_CODE,
    HttpProvider,
)

REQUEST = ProviderCallRequest(
    config=openai_chat_config(model="m"),
    transcript=Transcript(
        messages=(PromptMessage(role=MessageRole.USER, content="write add"),)
    ),
)
ENCODED_REQUEST = (
    b'{"model":"m","messages":[{"role":"user","content":"write add"}]}'
)
SUCCESS_BODY: dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "m",
    "choices": [
        {
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
}
ENCODED_SUCCESS = json.dumps(
    SUCCESS_BODY,
    separators=(",", ":"),
).encode()
ERROR_BODY = {"error": {"message": "provider-controlled detail"}}
ENCODED_ERROR = json.dumps(ERROR_BODY, separators=(",", ":")).encode()


class ByteChunks(httpx.SyncByteStream):
    def __init__(self, content: bytes) -> None:
        self._content = content
        self.yielded = 0

    def __iter__(self) -> Iterator[bytes]:
        for byte in self._content:
            self.yielded += 1
            yield bytes((byte,))


def _policy(**overrides: Any) -> ProviderTransportPolicy:
    return ProviderTransportPolicy(
        api_key_env="TEST_API_KEY",
        base_url="https://example.test/v1",
        max_request_bytes=overrides.pop("max_request_bytes", 1024),
        max_response_bytes=overrides.pop("max_response_bytes", 1024),
        **overrides,
    )


def _provider(
    handler: Any,
    *,
    policy: ProviderTransportPolicy,
) -> HttpProvider:
    return HttpProvider(
        policy=policy,
        api_key="test-key",
        _client_factory=lambda **_kwargs: httpx.Client(
            transport=httpx.MockTransport(handler)
        ),
    )


@pytest.mark.parametrize(
    ("limit_delta", "expected"),
    [(-1, "rejected"), (0, "sent"), (1, "sent")],
    ids=("limit-minus-one", "exact-limit", "limit-plus-one"),
)
def test_exact_encoded_request_is_counted_and_sent_once(
    limit_delta: int,
    expected: str,
) -> None:
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        assert request.headers["content-type"] == "application/json"
        return httpx.Response(200, content=ENCODED_SUCCESS)

    policy = _policy(
        max_request_bytes=len(ENCODED_REQUEST) + limit_delta,
    )
    evidence = _provider(handler, policy=policy).invoke(REQUEST)

    assert evidence.max_request_bytes == policy.max_request_bytes
    assert evidence.max_response_bytes == policy.max_response_bytes
    if expected == "sent":
        assert evidence.http_request is not None
        assert evidence.http_request.body_bytes == len(ENCODED_REQUEST)
        assert seen == [ENCODED_REQUEST]
        assert isinstance(evidence.outcome, ProviderTransportResponse)
    else:
        assert evidence.http_request is None
        assert seen == []
        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.failure_class is FailureClass.RESOURCE_EXHAUSTION
        assert failure.code == REQUEST_TOO_LARGE_CODE
        assert failure.response_body is None
        assert failure.metadata == {
            "limit_bytes": len(ENCODED_REQUEST) - 1,
            "observed_bytes": len(ENCODED_REQUEST),
        }


@pytest.mark.parametrize(
    ("status_code", "encoded_body"),
    [(200, ENCODED_SUCCESS), (400, ENCODED_ERROR)],
    ids=("success", "error"),
)
@pytest.mark.parametrize(
    ("limit_delta", "expected"),
    [(-1, "rejected"), (0, "accepted"), (1, "accepted")],
    ids=("limit-minus-one", "exact-limit", "limit-plus-one"),
)
def test_streamed_response_exact_limit_boundaries(
    status_code: int,
    encoded_body: bytes,
    limit_delta: int,
    expected: str,
) -> None:
    stream = ByteChunks(encoded_body)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={"content-length": "1"},
            stream=stream,
        )

    limit = len(encoded_body) + limit_delta
    evidence = _provider(
        handler,
        policy=_policy(max_response_bytes=limit),
    ).invoke(REQUEST)

    if expected == "rejected":
        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert failure.failure_class is FailureClass.RESOURCE_EXHAUSTION
        assert failure.code == RESPONSE_TOO_LARGE_CODE
        assert failure.response_body is None
        assert failure.metadata == {
            "limit_bytes": limit,
            "observed_bytes": limit + 1,
        }
        assert evidence.response_bytes == limit + 1
        assert stream.yielded == limit + 1
        assert len(failure.message) <= MAX_FAILURE_MESSAGE_CHARS
        assert "provider-controlled detail" not in failure.message
    elif status_code == 200:
        assert isinstance(evidence.outcome, ProviderTransportResponse)
        assert evidence.response_bytes == len(encoded_body)
        assert evidence.outcome.response_body == SUCCESS_BODY
    else:
        failure = evidence.failure
        assert isinstance(failure, ProviderTransportFailure)
        assert evidence.response_bytes == len(encoded_body)
        assert failure.response_body == ERROR_BODY
        assert failure.message == "provider returned HTTP status 400"
        assert "provider-controlled detail" not in failure.message


def test_missing_content_length_does_not_prevent_complete_streaming() -> None:
    stream = ByteChunks(ENCODED_SUCCESS)
    provider = _provider(
        lambda _request: httpx.Response(200, stream=stream),
        policy=_policy(max_response_bytes=len(ENCODED_SUCCESS)),
    )

    evidence = provider.invoke(REQUEST)

    assert isinstance(evidence.outcome, ProviderTransportResponse)
    assert evidence.response_bytes == len(ENCODED_SUCCESS)
    assert stream.yielded == len(ENCODED_SUCCESS)


def test_bound_applies_to_decompressed_response_bytes() -> None:
    compressed = gzip.compress(ENCODED_SUCCESS)
    stream = ByteChunks(compressed)
    provider = _provider(
        lambda _request: httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=stream,
        ),
        policy=_policy(max_response_bytes=len(ENCODED_SUCCESS) - 1),
    )

    evidence = provider.invoke(REQUEST)

    failure = evidence.failure
    assert isinstance(failure, ProviderTransportFailure)
    assert failure.code == RESPONSE_TOO_LARGE_CODE
    assert evidence.response_bytes is not None
    assert evidence.response_bytes > len(ENCODED_SUCCESS) - 1
    assert failure.response_body is None


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (
            "00012",
            {"kind": "delta_seconds", "value": 12},
        ),
        (
            "Wed, 21 Oct 2015 07:28:00 GMT",
            {
                "kind": "http_date",
                "value": "Wed, 21 Oct 2015 07:28:00 GMT",
            },
        ),
        ("not a retry hint", None),
        ("x" * (MAX_RETRY_AFTER_HEADER_BYTES + 1), None),
    ],
    ids=("delta-seconds", "http-date", "malformed", "oversized"),
)
def test_retry_after_is_bounded_and_normalized(
    header: str,
    expected: dict[str, Any] | None,
) -> None:
    provider = _provider(
        lambda _request: httpx.Response(
            429,
            headers={"retry-after": header},
            content=ENCODED_ERROR,
        ),
        policy=_policy(),
    )

    evidence = provider.invoke(REQUEST)

    assert (
        None
        if evidence.retry_after is None
        else evidence.retry_after.model_dump(mode="json")
    ) == expected


def test_byte_accounting_and_hint_keys_are_pinned() -> None:
    provider = _provider(
        lambda _request: httpx.Response(
            429,
            headers={"retry-after": "12"},
            content=ENCODED_ERROR,
        ),
        policy=_policy(),
    )

    payload = provider.invoke(REQUEST).identity_payload()

    assert set(payload) == {
        "request_identity_hash",
        "policy_identity",
        "max_request_bytes",
        "max_response_bytes",
        "http_request",
        "response_bytes",
        "retry_after",
        "response",
        "failure",
    }
    assert set(payload["http_request"]) == {
        "method",
        "url",
        "headers",
        "body",
        "body_bytes",
    }
    assert payload["retry_after"] == {
        "kind": "delta_seconds",
        "value": 12,
    }
