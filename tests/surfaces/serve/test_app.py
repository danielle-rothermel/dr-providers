from typing import ClassVar

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import dr_providers  # noqa: E402
from dr_providers import ProviderTransportPolicy  # noqa: E402
from dr_providers.modeling.request import ProviderCallRequest  # noqa: E402
from dr_providers.modeling.route import ProviderKind  # noqa: E402
from dr_providers.outcomes.evidence import (  # noqa: E402
    ProviderInvocationEvidence,
)
from dr_providers.surfaces.serve import app as serve_app  # noqa: E402
from dr_providers.surfaces.serve.app import (  # noqa: E402
    ProviderChoice,
    ProviderChoiceKind,
)
from dr_providers.surfaces.serve.query import (  # noqa: E402
    QuerySpec,
    ServeProviderKind,
    build_request,
)
from dr_providers.surfaces.testing.scripted import (  # noqa: E402
    ScriptedOutcome,
    ScriptedProvider,
)
from dr_providers.transport.policy import (  # noqa: E402
    DEFAULT_API_KEY_ENVS,
    policy_for,
)

SPEC = {
    "provider_kind": "openrouter",
    "model": "test/model",
    "messages": [{"role": "user", "content": "Say hello."}],
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(serve_app.create_app())


class RecordingHttpProvider:
    instances: ClassVar[list["RecordingHttpProvider"]] = []

    def __init__(
        self, *, policy: ProviderTransportPolicy, api_key: str
    ) -> None:
        self.policy = policy
        self.api_key = api_key
        self.events: list[tuple[str, object | None]] = []
        self.instances.append(self)

    def __enter__(self) -> "RecordingHttpProvider":
        self.events.append(("enter", None))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object | None,
    ) -> None:
        self.events.append(("exit", exc_type))

    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.events.append(("invoke", request.config.route.model))
        return ScriptedProvider(
            [ScriptedOutcome(text="offline live response")]
        ).invoke(request)


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": dr_providers.__version__,
    }


def test_build_payload_previews_the_wire_format(client: TestClient) -> None:
    response = client.post("/build_payload", json={"spec": SPEC})
    assert response.status_code == 200
    payload = response.json()
    assert payload["endpoint_path"] == "/chat/completions"
    assert payload["payload"]["model"] == "test/model"


def test_query_with_scripted_scripted_outcomes(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={
            "spec": SPEC,
            "provider": {
                "kind": "scripted",
                "scripted_outcomes": [{"text": "scripted hello"}],
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    call_result = body["provider_call_result"]
    assert call_result["outcome"]["kind"] == "accepted"
    evidence = call_result["completed_invocations"][-1]["observation"][
        "evidence"
    ]
    assert evidence["response"]["text"] == "scripted hello"
    assert evidence["failure"] is None


def test_query_conformance_violation_is_reported(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={
            "spec": {**SPEC, "token_limit": 5},
            "provider": {
                "kind": "scripted",
                "scripted_outcomes": [
                    {"text": "way past budget", "completion_tokens": 50}
                ],
            },
        },
    )
    assert response.status_code == 200
    call_result = response.json()["provider_call_result"]
    evidence = call_result["completed_invocations"][-1]["observation"][
        "evidence"
    ]
    warnings = evidence["response"]["warnings"]
    assert any(w["code"] == "token_limit_exceeded" for w in warnings)


def test_query_failure_outcome_returns_failure_record(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={
            "spec": SPEC,
            "provider": {
                "kind": "scripted",
                "scripted_outcomes": [
                    {
                        "text": "",
                        "failure_code": "rate_limited",
                        "failure_message": "scripted",
                    }
                ],
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    call_result = body["provider_call_result"]
    assert call_result["outcome"]["kind"] == "invocation_outcome"
    evidence = call_result["completed_invocations"][-1]["observation"][
        "evidence"
    ]
    assert evidence["response"] is None
    assert evidence["failure"]["code"] == "rate_limited"


def test_live_provider_without_key_is_424(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    response = client.post(
        "/query",
        json={"spec": SPEC, "provider": {"kind": "live"}},
    )
    assert response.status_code == 424
    assert "missing_api_key" in response.json()["detail"]


@pytest.mark.parametrize(
    ("serve_kind", "expected_provider"),
    [
        pytest.param(
            ServeProviderKind.OPENROUTER,
            ProviderKind.OPENROUTER,
            id="openrouter",
        ),
        pytest.param(
            ServeProviderKind.OPENAI,
            ProviderKind.OPENAI,
            id="openai-chat",
        ),
        pytest.param(
            ServeProviderKind.OPENAI_RESPONSES,
            ProviderKind.OPENAI,
            id="openai-responses",
        ),
        pytest.param(
            ServeProviderKind.GEMINI,
            ProviderKind.GEMINI,
            id="gemini",
        ),
        pytest.param(
            ServeProviderKind.ANTHROPIC,
            ProviderKind.ANTHROPIC,
            id="anthropic",
        ),
    ],
)
def test_live_provider_maps_key_policy_and_closes_after_success(
    monkeypatch: pytest.MonkeyPatch,
    serve_kind: ServeProviderKind,
    expected_provider: ProviderKind,
) -> None:
    RecordingHttpProvider.instances.clear()
    monkeypatch.setattr(serve_app, "HttpProvider", RecordingHttpProvider)
    api_key = f"test-key-for-{serve_kind.value}"
    monkeypatch.setenv(str(DEFAULT_API_KEY_ENVS[expected_provider]), api_key)
    spec = QuerySpec(
        provider_kind=serve_kind,
        model="test/model",
        messages=(),
    )

    with serve_app.resolve_provider(
        ProviderChoice(kind=ProviderChoiceKind.LIVE), spec
    ) as provider:
        evidence = provider.invoke(build_request(spec))

    instance = RecordingHttpProvider.instances[0]
    assert instance.api_key == api_key
    assert instance.policy == policy_for(expected_provider)
    assert evidence.response is not None
    assert instance.events == [
        ("enter", None),
        ("invoke", "test/model"),
        ("exit", None),
    ]


def test_live_provider_closes_when_context_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingHttpProvider.instances.clear()
    monkeypatch.setattr(serve_app, "HttpProvider", RecordingHttpProvider)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    spec = QuerySpec(
        provider_kind=ServeProviderKind.OPENROUTER,
        model="test/model",
        messages=(),
    )

    with (
        pytest.raises(RuntimeError, match="context body failed"),
        serve_app.resolve_provider(
            ProviderChoice(kind=ProviderChoiceKind.LIVE), spec
        ),
    ):
        raise RuntimeError("context body failed")

    assert RecordingHttpProvider.instances[0].events == [
        ("enter", None),
        ("exit", RuntimeError),
    ]


def test_unknown_provider_kind_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"spec": SPEC, "provider": {"kind": "bogus"}},
    )
    assert response.status_code == 422


def test_variance_endpoint_reports_and_records(client: TestClient) -> None:
    response = client.post(
        "/variance",
        json={
            "prompt": "Say hello.",
            "models": ["model-a", "model-b"],
            "samples": 2,
            "provider": {
                "kind": "scripted",
                "scripted_outcomes": [
                    {"text": "alpha"},
                    {"text": "beta"},
                ],
            },
        },
    )
    assert response.status_code == 200
    report = response.json()
    assert report["samples_per_model"] == 2
    assert len(report["records"]) == 4
    assert len(report["per_model"]) == 2
    assert report["records"][0] == {
        "model": "model-a",
        "sample_index": 0,
        "ok": True,
        "text": "alpha",
        "finish_reason": "stop",
        "completion_tokens": None,
        "total_cost": None,
        "warning_codes": [],
        "failure_code": None,
    }


def test_live_variance_uses_one_provider_context_for_full_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    RecordingHttpProvider.instances.clear()
    monkeypatch.setattr(serve_app, "HttpProvider", RecordingHttpProvider)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    response = client.post(
        "/variance",
        json={
            "prompt": "Say hello.",
            "models": ["model-a", "model-b"],
            "samples": 2,
            "provider": {"kind": "live"},
        },
    )

    assert response.status_code == 200
    assert len(RecordingHttpProvider.instances) == 1
    assert RecordingHttpProvider.instances[0].events == [
        ("enter", None),
        ("invoke", "model-a"),
        ("invoke", "model-a"),
        ("invoke", "model-b"),
        ("invoke", "model-b"),
        ("exit", None),
    ]


@pytest.mark.parametrize(
    ("overrides", "expected_detail"),
    [
        pytest.param(
            {"samples": 0},
            f"samples must be 1..{serve_app.MAX_VARIANCE_SAMPLES}",
            id="samples-lower-bound",
        ),
        pytest.param(
            {"samples": serve_app.MAX_VARIANCE_SAMPLES + 1},
            f"samples must be 1..{serve_app.MAX_VARIANCE_SAMPLES}",
            id="samples-upper-bound",
        ),
        pytest.param(
            {"models": []},
            f"models must be 1..{serve_app.MAX_VARIANCE_MODELS}",
            id="models-lower-bound",
        ),
        pytest.param(
            {
                "models": [
                    f"model-{index}"
                    for index in range(serve_app.MAX_VARIANCE_MODELS + 1)
                ]
            },
            f"models must be 1..{serve_app.MAX_VARIANCE_MODELS}",
            id="models-upper-bound",
        ),
    ],
)
def test_variance_endpoint_validates_boundaries(
    client: TestClient,
    overrides: dict[str, object],
    expected_detail: str,
) -> None:
    response = client.post(
        "/variance",
        json={"prompt": "p", "models": ["m"], "samples": 1, **overrides},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == expected_detail


@pytest.mark.parametrize(
    ("overrides", "expected_record_count"),
    [
        pytest.param(
            {"samples": serve_app.MAX_VARIANCE_SAMPLES},
            serve_app.MAX_VARIANCE_SAMPLES,
            id="samples-maximum",
        ),
        pytest.param(
            {
                "models": [
                    f"model-{index}"
                    for index in range(serve_app.MAX_VARIANCE_MODELS)
                ]
            },
            serve_app.MAX_VARIANCE_MODELS,
            id="models-maximum",
        ),
    ],
)
def test_variance_endpoint_accepts_exact_upper_boundaries(
    client: TestClient,
    overrides: dict[str, object],
    expected_record_count: int,
) -> None:
    response = client.post(
        "/variance",
        json={"prompt": "p", "models": ["m"], "samples": 1, **overrides},
    )

    assert response.status_code == 200
    assert len(response.json()["records"]) == expected_record_count
