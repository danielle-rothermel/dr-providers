import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import dr_providers  # noqa: E402
from dr_providers.surfaces.serve.app import create_app  # noqa: E402

SPEC = {
    "provider_kind": "openrouter",
    "model": "test/model",
    "messages": [{"role": "user", "content": "Say hello."}],
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


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
    assert body["response"]["text"] == "scripted hello"
    assert body["failure"] is None


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
    warnings = response.json()["response"]["warnings"]
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
    assert body["response"] is None
    assert body["failure"]["code"] == "rate_limited"


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


def test_variance_caps_samples(client: TestClient) -> None:
    response = client.post(
        "/variance",
        json={"prompt": "p", "models": ["m"], "samples": 999},
    )
    assert response.status_code == 422
