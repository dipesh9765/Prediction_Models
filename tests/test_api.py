from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from predictors.api import create_app
from predictors.specs import SPECS

from .conftest import example_payload

KEYS = sorted(SPECS)


# ---- ops ------------------------------------------------------------------------------------


def test_liveness_and_readiness(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["models"] == KEYS


def test_degraded_service_reports_why_and_refuses_to_guess(empty_client: TestClient) -> None:
    ready = empty_client.get("/readyz")
    assert ready.status_code == 503
    assert set(ready.json()["errors"]) == set(KEYS)
    assert empty_client.get("/healthz").status_code == 200  # the process itself is fine

    answer = empty_client.post("/v1/car/predict", json=example_payload("car"))
    assert answer.status_code == 503
    assert "unavailable" in answer.json()["detail"]
    assert "123.45" not in answer.text

    listing = empty_client.get("/v1/models").json()
    assert all(not m["available"] and m["reason"] for m in listing)


# ---- discovery ------------------------------------------------------------------------------


def test_model_listing(client: TestClient) -> None:
    models = client.get("/v1/models").json()
    assert [m["key"] for m in models] == list(SPECS)
    assert all(m["available"] and m["test_metrics"] and m["version"] for m in models)


def test_model_detail_exposes_the_input_schema_and_card(client: TestClient) -> None:
    detail = client.get("/v1/models/car").json()
    names = [f["name"] for f in detail["features"]]
    assert names == SPECS["car"].feature_names
    fuel = next(f for f in detail["features"] if f["name"] == "Fuel_Type")
    assert fuel["choices"] == ["Petrol", "Diesel", "CNG"]
    assert detail["card"]["selected"] in SPECS["car"].candidates


def test_unknown_model_is_404(client: TestClient) -> None:
    assert client.get("/v1/models/nope").status_code == 404
    assert client.post("/v1/nope/predict", json={}).status_code == 404


def test_openapi_documents_every_prediction_route_with_typed_fields(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for key in KEYS:
        assert f"/v1/{key}/predict" in schema["paths"]
    car = schema["components"]["schemas"]["CarRequest"]
    assert car["additionalProperties"] is False
    assert car["properties"]["Fuel_Type"]["enum"] == ["Petrol", "Diesel", "CNG"]
    assert car["properties"]["Owner"]["maximum"] == 3
    assert "Present_Price" in car["required"]


# ---- predictions ----------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_prediction_for_every_model(client: TestClient, key: str) -> None:
    response = client.post(f"/v1/{key}/predict", json=example_payload(key))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == key
    assert len(body["version"]) == 12
    assert body["unit"] == SPECS[key].target_unit
    if SPECS[key].task == "classification":
        assert 0 <= body["probability"] <= 1
        assert body["prediction"] == body["probability"]
        assert body["label"]
        assert body["lower"] is None
    else:
        assert body["lower"] <= body["upper"]
        assert body["interval_level"] == 0.9
        assert body["probability"] is None


def test_field_order_in_the_request_does_not_matter(client: TestClient) -> None:
    payload = example_payload("medical_insurance")
    backwards = dict(reversed(list(payload.items())))
    a = client.post("/v1/medical_insurance/predict", json=payload).json()
    b = client.post("/v1/medical_insurance/predict", json=backwards).json()
    assert a["prediction"] == b["prediction"]


def test_optional_fields_may_be_omitted_or_null(client: TestClient) -> None:
    payload = {k: v for k, v in example_payload("diabetes").items() if k != "Insulin"}
    assert client.post("/v1/diabetes/predict", json=payload).status_code == 200
    payload["SkinThickness"] = None
    assert client.post("/v1/diabetes/predict", json=payload).status_code == 200


def test_extrapolation_is_reported_in_the_response(client: TestClient) -> None:
    body = client.post("/v1/car/predict", json={**example_payload("car"), "Year": 1991}).json()
    assert body["warnings"]


# ---- validation (the original API accepted anything and returned 500) -----------------------


@pytest.mark.parametrize(
    ("key", "change"),
    [
        ("car", {"Year": 1850}),  # below range
        ("car", {"Owner": 9}),
        ("car", {"Fuel_Type": "Hydrogen"}),  # not a known category
        ("car", {"Kms_Driven": -5}),
        ("car", {"Kms_Driven": "lots"}),  # wrong type
        ("car", {"Owner": 1.5}),  # integer field
        ("car", {"Year": None}),  # required
        ("diabetes", {"Glucose": 0}),  # zero used to mean missing; no longer accepted
        ("diabetes", {"BMI": 500}),
        ("exercise", {"Gender": "robot"}),
        ("california_housing", {"Latitude": 80}),
        ("medical_insurance", {"smoker": "maybe"}),
    ],
)
def test_invalid_values_get_a_422_naming_the_field(
    client: TestClient, key: str, change: dict[str, object]
) -> None:
    response = client.post(f"/v1/{key}/predict", json={**example_payload(key), **change})
    assert response.status_code == 422
    fields = {err["loc"][-1] for err in response.json()["detail"]}
    assert set(change) <= fields


def test_missing_required_fields_are_listed(client: TestClient) -> None:
    response = client.post("/v1/car/predict", json={"Year": 2015})
    missing = {e["loc"][-1] for e in response.json()["detail"] if e["type"] == "missing"}
    assert missing == set(SPECS["car"].feature_names) - {"Year"}


def test_unknown_fields_are_rejected_not_ignored(client: TestClient) -> None:
    response = client.post("/v1/car/predict", json={**example_payload("car"), "Colour": "red"})
    assert response.status_code == 422


def test_non_finite_numbers_are_rejected(client: TestClient) -> None:
    raw = (
        '{"MedInc": NaN, "HouseAge": 20, "AveRooms": 5, "AveBedrms": 1, "Population": 900,'
        ' "AveOccup": 3, "Latitude": 34, "Longitude": -118}'
    )
    response = client.post(
        "/v1/california_housing/predict", content=raw, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_malformed_json_and_wrong_shape_are_client_errors(client: TestClient) -> None:
    bad = client.post(
        "/v1/car/predict", content="{oops", headers={"Content-Type": "application/json"}
    )
    assert bad.status_code == 422
    assert client.post("/v1/car/predict", json=[1, 2, 3]).status_code == 422
    assert client.post("/v1/car/predict", data={"Year": "2015"}).status_code == 422


def test_get_on_a_prediction_route_is_not_allowed(client: TestClient) -> None:
    assert client.get("/v1/car/predict").status_code == 405


def test_old_routes_are_gone(client: TestClient) -> None:
    for path in ("/predict", "/d_predict", "/Car/c_predict_api"):
        assert client.post(path, json={}).status_code == 404


# ---- cross-cutting --------------------------------------------------------------------------


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    assert len(client.get("/healthz").headers["X-Request-ID"]) == 32
    echoed = client.get("/healthz", headers={"X-Request-ID": "trace-123"})
    assert echoed.headers["X-Request-ID"] == "trace-123"


def test_security_headers_are_set(client: TestClient) -> None:
    headers = client.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_feature_values_are_never_written_to_the_log(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    client.post("/v1/diabetes/predict", json={**example_payload("diabetes"), "Glucose": 187})
    client.post("/v1/diabetes/predict", json={**example_payload("diabetes"), "Glucose": 7777})
    text = caplog.text
    assert "/v1/diabetes/predict" in text  # the request itself is logged...
    assert "187" not in text.replace("0.187", "")  # ...but not what the person typed
    assert "7777" not in text


def test_cors_is_off_by_default_and_opt_in(
    small_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = {"Origin": "https://example.org"}
    with TestClient(create_app(small_artifacts)) as default:
        assert "access-control-allow-origin" not in default.get("/healthz", headers=origin).headers
    monkeypatch.setenv("PREDICTORS_CORS_ORIGINS", "https://example.org")
    with TestClient(create_app(small_artifacts)) as enabled:
        allowed = enabled.get("/healthz", headers=origin).headers
        assert allowed["access-control-allow-origin"] == "https://example.org"
        other = enabled.get("/healthz", headers={"Origin": "https://evil.test"}).headers
        assert "access-control-allow-origin" not in other


def test_artifact_directory_can_come_from_the_environment(
    small_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PREDICTORS_ARTIFACTS", str(small_artifacts))
    with TestClient(create_app()) as env_client:
        assert env_client.get("/readyz").status_code == 200


def test_the_web_ui_is_served_from_the_same_origin(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "/static/app.js" in page.text
    assert "<script>" not in page.text, "inline scripts would break the CSP"
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.js").headers["Cache-Control"] == "no-cache"
    assert "Cache-Control" not in client.get("/healthz").headers
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/../../pyproject.toml").status_code == 404
