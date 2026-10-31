"""Checks against the real, fully trained artifacts in ./artifacts (CI runs `make train` first)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from predictors.api import create_app
from predictors.check_cards import main as check_main
from predictors.specs import SPECS

from .conftest import ROOT, example_payload

pytestmark = pytest.mark.integration

ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="module")
def real_client() -> TestClient:
    with TestClient(create_app(ARTIFACTS)) as client:
        return client


def test_every_trained_model_clears_its_quality_bar() -> None:
    assert check_main(["--artifacts", str(ARTIFACTS)]) == 0


def test_the_service_is_ready_with_real_artifacts(real_client: TestClient) -> None:
    assert real_client.get("/readyz").status_code == 200


@pytest.mark.parametrize(
    ("key", "payload", "low", "high"),
    [
        # A 2015 car whose new equivalent costs 6.5 lakh sells for a few lakh, not hundreds.
        ("car", None, 1.5, 7.0),
        # 20 minutes at 100 bpm for a 70 kg adult is a modest workout.
        ("exercise", None, 40, 140),
        ("medical_insurance", None, 2000, 12000),
        ("california_housing", None, 1.0, 4.5),
    ],
)
def test_example_predictions_are_plausible(
    real_client: TestClient, key: str, payload: dict[str, object] | None, low: float, high: float
) -> None:
    body = real_client.post(f"/v1/{key}/predict", json=payload or example_payload(key)).json()
    assert low <= body["prediction"] <= high, body


def test_domain_relationships_hold(real_client: TestClient) -> None:
    """Directional sanity checks that would break if columns were mixed up."""

    def price(**changes: object) -> float:
        body = {**example_payload("car"), **changes}
        return float(real_client.post("/v1/car/predict", json=body).json()["prediction"])

    assert price(Year=2010) < price(Year=2016)  # newer is worth more
    assert price(Kms_Driven=150000) < price(Kms_Driven=10000)  # fewer kilometres is worth more

    def charge(**changes: object) -> float:
        body = {**example_payload("medical_insurance"), **changes}
        return float(
            real_client.post("/v1/medical_insurance/predict", json=body).json()["prediction"]
        )

    assert charge(smoker="yes") > 2 * charge(smoker="no")

    def risk(**changes: object) -> float:
        body = {**example_payload("diabetes"), **changes}
        return float(real_client.post("/v1/diabetes/predict", json=body).json()["probability"])

    assert risk(Glucose=190) > risk(Glucose=85)


def test_model_cards_are_committed_for_every_model() -> None:
    for key in SPECS:
        assert (ROOT / "docs" / "model_cards" / f"{key}.md").exists()


def test_committed_cards_match_the_trained_artifacts() -> None:
    """If the data or code changed, the committed cards must be regenerated (`make train`)."""
    for key in SPECS:
        card = json.loads((Path(ARTIFACTS) / f"{key}.json").read_text())
        text = (ROOT / "docs" / "model_cards" / f"{key}.md").read_text(encoding="utf-8")
        assert card["data"]["sha256"][:12] in text, f"{key}: card is stale, rerun `make train`"
