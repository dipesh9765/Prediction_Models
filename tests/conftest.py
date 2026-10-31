"""Shared fixtures. Unit and API tests train small models into a temp directory, so they run in
seconds and never depend on ./artifacts. Tests marked ``integration`` use the real artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from predictors.api import create_app
from predictors.specs import SPECS
from predictors.train import train_one

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def small_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """All five models trained on a 500-row sample with 3-fold CV."""
    out = tmp_path_factory.mktemp("artifacts")
    cards = out / "cards"
    for spec in SPECS.values():
        train_one(spec, out, cards, cv_folds=3, max_rows=500)
    return out


@pytest.fixture
def client(small_artifacts: Path) -> TestClient:
    with TestClient(create_app(small_artifacts)) as test_client:
        return test_client


@pytest.fixture
def empty_client(tmp_path: Path) -> TestClient:
    """An app whose artifact directory is empty: every model is unavailable."""
    with TestClient(create_app(tmp_path)) as test_client:
        return test_client


def example_payload(key: str) -> dict[str, object]:
    """A valid request built from the spec's own examples (optional fields omitted)."""
    return {f.name: f.example for f in SPECS[key].features if f.example is not None}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if (ROOT / "artifacts" / "car.json").exists():
        return
    skip = pytest.mark.skip(reason="no trained artifacts in ./artifacts (run `make train`)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
