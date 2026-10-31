from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from predictors import train
from predictors.check_cards import main as check_main
from predictors.check_cards import problems_for
from predictors.data import load_dataset
from predictors.specs import SPECS


def test_scaler_is_fitted_on_the_training_rows_only() -> None:
    """Regression test for the original diabetes notebook, which fitted its scaler on all rows."""
    spec = SPECS["diabetes"]
    X, y, _ = load_dataset(spec)
    first, rest = X.iloc[:100], X.iloc[100:]
    pipeline = train.build_pipeline(spec, LinearRegression())
    pipeline.fit(first, y.iloc[:100])
    scaler = pipeline.named_steps["pre"].named_transformers_["numeric"].named_steps["scale"]
    expected = first.median().to_numpy()  # imputed with the training median first
    imputed = first.fillna(first.median())
    np.testing.assert_allclose(scaler.mean_, imputed.mean().to_numpy())
    assert not np.allclose(scaler.mean_, rest.fillna(rest.median()).mean().to_numpy())
    assert expected.shape == scaler.mean_.shape


def test_pipeline_rejects_unknown_categories_instead_of_guessing() -> None:
    spec = SPECS["car"]
    X, y, _ = load_dataset(spec)
    pipeline = train.build_pipeline(spec, LinearRegression()).fit(X, y)
    bad = X.iloc[[0]].copy()
    bad["Fuel_Type"] = "Hydrogen"
    with pytest.raises(ValueError, match="Hydrogen"):
        pipeline.predict(bad)


def test_pipeline_handles_missing_optional_values() -> None:
    spec = SPECS["diabetes"]
    X, y, _ = load_dataset(spec)
    pipeline = train.build_pipeline(
        spec, train.CANDIDATES["classification"]["logistic_regression"]()
    )
    pipeline.fit(X, y)
    row = X.iloc[[0]].copy()
    row[["Insulin", "SkinThickness"]] = np.nan
    assert 0.0 <= pipeline.predict_proba(row)[0, 1] <= 1.0


@pytest.fixture(scope="module")
def car_card(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path]:
    out = tmp_path_factory.mktemp("car")
    return train.train_one(SPECS["car"], out, out, cv_folds=3, max_rows=250), out


def test_card_records_provenance_and_artifact_checksum(
    car_card: tuple[dict[str, object], Path],
) -> None:
    card, out = car_card
    artifact = out / "car.joblib"
    assert card["artifact"] == {
        "file": "car.joblib",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    assert json.loads((out / "car.json").read_text()) == card
    data: Any = card["data"]
    assert data["train_rows"] + data["test_rows"] == data["rows_used"]
    assert len(data["sha256"]) == 64
    assert card["libraries"]["scikit-learn"]  # type: ignore[index]


def test_selected_model_is_the_best_cv_candidate(car_card: tuple[dict[str, object], Path]) -> None:
    card, _ = car_card
    candidates = card["validation"]["candidates"]  # type: ignore[index]
    best = min(candidates, key=lambda c: c["cv_rmse_mean"])
    assert card["selected"] == best["name"]
    assert [c["name"] for c in candidates] == list(SPECS["car"].candidates)


def test_regression_card_has_interval_and_importance(
    car_card: tuple[dict[str, object], Path],
) -> None:
    card, _ = car_card
    interval = card["interval"]
    assert interval["residual_low"] < 0 < interval["residual_high"]  # type: ignore[index]
    assert 0 <= interval["observed_test_coverage"] <= 1  # type: ignore[index]
    assert set(card["permutation_importance"]) == set(SPECS["car"].feature_names)  # type: ignore[call-overload]


def test_markdown_card_contains_the_essentials(car_card: tuple[dict[str, object], Path]) -> None:
    _, out = car_card
    text = (out / "car.md").read_text(encoding="utf-8")
    for needle in (
        "# Model card:",
        "Held-out test results",
        "Naive baseline",
        "## Limitations",
        "Seen in training",
        "Reproduce",
    ):
        assert needle in text
    assert "\\" not in text.split("## Summary")[1].split("## Held-out")[0], "no windows paths"


def test_health_models_carry_a_disclaimer(small_artifacts: Path) -> None:
    for key in ("diabetes", "medical_insurance"):
        text = (small_artifacts / "cards" / f"{key}.md").read_text(encoding="utf-8")
        assert "Not for real-world use" in text


def test_classification_card_has_no_interval(small_artifacts: Path) -> None:
    card = json.loads((small_artifacts / "diabetes.json").read_text())
    assert card["interval"] is None
    assert {"roc_auc", "recall", "brier"} <= set(card["metrics"]["test"])


def test_training_is_deterministic(tmp_path: Path) -> None:
    a = train.train_one(SPECS["diabetes"], tmp_path / "a", None, cv_folds=3, max_rows=300)
    b = train.train_one(SPECS["diabetes"], tmp_path / "b", None, cv_folds=3, max_rows=300)
    assert a["metrics"] == b["metrics"]
    assert (
        a["validation"]["candidates"][0]["cv_roc_auc_mean"]
        == (b["validation"]["candidates"][0]["cv_roc_auc_mean"])
    )


def test_train_cli_writes_artifacts_and_cards(tmp_path: Path) -> None:
    train.main(
        [
            "--model",
            "car",
            "--out",
            str(tmp_path / "a"),
            "--cards",
            str(tmp_path / "c"),
            "--cv",
            "3",
            "--max-rows",
            "200",
        ]
    )
    assert (tmp_path / "a" / "car.joblib").exists()
    assert (tmp_path / "c" / "car.md").exists()


def test_cli_requires_a_target() -> None:
    with pytest.raises(SystemExit):
        train.main([])


# ---- quality gate ---------------------------------------------------------------------------


def _card(task: str, **overrides: object) -> dict[str, object]:
    if task == "regression":
        card: dict[str, object] = {
            "task": "regression",
            "selected": "m",
            "validation": {"candidates": [{"name": "m", "cv_r2_mean": 0.9}]},
            "metrics": {"test": {"rmse": 1.0}, "baseline": {"rmse": 3.0}},
            "interval": {"observed_test_coverage": 0.9},
            "data": {"test_rows": 1000},
        }
    else:
        card = {
            "task": "classification",
            "selected": "m",
            "validation": {"candidates": [{"name": "m", "cv_roc_auc_mean": 0.85}]},
            "metrics": {"test": {"roc_auc": 0.84}, "baseline": {"roc_auc": 0.5}},
            "interval": None,
            "data": {"test_rows": 150},
        }
    card.update(overrides)
    return card


def test_gate_passes_a_good_card() -> None:
    assert problems_for(_card("regression"), 0.8) == []
    assert problems_for(_card("classification"), 0.8) == []


def test_gate_flags_each_failure_mode() -> None:
    low_cv = _card("regression", validation={"candidates": [{"name": "m", "cv_r2_mean": 0.5}]})
    assert any("below the floor" in p for p in problems_for(low_cv, 0.8))

    no_better = _card("regression", metrics={"test": {"rmse": 2.9}, "baseline": {"rmse": 3.0}})
    assert any("baseline" in p for p in problems_for(no_better, 0.8))

    bad_cover = _card("regression", interval={"observed_test_coverage": 0.6})
    assert any("coverage" in p for p in problems_for(bad_cover, 0.8))

    weak_auc = _card(
        "classification", metrics={"test": {"roc_auc": 0.55}, "baseline": {"roc_auc": 0.5}}
    )
    assert any("baseline" in p for p in problems_for(weak_auc, 0.8))


def test_gate_ignores_coverage_on_tiny_test_sets() -> None:
    card = _card("regression", interval={"observed_test_coverage": 0.6}, data={"test_rows": 60})
    assert problems_for(card, 0.8) == []


def test_gate_cli_reports_missing_cards(tmp_path: Path) -> None:
    assert check_main(["--artifacts", str(tmp_path)]) == 1
