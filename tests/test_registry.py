from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from predictors.registry import ArtifactError, ModelRegistry, load_model, predict
from predictors.specs import SPECS

from .conftest import example_payload


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_registry_loads_every_model(small_artifacts: Path) -> None:
    registry = ModelRegistry(small_artifacts)
    assert registry.ready
    assert set(registry.models) == set(SPECS)
    assert registry.errors == {}


def test_missing_artifacts_are_reported_not_replaced_by_a_mock(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    assert not registry.ready
    assert registry.models == {}
    assert "make train" in registry.errors["car"]


def test_a_tampered_artifact_is_refused(small_artifacts: Path, tmp_path: Path) -> None:
    directory = _copy(small_artifacts, tmp_path / "t")
    with (directory / "car.joblib").open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ArtifactError, match="checksum"):
        load_model(SPECS["car"], directory)
    registry = ModelRegistry(directory)
    assert "car" in registry.errors
    assert "diabetes" in registry.models  # one bad file does not take the others down


def test_an_artifact_from_another_scikit_learn_version_is_refused(
    small_artifacts: Path, tmp_path: Path
) -> None:
    directory = _copy(small_artifacts, tmp_path / "v")
    card_path = directory / "car.json"
    card = json.loads(card_path.read_text())
    card["libraries"]["scikit-learn"] = "1.0.2"
    card_path.write_text(json.dumps(card))
    with pytest.raises(ArtifactError, match=r"trained with scikit-learn 1\.0\.2"):
        load_model(SPECS["car"], directory)


def test_a_card_without_its_artifact_is_refused(small_artifacts: Path, tmp_path: Path) -> None:
    directory = _copy(small_artifacts, tmp_path / "m")
    (directory / "car.joblib").unlink()
    with pytest.raises(ArtifactError, match="missing artifact"):
        load_model(SPECS["car"], directory)


def test_a_corrupt_card_is_reported(small_artifacts: Path, tmp_path: Path) -> None:
    directory = _copy(small_artifacts, tmp_path / "c")
    (directory / "car.json").write_text("{not json")
    assert "car" in ModelRegistry(directory).errors


@pytest.fixture
def models(small_artifacts: Path) -> ModelRegistry:
    return ModelRegistry(small_artifacts)


@pytest.mark.parametrize("key", sorted(SPECS))
def test_every_model_predicts_from_its_example(models: ModelRegistry, key: str) -> None:
    result = predict(models.models[key], example_payload(key))
    assert result.value >= 0
    if SPECS[key].task == "classification":
        assert result.probability is not None
        assert 0 <= result.probability <= 1
        assert result.label in {SPECS[key].positive_label, SPECS[key].negative_label}
    else:
        assert result.lower is not None
        assert result.upper is not None
        assert result.lower <= result.upper


def test_prediction_does_not_depend_on_dict_order(models: ModelRegistry) -> None:
    payload = example_payload("car")
    reordered = dict(reversed(list(payload.items())))
    model = models.models["car"]
    assert predict(model, payload).value == predict(model, reordered).value


def test_each_feature_changes_the_prediction_it_should(models: ModelRegistry) -> None:
    """A mixed-up column order would make an obviously monotone feature behave oddly."""
    model = models.models["exercise"]
    short = predict(model, {**example_payload("exercise"), "Duration": 5}).value
    long = predict(model, {**example_payload("exercise"), "Duration": 60}).value
    assert long > short


def test_optional_features_can_be_missing(models: ModelRegistry) -> None:
    payload = example_payload("diabetes")
    complete = predict(models.models["diabetes"], payload)
    partial = predict(
        models.models["diabetes"], {**payload, "Insulin": None, "SkinThickness": None}
    )
    assert 0 <= partial.value <= 1
    assert partial.value != complete.value


def test_values_outside_the_training_range_produce_a_warning(models: ModelRegistry) -> None:
    result = predict(models.models["car"], {**example_payload("car"), "Year": 1992})
    assert any("Year" in w and "extrapolation" in w for w in result.warnings)
    assert predict(models.models["car"], example_payload("car")).warnings == []


class _FakePipeline:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: Any) -> list[float]:
        return [self.value]


def test_negative_predictions_are_clipped_to_zero(models: ModelRegistry) -> None:
    model = replace(models.models["car"], pipeline=_FakePipeline(-3.0))
    result = predict(model, example_payload("car"))
    assert result.value == 0.0
    assert result.lower == 0.0


def test_a_non_finite_prediction_is_an_error(models: ModelRegistry) -> None:
    model = replace(models.models["car"], pipeline=_FakePipeline(float("nan")))
    with pytest.raises(ArtifactError):
        predict(model, example_payload("car"))


def test_version_is_a_short_checksum(models: ModelRegistry) -> None:
    model = models.models["car"]
    assert model.version == model.card["artifact"]["sha256"][:12]
