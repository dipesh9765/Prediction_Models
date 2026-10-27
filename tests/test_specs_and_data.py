from __future__ import annotations

import pandas as pd
import pytest

from predictors.data import PIMA_ZERO_IS_MISSING, data_sha256, load_dataset
from predictors.specs import SPECS, Feature, ModelSpec


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_spec_examples_satisfy_their_own_constraints(spec: ModelSpec) -> None:
    names = [f.name for f in spec.features]
    assert len(names) == len(set(names))
    for f in spec.features:
        if f.kind == "number":
            assert f.example is not None
            assert f.minimum is not None
            assert f.maximum is not None
            assert f.minimum <= float(f.example) <= f.maximum
        else:
            assert f.example in f.choices


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_dataset_matches_spec(spec: ModelSpec) -> None:
    X, y, raw_rows = load_dataset(spec)
    assert list(X.columns) == spec.feature_names
    assert len(X) == len(y) <= raw_rows
    assert not y.isna().any()
    # no exact duplicate rows may survive, otherwise a record could be on both sides of the split
    if spec.key != "exercise":  # distinct people may share identical measurements
        assert not pd.concat([X, y], axis=1).duplicated().any()


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_data_inside_the_declared_valid_range(spec: ModelSpec) -> None:
    """The API must not reject values the model was trained on."""
    X, _, _ = load_dataset(spec)
    for f in spec.numeric:
        column = X[f.name].dropna()
        assert column.min() >= f.minimum, f"{f.name} below declared minimum"
        assert column.max() <= f.maximum, f"{f.name} above declared maximum"
    for f in spec.categorical:
        assert set(X[f.name].unique()) <= set(f.choices)


def test_pima_zero_is_treated_as_missing() -> None:
    X, _, _ = load_dataset(SPECS["diabetes"])
    raw = pd.read_csv(SPECS["diabetes"].data_paths()[0])
    for column in PIMA_ZERO_IS_MISSING:
        assert X[column].isna().sum() == (raw[column] == 0).sum()
        assert not (X[column] == 0).any()
    # pregnancies legitimately contains zero and must be untouched
    assert (X["Pregnancies"] == 0).any()


def test_exercise_join_keeps_every_user_once() -> None:
    X, y, raw_rows = load_dataset(SPECS["exercise"])
    assert raw_rows == 15000
    assert len(X) == 15000
    assert y.notna().all()


def test_duplicates_are_removed_from_car_and_insurance_data() -> None:
    assert load_dataset(SPECS["car"])[2] - len(load_dataset(SPECS["car"])[0]) == 2
    assert (
        load_dataset(SPECS["medical_insurance"])[2]
        - len(load_dataset(SPECS["medical_insurance"])[0])
        == 1
    )


def test_data_fingerprint_is_stable_and_distinct() -> None:
    hashes = {key: data_sha256(spec) for key, spec in SPECS.items()}
    assert hashes == {key: data_sha256(spec) for key, spec in SPECS.items()}
    assert len(set(hashes.values())) == len(hashes)


def test_housing_data_has_no_demographic_proxy_column() -> None:
    """The replaced Boston dataset had column B, derived from the share of Black residents."""
    columns = set(pd.read_csv(SPECS["california_housing"].data_paths()[0]).columns)
    assert "B" not in columns
    assert columns == {f.name for f in SPECS["california_housing"].features} | {"MedHouseVal"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "category"},  # no choices
        {"kind": "number"},  # no range
        {"kind": "number", "minimum": 0.0},  # half a range
    ],
)
def test_incomplete_feature_definitions_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="need"):
        Feature(name="x", label="x", **kwargs)  # type: ignore[arg-type]
