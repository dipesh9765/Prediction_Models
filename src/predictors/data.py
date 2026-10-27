"""Dataset loading and cleaning. One function per dataset, selected by model key."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from predictors.specs import ModelSpec

#: Pima columns where the source data uses 0 to mean "not measured".
PIMA_ZERO_IS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def _car(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return frames[0].drop(columns=["Car_Name"])


def _diabetes(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = frames[0].copy()
    df[list(PIMA_ZERO_IS_MISSING)] = df[list(PIMA_ZERO_IS_MISSING)].replace(0, np.nan)
    return df


def _exercise(frames: list[pd.DataFrame]) -> pd.DataFrame:
    exercise, calories = frames
    merged = exercise.merge(calories, on="User_ID", how="inner", validate="one_to_one")
    # User_ID is kept until de-duplication: two different people with identical measurements
    # are two valid records, not a duplicate.
    return merged


def _passthrough(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return frames[0]


_CLEANERS: dict[str, Callable[[list[pd.DataFrame]], pd.DataFrame]] = {
    "car": _car,
    "diabetes": _diabetes,
    "exercise": _exercise,
    "medical_insurance": _passthrough,
    "california_housing": _passthrough,
}


def data_sha256(spec: ModelSpec) -> str:
    """Fingerprint of the raw files, so a model card says exactly which data it was trained on."""
    digest = hashlib.sha256()
    for path in spec.data_paths():
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_dataset(spec: ModelSpec) -> tuple[pd.DataFrame, pd.Series, int]:
    """Return ``(features, target, raw_rows)`` after cleaning and removing duplicate rows.

    Duplicates are dropped before any split so the same record can never sit on both sides of the
    train/test boundary.
    """
    frames = [pd.read_csv(path) for path in spec.data_paths()]
    df = _CLEANERS[spec.key](frames)
    raw_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    missing = [c for c in [*spec.feature_names, spec.target] if c not in df.columns]
    if missing:
        raise ValueError(f"{spec.key}: dataset is missing columns {missing}")
    return df[spec.feature_names], df[spec.target], raw_rows
