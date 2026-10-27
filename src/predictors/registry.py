"""Load trained artifacts for serving and turn validated inputs into predictions.

There is deliberately no fallback model. The original service replaced any pickle that failed to
load with a ``MockModel`` that returned 123.45 for every request. Here a model that cannot be
verified is reported as unavailable and its endpoint answers 503.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from predictors.specs import SPECS, ModelSpec

logger = logging.getLogger(__name__)


class ArtifactError(Exception):
    """An artifact is missing, altered, or was trained with an incompatible library."""


@dataclass(frozen=True)
class LoadedModel:
    spec: ModelSpec
    pipeline: Any
    card: dict[str, Any]

    @property
    def version(self) -> str:
        """Short identifier of the exact artifact, returned with every prediction."""
        return str(self.card["artifact"]["sha256"])[:12]


@dataclass(frozen=True)
class Prediction:
    value: float
    lower: float | None = None
    upper: float | None = None
    probability: float | None = None
    label: str | None = None
    warnings: list[str] = field(default_factory=list)


def load_model(spec: ModelSpec, artifacts_dir: Path) -> LoadedModel:
    """Load and verify one artifact. Raises :class:`ArtifactError` if anything is off."""
    card_path = artifacts_dir / f"{spec.key}.json"
    if not card_path.exists():
        raise ArtifactError(f"no model card at {card_path}; run `make train`")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    artifact = artifacts_dir / card["artifact"]["file"]
    if not artifact.exists():
        raise ArtifactError(f"missing artifact {artifact}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != card["artifact"]["sha256"]:
        raise ArtifactError("artifact checksum does not match its model card")
    trained_with = card["libraries"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        raise ArtifactError(
            f"trained with scikit-learn {trained_with} but running {sklearn.__version__}"
        )
    # Safe only because the checksum above ties this file to the card written by our own training.
    pipeline = joblib.load(artifact)
    # Training uses all cores; one-row requests only pay thread start-up for that.
    if "model__n_jobs" in pipeline.get_params():
        pipeline.set_params(model__n_jobs=1)
    return LoadedModel(spec, pipeline, card)


class ModelRegistry:
    """Holds every model that loaded successfully and the reason for each one that did not."""

    def __init__(self, artifacts_dir: Path, specs: Mapping[str, ModelSpec] | None = None) -> None:
        self.models: dict[str, LoadedModel] = {}
        self.errors: dict[str, str] = {}
        for key, spec in (specs or SPECS).items():
            try:
                self.models[key] = load_model(spec, artifacts_dir)
            except (ArtifactError, OSError, ValueError, KeyError) as exc:
                logger.error("model %s unavailable: %s", key, exc)
                self.errors[key] = str(exc)

    @property
    def ready(self) -> bool:
        return not self.errors and bool(self.models)


def _extrapolation_warnings(model: LoadedModel, values: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    for feat, fspec in zip(model.card["features"], model.spec.features, strict=True):
        value = values.get(feat["name"])
        if feat["kind"] != "number" or value is None:
            continue
        low, high = feat["training_min"], feat["training_max"]
        if value < low or value > high:
            warnings.append(
                f"{fspec.label} = {value:g} is outside the range seen in training "
                f"({low:g} to {high:g}); the estimate is an extrapolation."
            )
    return warnings


def predict(model: LoadedModel, values: Mapping[str, Any]) -> Prediction:
    """Predict from already-validated, named inputs (order does not matter)."""
    spec = model.spec
    row = {name: values.get(name) for name in spec.feature_names}
    frame = pd.DataFrame([row], columns=spec.feature_names)
    for feature in spec.numeric:
        frame[feature.name] = pd.to_numeric(frame[feature.name], errors="coerce")

    warnings = _extrapolation_warnings(model, values)
    if spec.task == "classification":
        probability = float(model.pipeline.predict_proba(frame)[0, 1])
        label = spec.positive_label if probability >= 0.5 else spec.negative_label
        return Prediction(
            value=probability, probability=probability, label=label, warnings=warnings
        )

    value = float(model.pipeline.predict(frame)[0])
    interval = model.card.get("interval")
    lower = upper = None
    if interval:
        lower = max(0.0, value + float(interval["residual_low"]))
        upper = max(0.0, value + float(interval["residual_high"]))
    if not np.isfinite(value):
        raise ArtifactError("model produced a non-finite prediction")
    value = max(0.0, value)
    return Prediction(value=value, lower=lower, upper=upper, warnings=warnings)
