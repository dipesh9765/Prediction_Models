"""Training pipeline: compare candidates by cross-validation, evaluate once on held-out data,
write a versioned artifact and a model card.

Run ``python -m predictors.train --all`` (or ``--model car``).

Methodology, in order, for every model:

1. Load, clean and de-duplicate the data, then split 80/20 (stratified for classification). The
   test set is not touched again until step 4.
2. Build one ``Pipeline`` per candidate. Imputation, scaling and one-hot encoding are *inside* the
   pipeline, so they are fitted on training folds only. The original notebooks fitted the
   diabetes scaler on the full dataset before splitting, which leaks test statistics.
3. Compare candidates with 5-fold cross-validation on the training set and select the best by the
   primary metric (RMSE or ROC AUC).
4. Refit the winner on the whole training set and score the untouched test set next to a naive
   baseline (predict the mean / the class prior).
5. For regression, derive a 90% prediction interval from out-of-fold residuals and measure its
   real coverage on the test set. Compute permutation importance on the test set.
6. Dump the pipeline with joblib and record its SHA-256, the data fingerprint and library
   versions in the card, so serving can refuse an artifact that does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictors.card import render_markdown
from predictors.data import data_sha256, load_dataset
from predictors.specs import SPECS, ModelSpec

logger = logging.getLogger(__name__)

SEED = 42
TEST_SIZE = 0.2
INTERVAL_LEVEL = 0.90

Factory = Callable[[], BaseEstimator]

CANDIDATES: dict[str, dict[str, Factory]] = {
    "regression": {
        "ridge": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=SEED
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=SEED),
    },
    "classification": {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=3, random_state=SEED
        ),
    },
}


def build_pipeline(spec: ModelSpec, estimator: BaseEstimator) -> Pipeline:
    """Preprocessing + estimator as one object, so serving applies exactly what training fitted."""
    transformers: list[tuple[str, Any, list[str]]] = []
    if spec.numeric:
        numeric = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        transformers.append(("numeric", numeric, [f.name for f in spec.numeric]))
    if spec.categorical:
        encoder = OneHotEncoder(
            categories=[list(f.choices) for f in spec.categorical],
            handle_unknown="error",
            sparse_output=False,
        )
        transformers.append(("categorical", encoder, [f.name for f in spec.categorical]))
    pre = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("model", estimator)])


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": _round(root_mean_squared_error(y_true, pred)),
        "mae": _round(mean_absolute_error(y_true, pred)),
        "r2": _round(r2_score(y_true, pred)),
    }


def _classification_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    label = (proba >= 0.5).astype(int)
    return {
        "roc_auc": _round(roc_auc_score(y_true, proba)),
        "accuracy": _round(accuracy_score(y_true, label)),
        "precision": _round(precision_score(y_true, label, zero_division=0)),
        "recall": _round(recall_score(y_true, label, zero_division=0)),
        "f1": _round(f1_score(y_true, label, zero_division=0)),
        "brier": _round(brier_score_loss(y_true, proba)),
    }


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_candidates(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv: KFold | StratifiedKFold,
    scoring: dict[str, str],
    primary: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cross-validate every candidate on the training split; return all results and the best."""
    results: list[dict[str, Any]] = []
    for name in spec.candidates:
        pipe = build_pipeline(spec, CANDIDATES[spec.task][name]())
        t0 = time.perf_counter()
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        entry: dict[str, Any] = {"name": name, "fit_seconds": _round(time.perf_counter() - t0, 2)}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            values = -values if metric == "rmse" else values
            entry[f"cv_{metric}_mean"] = _round(values.mean())
            entry[f"cv_{metric}_std"] = _round(values.std())
        results.append(entry)
        logger.info("%s / %s: cv %s = %s", spec.key, name, primary, entry[f"cv_{primary}_mean"])
    key = f"cv_{primary}_mean"
    pick = max if spec.task == "classification" else min
    return results, pick(results, key=lambda r: r[key])


def train_one(
    spec: ModelSpec,
    out_dir: Path,
    cards_dir: Path | None,
    *,
    cv_folds: int = 5,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Train one model and return its card (also written next to the artifact)."""
    started = time.perf_counter()
    X, y, raw_rows = load_dataset(spec)
    if max_rows is not None and len(X) > max_rows:
        sample = X.sample(n=max_rows, random_state=SEED).index
        X, y = X.loc[sample].reset_index(drop=True), y.loc[sample].reset_index(drop=True)

    is_clf = spec.task == "classification"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y if is_clf else None
    )
    cv = (
        StratifiedKFold(cv_folds, shuffle=True, random_state=SEED)
        if is_clf
        else KFold(cv_folds, shuffle=True, random_state=SEED)
    )
    scoring = (
        {"roc_auc": "roc_auc", "accuracy": "accuracy"}
        if is_clf
        else {"rmse": "neg_root_mean_squared_error", "r2": "r2"}
    )
    primary = "roc_auc" if is_clf else "rmse"

    results, best = _compare_candidates(
        spec, X_train, y_train, cv=cv, scoring=scoring, primary=primary
    )
    pipeline = build_pipeline(spec, CANDIDATES[spec.task][best["name"]]())
    pipeline.fit(X_train, y_train)

    if is_clf:
        baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
        test_metrics = _classification_metrics(y_test, pipeline.predict_proba(X_test)[:, 1])
        baseline_metrics = _classification_metrics(y_test, baseline.predict_proba(X_test)[:, 1])
        interval: dict[str, float] | None = None
        importance_scoring = "roc_auc"
    else:
        baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        test_metrics = _regression_metrics(y_test, pred)
        baseline_metrics = _regression_metrics(y_test, baseline.predict(X_test))
        oof = cross_val_predict(clone(pipeline), X_train, y_train, cv=cv)
        residuals = y_train.to_numpy() - oof
        low_q, high_q = (1 - INTERVAL_LEVEL) / 2, 1 - (1 - INTERVAL_LEVEL) / 2
        q_low, q_high = (float(q) for q in np.quantile(residuals, [low_q, high_q]))
        inside = (y_test.to_numpy() >= pred + q_low) & (y_test.to_numpy() <= pred + q_high)
        interval = {
            "level": INTERVAL_LEVEL,
            "residual_low": _round(q_low),
            "residual_high": _round(q_high),
            "observed_test_coverage": _round(inside.mean()),
        }
        importance_scoring = "neg_root_mean_squared_error"

    imp = permutation_importance(
        pipeline, X_test, y_test, scoring=importance_scoring, n_repeats=5, random_state=SEED
    )
    importance = {
        name: _round(score)
        for name, score in sorted(
            zip(spec.feature_names, imp.importances_mean, strict=True), key=lambda kv: -kv[1]
        )
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"{spec.key}.joblib"
    joblib.dump(pipeline, artifact, compress=3)

    card: dict[str, Any] = {
        "key": spec.key,
        "title": spec.title,
        "task": spec.task,
        "description": spec.description,
        "target": {"name": spec.target, "label": spec.target_label, "unit": spec.target_unit},
        "trained_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "seed": SEED,
        "libraries": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data": {
            "files": [p.relative_to(p.parents[1]).as_posix() for p in spec.data_paths()],
            "sha256": data_sha256(spec),
            "raw_rows": raw_rows,
            "rows_used": len(X),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "positive_rate": _round(y.mean()) if is_clf else None,
        },
        "features": [
            {
                "name": f.name,
                "kind": f.kind,
                "unit": f.unit,
                "choices": list(f.choices),
                "optional": f.optional,
                "training_min": _round(X_train[f.name].min()) if f.kind == "number" else None,
                "training_max": _round(X_train[f.name].max()) if f.kind == "number" else None,
            }
            for f in spec.features
        ],
        "validation": {"cv_folds": cv_folds, "primary_metric": primary, "candidates": results},
        "selected": best["name"],
        "metrics": {"test": test_metrics, "baseline": baseline_metrics},
        "interval": interval,
        "permutation_importance": importance,
        "importance_metric": importance_scoring,
        "limitations": list(spec.limitations),
        "disclaimer": spec.disclaimer,
        "artifact": {"file": artifact.name, "sha256": _fingerprint(artifact)},
        "train_seconds": _round(time.perf_counter() - started, 1),
    }
    (out_dir / f"{spec.key}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    if cards_dir is not None:
        cards_dir.mkdir(parents=True, exist_ok=True)
        (cards_dir / f"{spec.key}.md").write_text(render_markdown(card, spec), encoding="utf-8")
    return card


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="train every model")
    group.add_argument("--model", choices=sorted(SPECS), help="train one model")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--cards", type=Path, default=Path("docs/model_cards"))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None, help="subsample (for quick runs)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keys = sorted(SPECS) if args.all else [args.model]
    for key in keys:
        card = train_one(SPECS[key], args.out, args.cards, cv_folds=args.cv, max_rows=args.max_rows)
        test = card["metrics"]["test"]
        logger.info("== %s: selected %s, test %s", key, card["selected"], test)


if __name__ == "__main__":
    main()
