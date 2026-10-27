# Legacy notebooks

The original exploratory notebooks, kept for reference. They are **not** the source of the served models and should not be re-run as a recipe:

- they fit scalers outside the model (and, for diabetes, on all rows before the split),
- they evaluate on a single random split,
- `house_boston_xgboost.ipynb` needs `sklearn.datasets.load_boston`, which no longer exists (see [ADR 0004](../../docs/adr/0004-replace-boston-housing.md)),
- they read CSVs from their old folders, which moved to `data/`.

The supported pipeline is `python -m predictors.train --all`.
