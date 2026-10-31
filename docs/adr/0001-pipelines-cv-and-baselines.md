# ADR 0001: Preprocessing inside the pipeline, cross-validated selection, baseline for every metric

Status: accepted

## Context
The original notebooks scaled features outside the model and pickled the scaler and the estimator separately. The diabetes notebook called `scaler.fit(X)` on the whole dataset *before* `train_test_split`, so test-set statistics leaked into training. Each model was evaluated on one random split (60 rows for the car data), compared with nothing, and the number that shipped was whatever the last cell printed.

## Decision
- One scikit-learn `Pipeline` per model: `ColumnTransformer` (median imputer + `StandardScaler` for numbers, `OneHotEncoder` with categories fixed by the spec) followed by the estimator. Everything is fitted on training data only; serving applies the identical object.
- Compare three candidates per task by 5-fold cross-validation on the training split (RMSE for regression, ROC AUC for classification), pick the best, refit, and evaluate on the untouched 20% **once**.
- Report every metric beside a `DummyRegressor`/`DummyClassifier` baseline, and fail CI (`check_cards`) if the model is not clearly better and its cross-validated R² / AUC is below the floor declared in its spec.
- Remove exact duplicates before splitting so a record cannot be on both sides.
- Treat Pima zeros in glucose, blood pressure, skin thickness, insulin and BMI as missing.

## Alternatives tried and rejected
- **Log-transformed target** for car and insurance prices (skewed). Cross-validated RMSE improved for the tree models (the two random forests that were later selected by only 1.5% on cars and 2.8% on insurance; gradient boosting on cars by 11%) but got much worse for ridge (car: 1.64 to 20.9, because `expm1` amplifies extrapolation errors). The gain for the winning models was too small to justify the extra moving part; selection stays on the raw target.
- **Hyper-parameter search.** Defaults were close enough that a search would mostly fit noise on 299 to 1,337 rows; the cards show the CV standard deviations, which are larger than the differences a search would find.

## Consequences
- Results are reproducible (fixed seed, deterministic test) and comparable across models.
- Small datasets still give noisy test scores; the card says when the test split has fewer than 500 rows and points to the CV table instead.
