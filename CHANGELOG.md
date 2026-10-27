# Changelog

## 1.0.0

### Added
- One reproducible training pipeline: scikit-learn `Pipeline` per model, 5-fold cross-validated candidate selection, held-out test results beside a naive baseline, 90% prediction intervals with measured coverage, permutation importance.
- Generated model cards (`docs/model_cards`) and JSON cards with data fingerprint, library versions and artifact checksum.
- Quality gate (`predictors.check_cards`) used in CI and the Docker build.
- FastAPI service with one typed route per model, OpenAPI schemas, `/healthz`, `/readyz`, `/v1/models`, request ids and security headers.
- A single web UI generated from the API schema.
- 104 tests (97.5% coverage), ruff, strict mypy, Dockerfile, CI workflow, ADRs, SECURITY and CONTRIBUTING.

### Changed
- Gradient boosting now comes from scikit-learn (`HistGradientBoosting`); the XGBoost dependency is gone.
- The house model now predicts California housing values (see ADR 0004).
- Duplicate rows are removed before splitting (2 in the car data, 1 in the insurance data).
- Routes are now `POST /v1/{model}/predict` with named JSON fields.

### Fixed
- The shipped pickles (scikit-learn 1.0.2/1.1.2) did not match the pinned library and the insurance model could not be unpickled; the service silently answered `123.45` through a `MockModel`.
- The diabetes scaler was fitted on all rows before the train/test split.
- Features were passed to models in client key order, with no names, types or range checks.
- Failures returned HTTP 500 with the raw exception text.
- Pima zeros (missing measurements) were treated as real values.
- CI ran `python manage.py test` on Python 3.7-3.9 in a project that is not Django.

### Removed
- The Flask app, `MockModel`, six duplicated HTML pages, the GitHub Pages workflow, `Procfile`, the manual `test_api.py`, the ten pickle files, and the Boston housing data dependency. Notebooks moved to `notebooks/legacy`.
