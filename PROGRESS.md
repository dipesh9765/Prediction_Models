# Progress log

Branch: `overhaul/v1` (nothing pushed). Baseline: a Flask app serving five pickled scikit-learn/XGBoost models, six hand-copied static HTML pages, notebooks, no tests, a CI workflow that runs `python manage.py test` on Python 3.7-3.9 (this is not a Django project, so it can never pass).

## Audit

### The deployed models cannot work as shipped (verified by loading them)
| Finding | Evidence |
|---------|----------|
| All 10 pickles were produced with scikit-learn **1.0.2 / 1.1.2**; `requirements.txt` pins **1.8.0** | `InconsistentVersionWarning` on every scaler and linear/SVC model |
| The medical-insurance GradientBoosting pickle **cannot be unpickled at all** | `ModuleNotFoundError: sklearn.ensemble._gb_losses` (module removed after 1.0) |
| `views.py` swallows the failure and substitutes `MockModel`, which returns **123.45 for every input** | `load_pickle_model(...) or MockModel()` |
| Pickle files are loaded blindly from the repo | arbitrary-code-execution surface if an artifact is ever replaced; no integrity check |

### ML methodology
- **Data leakage** in the diabetes notebook: `StandardScaler.fit(X)` is called on the whole dataset *before* `train_test_split`.
- One random split per model, no cross-validation, no baseline, no model comparison recorded; the metric that shipped is whatever the last notebook cell printed.
- **House model is not reproducible and ethically unsuitable**: it uses `sklearn.datasets.load_boston`, removed from scikit-learn 1.2 because the dataset contains a feature (`B`) derived from the proportion of Black residents. There is no CSV in the repo.
- Diabetes data uses `0` as a missing-value code for glucose, blood pressure, skin thickness, insulin and BMI; nothing handles it.
- Car data has duplicate rows spanning the train/test boundary; 301 rows with a single split gives a very noisy estimate.
- Medical and diabetes are health-related predictors with no disclaimer, model card, or statement of intended use.

### API and code quality
- Inputs are read with `list(data.values())`: the model gets features **in whatever order the client sent them**, with no names, no types, no range checks. The form endpoints use `request.form.values()` the same way.
- Five models x two endpoints each, copy-pasted (~250 lines); inconsistent routes (`/predict`, `/d_predict`, `/Diabetes/d_predict_api`, `/Medical/m_predict_api`) and dead "deprecated" routes returning 404.
- `except Exception` returns `str(e)` to the client with HTTP 500, including for bad input.
- No health endpoint, no request logging, no config, `debug=True` in `main.py`.
- Categorical encodings (e.g. `Petrol=0, Diesel=1, CNG=2`) live only in notebook cells and in six duplicated HTML files, so they can drift.
- `test_api.py` is a manual script that needs a running server.

## Plan
- **A. Structure and tooling.** `src/predictors` package, `data/` with provenance, notebooks archived, pyproject, ruff, mypy, pytest-cov, Makefile; delete pickles, Flask site and the broken workflows.
- **B. Training pipeline.** One declarative `ModelSpec` per model (typed features with valid ranges, target, candidates). Training builds a scikit-learn `Pipeline` (imputer, scaler, one-hot) so preprocessing cannot leak or drift, compares candidates with 5-fold CV, evaluates once on a held-out test set against a baseline, estimates a 90% prediction interval from out-of-fold residuals, computes permutation importance, and writes a versioned artifact plus a **model card** (JSON and Markdown).
- **C. Replace the house data** with California Housing (public domain, no demographic proxies).
- **D. Typed inference API (FastAPI).** One route per model generated from the spec, pydantic validation with named fields, categorical choices and ranges; extrapolation warnings; artifact SHA-256 and library-version check on load; no mock fallback (a missing/invalid artifact is reported by `/readyz` and returns 503); request-ID logging that never records feature values.
- **E. One generated web UI** (replaces six HTML files) built from the API's own schema.
- **F. Tests, Docker, CI.** Unit and API tests, a quality gate on the model cards, multi-stage Docker that trains at build time, GitHub Actions.
- **G. Docs.** README, ARCHITECTURE, ADRs, model cards, SECURITY, CONTRIBUTING, CHANGELOG, LICENSE, resume bullets.

## Status
- [x] Audit
- [ ] A Structure and tooling
- [ ] B Training pipeline and model cards
- [ ] C Housing dataset
- [ ] D API
- [ ] E UI
- [ ] F Tests, Docker, CI
- [ ] G Docs
