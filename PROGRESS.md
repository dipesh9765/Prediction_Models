# Progress log

Branch: `overhaul/v1` (nothing pushed), tag `v1.0.0`. Baseline at the start: a Flask app serving five pickled models, six hand-copied static HTML pages, notebooks, no tests, and a CI workflow that ran `python manage.py test` on Python 3.7-3.9 (this is not a Django project, so it could never pass).

## Audit

### The deployed models could not work as shipped (verified by loading them)
| Finding | Evidence |
|---------|----------|
| All 10 pickles were produced with scikit-learn **1.0.2 / 1.1.2**; `requirements.txt` pinned **1.8.0** | `InconsistentVersionWarning` on every scaler and linear/SVC model |
| The insurance GradientBoosting pickle **cannot be unpickled at all** | `ModuleNotFoundError: sklearn.ensemble._gb_losses` |
| `views.py` swallowed the failure and substituted `MockModel`, which returns **123.45 for every input** | `load_pickle_model(...) or MockModel()` |
| Pickles were loaded blindly from the repo | code-execution surface, no integrity check |

### ML methodology
- Data leakage in the diabetes notebook: `StandardScaler.fit(X)` on all rows before `train_test_split`.
- One random split per model, no cross-validation, no baseline, no recorded comparison.
- The house model was not reproducible and ethically unsuitable: `load_boston` was removed from scikit-learn 1.2 because column `B` is derived from the proportion of Black residents; no CSV existed.
- Pima zeros (missing measurements) used as real values.
- Duplicate rows (2 car, 1 insurance) could straddle the split.
- Health and insurance predictors had no disclaimer or model card.

### API and code quality
- Features were read with `list(data.values())`: the model got them in client order, with no names, types or ranges.
- Five models x two endpoints, copy-pasted (~250 lines); inconsistent routes; dead 404 routes.
- `except Exception` returned `str(e)` to clients with HTTP 500, including for bad input.
- No health endpoint, no request logging, `debug=True`.
- Encodings (`Petrol=0, Diesel=1, CNG=2`) existed only in notebook cells and six HTML files.

## Delivered

| Step | Result |
|------|--------|
| A Structure and tooling | `src/predictors` package, pyproject, pinned requirements, ruff, strict mypy, pytest-cov, Makefile; pickles, Flask site, six HTML pages and the broken workflows removed; notebooks archived |
| B Training pipeline | `ModelSpec` per model; sklearn `Pipeline`; 5-fold CV over three candidates; test vs baseline; 90% intervals with measured coverage; permutation importance; JSON and Markdown model cards; quality gate |
| C Housing data | California Housing replaces Boston (ADR 0004) |
| D API | FastAPI, one typed route per model generated from the spec, SHA-256 and version-checked artifacts, no mock, request ids, security headers, no input logging, safe validation errors |
| E UI | One page generated from the API schema (no inline script, strict CSP) |
| F Tests, Docker, CI | 104 tests (97.5% coverage), Dockerfile that trains and gates inside the build, CI on Python 3.12 and 3.13 plus a Docker smoke job |
| G Docs | README, ARCHITECTURE, 4 ADRs, 5 model cards, SECURITY, CONTRIBUTING, CHANGELOG, LICENSE, data provenance |

### Found while building (each has a regression test)
- A request containing `NaN` made FastAPI's own 422 response crash with a 500 (it tried to echo the non-finite input). Replaced by a handler that returns only field, message and type.
- `max(0.0, nan)` returns `0.0`, so a NaN model output would have been silently clipped to a valid-looking zero; the finiteness check now runs first.
- The first housing input ranges rejected real training values (census outliers such as 1,243 occupants per household); a test now asserts every training value is inside the declared range.
- A browser cached a stale `app.js` after a change; UI assets now send `Cache-Control: no-cache`.
- Inference was slow (4 s first request) because forests were serialised with `n_jobs=-1`; they are forced to one thread at load.
- Exercise de-duplication dropped a legitimate record (two people with identical measurements); de-duplication now happens before `User_ID` is removed.

### Results (held-out test, from the generated cards)
| Model | Selected | Test | Baseline | CV |
|-------|----------|------|----------|----|
| car | random forest | RMSE 3.17, R2 0.61 (60 rows) | RMSE 5.08 | R2 0.89 |
| diabetes | logistic regression | AUC 0.81 | AUC 0.50 | AUC 0.84 |
| exercise | gradient boosting | RMSE 1.95, R2 0.999 | RMSE 63.6 | R2 0.999 |
| medical insurance | random forest | RMSE 4,449, R2 0.89 | RMSE 13,612 | R2 0.83 |
| california housing | gradient boosting | RMSE 0.46, R2 0.84 | RMSE 1.14 | R2 0.83 |

## Verified
`ruff check`, `ruff format --check`, strict `mypy`, `pytest --cov` (104 passed, 97.5%) and `check_cards` all pass. The UI was exercised in a real browser (car and diabetes tabs: form prefilled, prediction, interval, disclaimer).

## Not verified / open items
- **The Docker image was never built** (no Docker on this machine). The CI `docker` job builds it and smoke-tests it; the first run may need a fix.
- No screenshots in the README.
- Car test metrics are noisy (60 test rows) and its interval covered 80% of them against a 90% target; documented in the card.
- Dataset licences and sources are best-effort attributions (`data/README.md`) and need checking before redistribution.
- No authentication, rate limiting or drift monitoring; artifacts are checksum-verified, not signed.
- LICENSE is MIT (the original repository had none).

## Status
- [x] Audit
- [x] A Structure and tooling
- [x] B Training pipeline and model cards
- [x] C Housing dataset
- [x] D API
- [x] E UI
- [x] F Tests, Docker, CI
- [x] G Docs
