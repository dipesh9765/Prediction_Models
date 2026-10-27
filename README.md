# Prediction Models

**Five machine-learning models trained by one reproducible pipeline, documented by generated model cards, and served through a typed FastAPI service with a web UI that builds itself from the API schema.**

[![CI](https://github.com/dipesh9765/Prediction_Models/actions/workflows/ci.yml/badge.svg)](https://github.com/dipesh9765/Prediction_Models/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-typed%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

| Model | Task | Selected by CV | Held-out test | Naive baseline | Card |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Used car price | regression, 299 rows | random forest | RMSE 3.17 lakh, R² 0.61 (CV R² **0.89**) | RMSE 5.08 | [card](docs/model_cards/car.md) |
| Diabetes risk | classification, 768 rows | logistic regression | ROC AUC **0.81** (CV 0.84) | AUC 0.50 | [card](docs/model_cards/diabetes.md) |
| Calories burnt | regression, 15,000 rows | gradient boosting | RMSE 1.95 kcal, R² **0.999** | RMSE 63.6 | [card](docs/model_cards/exercise.md) |
| Medical insurance | regression, 1,337 rows | random forest | RMSE $4,449, R² **0.89** | RMSE $13,612 | [card](docs/model_cards/medical_insurance.md) |
| California housing | regression, 20,640 rows | gradient boosting | RMSE 0.46 (x $100k), R² **0.84** | RMSE 1.14 | [card](docs/model_cards/california_housing.md) |

The car test split has only 60 rows, so one unusual car moves its test numbers a lot; its card says so and points to cross-validation as the better estimate.

## What this project demonstrates

- **Methodology you can defend.** Preprocessing lives *inside* each scikit-learn `Pipeline`, so scalers and imputers are fitted on training folds only (the original diabetes notebook fitted its scaler on all rows before splitting). Candidates are compared with 5-fold cross-validation, the test set is used once, and every result sits next to a naive baseline.
- **Honest uncertainty.** Regression answers come with a 90% prediction interval built from out-of-fold residuals, whose real coverage on the test split is reported in the card (90-91% on the three larger datasets).
- **A model that cannot silently lie.** The original service replaced any model that failed to load with a mock returning `123.45` for every request, and the shipped pickles were in fact unloadable. Here each artifact is verified (SHA-256 and library version) at start-up; a model that fails verification is reported by `/readyz` and answers `503`.
- **One source of truth.** A declarative [`ModelSpec`](src/predictors/specs.py) per model defines features, units, valid ranges and categories. The API schema, request validation, the web form, extrapolation warnings and the model cards are all generated from it, replacing five copy-pasted endpoints and six HTML pages.
- **Responsible handling of health and financial data.** Inputs are never logged or echoed back, the diabetes and insurance models carry "not for real-world use" disclaimers, and the Boston housing data (which contains a column derived from the racial make-up of neighbourhoods) was replaced by California Housing.
- **Quality gates in CI.** Ruff, strict mypy, 104 tests at 97.5% coverage, a gate that fails the build if any trained model drops below its cross-validated quality floor, and a Docker build that trains and gates the models inside the image.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
make install                                    # pinned dependencies + editable install
make train                                      # ~1 minute: trains all models, writes artifacts/ and docs/model_cards/
make serve                                      # http://127.0.0.1:8000  (UI at /, API docs at /docs)
```

Or with Docker (training happens during the build, so the image always matches its dependencies):

```bash
docker build -t prediction-models .
docker run --rm -p 8000:8000 prediction-models
```

```bash
curl -X POST localhost:8000/v1/car/predict -H 'Content-Type: application/json' -d '{
  "Year": 2015, "Present_Price": 6.5, "Kms_Driven": 45000,
  "Fuel_Type": "Petrol", "Seller_Type": "Dealer", "Transmission": "Manual", "Owner": 0 }'
```

```json
{ "model": "car", "version": "f5a698bb1e61", "prediction": 4.26, "unit": "lakh INR",
  "lower": 3.01, "upper": 6.03, "interval_level": 0.9, "probability": null,
  "label": null, "warnings": [] }
```

## API

| Method and path | Purpose |
| :-- | :-- |
| `POST /v1/{model}/predict` | One typed route per model: `car`, `diabetes`, `exercise`, `medical_insurance`, `california_housing` |
| `GET /v1/models` | Models, availability, selected estimator, test metrics, artifact version |
| `GET /v1/models/{model}` | Input schema and the full model card as JSON |
| `GET /healthz`, `GET /readyz` | Liveness; readiness (every artifact verified, else `503` with reasons) |
| `GET /docs`, `/openapi.json` | Interactive OpenAPI documentation, generated per model |

Requests are named JSON objects (field order is irrelevant), unknown fields are rejected, numbers are range-checked, categories must be one of the trained values, and `NaN`/`Infinity` are refused. Values inside the valid range but outside what the model saw in training are accepted with a `warnings` entry. Errors return only the field, message and type.

### Configuration

| Variable | Default | Meaning |
| :-- | :-- | :-- |
| `PREDICTORS_ARTIFACTS` | `artifacts` | Directory with `*.joblib` and `*.json` produced by `make train` |
| `PREDICTORS_CORS_ORIGINS` | *(empty: CORS off)* | Comma-separated origins allowed to call the API from a browser |
| `PREDICTORS_LOG_LEVEL` | `INFO` | Log level |

## Project layout

```
src/predictors/
  specs.py       declarative description of every model and feature
  data.py        loading, cleaning (e.g. Pima zeros = missing), de-duplication, data fingerprint
  train.py       pipelines, CV model selection, evaluation, intervals, importance, artifacts
  card.py        Markdown model-card renderer
  registry.py    verified artifact loading and prediction
  api.py         FastAPI app generated from the specs
  check_cards.py quality gate used by CI and the Docker build
  static/        the generated web UI (no inline scripts; strict CSP)
data/            CSV inputs with provenance notes
docs/model_cards generated, committed model cards
notebooks/legacy the original exploratory notebooks (superseded, kept for reference)
tests/           unit, API and integration tests
```

## Development

| Command | What it does |
| :-- | :-- |
| `make train` / `make check` | Train everything / enforce the quality floors |
| `make test` / `make cov` | Tests; tests with coverage (85% minimum, currently 97.5%) |
| `make lint` / `make format` / `make typecheck` | ruff, ruff format, strict mypy |

Unit and API tests train small models into a temp directory (seconds); tests marked `integration` run against the real `artifacts/` and are skipped until you run `make train`. See [ARCHITECTURE.md](ARCHITECTURE.md) and the [ADRs](docs/adr).

## Limitations

- These are teaching-scale datasets (299 to 20,640 rows). The models are not production-grade predictors and the diabetes and insurance models must not be used for decisions about people.
- The car dataset is tiny and from 2003-2018; the housing data is the 1990 census.
- Prediction intervals have the same width for every input (an average of training errors), and the car interval's observed coverage on 60 test rows was 80%, below its 90% target.
- Models are retrained from scratch, not updated online, and there is no drift monitoring.
- The Docker image and the GitHub Actions workflow were written and checked against the same commands locally, but the image has not been built on this machine (no Docker); CI builds and smoke-tests it.

## License

[MIT](LICENSE). Dataset provenance is in [data/README.md](data/README.md).
