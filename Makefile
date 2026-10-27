.PHONY: install train check test serve lint format typecheck cov docker

PY ?= python

install:
	$(PY) -m pip install -r requirements-dev.txt -e .

train:            ## train every model, write artifacts/ and docs/model_cards/
	$(PY) -m predictors.train --all

check:            ## fail if any model card misses its quality bar
	$(PY) -m predictors.check_cards

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov --cov-report=term-missing

serve:
	$(PY) -m uvicorn predictors.api:create_app --factory --reload

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix . && $(PY) -m ruff format .

typecheck:
	$(PY) -m mypy

docker:
	docker build -t prediction-models .
