# syntax=docker/dockerfile:1

# ---- build: install pinned dependencies and train every model --------------------------------
# Training happens here, with the exact scikit-learn version the runtime image will use, so the
# artifacts can never be stale or mismatched. The quality gate fails the build if any model
# regresses.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY src ./src
COPY data ./data
RUN PYTHONPATH=/app/src:/install/lib/python3.12/site-packages \
    python -m predictors.train --all --out /app/artifacts --cards /tmp/cards \
 && PYTHONPATH=/app/src:/install/lib/python3.12/site-packages \
    python -m predictors.check_cards --artifacts /app/artifacts

# ---- runtime: no compilers, no data, no training code paths exercised --------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PREDICTORS_ARTIFACTS=/app/artifacts
WORKDIR /app

COPY --from=build /install /usr/local
COPY --from=build /app/src ./src
COPY --from=build /app/artifacts ./artifacts

RUN useradd --system --uid 10001 --no-create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status == 200 else 1)"]

CMD ["python", "-m", "uvicorn", "predictors.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
