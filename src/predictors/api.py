"""FastAPI service. Request and response schemas are generated from the model specs, so the API,
its OpenAPI document, the web form and the model cards cannot disagree about a feature.

Note: this module intentionally does not use ``from __future__ import annotations``; FastAPI needs
the dynamically created request models to be real objects in the handler annotations.
"""

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, create_model

from predictors import __version__
from predictors.registry import ArtifactError, LoadedModel, ModelRegistry, predict
from predictors.specs import SPECS, Feature, ModelSpec

logger = logging.getLogger("predictors.api")

STATIC_DIR = Path(__file__).parent / "static"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
}


class PredictionResponse(BaseModel):
    model: str = Field(description="Model key")
    version: str = Field(description="Short checksum of the exact artifact that answered")
    prediction: float = Field(description="Predicted value, or the probability for classifiers")
    unit: str = Field(description="Unit of `prediction` (empty for probabilities)")
    lower: float | None = Field(None, description="Lower bound of the prediction interval")
    upper: float | None = Field(None, description="Upper bound of the prediction interval")
    interval_level: float | None = Field(None, description="Nominal coverage of the interval")
    probability: float | None = Field(None, description="Probability of the positive class")
    label: str | None = Field(None, description="Human-readable class (classifiers)")
    warnings: list[str] = Field(default_factory=list, description="Inputs outside training range")


def _field_for(feature: Feature) -> tuple[Any, Any]:
    """Pydantic field definition for one feature: type, bounds, optionality and docs."""
    if feature.kind == "category":
        annotation: Any = Literal[feature.choices]
        info = Field(..., description=feature.label, examples=[feature.example])
        return annotation, info
    base = int if feature.integer else float
    annotation = base | None if feature.optional else base
    info = Field(
        None if feature.optional else ...,
        ge=feature.minimum,
        le=feature.maximum,
        allow_inf_nan=False,
        description=feature.label + (f" ({feature.unit})" if feature.unit else ""),
        examples=[feature.example],
    )
    return annotation, info


def request_model_for(spec: ModelSpec) -> type[BaseModel]:
    fields = {f.name: _field_for(f) for f in spec.features}
    example: dict[str, Any] = {f.name: f.example for f in spec.features if f.example is not None}
    model: type[BaseModel] = create_model(  # type: ignore[call-overload]
        f"{spec.key.title().replace('_', '')}Request",
        __config__=ConfigDict(extra="forbid", json_schema_extra={"examples": [example]}),
        **fields,
    )
    return model


def _summary(spec: ModelSpec, registry: ModelRegistry) -> dict[str, Any]:
    loaded = registry.models.get(spec.key)
    summary: dict[str, Any] = {
        "key": spec.key,
        "title": spec.title,
        "task": spec.task,
        "description": spec.description,
        "available": loaded is not None,
        "disclaimer": spec.disclaimer,
    }
    if loaded:
        summary["selected_model"] = loaded.card["selected"]
        summary["test_metrics"] = loaded.card["metrics"]["test"]
        summary["version"] = loaded.version
    else:
        summary["reason"] = registry.errors.get(spec.key, "not loaded")
    return summary


def _feature_json(feature: Feature) -> dict[str, Any]:
    return {
        "name": feature.name,
        "label": feature.label,
        "kind": feature.kind,
        "unit": feature.unit,
        "min": feature.minimum,
        "max": feature.maximum,
        "integer": feature.integer,
        "choices": list(feature.choices),
        "optional": feature.optional,
        "example": feature.example,
    }


def _registry_of(request: Request) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.registry
    return registry


def _require_model(request: Request, key: str) -> LoadedModel:
    if key not in SPECS:
        raise HTTPException(404, f"unknown model '{key}'")
    registry = _registry_of(request)
    model = registry.models.get(key)
    if model is None:
        raise HTTPException(503, f"model '{key}' is unavailable: {registry.errors.get(key)}")
    return model


def _install_middleware(app: FastAPI) -> None:
    """CORS (opt-in), request ids, security headers, access log and safe validation errors."""
    origins = [
        o.strip() for o in os.environ.get("PREDICTORS_CORS_ORIGINS", "").split(",") if o.strip()
    ]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            # Revalidate on every load so a redeploy is never masked by a cached script.
            response.headers["Cache-Control"] = "no-cache"
        # Method, path, status and timing only. Request bodies are health and financial data and
        # are never logged.
        logger.info(
            "%s %s -> %d in %.1f ms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default body echoes every submitted value back (`input`, `ctx`). That is
        # unnecessary for health data and crashes the JSON encoder on NaN. Return only where and
        # why the request was invalid.
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    """Application factory; the artifact directory defaults to ``$PREDICTORS_ARTIFACTS``."""
    directory = artifacts_dir or Path(os.environ.get("PREDICTORS_ARTIFACTS", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=os.environ.get("PREDICTORS_LOG_LEVEL", "INFO"))
        app.state.registry = ModelRegistry(directory)
        logger.info("loaded %d/%d models", len(app.state.registry.models), len(SPECS))
        yield

    app = FastAPI(
        title="Prediction Models API",
        version=__version__,
        description="Typed prediction endpoints for five models. Every response names the exact "
        "model version; every model has a published card under /v1/models/{key}.",
        lifespan=lifespan,
    )

    _install_middleware(app)

    @app.get("/healthz", tags=["ops"], summary="Liveness")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"], summary="Readiness: every model loaded and verified")
    def readyz(request: Request) -> JSONResponse:
        registry = _registry_of(request)
        body = {
            "status": "ready" if registry.ready else "degraded",
            "models": sorted(registry.models),
            "errors": registry.errors,
        }
        return JSONResponse(body, status_code=200 if registry.ready else 503)

    @app.get("/v1/models", tags=["models"], summary="List models")
    def list_models(request: Request) -> list[dict[str, Any]]:
        registry = _registry_of(request)
        return [_summary(spec, registry) for spec in SPECS.values()]

    @app.get("/v1/models/{key}", tags=["models"], summary="Model card and input schema")
    def get_model(request: Request, key: str) -> dict[str, Any]:
        spec = SPECS.get(key)
        if spec is None:
            raise HTTPException(404, f"unknown model '{key}'")
        registry = _registry_of(request)
        body = _summary(spec, registry)
        body["features"] = [_feature_json(f) for f in spec.features]
        body["target"] = {"label": spec.target_label, "unit": spec.target_unit}
        if key in registry.models:
            body["card"] = registry.models[key].card
        return body

    def add_predict_route(spec: ModelSpec) -> None:
        request_model = request_model_for(spec)

        def handler(payload: request_model, request: Request) -> PredictionResponse:  # type: ignore[valid-type]
            model = _require_model(request, spec.key)
            try:
                result = predict(model, payload.model_dump())  # type: ignore[attr-defined]
            except ArtifactError as exc:
                logger.error("model %s produced an invalid result: %s", spec.key, exc)
                raise HTTPException(503, "model produced an invalid result") from exc
            interval = model.card.get("interval")
            return PredictionResponse(
                model=spec.key,
                version=model.version,
                prediction=result.value,
                unit=spec.target_unit,
                lower=result.lower,
                upper=result.upper,
                interval_level=interval["level"] if interval else None,
                probability=result.probability,
                label=result.label,
                warnings=result.warnings,
            )

        app.post(
            f"/v1/{spec.key}/predict",
            response_model=PredictionResponse,
            tags=["predict"],
            summary=spec.title,
            description=spec.description
            + (f"\n\n**{spec.disclaimer}**" if spec.disclaimer else ""),
            name=f"predict_{spec.key}",
        )(handler)

    for model_spec in SPECS.values():
        add_predict_route(model_spec)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
