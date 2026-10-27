# ADR 0003: Generate the API schema, validation and web UI from one model spec

Status: accepted

## Context
The old service accepted `list(request.json["data"].values())`, so the model received features in whatever order the client sent them, with no names, types or ranges, and any failure became HTTP 500 with `str(e)`. The same five endpoints were copy-pasted twice each, and six HTML pages repeated field names, ranges and categorical encodings.

## Decision
- Describe every model once as a `ModelSpec` (typed `Feature`s with units, valid ranges, categories, optionality, examples).
- FastAPI and pydantic: one route per model, `POST /v1/{key}/predict`, with a request model built by `create_model` from the spec: named fields, `extra="forbid"`, numeric bounds, `Literal` categories, `allow_inf_nan=False`. OpenAPI documents each model's exact schema for free.
- Valid does not mean typical: values inside the allowed range but outside the training range are accepted and returned with a `warnings` entry.
- One static page (vanilla JS, no framework, no inline scripts, CSP `default-src 'self'`) builds its forms from `GET /v1/models/{key}`.
- Validation errors return only `loc`, `msg` and `type`. FastAPI's default echoes the submitted values, which is unnecessary for health data and crashed the JSON encoder when the input was `NaN`.

## Alternatives
- Keep Flask and add marshmallow/pydantic by hand: more code, no generated schema.
- A JS framework for the UI: a build step and a dependency tree for five forms.

## Consequences
Adding a model is one spec plus a dataset loader. The cost is a dynamic request model, which is why `api.py` cannot use postponed annotations (documented at the top of the file).
