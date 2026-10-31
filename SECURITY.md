# Security

## Reporting a vulnerability
Open a private security advisory on GitHub (Security, then Advisories, then Report a vulnerability). Include the request that triggers the problem and the version string returned by `/v1/models`.

## Threat model

| Threat | Mitigation | Residual risk |
| :-- | :-- | :-- |
| Malformed, hostile or absurd input | Pydantic validation per model: named fields, `extra="forbid"`, numeric bounds, fixed categories, `allow_inf_nan=False`; errors return only field, message and type | None known |
| Unsafe deserialisation of model files | Artifact SHA-256 and library version are checked against a card before `joblib.load`; artifacts are built by the project's own training step and, in Docker, inside the image | An attacker who can write both the `.joblib` and its `.json` defeats the check; see ADR 0002 (signing or ONNX would close this) |
| A broken model returning plausible garbage | No mock or fallback; failed models are reported by `/readyz` and answer 503; negative and non-finite outputs are handled explicitly | None known |
| Sensitive inputs (health, income) leaking | Bodies are never logged or echoed in errors; logs contain method, path, status, latency and a request id | Reverse proxies or clients may log bodies; deploy accordingly |
| Browser abuse | CORS is off unless `PREDICTORS_CORS_ORIGINS` lists origins; CSP `default-src 'self'`, `X-Frame-Options: DENY`, `nosniff`, no-referrer; the UI renders all dynamic text with `textContent` | No authentication or rate limiting: put the service behind a gateway if it is exposed publicly |
| Dependency drift | Runtime dependencies are pinned in `requirements.txt`; the scikit-learn pin is also the training/serving contract | Automated dependency updates are not configured |
| Container | Non-root user (uid 10001), no compilers or training data in the runtime stage, healthcheck on `/readyz` | Image not built on the author's machine; CI builds and smoke-tests it |
| Misuse of health or insurance predictions | Disclaimers in the UI, the API description, the cards and the README | A disclaimer cannot prevent misuse |

## History note
The original repository contained no credentials. It did contain ten pickled model files that could not be trusted or loaded; they were deleted from the tree (they remain in git history).
