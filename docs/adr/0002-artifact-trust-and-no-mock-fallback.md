# ADR 0002: Verify artifacts at load time and never fall back to a mock

Status: accepted

## Context
`views.py` loaded each pickle with `load_pickle_model(path) or MockModel()`, and `MockModel.predict` returned `[123.45]`. The shipped pickles were produced by scikit-learn 1.0.2/1.1.2 while the app pinned 1.8.0, and the insurance model referenced a module (`sklearn.ensemble._gb_losses`) that no longer exists, so it could not be loaded at all. The service would have answered every insurance request with 123.45 and returned HTTP 200. Separately, unpickling arbitrary files from a directory is a code-execution risk if the files can be replaced.

## Decision
- Training writes `key.joblib` and `key.json`. The card records the file's SHA-256, the scikit-learn version and the data fingerprint.
- `load_model` refuses an artifact if the card is missing, the checksum does not match, or the running scikit-learn version differs. Only after those checks is the file unpickled.
- There is no fallback. `ModelRegistry` records the reason for each failure; `/readyz` returns 503 and lists them, `/v1/models` shows `available: false`, and the predict route answers 503.
- The Docker build trains inside the image, so the artifact and the runtime cannot disagree.
- Every response includes the first 12 characters of the artifact checksum as `version`.

## Alternatives
- **ONNX or skops**: remove the unpickling risk entirely. Not adopted because the pipelines include pandas-aware preprocessing and the extra runtime would outweigh the benefit for a service whose artifacts are built by its own CI; recorded as the next step if artifacts ever come from outside.
- **Signing artifacts with a key**: stronger than a checksum stored beside the file (an attacker who can replace the `.joblib` can also replace the `.json`). Out of scope here; the threat model in SECURITY.md states this.

## Consequences
A deployment with stale or tampered files fails loudly at start-up instead of returning plausible-looking wrong numbers.
