# Contributing

## Setup
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
make install
make train          # needed once for the integration tests and to run the API
```

## Before opening a pull request
```bash
make lint typecheck test
make train check    # if you touched data, specs or training code
```
CI runs the same steps on Python 3.12 and 3.13.

## Adding a model
1. Put the CSV in `data/<name>/` and describe its source in `data/README.md`.
2. Add a cleaner to `data.py` and a `ModelSpec` to `specs.py`: features with units, **valid ranges**, categories and examples; candidates; a `quality_floor`; limitations; a disclaimer if it concerns people's health, money or safety.
3. Run `make train check`. If the model does not clear its floor, fix the data or the approach; do not lower the floor to pass.
4. Commit the regenerated `docs/model_cards/<name>.md`.

`tests/test_specs_and_data.py` already checks that examples satisfy their constraints and that every training value lies inside the declared ranges.

## Conventions
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `build:`).
- Preprocessing belongs inside the pipeline; never fit a transformer on data that includes the test split.
- Never log request bodies. Never add a fallback that returns a prediction when a model failed to load.
- Do not commit trained artifacts (`artifacts/` is ignored); commit the model cards.
- Use named fields everywhere. Positional feature arrays are how the original service broke.
