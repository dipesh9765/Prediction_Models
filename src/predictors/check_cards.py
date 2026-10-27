"""Quality gate: fail if a trained model does not clear its bar. Run after ``make train``.

A model passes when (1) the selected candidate's cross-validated R2 / ROC AUC reaches the floor
declared in its spec, (2) it beats the naive baseline on the held-out test set by a clear margin,
and (3) for regression, the observed interval coverage is not badly below the nominal 90%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from predictors.specs import SPECS


def problems_for(card: dict[str, Any], floor: float) -> list[str]:
    problems: list[str] = []
    is_clf = card["task"] == "classification"
    selected = next(c for c in card["validation"]["candidates"] if c["name"] == card["selected"])
    metric = "roc_auc" if is_clf else "r2"
    cv_score = selected[f"cv_{metric}_mean"]
    if cv_score < floor:
        problems.append(f"cross-validated {metric} {cv_score} is below the floor {floor}")

    test, base = card["metrics"]["test"], card["metrics"]["baseline"]
    if is_clf:
        if test["roc_auc"] < base["roc_auc"] + 0.2:
            problems.append(f"test ROC AUC {test['roc_auc']} does not clearly beat the baseline")
    elif test["rmse"] > 0.8 * base["rmse"]:
        problems.append(f"test RMSE {test['rmse']} is not 20% better than baseline {base['rmse']}")

    interval = card.get("interval")
    # Small test sets are noisy, so only enforce coverage where there are enough rows.
    if interval and card["data"]["test_rows"] >= 250 and interval["observed_test_coverage"] < 0.85:
        problems.append(f"interval coverage {interval['observed_test_coverage']} is below 0.85")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    failed = False
    for key, spec in sorted(SPECS.items()):
        path = args.artifacts / f"{key}.json"
        if not path.exists():
            print(f"FAIL {key}: no model card at {path}")
            failed = True
            continue
        problems = problems_for(json.loads(path.read_text(encoding="utf-8")), spec.quality_floor)
        status = "FAIL" if problems else "ok  "
        print(f"{status} {key}" + "".join(f"\n       - {p}" for p in problems))
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
