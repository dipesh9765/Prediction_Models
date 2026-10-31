"""Render a model card (Mitchell et al., 2019 style) as Markdown from the training results."""

from __future__ import annotations

from typing import Any

from predictors.specs import ModelSpec


def _num(value: float | None) -> str:
    """Compact number without exponent notation, e.g. 1000000 -> 1,000,000."""
    if value is None:
        return "-"
    return f"{value:,.6g}" if abs(value) < 1e6 else f"{value:,.0f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(":--" for _ in headers) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(card: dict[str, Any], spec: ModelSpec) -> str:
    is_clf = card["task"] == "classification"
    primary = card["validation"]["primary_metric"]
    primary_name = "RMSE" if primary == "rmse" else "ROC AUC"
    data = card["data"]
    test = card["metrics"]["test"]
    base = card["metrics"]["baseline"]

    out: list[str] = [f"# Model card: {card['title']}", "", card["description"], ""]
    if card["disclaimer"]:
        out += [f"> **Not for real-world use.** {card['disclaimer']}", ""]

    out += [
        "## Summary",
        "",
        f"- **Task:** {card['task']} (`{card['target']['name']}`"
        + (f", in {card['target']['unit']}" if card["target"]["unit"] else "")
        + ")",
        f"- **Selected model:** `{card['selected']}` (best "
        f"{card['validation']['cv_folds']}-fold cross-validated {primary_name} on the training "
        "split)",
        f"- **Data:** {', '.join(f'`{f}`' for f in data['files'])}, {data['raw_rows']:,} rows "
        f"({data['rows_used']:,} after removing duplicates), {data['train_rows']:,} train / "
        f"{data['test_rows']:,} test, SHA-256 `{data['sha256'][:12]}`",
        f"- **Libraries:** scikit-learn {card['libraries']['scikit-learn']}, "
        f"pandas {card['libraries']['pandas']}, Python {card['libraries']['python']}; "
        f"seed {card['seed']}",
        "",
        "## Held-out test results",
        "",
    ]
    metric_names = list(test)
    out.append(
        _table(
            ["", *metric_names],
            [
                ["**Model**", *[f"**{test[m]}**" for m in metric_names]],
                ["Naive baseline", *[str(base[m]) for m in metric_names]],
            ],
        )
    )
    out += [
        "",
        "The baseline predicts the training mean (regression) or class prior (classification)."
        " The test split was used once, after model selection."
        + (
            f" With only {data['test_rows']} test rows one unusual record moves these numbers a "
            "lot; the cross-validation table below is the more reliable estimate."
            if data["test_rows"] < 500
            else ""
        ),
        "",
        "## Model selection (training split, cross-validation)",
        "",
    ]
    metrics = ["roc_auc", "accuracy"] if is_clf else ["rmse", "r2"]
    rows = []
    for cand in card["validation"]["candidates"]:
        name = f"`{cand['name']}`" + (" ✔" if cand["name"] == card["selected"] else "")
        rows.append(
            [name]
            + [f"{cand[f'cv_{m}_mean']} ± {cand[f'cv_{m}_std']}" for m in metrics]
            + [f"{cand['fit_seconds']} s"]
        )
    out += [_table(["Candidate", *[f"CV {m}" for m in metrics], "CV time"], rows), ""]

    if card["interval"]:
        iv = card["interval"]
        out += [
            "## Prediction interval",
            "",
            f"The API returns a {int(iv['level'] * 100)}% interval built from out-of-fold training "
            f"residuals: prediction {iv['residual_low']:+} to {iv['residual_high']:+} "
            f"{card['target']['unit']}. Observed coverage on the test split: "
            f"**{iv['observed_test_coverage']:.0%}**. The interval is the same width for every "
            "input, so it is too wide for easy cases and too narrow for hard ones.",
            "",
        ]

    out += ["## Inputs", ""]
    rows = []
    for feat, fspec in zip(card["features"], spec.features, strict=True):
        if feat["kind"] == "number":
            valid = f"{_num(fspec.minimum)} to {_num(fspec.maximum)}"
            seen = f"{_num(feat['training_min'])} to {_num(feat['training_max'])}"
        else:
            valid = ", ".join(feat["choices"])
            seen = "-"
        rows.append(
            [
                f"`{feat['name']}`",
                fspec.label,
                feat["unit"] or "-",
                valid,
                seen,
                "optional" if feat["optional"] else "required",
            ]
        )
    out += [
        _table(["Name", "Meaning", "Unit", "Accepted values", "Seen in training", ""], rows),
        "",
        "Values outside *Seen in training* are accepted but returned with a warning, because the "
        "model is extrapolating.",
        "",
        "## What drives the prediction",
        "",
        "Permutation importance on the test split: how much "
        + ("ROC AUC drops" if is_clf else f"RMSE ({card['target']['unit']}) rises")
        + " when one feature's values are shuffled (larger = the model relies on it more):",
        "",
        _table(
            ["Feature", "Importance"],
            [[f"`{k}`", str(v)] for k, v in card["permutation_importance"].items()],
        ),
        "",
        "## Limitations",
        "",
        *[f"- {text}" for text in card["limitations"]],
        "- Correlation, not causation: feature importance does not mean changing a feature changes "
        "the outcome.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python -m predictors.train --model {card['key']}",
        "```",
        "",
        f"Artifact SHA-256: `{card['artifact']['sha256']}`",
        "",
    ]
    return "\n".join(out)
