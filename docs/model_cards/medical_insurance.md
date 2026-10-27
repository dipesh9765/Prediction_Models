# Model card: Medical insurance charges

Estimates the yearly medical insurance charge billed to a person in the US from age, BMI, smoking status and a few demographics.

> **Not for real-world use.** Illustrative model trained on a small public sample. It must not be used to price insurance, set premiums or make decisions about individuals.

## Summary

- **Task:** regression (`charges`, in USD)
- **Selected model:** `random_forest` (best 5-fold cross-validated RMSE on the training split)
- **Data:** `medical_insurance/insurance.csv`, 1,338 rows (1,337 after removing duplicates), 1,069 train / 268 test, SHA-256 `388eff679557`
- **Libraries:** scikit-learn 1.9.1, pandas 3.1.0, Python 3.13.15; seed 42

## Held-out test results

|  | rmse | mae | r2 |
| :-- | :-- | :-- | :-- |
| **Model** | **4448.5857** | **2476.168** | **0.8923** |
| Naive baseline | 13612.426 | 9861.7979 | -0.0084 |

The baseline predicts the training mean (regression) or class prior (classification). The test split was used once, after model selection. With only 268 test rows one unusual record moves these numbers a lot; the cross-validation table below is the more reliable estimate.

## Model selection (training split, cross-validation)

| Candidate | CV rmse | CV r2 | CV time |
| :-- | :-- | :-- | :-- |
| `ridge` | 6123.6479 ± 188.42 | 0.7228 ± 0.028 | 0.08 s |
| `random_forest` ✔ | 4738.9647 ± 200.7587 | 0.8324 ± 0.0284 | 1.12 s |
| `hist_gradient_boosting` | 4937.5247 ± 188.1704 | 0.818 ± 0.0303 | 1.42 s |

## Prediction interval

The API returns a 90% interval built from out-of-fold training residuals: prediction -4916.2031 to +12482.981 USD. Observed coverage on the test split: **91%**. The interval is the same width for every input, so it is too wide for easy cases and too narrow for hard ones.

## Inputs

| Name | Meaning | Unit | Accepted values | Seen in training |  |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `age` | Age | years | 18 to 100 | 18 to 64 | required |
| `sex` | Sex | - | female, male | - | required |
| `bmi` | Body mass index | kg/m2 | 10 to 70 | 15.96 to 53.13 | required |
| `children` | Dependants | - | 0 to 10 | 0 to 5 | required |
| `smoker` | Smoker | - | no, yes | - | required |
| `region` | Region | - | northeast, northwest, southeast, southwest | - | required |

Values outside *Seen in training* are accepted but returned with a warning, because the model is extrapolating.

## What drives the prediction

Permutation importance on the test split: how much RMSE (USD) rises when one feature's values are shuffled (larger = the model relies on it more):

| Feature | Importance |
| :-- | :-- |
| `smoker` | 11954.1698 |
| `bmi` | 3649.3309 |
| `age` | 2584.3763 |
| `children` | 188.757 |
| `region` | 69.4557 |
| `sex` | -54.9488 |

## Limitations

- 1,337 people; the charge distribution is strongly skewed and driven by smoking status, so errors are large for high-cost individuals.
- Sex and region are included because they are in the data, not because they should drive pricing decisions.
- Correlation, not causation: feature importance does not mean changing a feature changes the outcome.

## Reproduce

```bash
python -m predictors.train --model medical_insurance
```

Artifact SHA-256: `1c93c8191f3ba2019b4e3338157f739dc30f4966247c8e8072928de2ca7673c1`
