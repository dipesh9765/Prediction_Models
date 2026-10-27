# Model card: California house value

Estimates the median house value of a California census block group from its income, housing and location statistics (1990 census).

## Summary

- **Task:** regression (`MedHouseVal`, in x $100,000)
- **Selected model:** `hist_gradient_boosting` (chosen by 5-fold cross-validated RMSE on the training split)
- **Data:** `california_housing/california_housing.csv`, 20,640 rows (20,640 after removing duplicates), 16,512 train / 4,128 test, SHA-256 `a61475863454`
- **Libraries:** scikit-learn 1.9.1, pandas 3.1.0, Python 3.13.15; seed 42

## Held-out test results

|  | rmse | mae | r2 |
| :-- | :-- | :-- | :-- |
| **Model** | **0.4643** | **0.3103** | **0.8355** |
| Naive baseline | 1.1449 | 0.9061 | -0.0002 |

The baseline predicts the training mean (regression) or class prior (classification). The test split was used once, after model selection.

## Model selection (training split, cross-validation)

| Candidate | CV rmse | CV r2 | CV time |
| :-- | :-- | :-- | :-- |
| `ridge` | 0.7205 ± 0.0139 | 0.6115 ± 0.0124 | 0.41 s |
| `random_forest` | 0.51 ± 0.0121 | 0.8054 ± 0.0067 | 8.91 s |
| `hist_gradient_boosting` ✔ | 0.4732 ± 0.0123 | 0.8325 ± 0.0058 | 1.88 s |

## Prediction interval

The API returns a 90% interval built from out-of-fold training residuals: prediction -0.6519 to +0.7928 x $100,000. Observed coverage on the test split: **90%**. The interval is the same width for every input, so it is too wide for easy cases and too narrow for hard ones.

## Inputs

| Name | Meaning | Unit | Accepted values | Seen in training |  |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `MedInc` | Median income | x $10,000 | 0 to 20 | 0.4999 to 15.0001 | required |
| `HouseAge` | Median house age | years | 1 to 100 | 1 to 52 | required |
| `AveRooms` | Average rooms per household | - | 0.5 to 150 | 0.8889 to 141.909 | required |
| `AveBedrms` | Average bedrooms per household | - | 0.3 to 40 | 0.3333 to 25.6364 | required |
| `Population` | Block group population | - | 3 to 40,000 | 3 to 35,682 | required |
| `AveOccup` | Average occupants per household | - | 0.5 to 1,500 | 0.6923 to 1,243.33 | required |
| `Latitude` | Latitude | deg | 32 to 42.5 | 32.55 to 41.95 | required |
| `Longitude` | Longitude | deg | -125 to -114 | -124.35 to -114.31 | required |

Values outside *Seen in training* are accepted but returned with a warning, because the model is extrapolating.

## What drives the prediction

Permutation importance on the test split: how much RMSE (x $100,000) rises when one feature's values are shuffled (larger = the model relies on it more):

| Feature | Importance |
| :-- | :-- |
| `Latitude` | 0.642 |
| `Longitude` | 0.5738 |
| `MedInc` | 0.5024 |
| `AveOccup` | 0.1835 |
| `HouseAge` | 0.054 |
| `AveRooms` | 0.0508 |
| `Population` | 0.0118 |
| `AveBedrms` | 0.0091 |

## Limitations

- 1990 census block groups: prices are decades old and only meaningful in California.
- Target values are censored at $500,000 in the source data, so the model cannot predict above that.
- Replaces the Boston housing data used by the original project; that dataset was removed from scikit-learn because one column encodes the racial make-up of neighbourhoods.
- Correlation, not causation: feature importance does not mean changing a feature changes the outcome.

## Reproduce

```bash
python -m predictors.train --model california_housing
```

Artifact SHA-256: `f8a1145f684e73294f1a65d888489280883053b75add030e6dfdc0d443bea9ac`
