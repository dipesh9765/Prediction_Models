# Model card: Calories burnt during exercise

Estimates the energy spent in a workout from the person's body measurements and the session's duration, heart rate and body temperature.

## Summary

- **Task:** regression (`Calories`, in kcal)
- **Selected model:** `hist_gradient_boosting` (chosen by 5-fold cross-validated RMSE on the training split)
- **Data:** `exercise/exercise.csv`, `exercise/calories.csv`, 15,000 rows (15,000 after removing duplicates), 12,000 train / 3,000 test, SHA-256 `6d4ce2a06478`
- **Libraries:** scikit-learn 1.9.1, pandas 3.1.0, Python 3.13.15; seed 42

## Held-out test results

|  | rmse | mae | r2 |
| :-- | :-- | :-- | :-- |
| **Model** | **1.9499** | **1.3377** | **0.9991** |
| Naive baseline | 63.5667 | 53.9279 | -0.0012 |

The baseline predicts the training mean (regression) or class prior (classification). The test split was used once, after model selection.

## Model selection (training split, cross-validation)

| Candidate | CV rmse | CV r2 | CV time |
| :-- | :-- | :-- | :-- |
| `ridge` | 11.2701 ± 0.1461 | 0.9671 ± 0.0009 | 0.11 s |
| `random_forest` | 3.104 ± 0.1327 | 0.9975 ± 0.0002 | 3.35 s |
| `hist_gradient_boosting` ✔ | 2.0886 ± 0.1257 | 0.9989 ± 0.0001 | 1.66 s |

## Prediction interval

The API returns a 90% interval built from out-of-fold training residuals: prediction -3.0953 to +3.0173 kcal. Observed coverage on the test split: **91%**. The interval is the same width for every input, so it is too wide for easy cases and too narrow for hard ones.

## Inputs

| Name | Meaning | Unit | Accepted values | Seen in training |  |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `Gender` | Gender | - | male, female | - | required |
| `Age` | Age | years | 10 to 100 | 20 to 79 | required |
| `Height` | Height | cm | 100 to 230 | 123 to 218 | required |
| `Weight` | Weight | kg | 25 to 250 | 36 to 132 | required |
| `Duration` | Exercise duration | min | 1 to 300 | 1 to 30 | required |
| `Heart_Rate` | Average heart rate | bpm | 40 to 220 | 67 to 128 | required |
| `Body_Temp` | Body temperature | C | 35 to 42 | 37.1 to 41.5 | required |

Values outside *Seen in training* are accepted but returned with a warning, because the model is extrapolating.

## What drives the prediction

Permutation importance on the test split: how much RMSE (kcal) rises when one feature's values are shuffled (larger = the model relies on it more):

| Feature | Importance |
| :-- | :-- |
| `Duration` | 66.2152 |
| `Heart_Rate` | 24.4688 |
| `Age` | 13.6036 |
| `Gender` | 6.4714 |
| `Weight` | 3.6175 |
| `Body_Temp` | 0.0133 |
| `Height` | 0.0091 |

## Limitations

- 15,000 records from a public fitness dataset that looks synthetic or heavily smoothed: calories are almost a deterministic function of duration and heart rate.
- Not validated against calorimetry; use for illustration, not for training plans.
- Correlation, not causation: feature importance does not mean changing a feature changes the outcome.

## Reproduce

```bash
python -m predictors.train --model exercise
```

Artifact SHA-256: `876335a3e4f457bb9e3433a5acbb4d0325ff11e3e7bda27c848445c3201ce9a1`
