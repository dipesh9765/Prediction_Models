# Model card: Diabetes risk screening

Estimates the probability that a woman of Pima Indian heritage has diabetes from eight routine measurements (Pima Indians Diabetes dataset).

> **Not for real-world use.** Educational demonstration only. This is not a medical device and must not be used to diagnose, screen or make decisions about anyone's health.

## Summary

- **Task:** classification (`Outcome`)
- **Selected model:** `logistic_regression` (chosen by 5-fold cross-validated ROC_AUC on the training split)
- **Data:** `diabetes/diabetes.csv`, 768 rows (768 after removing duplicates), 614 train / 154 test, SHA-256 `698c203a14aa`
- **Libraries:** scikit-learn 1.9.1, pandas 3.1.0, Python 3.13.15; seed 42

## Held-out test results

|  | roc_auc | accuracy | precision | recall | f1 | brier |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **Model** | **0.813** | **0.7078** | **0.6** | **0.5** | **0.5455** | **0.1754** |
| Naive baseline | 0.5 | 0.6494 | 0.0 | 0.0 | 0.0 | 0.2277 |

The baseline predicts the training mean (regression) or class prior (classification). The test split was used once, after model selection. With only 154 test rows one unusual record moves these numbers a lot; the cross-validation table below is the more reliable estimate.

## Model selection (training split, cross-validation)

| Candidate | CV roc_auc | CV accuracy | CV time |
| :-- | :-- | :-- | :-- |
| `logistic_regression` ✔ | 0.8434 ± 0.0185 | 0.7882 ± 0.0191 | 0.1 s |
| `random_forest` | 0.8272 ± 0.0252 | 0.7687 ± 0.0195 | 2.59 s |
| `hist_gradient_boosting` | 0.8227 ± 0.0175 | 0.7622 ± 0.0264 | 0.66 s |

## Inputs

| Name | Meaning | Unit | Accepted values | Seen in training |  |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `Pregnancies` | Pregnancies | - | 0 to 20 | 0 to 17 | required |
| `Glucose` | Plasma glucose | mg/dL | 40 to 400 | 56 to 199 | required |
| `BloodPressure` | Diastolic blood pressure | mmHg | 20 to 200 | 24 to 122 | required |
| `SkinThickness` | Triceps skin fold thickness | mm | 1 to 100 | 7 to 99 | optional |
| `Insulin` | 2-hour serum insulin | mu U/ml | 1 to 1,000 | 15 to 744 | optional |
| `BMI` | Body mass index | kg/m2 | 10 to 70 | 18.2 to 67.1 | required |
| `DiabetesPedigreeFunction` | Diabetes pedigree function | - | 0.01 to 3 | 0.084 to 2.329 | required |
| `Age` | Age | years | 18 to 100 | 21 to 81 | required |

Values outside *Seen in training* are accepted but returned with a warning, because the model is extrapolating.

## What drives the prediction

Permutation importance on the test split: how much ROC AUC drops when one feature's values are shuffled (larger = the model relies on it more):

| Feature | Importance |
| :-- | :-- |
| `Glucose` | 0.1667 |
| `BMI` | 0.0374 |
| `Pregnancies` | 0.0247 |
| `DiabetesPedigreeFunction` | 0.0091 |
| `Insulin` | 0.0024 |
| `Age` | 0.0013 |
| `BloodPressure` | 0.0012 |
| `SkinThickness` | 0.0003 |

## Limitations

- 768 patients from a single population (women of Pima heritage aged 21+); it does not generalise to other groups.
- In the source data zero is used as a missing-value code for glucose, blood pressure, skin thickness, insulin and BMI. These are treated as missing and imputed with the training median.
- Labels come from a 1990s study and a 5-fold ROC AUC of about 0.8 is typical for this dataset; it is not a clinical-grade score.
- Correlation, not causation: feature importance does not mean changing a feature changes the outcome.

## Reproduce

```bash
python -m predictors.train --model diabetes
```

Artifact SHA-256: `50e6909736e47a2bc0fa463e634af035bf44a4e3b009689be4a3363f912a7d0a`
