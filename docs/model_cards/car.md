# Model card: Used car resale price

Estimates what a used car in the Indian market will sell for, from its age, mileage, fuel, transmission and the price of a comparable new car.

## Summary

- **Task:** regression (`Selling_Price`, in lakh INR)
- **Selected model:** `random_forest` (chosen by 5-fold cross-validated RMSE on the training split)
- **Data:** `car/car_data.csv`, 301 rows (299 after removing duplicates), 239 train / 60 test, SHA-256 `00b477def120`
- **Libraries:** scikit-learn 1.9.1, pandas 3.1.0, Python 3.13.15; seed 42

## Held-out test results

|  | rmse | mae | r2 |
| :-- | :-- | :-- | :-- |
| **Model** | **3.1661** | **1.3551** | **0.6111** |
| Naive baseline | 5.0768 | 3.3269 | -0.0 |

The baseline predicts the training mean (regression) or class prior (classification). The test split was used once, after model selection. With only 60 test rows one unusual record moves these numbers a lot; the cross-validation table below is the more reliable estimate.

## Model selection (training split, cross-validation)

| Candidate | CV rmse | CV r2 | CV time |
| :-- | :-- | :-- | :-- |
| `ridge` | 1.6384 ± 0.4307 | 0.8858 ± 0.0285 | 0.08 s |
| `random_forest` ✔ | 1.5739 ± 0.8035 | 0.8919 ± 0.0668 | 1.05 s |
| `hist_gradient_boosting` | 2.1483 ± 0.936 | 0.8049 ± 0.0954 | 0.59 s |

## Prediction interval

The API returns a 90% interval built from out-of-fold training residuals: prediction -1.2436 to +1.7715 lakh INR. Observed coverage on the test split: **80%**. The interval is the same width for every input, so it is too wide for easy cases and too narrow for hard ones.

## Inputs

| Name | Meaning | Unit | Accepted values | Seen in training |  |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `Year` | Year of manufacture | - | 1,990 to 2,030 | 2,003 to 2,017 | required |
| `Present_Price` | Price of the equivalent new car | lakh INR | 0.1 to 200 | 0.32 to 92.6 | required |
| `Kms_Driven` | Kilometres driven | km | 0 to 1,000,000 | 500 to 213,000 | required |
| `Fuel_Type` | Fuel | - | Petrol, Diesel, CNG | - | required |
| `Seller_Type` | Seller | - | Dealer, Individual | - | required |
| `Transmission` | Transmission | - | Manual, Automatic | - | required |
| `Owner` | Previous owners | - | 0 to 3 | 0 to 1 | required |

Values outside *Seen in training* are accepted but returned with a warning, because the model is extrapolating.

## What drives the prediction

Permutation importance on the test split: how much RMSE (lakh INR) rises when one feature's values are shuffled (larger = the model relies on it more):

| Feature | Importance |
| :-- | :-- |
| `Present_Price` | 3.7598 |
| `Year` | 0.6573 |
| `Kms_Driven` | 0.239 |
| `Fuel_Type` | 0.0218 |
| `Transmission` | 0.0073 |
| `Seller_Type` | 0.0044 |
| `Owner` | 0.0 |

## Limitations

- Only 299 distinct listings, almost all from 2003-2018; prices outside that period are extrapolations.
- The data comes from one marketplace and does not include model, city or condition.
- Correlation, not causation: feature importance does not mean changing a feature changes the outcome.

## Reproduce

```bash
python -m predictors.train --model car
```

Artifact SHA-256: `f5a698bb1e61d9a4a727480ae436a98f282b9cc85d20d610fd233578b79978e5`
