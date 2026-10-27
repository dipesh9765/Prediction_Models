# Data

All files are small public teaching datasets. Training reads them from here, so it needs no network access.

| Folder | File(s) | Rows | Origin | Notes |
| :-- | :-- | :-- | :-- | :-- |
| `car/` | `car_data.csv` | 301 (299 distinct) | CarDekho used-car listings, redistributed on Kaggle | Prices in lakh INR, 2003-2018 |
| `diabetes/` | `diabetes.csv` | 768 | Pima Indians Diabetes Database (UCI / NIDDK) | Zeros in glucose, blood pressure, skin thickness, insulin and BMI mean "not measured" |
| `exercise/` | `exercise.csv`, `calories.csv` | 15,000 | Public fitness dataset on Kaggle ("Calories Burnt Prediction") | Joined on `User_ID`; looks synthetic or heavily smoothed |
| `medical_insurance/` | `insurance.csv` | 1,338 (1,337 distinct) | "Medical Cost Personal Datasets" from Machine Learning with R (Lantz), on Kaggle | US charges in USD |
| `california_housing/` | `california_housing.csv` | 20,640 | 1990 US census, Pace and Barry (1997), via `sklearn.datasets.fetch_california_housing` | Target is median value in units of $100,000, capped at 5.0 |

The original repository did not document sources. The descriptions above are best-effort attributions and should be checked against each dataset's own licence before redistribution. Each model card records the SHA-256 of the exact files it was trained on.
