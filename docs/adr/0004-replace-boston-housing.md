# ADR 0004: Replace the Boston housing data with California Housing

Status: accepted

## Context
The house-price notebook called `sklearn.datasets.load_boston`, which was removed in scikit-learn 1.2. The dataset contains a column `B = 1000(Bk - 0.63)^2` where `Bk` is the proportion of Black residents, a variable constructed to model "racial self-segregation" and the stated reason for its removal. The repository had no CSV, so the model could not be retrained, and the 1.0.2-era pickle could not be loaded by the pinned library.

## Decision
Use the California Housing dataset (1990 census block groups, 20,640 rows, 8 numeric features, no demographic proxies), fetched once with `fetch_california_housing`, saved to `data/california_housing/california_housing.csv` (1.2 MB) and committed so training is offline and reproducible. A test asserts that no column `B` exists.

## Consequences
- The house model changes target and units (median value in units of $100,000) and the feature set; this is a different model from the original, not a re-training.
- The source data caps house values at $500,000, noted in the card.
- Its extreme census outliers (for example 1,243 occupants per household) forced the declared input ranges to be wider than a person would expect; a test checks that every training value is inside them so the API never rejects what the model learned from.
