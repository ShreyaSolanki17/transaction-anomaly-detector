# Phase 2 — Preprocessing & Feature Engineering

## Split
- Train/test split: 80%/20%, stratified on `Class`, random_state=42
- Train (before SMOTE): 227,845 rows, 394 fraud
- Test (untouched, real-world distribution): 56,962 rows, 98 fraud

## Feature engineering
- Added `Time_since_last`: seconds since the previous transaction (rows sorted by `Time`).
  No per-account ID exists in this dataset, so this is computed on global transaction
  order rather than per-account, as flagged in Phase 1.

## Scaling
- `Time`, `Amount`, `Time_since_last` scaled with `StandardScaler`, fit on the training
  fold only and applied to test to avoid leakage. `V1`-`V28` are already PCA components
  (pre-scaled) and left as-is.

## Class imbalance
- SMOTE applied to the training fold only, after the split and after scaling.
- Train after SMOTE: 454,902 rows, 227451 fraud (50.0%)
- Test set is left at its natural ~0.17% fraud rate so Phase 3 metrics (PR-AUC,
  recall@precision) reflect real-world performance.

## Output
- `data/processed/train.csv`, `data/processed/test.csv` (gitignored; regenerate via
  `python src/preprocessing.py`)
