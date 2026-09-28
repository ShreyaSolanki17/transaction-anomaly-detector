# Phase 3 — Model Training & Evaluation

## Model
- XGBClassifier trained on SMOTE-balanced training set (454,902 rows)
- Evaluated on untouched test set (56,962 rows, 98 real fraud)
  at its natural ~0.17% fraud rate, per Phase 1/2 notes (accuracy is not a useful metric here).

## Results
- PR-AUC: 0.8570
- Recall at precision >= 0.5: 0.8673
- Recall at precision >= 0.9: 0.7959

## Confusion matrix (threshold = 0.5)
- True negatives: 56,839, False positives: 25
- False negatives: 14, True positives: 84

## Threshold comparison
Same model, scored at different cutoffs — pick the threshold that fits the
business cost of a false positive (blocked legit transaction) vs. a false
negative (missed fraud).

| Threshold | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| 0.1 | 0.515 | 0.867 | 85 | 80 | 13 |
| 0.2 | 0.616 | 0.867 | 85 | 53 | 13 |
| 0.3 | 0.714 | 0.867 | 85 | 34 | 13 |
| 0.4 | 0.726 | 0.867 | 85 | 32 | 13 |
| 0.5 | 0.771 | 0.857 | 84 | 25 | 14 |
| 0.6 | 0.808 | 0.857 | 84 | 20 | 14 |
| 0.7 | 0.812 | 0.837 | 82 | 19 | 16 |
| 0.8 | 0.860 | 0.816 | 80 | 13 | 18 |
| 0.9 | 0.898 | 0.806 | 79 | 9 | 19 |

## Output
- `models/xgb_fraud.joblib` (gitignored; regenerate via `python src/train.py`)
