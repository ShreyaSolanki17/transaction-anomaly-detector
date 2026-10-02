# Transaction Anomaly Detector

Fraud detection on the Kaggle `mlg-ulb/creditcardfraud` dataset (284,807 transactions,
0.173% fraud). Built in four phases; each writes a report to `reports/`.

## Setup

```
pip install -r requirements.txt
```

Place the raw dataset at `data/raw/creditcard.csv` (not committed — see `.gitignore`).

## Pipeline

Run in order; each step reads the previous step's output.

| Step | Script | Output |
|---|---|---|
| 1. EDA | `python src/eda.py` | `reports/phase1_eda.md`, `reports/figures/*.png` |
| 2. Preprocessing | `python src/preprocessing.py` | `data/processed/{train,test}.csv`, `reports/phase2_preprocessing.md` |
| 3. Training | `python src/train.py` | `models/xgb_fraud.joblib`, `reports/phase3_model.md` |
| 4. Explainability | `python src/explain.py` | `reports/figures/shap_*.png`, `reports/phase4_explainability.md` |

## Approach

- **Imbalance**: SMOTE applied to the training fold only (after the split); the test
  set keeps its real ~0.17% fraud rate so metrics reflect real-world performance.
- **Metric**: PR-AUC and recall-at-precision, not accuracy — a model predicting
  "always legit" would score 99.83% accuracy while catching zero fraud.
- **Model**: XGBoost classifier. PR-AUC 0.857 on the held-out test set; see
  `reports/phase3_model.md` for the full precision/recall threshold comparison.
- **Explainability**: SHAP confirms the model relies on the same PCA components
  (V14, V4, V12) that Phase 1's correlation analysis flagged as most predictive.
