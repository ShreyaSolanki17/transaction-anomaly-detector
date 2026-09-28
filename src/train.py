"""Phase 3: train and evaluate a fraud classifier.

Run: python src/train.py
Reads data/processed/{train,test}.csv, writes:
  - models/xgb_fraud.joblib
  - reports/phase3_model.md
"""
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "phase3_model.md"

RANDOM_STATE = 42
PRECISION_TARGETS = [0.5, 0.9]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def recall_at_precision(precision, recall, target):
    """Best recall among points where precision >= target."""
    mask = precision >= target
    return recall[mask].max() if mask.any() else 0.0


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="aucpr")
    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_scores)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    recalls = {t: recall_at_precision(precision, recall, t) for t in PRECISION_TARGETS}

    y_pred = (y_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    threshold_rows = []
    for t in THRESHOLDS:
        pred = (y_scores >= t).astype(int)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_test, pred).ravel()
        t_precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) else 0.0
        t_recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) else 0.0
        threshold_rows.append((t, t_precision, t_recall, t_tp, t_fp, t_fn))

    joblib.dump(model, MODEL_DIR / "xgb_fraud.joblib")

    threshold_table = "\n".join(
        f"| {t:.1f} | {p:.3f} | {r:.3f} | {tp_:,} | {fp_:,} | {fn_:,} |"
        for t, p, r, tp_, fp_, fn_ in threshold_rows
    )

    report = f"""# Phase 3 — Model Training & Evaluation

## Model
- XGBClassifier trained on SMOTE-balanced training set ({len(train_df):,} rows)
- Evaluated on untouched test set ({len(test_df):,} rows, {int(y_test.sum())} real fraud)
  at its natural ~0.17% fraud rate, per Phase 1/2 notes (accuracy is not a useful metric here).

## Results
- PR-AUC: {pr_auc:.4f}
- Recall at precision >= 0.5: {recalls[0.5]:.4f}
- Recall at precision >= 0.9: {recalls[0.9]:.4f}

## Confusion matrix (threshold = 0.5)
- True negatives: {tn:,}, False positives: {fp:,}
- False negatives: {fn:,}, True positives: {tp:,}

## Threshold comparison
Same model, scored at different cutoffs — pick the threshold that fits the
business cost of a false positive (blocked legit transaction) vs. a false
negative (missed fraud).

| Threshold | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
{threshold_table}

## Output
- `models/xgb_fraud.joblib` (gitignored; regenerate via `python src/train.py`)
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Recall@precision>=0.5: {recalls[0.5]:.4f}, Recall@precision>=0.9: {recalls[0.9]:.4f}")
    print(f"Confusion matrix @0.5 -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")


if __name__ == "__main__":
    main()
