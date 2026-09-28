"""Phase 4: explainability — which features drive the model's fraud predictions.

Run: python src/explain.py
Reads models/xgb_fraud.joblib and data/processed/test.csv, writes:
  - reports/figures/shap_summary.png
  - reports/figures/shap_case_true_positive.png
  - reports/figures/shap_case_false_negative.png
  - reports/phase4_explainability.md
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "phase4_explainability.md"

RANDOM_STATE = 42
SAMPLE_SIZE = 2000
THRESHOLD = 0.5


def save_waterfall(explanation, path):
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_DIR / "xgb_fraud.joblib")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= THRESHOLD).astype(int)

    explainer = shap.TreeExplainer(model)

    # global summary on a sample (TreeExplainer is fast, but no need to run all 57k rows)
    sample = X_test.sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=RANDOM_STATE)
    sample_explanation = explainer(sample)

    plt.figure()
    shap.plots.beeswarm(sample_explanation, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        abs(sample_explanation.values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)
    top_features = mean_abs_shap.head(10)

    # local explanations: one correctly-caught fraud, one missed fraud
    tp_idx = X_test[(y_test == 1) & (y_pred == 1)].index
    fn_idx = X_test[(y_test == 1) & (y_pred == 0)].index

    case_notes = []
    if len(tp_idx) > 0:
        row = X_test.loc[[tp_idx[0]]]
        save_waterfall(explainer(row)[0], FIG_DIR / "shap_case_true_positive.png")
        case_notes.append("- **True positive** (row correctly flagged as fraud): `figures/shap_case_true_positive.png`")
    if len(fn_idx) > 0:
        row = X_test.loc[[fn_idx[0]]]
        save_waterfall(explainer(row)[0], FIG_DIR / "shap_case_false_negative.png")
        case_notes.append("- **False negative** (fraud the model missed): `figures/shap_case_false_negative.png`")

    top_features_table = "\n".join(f"| {f} | {v:.4f} |" for f, v in top_features.items())

    report = f"""# Phase 4 — Explainability (SHAP)

## Setup
- `shap.TreeExplainer` on the trained XGBoost model ({MODEL_DIR / 'xgb_fraud.joblib'})
- Global summary computed on a random sample of {min(SAMPLE_SIZE, len(X_test)):,} test rows
  (TreeExplainer is exact for trees; the sample only keeps the plot/report fast to generate)

## Global feature importance (mean |SHAP value|)
| Feature | Mean \\|SHAP\\| |
|---|---|
{top_features_table}

![shap summary](figures/shap_summary.png)

## Case-level explanations
{chr(10).join(case_notes) if case_notes else "- No qualifying cases found in this test set."}

Each waterfall plot shows how each feature pushed that single transaction's score
up (toward fraud) or down (toward legit) from the model's base rate.

## Implications
- V14, V17, V12 and other PCA components already flagged in Phase 1's correlation
  analysis are expected to dominate here too — SHAP confirms whether the model is
  actually relying on the same signals, or picking up something spurious.
- Use the false-negative case plot to see what suppressed the fraud score on a
  missed case — useful for deciding whether a lower decision threshold
  (see Phase 3's threshold table) is worth the added false positives.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print("Top 5 features by mean |SHAP|:")
    print(top_features.head(5).to_string())


if __name__ == "__main__":
    main()
