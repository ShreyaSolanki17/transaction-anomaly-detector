"""Phase 1: EDA on the Kaggle credit card fraud dataset.

Run: python src/eda.py
Writes findings to reports/phase1_eda.md and plots to reports/figures/.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "phase1_eda.md"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    n_fraud = int(df["Class"].sum())
    n_legit = n_rows - n_fraud
    fraud_pct = 100 * n_fraud / n_rows

    # class balance
    plt.figure(figsize=(4, 4))
    df["Class"].value_counts().plot(kind="bar")
    plt.xticks([0, 1], ["Legit (0)", "Fraud (1)"], rotation=0)
    plt.title("Class balance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png")
    plt.close()

    # amount distribution, fraud vs legit
    plt.figure(figsize=(8, 4))
    sns.histplot(df[df.Class == 0]["Amount"], bins=50, color="steelblue", label="Legit", stat="density", log_scale=(False, True))
    sns.histplot(df[df.Class == 1]["Amount"], bins=50, color="crimson", label="Fraud", stat="density", log_scale=(False, True))
    plt.xlim(0, 2000)
    plt.legend()
    plt.title("Transaction amount distribution (legit vs fraud)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "amount_distribution.png")
    plt.close()

    # correlation of each feature with Class
    corr_with_class = df.corr(numeric_only=True)["Class"].drop("Class").sort_values()
    plt.figure(figsize=(6, 8))
    corr_with_class.plot(kind="barh")
    plt.title("Feature correlation with fraud label")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_with_class.png")
    plt.close()

    top_pos = corr_with_class.tail(5)
    top_neg = corr_with_class.head(5)

    amount_stats = df.groupby("Class")["Amount"].describe()
    time_span_hours = (df["Time"].max() - df["Time"].min()) / 3600

    report = f"""# Phase 1 — EDA Findings

Dataset: Kaggle `mlg-ulb/creditcardfraud` ({n_rows:,} rows x {n_cols} columns)

## Shape & quality
- Rows: {n_rows:,}, Columns: {n_cols}
- Missing values: {n_missing} (none)
- Columns: `Time`, `V1`..`V28` (PCA-anonymized), `Amount`, `Class`

## Class imbalance
- Legit: {n_legit:,} ({100 - fraud_pct:.3f}%)
- Fraud: {n_fraud:,} ({fraud_pct:.3f}%)
- This is a ~{n_legit // n_fraud}:1 imbalance — standard accuracy is meaningless here;
  use PR-AUC / recall-at-precision as the evaluation metric, and address imbalance
  in Phase 2 (SMOTE / class weighting) before training the supervised model.

![class balance](figures/class_balance.png)

## Amount
- Legit mean amount: {amount_stats.loc[0, 'mean']:.2f}, median: {amount_stats.loc[0, '50%']:.2f}
- Fraud mean amount: {amount_stats.loc[1, 'mean']:.2f}, median: {amount_stats.loc[1, '50%']:.2f}
- Fraud amounts skew lower and are more concentrated in the small-transaction range
  (see plot) — `Amount` alone is a weak signal but worth scaling and keeping as a feature.

![amount distribution](figures/amount_distribution.png)

## Time
- Data spans {time_span_hours:.1f} hours ({time_span_hours / 24:.1f} days) as seconds elapsed since the first transaction.
- No per-user/account ID is present in this dataset — `Time` is the only ordering signal,
  so Phase 2's rolling/lag features will be computed on transaction order rather than
  a real per-account key (this dataset doesn't provide one; noted as a limitation, not
  something to work around here).

## Feature correlation with fraud (`Class`)
- V1-V28 are already PCA components (anonymized for privacy), so they aren't individually
  interpretable, but several are strongly predictive:
  - Most positively correlated: {', '.join(f'{k} ({v:.2f})' for k, v in top_pos.items())}
  - Most negatively correlated: {', '.join(f'{k} ({v:.2f})' for k, v in top_neg.items())}

![correlation with class](figures/correlation_with_class.png)

## Implications for later phases
- **Phase 2**: handle severe class imbalance (SMOTE on training fold only, or class
  weights); scale `Amount`/`Time`, engineer rolling features on transaction order.
- **Phase 3/4**: PR-AUC and recall@precision are the metrics to optimize for and report,
  not accuracy — with {fraud_pct:.3f}% positive rate, a model predicting all-legit would
  score {100 - fraud_pct:.2f}% accuracy while catching zero fraud.
- No missing values, no dtype cleanup needed — data is already clean.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Fraud rate: {fraud_pct:.4f}% ({n_fraud}/{n_rows})")


if __name__ == "__main__":
    main()
