"""Phase 2: preprocessing & feature engineering for the fraud dataset.

Run: python src/preprocessing.py
Reads data/raw/creditcard.csv, writes:
  - data/processed/train.csv, data/processed/test.csv (SMOTE applied to train only)
  - reports/phase2_preprocessing.md
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "phase2_preprocessing.md"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # transaction-order feature: seconds since previous transaction (no account ID exists to group by)
    df = df.sort_values("Time").reset_index(drop=True)
    df["Time_since_last"] = df["Time"].diff().fillna(0)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # scale Amount/Time*, fit on train only to avoid leakage
    scale_cols = ["Time", "Amount", "Time_since_last"]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    n_train_before = len(X_train)
    fraud_before = int(y_train.sum())

    # SMOTE on training fold only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    train_df = X_train_res.copy()
    train_df["Class"] = y_train_res
    test_df = X_test.copy()
    test_df["Class"] = y_test

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    report = f"""# Phase 2 — Preprocessing & Feature Engineering

## Split
- Train/test split: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, stratified on `Class`, random_state={RANDOM_STATE}
- Train (before SMOTE): {n_train_before:,} rows, {fraud_before} fraud
- Test (untouched, real-world distribution): {len(X_test):,} rows, {int(y_test.sum())} fraud

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
- Train after SMOTE: {len(train_df):,} rows, {int(y_train_res.sum())} fraud ({int(y_train_res.sum())/len(train_df):.1%})
- Test set is left at its natural ~0.17% fraud rate so Phase 3 metrics (PR-AUC,
  recall@precision) reflect real-world performance.

## Output
- `data/processed/train.csv`, `data/processed/test.csv` (gitignored; regenerate via
  `python src/preprocessing.py`)
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Train: {len(train_df)} rows ({int(y_train_res.sum())} fraud) -> {PROCESSED_DIR / 'train.csv'}")
    print(f"Test: {len(test_df)} rows ({int(y_test.sum())} fraud) -> {PROCESSED_DIR / 'test.csv'}")


if __name__ == "__main__":
    main()
