import argparse
import ipaddress
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LDA classifier for DoH detection")
    parser.add_argument("--csv", type=str, default="../CSVs/Firefox/all.csv", help="Input CSV path")
    parser.add_argument("--target", type=str, default="DoH", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output folder")
    parser.add_argument(
        "--drop-identifiers",
        action="store_true",
        default=True,
        help="Drop identifier/leakage columns like source/destination IP and timestamp (default: enabled)",
    )
    parser.add_argument(
        "--keep-identifiers",
        action="store_true",
        help="Keep identifier columns instead of dropping them",
    )
    return parser.parse_args()


def encode_ip(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(int(ipaddress.ip_address(str(value).strip())))
    except ValueError:
        return np.nan


def encode_target(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "doh": 1,
        "non-doh": 0,
        "nondoh": 0,
    }
    y = s.map(mapping)
    if y.isna().any():
        y = pd.Series(pd.factorize(s)[0], index=series.index)
    return y.astype(int)


def preprocess_features(df: pd.DataFrame, drop_identifiers: bool) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if drop_identifiers:
        drop_candidates = [
            c
            for c in out.columns
            if c.lower() in {"sourceip", "destinationip", "timestamp"}
        ]
        if drop_candidates:
            out = out.drop(columns=drop_candidates)

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue

        name = col.lower()
        if "ip" in name:
            out[col] = out[col].map(encode_ip)
            continue

        if "timestamp" in name:
            ts = pd.to_datetime(out[col], errors="coerce")
            out[col] = ts.astype("int64") / 1e9
            out.loc[ts.isna(), col] = np.nan
            continue

        numeric_col = pd.to_numeric(out[col], errors="coerce")
        if numeric_col.notna().mean() >= 0.95:
            out[col] = numeric_col
        else:
            out[col] = pd.factorize(out[col])[0].astype(float)

    return out


def main() -> None:
    args = parse_args()
    drop_identifiers = args.drop_identifiers and not args.keep_identifiers

    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {csv_path}")

    y = encode_target(df[args.target])
    x = preprocess_features(df.drop(columns=[args.target]), drop_identifiers=drop_identifiers)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    lda_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )

    lda_pipeline.fit(x_train, y_train)
    y_pred = lda_pipeline.predict(x_test)

    metrics = {
        "model": "LDA",
        "rows": int(len(df)),
        "num_features": int(x.shape[1]),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }

    if hasattr(lda_pipeline, "predict_proba"):
        y_prob = lda_pipeline.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = None

    summary_df = pd.DataFrame(
        [
            {
                "model": "LDA",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        ]
    )

    summary_path = output_dir / "lda_summary.csv"
    details_path = output_dir / "lda_details.json"
    summary_df.to_csv(summary_path, index=False)
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Loaded dataset: {csv_path}")
    print(f"Rows: {len(df):,} | Features used: {x.shape[1]}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(
        "LDA metrics: "
        f"acc={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )
    print("Saved outputs:")
    print(f"- {summary_path}")
    print(f"- {details_path}")


if __name__ == "__main__":
    main()
