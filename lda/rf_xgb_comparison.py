import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from lda_classifier import encode_ip, encode_target, preprocess_features

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost is required. Install it with:  pip install xgboost")


CSV_PATH = Path(__file__).resolve().parent / "all.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
TARGET_COL = "DoH"
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df.columns = [str(c).strip() for c in df.columns]

y = encode_target(df[TARGET_COL])
X = preprocess_features(df.drop(columns=[TARGET_COL]), drop_identifiers=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

lda_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
    ]
)

X_train_lda = lda_pipeline.fit_transform(X_train, y_train)
X_test_lda = lda_pipeline.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
)
rf.fit(X_train_lda, y_train)
rf_pred = rf.predict(X_test_lda)
rf_prob = rf.predict_proba(X_test_lda)[:, 1]

scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1,
)
xgb.fit(X_train_lda, y_train)
xgb_pred = xgb.predict(X_test_lda)
xgb_prob = xgb.predict_proba(X_test_lda)[:, 1]


def compute_metrics(name, y_true, y_pred, y_prob):
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


rf_metrics = compute_metrics("Random Forest + LDA", y_test, rf_pred, rf_prob)
xgb_metrics = compute_metrics("XGBoost + LDA", y_test, xgb_pred, xgb_prob)

winner = "Random Forest + LDA" if rf_metrics["f1"] >= xgb_metrics["f1"] else "XGBoost + LDA"
print(f"Best model (by F1-score): {winner}")

summary_df = pd.DataFrame(
    [
        {k: v for k, v in rf_metrics.items() if k not in ("confusion_matrix", "classification_report")},
        {k: v for k, v in xgb_metrics.items() if k not in ("confusion_matrix", "classification_report")},
    ]
)
summary_path = OUTPUT_DIR / "rf_xgb_comparison_summary.csv"
details_path = OUTPUT_DIR / "rf_xgb_comparison_details.json"

summary_df.to_csv(summary_path, index=False)
with details_path.open("w", encoding="utf-8") as f:
    json.dump({"random_forest": rf_metrics, "xgboost": xgb_metrics}, f, indent=2)

print(f"Saved: {summary_path}")
print(f"Saved: {details_path}")

COLOR_RF = "#2196F3"   
COLOR_XGB = "#FF9800"  
BG_COLOR = "#1a1a2e"
CARD_COLOR = "#16213e"
TEXT_COLOR = "#e0e0e0"

metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
rf_vals = [rf_metrics["accuracy"], rf_metrics["precision"], rf_metrics["recall"],
           rf_metrics["f1"], rf_metrics["roc_auc"]]
xgb_vals = [xgb_metrics["accuracy"], xgb_metrics["precision"], xgb_metrics["recall"],
            xgb_metrics["f1"], xgb_metrics["roc_auc"]]

fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor=BG_COLOR)
fig.suptitle("Random Forest vs XGBoost  (LDA Feature Extraction)",
             fontsize=20, fontweight="bold", color=TEXT_COLOR, y=0.97)

ax1 = axes[0, 0]
ax1.set_facecolor(CARD_COLOR)
x = np.arange(len(metric_names))
w = 0.35
bars_rf = ax1.bar(x - w / 2, rf_vals, w, label="Random Forest", color=COLOR_RF, edgecolor="white", linewidth=0.5)
bars_xgb = ax1.bar(x + w / 2, xgb_vals, w, label="XGBoost", color=COLOR_XGB, edgecolor="white", linewidth=0.5)

for bar in bars_rf:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)
for bar in bars_xgb:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

ax1.set_xticks(x)
ax1.set_xticklabels(metric_names, color=TEXT_COLOR, fontsize=11)
ax1.set_ylim(0, 1.12)
ax1.set_ylabel("Score", color=TEXT_COLOR, fontsize=12)
ax1.set_title("Metric Comparison", color=TEXT_COLOR, fontsize=14, pad=10)
ax1.legend(loc="upper right", fontsize=10)
ax1.tick_params(colors=TEXT_COLOR)
ax1.spines["bottom"].set_color(TEXT_COLOR)
ax1.spines["left"].set_color(TEXT_COLOR)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


ax2 = axes[0, 1]
ax2.set_facecolor(CARD_COLOR)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
ax2.plot(fpr_rf, tpr_rf, color=COLOR_RF, lw=2.5,
         label=f"RF  (AUC={rf_metrics['roc_auc']:.4f})")
ax2.plot(fpr_xgb, tpr_xgb, color=COLOR_XGB, lw=2.5,
         label=f"XGB (AUC={xgb_metrics['roc_auc']:.4f})")
ax2.plot([0, 1], [0, 1], "w--", lw=0.8, alpha=0.4)
ax2.set_xlabel("False Positive Rate", color=TEXT_COLOR, fontsize=12)
ax2.set_ylabel("True Positive Rate", color=TEXT_COLOR, fontsize=12)
ax2.set_title("ROC Curve", color=TEXT_COLOR, fontsize=14, pad=10)
ax2.legend(loc="lower right", fontsize=10)
ax2.tick_params(colors=TEXT_COLOR)
ax2.spines["bottom"].set_color(TEXT_COLOR)
ax2.spines["left"].set_color(TEXT_COLOR)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


ax3 = axes[1, 0]
ax3.set_facecolor(CARD_COLOR)
cm_rf = np.array(rf_metrics["confusion_matrix"])
cm_xgb = np.array(xgb_metrics["confusion_matrix"])

labels = ["Non-DoH", "DoH"]
combined = np.zeros((2, 4))
combined[:, :2] = cm_rf
combined[:, 2:] = cm_xgb

im = ax3.imshow(combined, cmap="Blues", aspect="auto")
for i in range(2):
    for j in range(4):
        ax3.text(j, i, f"{int(combined[i, j]):,}",
                 ha="center", va="center", fontsize=11, fontweight="bold",
                 color="white" if combined[i, j] > combined.max() * 0.5 else "black")

ax3.set_xticks([0.5, 2.5])
ax3.set_xticklabels(["Random Forest", "XGBoost"], fontsize=11, color=TEXT_COLOR)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(labels, fontsize=11, color=TEXT_COLOR)
ax3.set_title("Confusion Matrices", color=TEXT_COLOR, fontsize=14, pad=10)

ax3.axvline(x=1.5, color="white", linewidth=2)

for j, lbl in enumerate(["Pred\nNon-DoH", "Pred\nDoH", "Pred\nNon-DoH", "Pred\nDoH"]):
    ax3.text(j, -0.7, lbl, ha="center", va="center", fontsize=8, color=TEXT_COLOR)

ax3.tick_params(colors=TEXT_COLOR)


ax4 = axes[1, 1]
ax4.remove()
ax4 = fig.add_subplot(2, 2, 4, polar=True, facecolor=CARD_COLOR)

angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
rf_vals_r = rf_vals + [rf_vals[0]]
xgb_vals_r = xgb_vals + [xgb_vals[0]]
angles += [angles[0]]

ax4.plot(angles, rf_vals_r, "o-", color=COLOR_RF, lw=2, label="Random Forest")
ax4.fill(angles, rf_vals_r, alpha=0.15, color=COLOR_RF)
ax4.plot(angles, xgb_vals_r, "o-", color=COLOR_XGB, lw=2, label="XGBoost")
ax4.fill(angles, xgb_vals_r, alpha=0.15, color=COLOR_XGB)
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metric_names, color=TEXT_COLOR, fontsize=10)
ax4.set_ylim(0, 1.05)
ax4.set_title("Performance Radar", color=TEXT_COLOR, fontsize=14, pad=20)
ax4.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
ax4.tick_params(colors=TEXT_COLOR)

plt.tight_layout(rect=[0, 0, 1, 0.94])
chart_path = OUTPUT_DIR / "rf_vs_xgb_comparison.png"
plt.savefig(chart_path, dpi=200, facecolor=BG_COLOR, bbox_inches="tight")
print(f"Saved chart: {chart_path}")
plt.show()
print("\nDone!")
