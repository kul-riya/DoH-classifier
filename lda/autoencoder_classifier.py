"""
autoencoder_classifier.py
─────────────────────────
PyTorch autoencoder for DoH traffic detection.

What this module does
─────────────────────
1. Trains a deep autoencoder on flow-level features (unsupervised).
2. Extracts the bottleneck (latent) representation for every sample.
3. Trains Random Forest + XGBoost on those latent features, exactly
   mirroring rf_xgb_comparison.py so results are directly comparable.
4. Computes per-sample reconstruction error — useful as an anomaly
   score independent of any label.
5. Visualises the latent space (2-D UMAP projection, coloured by DoH),
   analogous to the PCA scatter in doh.py.
6. Saves a side-by-side metric comparison: AE pipeline vs LDA pipeline.

Usage
─────
    python autoencoder_classifier.py
    python autoencoder_classifier.py --csv path/to/all.csv --latent-dim 32
    python autoencoder_classifier.py --epochs 60 --batch-size 256 --no-umap
    python autoencoder_classifier.py --compare-lda results/rf_xgb_comparison_summary.csv

Dependencies
────────────
    pip install torch scikit-learn xgboost matplotlib pandas numpy
    pip install umap-learn          # optional — skipped if --no-umap
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost is required:  pip install xgboost")

# Reuse helpers already written in lda_classifier.py
from lda_classifier import encode_target, preprocess_features


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autoencoder-based DoH classifier (PyTorch)")
    p.add_argument("--csv",          default="../all.csv",   help="Input CSV path")
    p.add_argument("--target",       default="DoH",          help="Target column name")
    p.add_argument("--test-size",    type=float, default=0.2)
    p.add_argument("--random-state", type=int,   default=42)
    p.add_argument("--output-dir",   default="results",      help="Output folder")

    # Autoencoder architecture
    p.add_argument("--latent-dim",   type=int,   default=32,
                   help="Bottleneck size (latent dimension). Default: 32")
    p.add_argument("--hidden-dims",  nargs="+", type=int, default=[256, 128, 64],
                   help="Encoder hidden layer sizes (decoder mirrors). Default: 256 128 64")
    p.add_argument("--dropout",      type=float, default=0.2)

    # Training
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--patience",     type=int,   default=8,
                   help="Early-stopping patience (epochs without val-loss improvement)")

    # Downstream classifiers
    p.add_argument("--rf-estimators", type=int, default=200)
    p.add_argument("--xgb-estimators", type=int, default=200)

    # Extras
    p.add_argument("--no-umap",  action="store_true",
                   help="Skip UMAP visualisation (use PCA fallback instead)")
    p.add_argument("--compare-lda", default=None,
                   help="Path to rf_xgb_comparison_summary.csv for side-by-side table")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Compute device. 'auto' picks CUDA > MPS > CPU.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Autoencoder model
# ─────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Symmetric autoencoder with BatchNorm + Dropout.

    Encoder: input_dim → hidden_dims → latent_dim
    Decoder: latent_dim → hidden_dims (reversed) → input_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # ── Encoder ─────────────────────────────────────────
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ─────────────────────────────────────────
        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────

def select_device(preference: str) -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def train_autoencoder(
    model: Autoencoder,
    x_train_t: torch.Tensor,
    x_val_t: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(x_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        train_losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ─────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            x_val_d = x_val_t.to(device)
            x_hat_val, _ = model(x_val_d)
            val_loss = criterion(x_hat_val, x_val_d).item()

        train_loss = float(np.mean(train_losses))
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}"
            )

        # ── Early stopping ───────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def extract_latent(
    model: Autoencoder,
    x_t: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    model.to(device)
    parts = []
    for i in range(0, len(x_t), batch_size):
        xb = x_t[i : i + batch_size].to(device)
        parts.append(model.encode(xb).cpu().numpy())
    return np.vstack(parts)


@torch.no_grad()
def compute_reconstruction_error(
    model: Autoencoder,
    x_t: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Per-sample MSE between input and reconstruction."""
    model.eval()
    model.to(device)
    errors = []
    for i in range(0, len(x_t), batch_size):
        xb = x_t[i : i + batch_size].to(device)
        x_hat, _ = model(xb)
        mse = ((x_hat - xb) ** 2).mean(dim=1).cpu().numpy()
        errors.append(mse)
    return np.concatenate(errors)


# ─────────────────────────────────────────────────────────────
# Classifier helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(name: str, y_true, y_pred, y_prob) -> dict:
    return {
        "model": name,
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

BG    = "#1a1a2e"
CARD  = "#16213e"
TEXT  = "#e0e0e0"
C_RF  = "#2196F3"
C_XGB = "#FF9800"
C_DOH = "#E91E63"
C_NON = "#00BCD4"


def plot_training_curve(history: list[dict], out_path: Path) -> None:
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(CARD)
    ax.plot(epochs, train_loss, color=C_RF,  lw=2,   label="Train loss")
    ax.plot(epochs, val_loss,   color=C_XGB, lw=2,   label="Val loss")
    ax.set_xlabel("Epoch", color=TEXT)
    ax.set_ylabel("MSE loss", color=TEXT)
    ax.set_title("Autoencoder training curve", color=TEXT, fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(colors=TEXT)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(TEXT)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_reconstruction_error(errors: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(CARD)

    for val, label, color in [(0, "Non-DoH", C_NON), (1, "DoH", C_DOH)]:
        mask = labels == val
        ax.hist(errors[mask], bins=80, alpha=0.65, color=color,
                label=f"{label} (n={mask.sum():,})", density=True)

    ax.set_xlabel("Reconstruction error (MSE)", color=TEXT)
    ax.set_ylabel("Density", color=TEXT)
    ax.set_title("Reconstruction error distribution", color=TEXT, fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(colors=TEXT)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(TEXT)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_latent_space(z: np.ndarray, labels: np.ndarray, method: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(CARD)

    for val, label, color in [(0, "Non-DoH", C_NON), (1, "DoH", C_DOH)]:
        mask = labels == val
        ax.scatter(z[mask, 0], z[mask, 1],
                   c=color, label=label, s=6, alpha=0.5, linewidths=0)

    ax.set_xlabel(f"{method} dim 1", color=TEXT)
    ax.set_ylabel(f"{method} dim 2", color=TEXT)
    ax.set_title(f"Latent space ({method} projection)", color=TEXT, fontsize=13)
    ax.legend(fontsize=10, markerscale=3)
    ax.tick_params(colors=TEXT)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(TEXT)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_comparison_dashboard(
    rf_m: dict, xgb_m: dict, history: list[dict],
    y_test: np.ndarray, rf_prob: np.ndarray, xgb_prob: np.ndarray,
    out_path: Path,
    lda_summary_path: str | None = None,
) -> None:
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rf_vals  = [rf_m["accuracy"],  rf_m["precision"],  rf_m["recall"],  rf_m["f1"],  rf_m["roc_auc"]]
    xgb_vals = [xgb_m["accuracy"], xgb_m["precision"], xgb_m["recall"], xgb_m["f1"], xgb_m["roc_auc"]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), facecolor=BG)
    fig.suptitle("Autoencoder pipeline — RF vs XGBoost", color=TEXT, fontsize=18, y=0.97)

    # ── Top-left: metric bars ────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor(CARD)
    x  = np.arange(len(metric_names))
    w  = 0.32
    b1 = ax1.bar(x - w / 2, rf_vals,  w, label="RF + AE",  color=C_RF,  edgecolor="white", lw=0.4)
    b2 = ax1.bar(x + w / 2, xgb_vals, w, label="XGB + AE", color=C_XGB, edgecolor="white", lw=0.4)
    for bars in (b1, b2):
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.008,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=8, color=TEXT)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, color=TEXT, fontsize=11)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Score", color=TEXT)
    ax1.set_title("Metric comparison", color=TEXT, fontsize=13)
    ax1.legend(fontsize=10)
    ax1.tick_params(colors=TEXT)
    for s in ["top", "right"]:  ax1.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax1.spines[s].set_color(TEXT)

    # ── Top-right: ROC curves ────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor(CARD)
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_prob)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
    ax2.plot(fpr_rf,  tpr_rf,  color=C_RF,  lw=2.5, label=f"RF  (AUC={rf_m['roc_auc']:.4f})")
    ax2.plot(fpr_xgb, tpr_xgb, color=C_XGB, lw=2.5, label=f"XGB (AUC={xgb_m['roc_auc']:.4f})")
    ax2.plot([0, 1], [0, 1], "w--", lw=0.8, alpha=0.4)
    ax2.set_xlabel("FPR", color=TEXT)
    ax2.set_ylabel("TPR", color=TEXT)
    ax2.set_title("ROC curve", color=TEXT, fontsize=13)
    ax2.legend(fontsize=10)
    ax2.tick_params(colors=TEXT)
    for s in ["top", "right"]:   ax2.spines[s].set_visible(False)
    for s in ["bottom", "left"]:  ax2.spines[s].set_color(TEXT)

    # ── Bottom-left: training loss curve ────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor(CARD)
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    ax3.plot(epochs, train_loss, color=C_RF,  lw=2, label="Train MSE")
    ax3.plot(epochs, val_loss,   color=C_XGB, lw=2, label="Val MSE")
    ax3.set_xlabel("Epoch", color=TEXT)
    ax3.set_ylabel("MSE loss", color=TEXT)
    ax3.set_title("Autoencoder training curve", color=TEXT, fontsize=13)
    ax3.legend(fontsize=10)
    ax3.tick_params(colors=TEXT)
    for s in ["top", "right"]:   ax3.spines[s].set_visible(False)
    for s in ["bottom", "left"]:  ax3.spines[s].set_color(TEXT)

    # ── Bottom-right: comparison table vs LDA pipeline ──────
    ax4 = axes[1, 1]
    ax4.set_facecolor(CARD)
    ax4.axis("off")

    rows = [
        ["Model",           "Acc",    "Prec",   "Rec",    "F1",     "AUC"],
        ["RF + AE",
         f"{rf_m['accuracy']:.3f}", f"{rf_m['precision']:.3f}",
         f"{rf_m['recall']:.3f}",  f"{rf_m['f1']:.3f}",  f"{rf_m['roc_auc']:.3f}"],
        ["XGB + AE",
         f"{xgb_m['accuracy']:.3f}", f"{xgb_m['precision']:.3f}",
         f"{xgb_m['recall']:.3f}",  f"{xgb_m['f1']:.3f}", f"{xgb_m['roc_auc']:.3f}"],
    ]

    # Optionally append LDA rows for comparison
    if lda_summary_path:
        try:
            lda_df = pd.read_csv(lda_summary_path)
            for _, row in lda_df.iterrows():
                rows.append([
                    f"{row['model']} (LDA)",
                    f"{row['accuracy']:.3f}", f"{row['precision']:.3f}",
                    f"{row['recall']:.3f}",   f"{row['f1']:.3f}",
                    f"{row['roc_auc']:.3f}",
                ])
        except Exception:
            pass

    tbl = ax4.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(CARD if r % 2 == 0 else "#0f3460")
        cell.set_text_props(color=TEXT)
        cell.set_edgecolor("#2a2a4a")
    ax4.set_title("Pipeline comparison (AE vs LDA)", color=TEXT, fontsize=13, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    device     = select_device(args.device)
    rng        = args.random_state
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  DoH Autoencoder — PyTorch")
    print(f"  Device        : {device}")
    print(f"  Latent dim    : {args.latent_dim}")
    print(f"  Hidden layers : {args.hidden_dims}")
    print(f"{'─'*55}\n")

    # ── 1. Load & preprocess ─────────────────────────────────
    csv_path = Path(args.csv).resolve()
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    y = encode_target(df[args.target])
    X = preprocess_features(df.drop(columns=[args.target]), drop_identifiers=True)

    print(f"Loaded: {csv_path}")
    print(f"Rows: {len(df):,}  |  Features: {X.shape[1]}  |  DoH ratio: {y.mean():.3f}\n")

    # ── 2. Train/test split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=rng, stratify=y
    )

    # ── 3. Impute → scale ────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_test_sc  = scaler.transform(X_test_imp).astype(np.float32)

    # Use 10% of train for validation during AE training
    val_split = int(len(X_train_sc) * 0.1)
    X_val_sc  = X_train_sc[:val_split]
    X_tr_sc   = X_train_sc[val_split:]

    X_tr_t  = torch.from_numpy(X_tr_sc)
    X_val_t = torch.from_numpy(X_val_sc)
    X_test_t = torch.from_numpy(X_test_sc)
    X_all_t  = torch.from_numpy(X_train_sc)  # full train for latent extraction

    input_dim = X_train_sc.shape[1]

    # ── 4. Build & train autoencoder ────────────────────────
    print("Training autoencoder …")
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    history = train_autoencoder(
        model, X_tr_t, X_val_t,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    # Save model weights
    model_path = output_dir / "autoencoder_weights.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    # ── 5. Extract latent representations ───────────────────
    print("\nExtracting latent features …")
    z_train = extract_latent(model, X_all_t,  device)
    z_test  = extract_latent(model, X_test_t, device)
    print(f"  z_train shape: {z_train.shape}")
    print(f"  z_test  shape: {z_test.shape}")

    # ── 6. Reconstruction error ──────────────────────────────
    print("\nComputing reconstruction errors …")
    err_train = compute_reconstruction_error(model, X_all_t,  device)
    err_test  = compute_reconstruction_error(model, X_test_t, device)

    recon_path = output_dir / "ae_reconstruction_error.csv"
    pd.DataFrame({
        "reconstruction_error": np.concatenate([err_train, err_test]),
        "split": ["train"] * len(err_train) + ["test"] * len(err_test),
        "label": np.concatenate([y_train.values, y_test.values]),
    }).to_csv(recon_path, index=False)
    print(f"  Saved: {recon_path}")

    # ── 7. Train RF + XGBoost on latent features ────────────
    print("\nTraining classifiers on latent features …")

    rf = RandomForestClassifier(
        n_estimators=args.rf_estimators,
        max_depth=15,
        random_state=rng,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(z_train, y_train)
    rf_pred = rf.predict(z_test)
    rf_prob = rf.predict_proba(z_test)[:, 1]

    scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb = XGBClassifier(
        n_estimators=args.xgb_estimators,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        random_state=rng,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb.fit(z_train, y_train)
    xgb_pred = xgb.predict(z_test)
    xgb_prob = xgb.predict_proba(z_test)[:, 1]

    rf_metrics  = compute_metrics("RF + AE",  y_test, rf_pred,  rf_prob)
    xgb_metrics = compute_metrics("XGB + AE", y_test, xgb_pred, xgb_prob)

    winner = "RF + AE" if rf_metrics["f1"] >= xgb_metrics["f1"] else "XGB + AE"
    print(f"\n  Best model (by F1): {winner}")
    for m in (rf_metrics, xgb_metrics):
        print(
            f"  {m['model']:10s}  acc={m['accuracy']:.4f}  "
            f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  "
            f"f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}"
        )

    # Save summary CSV & detailed JSON
    summary_df = pd.DataFrame([
        {k: v for k, v in m.items() if k not in ("confusion_matrix", "classification_report")}
        for m in (rf_metrics, xgb_metrics)
    ])
    summary_path = output_dir / "ae_comparison_summary.csv"
    details_path = output_dir / "ae_comparison_details.json"
    summary_df.to_csv(summary_path, index=False)
    with details_path.open("w") as f:
        json.dump({"random_forest": rf_metrics, "xgboost": xgb_metrics}, f, indent=2)
    print(f"\n  Saved: {summary_path}")
    print(f"  Saved: {details_path}")

    # ── 8. Latent space visualisation ───────────────────────
    print("\nVisualising latent space …")
    z_all    = np.vstack([z_train, z_test])
    y_all    = np.concatenate([y_train.values, y_test.values])

    use_umap = not args.no_umap
    proj_2d  = None
    method   = "PCA"

    if use_umap:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=rng, n_jobs=1)
            proj_2d = reducer.fit_transform(z_all)
            method  = "UMAP"
        except ImportError:
            print("  umap-learn not installed — falling back to PCA (pip install umap-learn)")

    if proj_2d is None:
        from sklearn.decomposition import PCA as skPCA
        proj_2d = skPCA(n_components=2, random_state=rng).fit_transform(z_all)

    plot_latent_space(
        proj_2d, y_all, method,
        output_dir / f"ae_latent_{method.lower()}.png",
    )

    # ── 9. Plots ─────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_training_curve(history, output_dir / "ae_training_curve.png")
    plot_reconstruction_error(
        np.concatenate([err_train, err_test]),
        y_all,
        output_dir / "ae_reconstruction_error_dist.png",
    )
    plot_comparison_dashboard(
        rf_metrics, xgb_metrics, history,
        y_test.values, rf_prob, xgb_prob,
        output_dir / "ae_dashboard.png",
        lda_summary_path=args.compare_lda,
    )

    # ── 10. Summary ──────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  All outputs saved to:", output_dir)
    print(f"{'─'*55}")
    print("  ae_training_curve.png          — AE loss over epochs")
    print("  ae_reconstruction_error.csv    — per-sample MSE")
    print("  ae_reconstruction_error_dist.png — DoH vs Non-DoH MSE dist")
    print(f"  ae_latent_{method.lower()}.png           — latent space scatter")
    print("  ae_comparison_summary.csv      — RF+AE vs XGB+AE metrics")
    print("  ae_comparison_details.json     — full metrics + conf matrices")
    print("  ae_dashboard.png               — combined comparison chart")
    print("  autoencoder_weights.pt         — saved model weights")
    if args.compare_lda:
        print("  (dashboard includes LDA pipeline rows for comparison)")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
