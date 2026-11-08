"""
RingBreaker - GNN Model Training (GraphSAGE, Phase 2 Hardened)
- Model checkpoint selected by validation loss / PR-AUC with minimum 30 epochs and early stopping (C5).
- Per-epoch training log saved to ml/artifacts/train_log.json.
- Warnings if best_epoch < 10.
- Platt / Temperature probability calibration fit strictly on validation.
- Clean JSON serialization (no pickles for scaler/metadata, weights loaded safely).
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

from ml.data_utils import load_dataset, save_scaler_to_json, SEED
from ml.features import FEATURE_NAMES

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RingBreakerGNN(nn.Module):
    """
    3-layer GraphSAGE node classifier for fraud ring detection.
    Aggregates multi-hop neighborhood transaction flow.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv3(h, edge_index)
        h = F.relu(h)
        h = self.fc(h)
        return h.squeeze(-1)  # Logits of shape (num_nodes,)


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)


def fit_platt_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Fits Platt scaling (logistic regression on validation logits).
    Returns (slope, intercept, brier_calibrated, ece_calibrated).
    """
    clf = LogisticRegression(C=1.0, solver="lbfgs")
    clf.fit(val_logits.reshape(-1, 1), val_labels)
    slope = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])

    cal_probs = clf.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    brier = float(brier_score_loss(val_labels, cal_probs))
    ece = compute_ece(cal_probs, val_labels)

    return slope, intercept, brier, ece


def train_gnn(
    data_dir: str = "data",
    artifacts_dir: str = "ml/artifacts",
    seed: int = 42,
    epochs: int = 100,
    lr: float = 0.005,
    hidden_dim: int = 64,
    min_epochs: int = 30,
    patience: int = 25
) -> Tuple[RingBreakerGNN, Dict[str, Any]]:
    """
    Trains GraphSAGE with class-weighted loss, honest validation checkpointing,
    per-epoch train log, and probability calibration.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"[RingBreaker GNN Training] Loading dataset from '{data_dir}' (seed={seed})...")
    data, scaler, node_to_idx, idx_to_node = load_dataset(data_dir=data_dir, seed=seed)

    # Class weight from train split only
    y_train = data.y[data.train_mask]
    num_pos = float((y_train == 1).sum().item())
    num_neg = float((y_train == 0).sum().item())
    pos_weight_val = num_neg / max(1.0, num_pos)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"[RingBreaker GNN Training] Train set: {int(num_neg)} normal, {int(num_pos)} fraud nodes (pos_weight: {pos_weight_val:.2f})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    in_channels = data.x.size(1)
    model = RingBreakerGNN(in_channels=in_channels, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_val_pr_auc = -1.0
    best_state_dict = None
    best_epoch = 0
    patience_counter = 0

    train_log = []

    print(f"[RingBreaker GNN Training] Training 3-layer GraphSAGE (min_epochs={min_epochs}, max_epochs={epochs}, patience={patience})...")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        train_loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            full_logits = model(data.x, data.edge_index)
            val_logits_t = full_logits[data.val_mask]
            val_loss_t = criterion(val_logits_t, data.y[data.val_mask])
            val_loss = float(val_loss_t.item())

            val_probs = torch.sigmoid(val_logits_t).cpu().numpy()
            y_val = data.y[data.val_mask].cpu().numpy().astype(int)
            val_pr_auc = float(average_precision_score(y_val, val_probs))

        train_log.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss.item()), 4),
            "val_loss": round(val_loss, 4),
            "val_pr_auc": round(val_pr_auc, 4)
        })

        # Checkpoint selection: break PR-AUC ties with lower val_loss (C5)
        is_better = False
        if val_pr_auc > best_val_pr_auc + 1e-4:
            is_better = True
        elif abs(val_pr_auc - best_val_pr_auc) <= 1e-4 and val_loss < best_val_loss:
            is_better = True

        if is_better:
            best_val_pr_auc = val_pr_auc
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            if epoch >= min_epochs:
                patience_counter += 1

        if epoch % 10 == 0 or epoch == epochs or epoch == best_epoch:
            print(f"  Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} (Best: {best_val_pr_auc:.4f} @ Ep {best_epoch})")

        if epoch >= min_epochs and patience_counter >= patience:
            print(f"[RingBreaker GNN Training] Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    # C5 Integrity warning
    if best_epoch < 10:
        print(f"[INTEGRITY WARNING C5] Best checkpoint selected at epoch {best_epoch} (< 10). Possible underfitting or freeze!")
    else:
        print(f"[INTEGRITY PASS C5] Best checkpoint at epoch {best_epoch} (>= 10).")

    # Load best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 4. Fit probability calibration on validation split
    model.eval()
    with torch.no_grad():
        final_val_logits = model(data.x, data.edge_index)[data.val_mask].cpu().numpy()
        raw_val_probs = torch.sigmoid(torch.tensor(final_val_logits)).numpy()
        y_val_np = data.y[data.val_mask].cpu().numpy().astype(int)

    brier_uncal = float(brier_score_loss(y_val_np, raw_val_probs))
    ece_uncal = compute_ece(raw_val_probs, y_val_np)

    cal_slope, cal_intercept, brier_cal, ece_cal = fit_platt_scaling(final_val_logits, y_val_np)
    print(f"[Calibration] Uncalibrated Val Brier: {brier_uncal:.4f}, ECE: {ece_uncal:.4f}")
    print(f"[Calibration] Platt-Calibrated Val Brier: {brier_cal:.4f}, ECE: {ece_cal:.4f} (slope={cal_slope:.4f}, intercept={cal_intercept:.4f})")

    # 5. Save all artifacts without pickles (I6)
    model_path = os.path.join(artifacts_dir, "gnn_model.pt")
    scaler_path = os.path.join(artifacts_dir, "scaler.json")
    node_mapping_path = os.path.join(artifacts_dir, "node_mapping.json")
    metadata_path = os.path.join(artifacts_dir, "model_meta.json")
    calibrator_path = os.path.join(artifacts_dir, "calibrator.json")
    train_log_path = os.path.join(artifacts_dir, "train_log.json")

    torch.save(model.state_dict(), model_path)
    save_scaler_to_json(scaler, scaler_path, FEATURE_NAMES)

    with open(node_mapping_path, "w", encoding="utf-8") as f:
        json.dump({"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}, f, indent=2)

    with open(calibrator_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "platt_scaling",
            "slope": cal_slope,
            "intercept": cal_intercept,
            "val_brier_uncalibrated": brier_uncal,
            "val_brier_calibrated": brier_cal,
            "val_ece_uncalibrated": ece_uncal,
            "val_ece_calibrated": ece_cal
        }, f, indent=2)

    with open(train_log_path, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    meta = {
        "architecture": "RingBreakerGNN (3-layer GraphSAGE)",
        "in_channels": in_channels,
        "hidden_dim": hidden_dim,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_pr_auc": round(float(best_val_pr_auc), 4),
        "min_epochs_enforced": min_epochs,
        "calibration": {
            "slope": round(cal_slope, 4),
            "intercept": round(cal_intercept, 4),
            "brier_calibrated": round(brier_cal, 4),
            "ece_calibrated": round(ece_cal, 4)
        },
        "artifacts": {
            "weights": "ml/artifacts/gnn_model.pt",
            "scaler": "ml/artifacts/scaler.json",
            "node_mapping": "ml/artifacts/node_mapping.json",
            "calibrator": "ml/artifacts/calibrator.json",
            "train_log": "ml/artifacts/train_log.json"
        }
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[RingBreaker GNN Training] Successfully saved weights to '{model_path}'")
    print(f"[RingBreaker GNN Training] Saved scaler to '{scaler_path}'")
    print(f"[RingBreaker GNN Training] Saved training log to '{train_log_path}'")

    return model, meta


if __name__ == "__main__":
    train_gnn()
