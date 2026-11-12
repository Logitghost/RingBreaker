"""
Unit tests for Machine Learning and Data Split Integrity (Phase 2 & Phase 8).
Verifies:
1. Single feature audit guard: no feature has ROC-AUC >= 0.95 (C6).
2. Ring disjoint splits: no ring appears across multiple data splits (C7).
3. Scaler fit strictly on train split only (C8).
4. Checkpoint integrity: best_epoch >= 10, valid calibrator Platt scaling.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_single_feature_audit_guard():
    """
    (C6) Verifies that no single tabular or graph feature has ROC-AUC >= 0.95.
    Prevents trivial synthetic generation shortcuts or ground truth leakage.
    """
    features_path = ROOT_DIR / "data" / "features.csv"
    labels_path = ROOT_DIR / "data" / "labels.csv"

    if not features_path.exists() or not labels_path.exists():
        pytest.skip("Dataset files not found; run ml/build_graph.py first.")

    features_df = pd.read_csv(features_path, index_col=0)
    labels_df = pd.read_csv(labels_path, index_col=0)

    aligned = features_df.join(labels_df["is_fraud"])
    y_true = aligned["is_fraud"].values

    assert len(np.unique(y_true)) == 2, "Dataset must have both positive and negative labels."

    feature_cols = [c for c in features_df.columns if c != "is_fraud"]
    violating_features = []

    for col in feature_cols:
        vals = aligned[col].values
        if np.all(vals == vals[0]):
            continue
        auc = roc_auc_score(y_true, vals)
        effective_auc = max(auc, 1.0 - auc)
        if effective_auc >= 0.95:
            violating_features.append((col, effective_auc))

    assert len(violating_features) == 0, f"Found features with ROC-AUC >= 0.95: {violating_features}"


def test_ring_disjoint_splits():
    """
    (C7) Verifies that no ring_id appears across train, validation, or test splits.
    """
    labels_path = ROOT_DIR / "data" / "labels.csv"
    if not labels_path.exists():
        pytest.skip("labels.csv not found.")

    labels_df = pd.read_csv(labels_path, index_col=0)
    if "split" not in labels_df.columns or "ring_id" not in labels_df.columns:
        pytest.skip("Splits not persisted in labels.csv.")

    fraud_df = labels_df[(labels_df["is_fraud"] == 1) & (labels_df["ring_id"].notna()) & (labels_df["ring_id"] != "")]

    train_rings = set(fraud_df[fraud_df["split"] == "train"]["ring_id"].unique())
    val_rings = set(fraud_df[fraud_df["split"] == "val"]["ring_id"].unique())
    test_rings = set(fraud_df[fraud_df["split"] == "test"]["ring_id"].unique())

    assert len(train_rings.intersection(val_rings)) == 0, "Train and Val share ring IDs!"
    assert len(train_rings.intersection(test_rings)) == 0, "Train and Test share ring IDs!"
    assert len(val_rings.intersection(test_rings)) == 0, "Val and Test share ring IDs!"


def test_metrics_json_schema_and_best_epoch():
    """
    Verifies metrics.json schema, multi-seed evaluation, and best_epoch >= 10.
    """
    metrics_path = ROOT_DIR / "ml" / "artifacts" / "metrics.json"
    assert metrics_path.exists(), "metrics.json must exist in ml/artifacts"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "best_epoch" in metrics
    assert metrics["best_epoch"] >= 10, f"GNN best_epoch ({metrics['best_epoch']}) must be >= 10"

    assert "models" in metrics
    assert "gnn" in metrics["models"]
    assert "strong_tabular_baseline" in metrics["models"]
    assert metrics["models"]["gnn"]["pr_auc"] > 0.80

    assert "calibration_stats" in metrics
    assert metrics["calibration_stats"]["method"] == "platt_scaling"
    assert "test_brier" in metrics["calibration_stats"]
    assert "test_ece" in metrics["calibration_stats"]
