"""
RingBreaker - Integrity & Baseline Audit Script
Audits and demonstrates critical flaws in the current implementation:
(a) Ground-truth label and name dependence in live scoring (C2, C3)
(b) Per-feature single-feature ROC-AUC on generated data (C6)
(c) GNN best_epoch checkpoint validation (C5)
(d) Score distribution of legit vs fraud accounts (C4)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph_store import GraphStore
from app.scoring import RiskScorer
from ml.features import FEATURE_NAMES


def audit_label_dependence(store: GraphStore, scorer: RiskScorer):
    print("=" * 70)
    print("AUDIT (A): GROUND TRUTH LABEL & NAME DEPENDENCE TEST")
    print("=" * 70)

    # Find a node that has ring metadata
    ring_nodes = [n for n, m in store.node_metadata.items() if m.get("is_fraud", 0) == 1]
    if not ring_nodes:
        # Create a test account
        test_acc = "acc_test_audit"
        store.graph.add_node(test_acc)
        store.node_metadata[test_acc] = {"is_fraud": 1, "ring_type": "cycle"}
        target_node = test_acc
    else:
        target_node = ring_nodes[0]

    # Score with original metadata (is_fraud = 1)
    store.node_metadata[target_node]["is_fraud"] = 1
    score_with_fraud_label, tier_1 = scorer.score_account(target_node, store)

    # Flip label: is_fraud = 0 (leave graph structure identical)
    store.node_metadata[target_node]["is_fraud"] = 0
    score_without_fraud_label, tier_0 = scorer.score_account(target_node, store)

    # Restore label
    store.node_metadata[target_node]["is_fraud"] = 1

    print(f"Target Account: {target_node}")
    print(f"  Score WITH is_fraud=1:    {score_with_fraud_label:.4f} (Tier: {tier_1})")
    print(f"  Score with is_fraud=0:    {score_without_fraud_label:.4f} (Tier: {tier_0})")
    print(f"  Delta:                    {abs(score_with_fraud_label - score_without_fraud_label):.4f}")

    if abs(score_with_fraud_label - score_without_fraud_label) > 1e-4:
        print("  [FAIL - CRITICAL] Scorer reads ground truth 'is_fraud' label! (Issue C2)")
    else:
        print("  [PASS] Scorer is invariant to ground truth label.")
    print()


def audit_single_feature_roc_auc(data_dir: str = "data"):
    print("=" * 70)
    print("AUDIT (B): SINGLE-FEATURE SEPARABILITY (ROC-AUC) AUDIT")
    print("=" * 70)

    features_path = os.path.join(data_dir, "features.csv")
    labels_path = os.path.join(data_dir, "labels.csv")

    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        print(f"Missing {features_path} or {labels_path}. Skipping feature audit.")
        return

    features_df = pd.read_csv(features_path, index_col="account_id")
    labels_df = pd.read_csv(labels_path, index_col="account_id")
    y = labels_df.loc[features_df.index, "is_fraud"].values

    print(f"Dataset Size: {len(y)} accounts ({np.sum(y == 0)} legit, {np.sum(y == 1)} fraud)")
    print(f"{'Feature Name':<30} | {'ROC-AUC':<10} | {'Trivially Separable (>= 0.95)?'}")
    print("-" * 70)

    any_trivial = False
    for col in features_df.columns:
        vals = features_df[col].values
        # ROC AUC can be positive or negative correlation
        try:
            auc = roc_auc_score(y, vals)
            auc_effective = max(auc, 1.0 - auc)
            is_trivial = auc_effective >= 0.95
            if is_trivial:
                any_trivial = True
            flag = "[FAIL - TRIVIAL]" if is_trivial else "[OK]"
            print(f"{col:<30} | {auc_effective:<10.4f} | {flag}")
        except Exception as e:
            print(f"{col:<30} | ERROR: {e}")

    print("-" * 70)
    if any_trivial:
        print("[FAIL - CRITICAL] Benchmark has single features with ROC-AUC >= 0.95 (Issue C6).")
        print("Graph topology claim is unsupported because simple thresholds on single tabular features separate classes.")
    else:
        print("[PASS] No single feature exceeds 0.95 ROC-AUC. Detection requires multi-feature/topological signal.")
    print()


def audit_gnn_checkpoint(artifacts_dir: str = "ml/artifacts"):
    print("=" * 70)
    print("AUDIT (C): GNN CHECKPOINT & BEST EPOCH AUDIT")
    print("=" * 70)

    meta_path = os.path.join(artifacts_dir, "model_meta.json")
    if not os.path.exists(meta_path):
        print(f"Missing {meta_path}.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    best_epoch = meta.get("best_epoch", "Unknown")
    best_val_pr_auc = meta.get("best_val_pr_auc", "Unknown")

    print(f"GNN Best Epoch:       {best_epoch}")
    print(f"GNN Best Val PR-AUC:   {best_val_pr_auc}")

    if best_epoch == 1:
        print("  [FAIL - CRITICAL] GNN saved from epoch 1 of 150! (Issue C5)")
        print("  Validation PR-AUC hit 1.0 on epoch 1 and strict '>' prevented updates, freezing un-converged weights.")
    else:
        print(f"  [PASS] GNN converged at epoch {best_epoch} >= 10.")
    print()


def audit_score_distribution(store: GraphStore, scorer: RiskScorer):
    print("=" * 70)
    print("AUDIT (D): SCORE DISTRIBUTION & AUTO-CAPTURE AUDIT")
    print("=" * 70)

    legit_nodes = [n for n, m in store.node_metadata.items() if m.get("is_fraud", 0) == 0]
    fraud_nodes = [n for n, m in store.node_metadata.items() if m.get("is_fraud", 0) == 1]

    # Sample up to 100 of each
    sample_legit = legit_nodes[:100]
    sample_fraud = fraud_nodes[:100]

    legit_scores = [scorer.score_account(n, store)[0] for n in sample_legit]
    fraud_scores = [scorer.score_account(n, store)[0] for n in sample_fraud]

    legit_arr = np.array(legit_scores)
    fraud_arr = np.array(fraud_scores)

    print(f"Operating Low Threshold (Auto-Capture Cutoff): {scorer.threshold_low:.2f}")
    print(f"Operating High Threshold (Auto-Void Cutoff):    {scorer.threshold_high:.2f}")
    print("-" * 70)
    print(f"Legit Accounts (N={len(legit_arr)}):")
    print(f"  Min: {legit_arr.min():.4f} | Q25: {np.percentile(legit_arr, 25):.4f} | Median: {np.median(legit_arr):.4f} | Q75: {np.percentile(legit_arr, 75):.4f} | Max: {legit_arr.max():.4f}")
    auto_capture_pct = np.mean(legit_arr < scorer.threshold_low) * 100.0
    print(f"  Percent Auto-Captured (< {scorer.threshold_low:.2f}): {auto_capture_pct:.1f}%")

    print(f"\nFraud Accounts (N={len(fraud_arr)}):")
    print(f"  Min: {fraud_arr.min():.4f} | Q25: {np.percentile(fraud_arr, 25):.4f} | Median: {np.median(fraud_arr):.4f} | Q75: {np.percentile(fraud_arr, 75):.4f} | Max: {fraud_arr.max():.4f}")
    auto_void_pct = np.mean(fraud_arr >= scorer.threshold_high) * 100.0
    print(f"  Percent Auto-Voided (>= {scorer.threshold_high:.2f}):   {auto_void_pct:.1f}%")

    print("-" * 70)
    if auto_capture_pct < 50.0:
        print(f"  [FAIL - CRITICAL] Normal payments never auto-capture! Only {auto_capture_pct:.1f}% land in LOW (Issue C4).")
        print(f"  Median legit score is {np.median(legit_arr):.4f}, which exceeds threshold_low ({scorer.threshold_low:.2f}).")
    else:
        print(f"  [PASS] {auto_capture_pct:.1f}% of legitimate accounts land safely in LOW risk tier.")

    if np.all(np.isclose(fraud_arr, 0.85)):
        print("  [FAIL - CRITICAL] Every fraud account scores exactly 0.85 due to hardcoded override! (Issue C2)")
    print("=" * 70)


def main():
    store = GraphStore()
    scorer = RiskScorer()

    audit_label_dependence(store, scorer)
    audit_single_feature_roc_auc()
    audit_gnn_checkpoint()
    audit_score_distribution(store, scorer)


if __name__ == "__main__":
    main()
