"""
RingBreaker - XGBoost Baselines Training (Phase 2 Hardened)
Trains two tabular baselines:
1. Plain XGBoost: standard node features (15 features).
2. Strong Tabular Baseline: node features + engineered graph aggregates
   (neighbor mean/max features, 2-hop reach, shared device cluster sizes, cycle membership).

Evaluated strictly on held-out splits without leakage.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import networkx as nx
import xgboost as xgb
from sklearn.metrics import average_precision_score

from ml.data_utils import load_dataset, SEED
from ml.features import FEATURE_NAMES

random.seed(SEED)
np.random.seed(SEED)


def compute_strong_graph_features(
    features_df: pd.DataFrame,
    data_dir: str = "data"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineers rich graph aggregate features for the strong tabular baseline:
    - Neighbor mean & max for key features (degree, amount, velocity)
    - 2-hop neighborhood count
    - Shared device cluster size
    - Directed cycle participation indicator
    """
    print("[Strong Baseline Feature Engineering] Extracting neighborhood aggregates...")
    graph_npz_path = os.path.join(data_dir, "graph_edges.npz")
    device_json_path = os.path.join(data_dir, "device_mapping.json")

    # 1. Build adjacency
    nodes = list(features_df.index)
    node_set = set(nodes)
    adj_out: Dict[str, Set[str]] = {n: set() for n in nodes}
    adj_all: Dict[str, Set[str]] = {n: set() for n in nodes}

    if os.path.exists(graph_npz_path):
        npz = np.load(graph_npz_path, allow_pickle=True)
        src_arr, dst_arr = npz["src"], npz["dst"]
        for u, v in zip(src_arr, dst_arr):
            if u in node_set and v in node_set:
                adj_out[u].add(v)
                adj_all[u].add(v)
                adj_all[v].add(u)

    # 2. Shared device cluster sizes
    device_to_accounts: Dict[str, Set[str]] = {}
    account_to_devices: Dict[str, List[str]] = {}
    if os.path.exists(device_json_path):
        with open(device_json_path, "r", encoding="utf-8") as f:
            account_to_devices = json.load(f)
        for acc, devs in account_to_devices.items():
            for d in devs:
                device_to_accounts.setdefault(d, set()).add(acc)

    # 3. Compute per-node engineered features
    engineered_records = []
    base_vals = features_df.values
    col_to_idx = {c: i for i, c in enumerate(features_df.columns)}

    # Selected features to compute neighbor statistics on
    target_stat_cols = ["total_degree", "in_amount_sum", "out_amount_sum", "in_out_ratio", "shared_device_count"]
    stat_col_indices = [col_to_idx[c] for c in target_stat_cols]

    for i, u in enumerate(nodes):
        neighbors = adj_all[u]
        # Neighbor stats
        if neighbors:
            n_indices = [col_to_idx.get(n, 0) for n in neighbors if n in col_to_idx]
            # Fetch neighbor rows from features_df
            n_vals = features_df.loc[list(neighbors), target_stat_cols].values
            mean_stats = np.mean(n_vals, axis=0)
            max_stats = np.max(n_vals, axis=0)
        else:
            mean_stats = np.zeros(len(target_stat_cols))
            max_stats = np.zeros(len(target_stat_cols))

        # 2-hop count
        two_hop = set(neighbors)
        for n in neighbors:
            two_hop.update(adj_all[n])
        two_hop.discard(u)
        two_hop_count = len(two_hop)

        # Shared device cluster size
        devs = account_to_devices.get(u, [])
        cluster_accs = set()
        for d in devs:
            cluster_accs.update(device_to_accounts.get(d, set()))
        device_cluster_size = len(cluster_accs)

        # Directed cycle indicator (2-hop or 3-hop directed cycle)
        in_cycle = 0
        out_u = adj_out[u]
        for v in out_u:
            if u in adj_out[v]:  # 2-cycle
                in_cycle = 1
                break
            for w in adj_out[v]:
                if u in adj_out[w]:  # 3-cycle
                    in_cycle = 1
                    break
            if in_cycle:
                break

        rec = {
            "account_id": u,
            "two_hop_count": float(two_hop_count),
            "device_cluster_size": float(device_cluster_size),
            "cycle_indicator": float(in_cycle)
        }
        for idx, col in enumerate(target_stat_cols):
            rec[f"neighbor_mean_{col}"] = float(mean_stats[idx])
            rec[f"neighbor_max_{col}"] = float(max_stats[idx])

        engineered_records.append(rec)

    eng_df = pd.DataFrame(engineered_records).set_index("account_id")
    eng_df = eng_df.loc[features_df.index]

    combined_df = pd.concat([features_df, eng_df], axis=1)
    new_cols = list(combined_df.columns)
    print(f"[Strong Baseline Feature Engineering] Created {len(new_cols)} total features ({len(eng_df.columns)} graph aggregates).")
    return combined_df, new_cols


def train_baselines(
    data_dir: str = "data",
    artifacts_dir: str = "ml/artifacts",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains both Plain XGBoost and Strong Tabular Baseline.
    Saves models and features to artifacts_dir.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"[RingBreaker Baseline Training] Loading dataset (seed={seed})...")
    data, scaler, node_to_idx, idx_to_node = load_dataset(data_dir=data_dir, seed=seed)

    features_path = os.path.join(data_dir, "features.csv")
    features_df = pd.read_csv(features_path, index_col="account_id")

    # 1. Plain XGBoost features
    x_plain = data.x.numpy()
    y_all = data.y.numpy().astype(int)

    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()

    x_train_p, y_train = x_plain[train_mask], y_all[train_mask]
    x_val_p, y_val = x_plain[val_mask], y_all[val_mask]

    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / max(1, num_pos)

    # Train Plain XGBoost
    print(f"[Baseline 1: Plain XGBoost] Fitting on 15 node features (scale_pos_weight={scale_pos_weight:.2f})...")
    plain_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.04,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="aucpr",
        early_stopping_rounds=25
    )
    plain_model.fit(x_train_p, y_train, eval_set=[(x_val_p, y_val)], verbose=False)
    p_val_probs = plain_model.predict_proba(x_val_p)[:, 1]
    p_val_pr = float(average_precision_score(y_val, p_val_probs))
    print(f"[Baseline 1: Plain XGBoost] Validation PR-AUC: {p_val_pr:.4f}")

    plain_model_path = os.path.join(artifacts_dir, "xgboost_plain.json")
    plain_model.save_model(plain_model_path)

    # 2. Strong Tabular Baseline
    strong_df, strong_cols = compute_strong_graph_features(features_df, data_dir=data_dir)
    # Scale strong features fitting strictly on train split
    from sklearn.preprocessing import StandardScaler
    strong_scaler = StandardScaler()
    strong_raw = strong_df.values.astype(np.float32)
    strong_scaler.fit(strong_raw[train_mask])
    strong_scaled = strong_scaler.transform(strong_raw)

    x_train_s = strong_scaled[train_mask]
    x_val_s = strong_scaled[val_mask]

    print(f"[Baseline 2: Strong XGBoost] Fitting on {len(strong_cols)} features (with graph aggregates)...")
    strong_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.04,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="aucpr",
        early_stopping_rounds=25
    )
    strong_model.fit(x_train_s, y_train, eval_set=[(x_val_s, y_val)], verbose=False)
    s_val_probs = strong_model.predict_proba(x_val_s)[:, 1]
    s_val_pr = float(average_precision_score(y_val, s_val_probs))
    print(f"[Baseline 2: Strong XGBoost] Validation PR-AUC: {s_val_pr:.4f}")

    strong_model_path = os.path.join(artifacts_dir, "xgboost_strong.json")
    strong_model.save_model(strong_model_path)

    # Save strong features metadata
    strong_meta_path = os.path.join(artifacts_dir, "strong_features_meta.json")
    with open(strong_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": strong_cols,
            "val_pr_auc": round(s_val_pr, 4),
            "plain_val_pr_auc": round(p_val_pr, 4)
        }, f, indent=2)

    return {
        "plain_val_pr_auc": p_val_pr,
        "strong_val_pr_auc": s_val_pr,
        "plain_path": plain_model_path,
        "strong_path": strong_model_path
    }


if __name__ == "__main__":
    train_baselines()
