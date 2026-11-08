"""
RingBreaker - ML Data Utilities (Phase 2 Group Splits & Leak-Free Scaler)
- Ring-level / cluster-level splits (C7): all nodes in a ring or hard-negative cluster
  are assigned to the SAME split (train, val, or test).
- Scaler is fit strictly on train split only.
- Serializes scaler to JSON (I6, no unsafe pickle).
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_NAMES

SEED = 42


def save_scaler_to_json(scaler: StandardScaler, output_path: str, feature_names: List[str]):
    """Saves StandardScaler parameters to JSON without pickle."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scaler_dict = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features": int(scaler.n_features_in_),
        "feature_names": feature_names
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f, indent=2)


def load_scaler_from_json(json_path: str) -> StandardScaler:
    """Loads StandardScaler from JSON parameters without pickle."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"], dtype=np.float64)
    scaler.scale_ = np.array(data["scale"], dtype=np.float64)
    scaler.var_ = np.array(data["var"], dtype=np.float64)
    scaler.n_features_in_ = int(data["n_features"])
    return scaler


def get_group_split_masks(
    labels_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assigns accounts to train, val, and test splits strictly at the group/ring level.
    All members of any ring or hard-negative cluster go to the SAME split (no data leakage).
    """
    rng = np.random.RandomState(seed)

    # If group_id is missing, fallback to account_id
    if "group_id" not in labels_df.columns:
        labels_df["group_id"] = labels_df.index

    # Gather nodes per group
    group_info: Dict[str, Dict[str, Any]] = {}
    for acc, row in labels_df.iterrows():
        gid = row["group_id"]
        is_f = int(row["is_fraud"])
        if gid not in group_info:
            group_info[gid] = {"is_fraud": is_f, "accounts": []}
        group_info[gid]["accounts"].append(acc)

    fraud_groups = [g for g, d in group_info.items() if d["is_fraud"] == 1]
    legit_groups = [g for g, d in group_info.items() if d["is_fraud"] == 0]

    # Shuffle groups deterministically by seed
    rng.shuffle(fraud_groups)
    rng.shuffle(legit_groups)

    def split_group_list(groups: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(groups)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_g = groups[:n_train]
        val_g = groups[n_train: n_train + n_val]
        test_g = groups[n_train + n_val:]
        return train_g, val_g, test_g

    train_fg, val_fg, test_fg = split_group_list(fraud_groups)
    train_lg, val_lg, test_lg = split_group_list(legit_groups)

    train_groups = set(train_fg + train_lg)
    val_groups = set(val_fg + val_lg)
    test_groups = set(test_fg + test_lg)

    node_list = list(labels_df.index)
    node_to_split = {}
    for g in train_groups:
        for acc in group_info[g]["accounts"]:
            node_to_split[acc] = "train"
    for g in val_groups:
        for acc in group_info[g]["accounts"]:
            node_to_split[acc] = "val"
    for g in test_groups:
        for acc in group_info[g]["accounts"]:
            node_to_split[acc] = "test"

    train_mask = np.array([node_to_split.get(acc) == "train" for acc in node_list], dtype=bool)
    val_mask = np.array([node_to_split.get(acc) == "val" for acc in node_list], dtype=bool)
    test_mask = np.array([node_to_split.get(acc) == "test" for acc in node_list], dtype=bool)

    return train_mask, val_mask, test_mask


def load_dataset(
    data_dir: str = "data",
    seed: int = 42
) -> Tuple[Data, StandardScaler, Dict[str, int], Dict[int, str]]:
    """
    Loads saved graph, feature matrices, and labels.
    Prepares PyTorch Geometric Data object with disjoint ring-level train/val/test masks.
    Fits feature scaler strictly on train set only (no data leakage).
    """
    features_path = os.path.join(data_dir, "features.csv")
    labels_path = os.path.join(data_dir, "labels.csv")
    graph_npz_path = os.path.join(data_dir, "graph_edges.npz")
    graph_pkl_path = os.path.join(data_dir, "graph.pkl")

    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        raise FileNotFoundError(f"Missing feature or label artifacts in {data_dir}. Run `python ml/build_graph.py` first.")

    features_df = pd.read_csv(features_path, index_col="account_id")
    labels_df = pd.read_csv(labels_path, index_col="account_id")

    # Consistent ordering of nodes
    common_nodes = [n for n in features_df.index if n in labels_df.index]
    features_df = features_df.loc[common_nodes]
    labels_df = labels_df.loc[common_nodes]

    node_list = common_nodes
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for i, node in enumerate(node_list)}
    num_nodes = len(node_list)

    y = labels_df["is_fraud"].values.astype(np.int64)

    # Ring-level splits (C7)
    train_mask_np, val_mask_np, test_mask_np = get_group_split_masks(labels_df, seed=seed)

    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

    # Scale features - FIT STRICTLY ON TRAIN SPLIT ONLY to prevent leakage
    scaler = StandardScaler()
    x_raw = features_df.values.astype(np.float32)
    scaler.fit(x_raw[train_mask_np])
    x_scaled = scaler.transform(x_raw)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Load edges: prefer NPZ (non-pickle) if available
    src_list, dst_list = [], []
    if os.path.exists(graph_npz_path):
        npz = np.load(graph_npz_path, allow_pickle=True)
        src_arr = npz["src"]
        dst_arr = npz["dst"]
        for u, v in zip(src_arr, dst_arr):
            if u in node_to_idx and v in node_to_idx:
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                src_list.append(u_idx)
                dst_list.append(v_idx)
                src_list.append(v_idx)
                dst_list.append(u_idx)
    elif os.path.exists(graph_pkl_path):
        import pickle
        with open(graph_pkl_path, "rb") as f:
            graph_data = pickle.load(f)
        graph = graph_data["graph"]
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                src_list.append(u_idx)
                dst_list.append(v_idx)
                src_list.append(v_idx)
                dst_list.append(u_idx)
    else:
        raise FileNotFoundError("No graph edges file found (neither graph_edges.npz nor graph.pkl).")

    if len(src_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    pyg_data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return pyg_data, scaler, node_to_idx, idx_to_node
