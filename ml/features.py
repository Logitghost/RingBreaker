"""
RingBreaker - Feature Extraction
Extracts behavioral and graph neighborhood features for accounts.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import networkx as nx


FEATURE_NAMES = [
    "in_degree",
    "out_degree",
    "total_degree",
    "in_amount_sum",
    "in_amount_mean",
    "in_amount_max",
    "out_amount_sum",
    "out_amount_mean",
    "out_amount_max",
    "in_out_ratio",
    "net_flow",
    "account_age_hours",
    "avg_time_between_tx_hours",
    "shared_device_count",
    "unique_counterparties",
]


def compute_node_features(
    graph: nx.MultiDiGraph,
    device_mapping: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Computes node-level features for all account nodes in the graph.
    
    Args:
        graph: NetworkX MultiDiGraph where directed edges represent transactions
               with attributes 'amount' (float) and 'timestamp' (float or int).
        device_mapping: Optional mapping from account_id -> list of device_ids.

    Returns:
        pd.DataFrame indexed by account_id with standardized float columns.
    """
    if device_mapping is None:
        device_mapping = {}

    # Precalculate device sharing across all accounts
    device_to_accounts: Dict[str, set] = {}
    for acc, devices in device_mapping.items():
        for dev in devices:
            device_to_accounts.setdefault(dev, set()).add(acc)

    account_shared_devices: Dict[str, int] = {}
    for acc, devices in device_mapping.items():
        shared = 0
        for dev in devices:
            if len(device_to_accounts.get(dev, set())) > 1:
                shared += 1
        account_shared_devices[acc] = shared

    records: List[Dict[str, Any]] = []
    
    # Process only account nodes
    nodes = list(graph.nodes())
    for node in nodes:
        in_edges = list(graph.in_edges(node, data=True))
        out_edges = list(graph.out_edges(node, data=True))
        
        in_amounts = [d.get("amount", 0.0) for _, _, d in in_edges]
        out_amounts = [d.get("amount", 0.0) for _, _, d in out_edges]
        
        in_times = [d.get("timestamp", 0.0) for _, _, d in in_edges if "timestamp" in d]
        out_times = [d.get("timestamp", 0.0) for _, _, d in out_edges if "timestamp" in d]
        all_times = sorted(in_times + out_times)

        in_degree = len(in_edges)
        out_degree = len(out_edges)
        total_degree = in_degree + out_degree

        in_sum = float(np.sum(in_amounts)) if in_amounts else 0.0
        in_mean = float(np.mean(in_amounts)) if in_amounts else 0.0
        in_max = float(np.max(in_amounts)) if in_amounts else 0.0

        out_sum = float(np.sum(out_amounts)) if out_amounts else 0.0
        out_mean = float(np.mean(out_amounts)) if out_amounts else 0.0
        out_max = float(np.max(out_amounts)) if out_amounts else 0.0

        # in/out ratio and net flow
        in_out_ratio = (out_sum + 1.0) / (in_sum + 1.0)
        net_flow = in_sum - out_sum

        # Timing features (hours)
        if len(all_times) >= 2:
            age_hours = (all_times[-1] - all_times[0]) / 3600.0
            time_deltas = np.diff(all_times) / 3600.0
            avg_time_between_tx = float(np.mean(time_deltas))
        else:
            age_hours = 0.0
            avg_time_between_tx = 0.0

        # Counterparties
        counterparties = set()
        for u, _, _ in in_edges:
            counterparties.add(u)
        for _, v, _ in out_edges:
            counterparties.add(v)
        unique_counterparties = len(counterparties)

        shared_dev = account_shared_devices.get(node, 0)

        record = {
            "account_id": node,
            "in_degree": float(in_degree),
            "out_degree": float(out_degree),
            "total_degree": float(total_degree),
            "in_amount_sum": in_sum,
            "in_amount_mean": in_mean,
            "in_amount_max": in_max,
            "out_amount_sum": out_sum,
            "out_amount_mean": out_mean,
            "out_amount_max": out_max,
            "in_out_ratio": in_out_ratio,
            "net_flow": net_flow,
            "account_age_hours": age_hours,
            "avg_time_between_tx_hours": avg_time_between_tx,
            "shared_device_count": float(shared_dev),
            "unique_counterparties": float(unique_counterparties),
        }
        records.append(record)

    df = pd.DataFrame(records).set_index("account_id")
    # Fill any potential NaN or Inf safely
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df
