"""
Unit tests for RingBreaker graph builder & ring injection.
"""

import os
import shutil
import pytest
import networkx as nx

from ml.build_graph import (
    generate_synthetic_transactions,
    inject_circular_rings,
    inject_fan_in_mule_clusters,
    inject_shared_device_clusters,
    build_and_save_graph
)


def test_synthetic_transactions_generation():
    graph, devices, metadata = generate_synthetic_transactions(
        num_legit_accounts=50,
        num_tx_per_account_mean=4.0,
        time_window_days=7
    )
    assert graph.number_of_nodes() == 50
    assert graph.number_of_edges() > 0
    assert len(devices) == 50
    assert len(metadata) == 50

    # Ensure all nodes have is_fraud = 0 initially
    for node, data in metadata.items():
        assert data["is_fraud"] == 0
        assert data["ring_type"] == "none"


def test_ring_injections():
    graph = nx.MultiDiGraph()
    devices = {}
    metadata = {}

    num_cycle_acc = inject_circular_rings(graph, devices, metadata, num_rings=2, ring_size_range=(3, 3))
    assert num_cycle_acc == 6
    cycle_nodes = [n for n, m in metadata.items() if m["ring_type"] == "cycle"]
    assert len(cycle_nodes) == 6

    num_mule_acc = inject_fan_in_mule_clusters(graph, devices, metadata, num_clusters=2, senders_per_cluster_range=(4, 4))
    mule_nodes = [n for n, m in metadata.items() if m["ring_type"] == "fan_in_mule"]
    # 2 clusters * (4 senders + 1 hub + 1 cashout) = 12 nodes
    assert len(mule_nodes) == 12

    num_dev_acc = inject_shared_device_clusters(graph, devices, metadata, num_clusters=2, accounts_per_cluster_range=(3, 3))
    dev_nodes = [n for n, m in metadata.items() if m["ring_type"] == "shared_device"]
    assert len(dev_nodes) == 6


def test_build_and_save_pipeline(tmp_path):
    output_dir = str(tmp_path / "test_data")
    summary = build_and_save_graph(output_dir=output_dir, num_legit=60)

    assert os.path.exists(os.path.join(output_dir, "graph.pkl"))
    assert os.path.exists(os.path.join(output_dir, "features.csv"))
    assert os.path.exists(os.path.join(output_dir, "labels.csv"))
    assert os.path.exists(os.path.join(output_dir, "summary.json"))

    assert summary["total_nodes"] > 60
    assert summary["fraud_nodes"] > 0
    assert summary["fraud_rate_pct"] > 0
