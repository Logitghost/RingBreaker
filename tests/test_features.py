"""
Unit tests for RingBreaker feature extraction.
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from ml.features import compute_node_features, FEATURE_NAMES


def test_compute_node_features_basic():
    # Construct a simple triangle graph: A -> B -> C -> A
    graph = nx.MultiDiGraph()
    graph.add_edge("acc_A", "acc_B", key="1", amount=100.0, timestamp=1000.0)
    graph.add_edge("acc_B", "acc_C", key="2", amount=100.0, timestamp=2000.0)
    graph.add_edge("acc_C", "acc_A", key="3", amount=100.0, timestamp=3000.0)

    devices = {
        "acc_A": ["dev_1"],
        "acc_B": ["dev_1"],  # Shares device with A
        "acc_C": ["dev_2"],
    }

    df = compute_node_features(graph, devices)

    assert isinstance(df, pd.DataFrame)
    assert set(df.index) == {"acc_A", "acc_B", "acc_C"}
    for feat in FEATURE_NAMES:
        assert feat in df.columns

    # Check acc_A degrees and amounts
    assert df.loc["acc_A", "in_degree"] == 1.0
    assert df.loc["acc_A", "out_degree"] == 1.0
    assert df.loc["acc_A", "total_degree"] == 2.0
    assert df.loc["acc_A", "in_amount_sum"] == 100.0
    assert df.loc["acc_A", "out_amount_sum"] == 100.0

    # Device sharing
    assert df.loc["acc_A", "shared_device_count"] == 1.0
    assert df.loc["acc_B", "shared_device_count"] == 1.0
    assert df.loc["acc_C", "shared_device_count"] == 0.0

    # No NaN or Inf
    assert not df.isna().any().any()
    assert not np.isinf(df.values).any()


def test_empty_isolated_node():
    graph = nx.MultiDiGraph()
    graph.add_node("acc_isolated")

    df = compute_node_features(graph, {})
    assert "acc_isolated" in df.index
    assert df.loc["acc_isolated", "total_degree"] == 0.0
    assert df.loc["acc_isolated", "in_amount_sum"] == 0.0
    assert not df.isna().any().any()
