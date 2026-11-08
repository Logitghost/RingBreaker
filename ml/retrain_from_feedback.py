"""
RingBreaker - Analyst Feedback Retraining Pipeline (Phase 3 Directive 3.5 / I11)
Reads human review decisions from SQLite, updates dataset labels,
and trains a versioned checkpoint without silently overwriting active models.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd
import torch

from app.graph_store import GraphStore
from ml.data_utils import load_dataset
from ml.train_gnn import RingBreakerGNN


def retrain_from_analyst_feedback(
    min_labels_required: int = 5,
    force: bool = False,
    data_dir: str = "data",
    artifacts_dir: str = "ml/artifacts"
) -> Dict[str, Any]:
    print("[RingBreaker Feedback Retraining] Checking analyst decisions in SQLite...")
    graph_store = GraphStore()

    with graph_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM review_decisions ORDER BY timestamp DESC")
        decisions = [dict(row) for row in cursor.fetchall()]

    num_decisions = len(decisions)
    print(f"[RingBreaker Feedback Retraining] Found {num_decisions} analyst decisions in review_decisions.")

    if num_decisions < min_labels_required and not force:
        msg = f"Insufficient analyst labels to trigger retrain (found {num_decisions}, required {min_labels_required}). Labels remain stored for future retraining."
        print(f"[RingBreaker Feedback Retraining] {msg}")
        return {"status": "skipped", "reason": msg, "decisions_count": num_decisions}

    # Versioned directory
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versions_dir = os.path.join(artifacts_dir, "versions")
    os.makedirs(versions_dir, exist_ok=True)
    versioned_weights_path = os.path.join(versions_dir, f"gnn_model_v{timestamp_str}.pt")

    # Load active model
    meta_path = os.path.join(artifacts_dir, "model_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = RingBreakerGNN(in_channels=meta["in_channels"], hidden_dim=meta["hidden_dim"])
    active_weights = os.path.join(artifacts_dir, "gnn_model.pt")
    model.load_state_dict(torch.load(active_weights, map_location="cpu", weights_only=True))

    data, scaler, node_to_idx, _ = load_dataset(data_dir=data_dir)

    # Apply analyst labels to train mask labels
    override_count = 0
    for d in decisions:
        acc = d.get("account_id")
        dec = d.get("decision")  # "fraud" or "legit"
        if acc in node_to_idx:
            idx = node_to_idx[acc]
            new_label = 1.0 if dec == "fraud" else 0.0
            data.y[idx] = new_label
            data.train_mask[idx] = True  # Include confirmed account in training
            override_count += 1

    print(f"[RingBreaker Feedback Retraining] Ingested {override_count} verified analyst ground-truth labels.")

    # Fine-tune model for 15 epochs with low learning rate
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, 16):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Save versioned artifact (NEVER overwrites silently)
    torch.save(model.state_dict(), versioned_weights_path)
    print(f"[RingBreaker Feedback Retraining] Saved versioned weights to: '{versioned_weights_path}'")

    retrain_log_path = os.path.join(artifacts_dir, "retrain_log.json")
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": f"v{timestamp_str}",
        "analyst_decisions_applied": override_count,
        "final_loss": round(float(loss.item()), 4),
        "versioned_path": versioned_weights_path
    }

    log_history = []
    if os.path.exists(retrain_log_path):
        try:
            with open(retrain_log_path, "r", encoding="utf-8") as f:
                log_history = json.load(f)
        except Exception:
            log_history = []
    log_history.append(log_entry)

    with open(retrain_log_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    return {
        "status": "success",
        "version": f"v{timestamp_str}",
        "analyst_labels_ingested": override_count,
        "versioned_artifact": versioned_weights_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain model from analyst feedback")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of label count")
    parser.add_argument("--min-labels", type=int, default=5, help="Minimum labels to trigger")
    args = parser.parse_args()

    retrain_from_analyst_feedback(min_labels_required=args.min_labels, force=args.force)
