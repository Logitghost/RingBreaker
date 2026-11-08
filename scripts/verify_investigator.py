"""
RingBreaker - Phase 5 Investigator Verification Script
Demonstrates factual extraction and structured JSON explanation on a flagged cluster.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from app.graph_store import GraphStore
from app.investigator import LLMInvestigator, extract_cluster_facts


def run_investigation_demo():
    print("\n" + "=" * 70)
    print("      RINGBREAKER PHASE 5: LLM INVESTIGATOR VERIFICATION")
    print("=" * 70)

    store = GraphStore()
    
    # Pick a flagged or recent transaction
    txs = store.get_recent_transactions(limit=10)
    target_tx = None
    for tx in txs:
        if tx["is_ring_tx"] or tx["risk_tier"] in ("HIGH", "MEDIUM"):
            target_tx = tx
            break

    if not target_tx and txs:
        target_tx = txs[0]

    if not target_tx:
        print("No transactions found in database to investigate.")
        return

    payment_id = target_tx["id"]
    print(f"Target Flagged Payment: {payment_id}")
    print(f"  Sender: {target_tx['sender_id']} | Receiver: {target_tx['receiver_id']}")
    print(f"  Amount: ${target_tx['amount']:.2f} | Risk Tier: {target_tx['risk_tier']} | Status: {target_tx['status']}")

    print("\n>>> Extracting Hard Factual Evidence from Live Graph...")
    facts = extract_cluster_facts(payment_id, store)
    print(f"  Cluster Accounts: {facts.cluster_accounts}")
    print(f"  Shared Devices: {facts.shared_devices}")
    print(f"  Detected Cycles: {facts.detected_cycles}")
    print(f"  Total Volume: ${facts.total_cluster_volume:,.2f}")

    print("\n>>> Generating Structured Explanation (Strict Pydantic Schema)...")
    investigator = LLMInvestigator()
    explanation = investigator.investigate(facts)

    print("\n" + json.dumps(explanation.model_dump(), indent=2))

    print("\n" + "=" * 70)
    print("               PHASE 5 VERIFICATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_investigation_demo()
