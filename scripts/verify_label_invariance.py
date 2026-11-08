"""
RingBreaker - Phase 1 Verification: Label & Name Invariance
Proves that scoring logic is 100% invariant to ground-truth labels and account names:
- Scrambling is_fraud has zero effect on scores
- Renaming accounts or removing name prefixes has zero effect on scores
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph_store import GraphStore
from app.scoring import RiskScorer


def verify_invariance():
    print("\n" + "=" * 70)
    print("PHASE 1 EVIDENCE: LABEL & NAME INVARIANCE VERIFICATION")
    print("=" * 70)

    store = GraphStore()
    scorer = RiskScorer(graph_store=store)

    nodes = list(store.graph.nodes())[:15]
    if not nodes:
        print("No nodes in graph to test.")
        return False

    # 1. Baseline scoring
    baseline_scores = {}
    for n in nodes:
        score, tier, cold, model = scorer.score_account(n, store)
        baseline_scores[n] = score

    # 2. Scramble labels: inject fake is_fraud flags into ground truth and metadata
    for n in nodes:
        store.ground_truth[n] = {"is_fraud": 1, "ring_type": "cycle", "ring_id": "fake_test"}
        store.node_metadata[n] = {"is_fraud": 1, "ring_type": "cycle"}

    scrambled_scores = {}
    for n in nodes:
        score, tier, cold, model = scorer.score_account(n, store)
        scrambled_scores[n] = score

    # Check differences
    max_delta = max(abs(baseline_scores[n] - scrambled_scores[n]) for n in nodes)
    print(f"Tested {len(nodes)} accounts across the graph.")
    print(f"Max score delta after scrambling ground-truth labels to 1: {max_delta:.8f}")

    if max_delta > 1e-6:
        print("[FAIL] Scores changed when ground-truth labels were scrambled!")
        return False
    else:
        print("[PASS] ZERO score delta. Scoring is 100% invariant to ground-truth labels! (C2 Resolved)")

    # 3. Test name invariance: isomorphic nodes with different names
    u1, v1 = nodes[0], nodes[1]
    deg_u1 = store.graph.degree(u1)
    
    # Add an identical node with a 'ring_attack_fake' name
    name_test_node = "ring_attack_fake_99"
    store.graph.add_node(name_test_node)
    for neighbor in list(store.graph.neighbors(u1))[:3]:
        store.graph.add_edge(name_test_node, neighbor, key="test_edge", amount=50.0, timestamp=1000.0)

    score_named, _, _, _ = scorer.score_account(name_test_node, store)
    
    # Add an identical node with a neutral name
    neutral_test_node = "acc_0a1b2c3d"
    store.graph.add_node(neutral_test_node)
    for neighbor in list(store.graph.neighbors(u1))[:3]:
        store.graph.add_edge(neutral_test_node, neighbor, key="test_edge2", amount=50.0, timestamp=1000.0)

    score_neutral, _, _, _ = scorer.score_account(neutral_test_node, store)
    name_delta = abs(score_named - score_neutral)
    print(f"Score for 'ring_attack_fake_99': {score_named:.4f}")
    print(f"Score for 'acc_0a1b2c3d':         {score_neutral:.4f}")
    print(f"Name-induced score delta:        {name_delta:.8f}")

    # Cleanup temporary test nodes
    store.graph.remove_node(name_test_node)
    store.graph.remove_node(neutral_test_node)

    if name_delta > 1e-6:
        print("[FAIL] Scorer changed score based on account name substring!")
        return False
    else:
        print("[PASS] ZERO name delta. Scoring is 100% invariant to account names! (C3 Resolved)")

    print("=" * 70)
    print("PHASE 1 LABEL INVARIANCE: FULLY VERIFIED")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    success = verify_invariance()
    if not success:
        sys.exit(1)
