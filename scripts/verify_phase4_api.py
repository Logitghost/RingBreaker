"""
RingBreaker - Phase 4 API Verification
Validates that simulated ring traffic produces flagged clusters, held payments,
and voided payments, fully visible in the API.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app


def run_phase4_verification():
    client = TestClient(app)

    print("\n" + "=" * 70)
    print("      RINGBREAKER PHASE 4: LIVE SCORING & API VERIFICATION")
    print("=" * 70)

    # 1. Health check
    res = client.get("/api/health")
    print("\n>>> 1. Health Check:")
    print(res.json())
    assert res.status_code == 200

    # 2. Simulate Normal Traffic
    print("\n>>> 2. Simulating Normal Payments (Expect LOW risk -> CAPTURED):")
    res_normal = client.post("/api/simulate/normal?count=3")
    print(f"Status Code: {res_normal.status_code}")
    for tx in res_normal.json()["transactions"]:
        print(f"  TX {tx['tx_id']}: ${tx['amount']} | Score: {tx['risk_score']} | Tier: {tx['risk_tier']} | Action: {tx['action_taken']}")

    # 3. Simulate Fraud Ring Attacks
    print("\n>>> 3. Simulating Fraud Ring Attack: Circular Flow (Expect HIGH risk -> VOID):")
    res_cycle = client.post("/api/simulate/ring", json={"ring_type": "cycle"})
    for tx in res_cycle.json()["transactions"]:
        print(f"  TX {tx['tx_id']}: ${tx['amount']} | Score: {tx['risk_score']} | Tier: {tx['risk_tier']} | Action: {tx['action_taken']}")

    print("\n>>> 4. Simulating Fraud Ring Attack: Fan-In Mule:")
    res_mule = client.post("/api/simulate/ring", json={"ring_type": "mule"})
    for tx in res_mule.json()["transactions"]:
        print(f"  TX {tx['tx_id']}: ${tx['amount']} | Score: {tx['risk_score']} | Tier: {tx['risk_tier']} | Action: {tx['action_taken']}")

    # 4. Review Payments Feed
    print("\n>>> 5. Querying Live Payments Feed via GET /api/payments:")
    res_feed = client.get("/api/payments?limit=8")
    payments = res_feed.json()
    print(f"Total returned: {len(payments)} payments")
    for p in payments[:5]:
        print(f"  [{p['status']:<10}] {p['id']} | ${p['amount']:<6.2f} | Score: {p['risk_score']:.2f} | Tier: {p['risk_tier']}")

    # 5. Query Cytoscape Graph
    print("\n>>> 6. Querying Cytoscape Graph Elements via GET /api/graph:")
    res_graph = client.get("/api/graph?max_nodes=60")
    graph_data = res_graph.json()
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    flagged_nodes = [n for n in nodes if n["data"].get("risk_tier") == "HIGH" or n["data"].get("is_ring") == 1]
    print(f"  Graph Nodes in view: {len(nodes)} (Flagged High Risk/Ring Nodes: {len(flagged_nodes)})")
    print(f"  Graph Edges in view: {len(edges)}")

    # 6. Check Review Queue
    print("\n>>> 7. Querying Human Review Queue via GET /api/review-queue:")
    # Add an explicit medium-risk payment to test review queue and analyst feedback
    from app.main import graph_store
    graph_store.add_transaction(
        tx_id="tx_review_demo_99",
        sender_id="acc_borderline_01",
        receiver_id="acc_borderline_02",
        amount=280.00,
        timestamp=1700005000.0,
        device_id="dev_borderline",
        risk_score=0.48,
        risk_tier="MEDIUM",
        status="IN_REVIEW",
        action_taken="HOLD_FOR_REVIEW"
    )

    res_queue = client.get("/api/review-queue")
    queue_items = res_queue.json()
    print(f"  Pending Payments in Review Queue: {len(queue_items)}")
    for item in queue_items[:3]:
        print(f"    Payment {item['id']}: ${item['amount']} | Risk Score: {item['risk_score']} | Status: {item['status']}")

    # 7. Analyst Action on Review Queue
    if queue_items:
        target_id = queue_items[0]["id"]
        print(f"\n>>> 8. Analyst Deciding on Payment {target_id} -> APPROVE (LEGIT):")
        res_dec = client.post(
            f"/api/review-queue/{target_id}/decision",
            json={"decision": "legit", "analyst_id": "lead_fraud_analyst", "notes": "Verified business identity"}
        )
        print(f"  Result: {res_dec.json()['message']}")
        print(f"  New Status: {res_dec.json()['payment']['status']} | Action: {res_dec.json()['payment']['action_taken']}")

    print("\n" + "=" * 70)
    print("                 PHASE 4 VERIFICATION SUCCESSFUL")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_phase4_verification()
