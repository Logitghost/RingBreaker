"""
RingBreaker - Reset Demo Script
Resets SQLite database and live graph to a clean state with zero seeded demo rows.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph_store import GraphStore
from app.scoring import RiskScorer
from app.simulator import TrafficSimulator
from app.paypal_client import PayPalClient


def reset_demo():
    print("[RingBreaker] Resetting database and live graph state...")
    store = GraphStore()
    store.reset_database()
    client = PayPalClient()
    scorer = RiskScorer(graph_store=store)
    sim = TrafficSimulator(store, scorer, client)
    sim._init_legit_pool()
    print("[RingBreaker] Reset complete. Zero fake rows. Clean state active.")


if __name__ == "__main__":
    reset_demo()
