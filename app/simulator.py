"""
RingBreaker - Checkout Simulator & Attack Ring Generator
Generates live traffic (both normal payments and simulated fraud ring attacks).
ZERO NAME OR LABEL LEAKS: Accounts receive neutral IDs (e.g. acc_7a1b2c).
Ground truth is stored strictly in graph_store.ground_truth for demo overlays, never in scoring.
"""

import os
import sys
import time
import uuid
import random
from typing import List, Dict, Any, Optional

from app.graph_store import GraphStore
from app.scoring import RiskScorer
from app.paypal_client import PayPalClient


class TrafficSimulator:
    def __init__(self, graph_store: GraphStore, scorer: RiskScorer, paypal_client: PayPalClient):
        self.graph_store = graph_store
        self.scorer = scorer
        self.paypal_client = paypal_client
        self._init_legit_pool()

    def _init_legit_pool(self) -> None:
        """
        Maintains an explicit list of known baseline accounts without inspecting names or is_fraud.
        """
        all_nodes = list(self.graph_store.graph.nodes())
        if len(all_nodes) >= 20:
            # Take the first 200 nodes as the baseline active customer pool
            self.baseline_pool = all_nodes[:200]
        else:
            self.baseline_pool = [f"acc_{uuid.uuid4().hex[:8]}" for _ in range(50)]
            for acc in self.baseline_pool:
                self.graph_store.graph.add_node(acc)

    def _generate_account_id(self) -> str:
        """
        Produces neutral account IDs without any semantic markers.
        """
        return f"acc_{uuid.uuid4().hex[:8]}"

    def _generate_device_id(self) -> str:
        return f"dev_{uuid.uuid4().hex[:8]}"

    def send_normal_payments(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Simulates legitimate customer checkouts among active pool accounts.
        """
        results = []
        for _ in range(count):
            sender = random.choice(self.baseline_pool)
            receiver = random.choice(self.baseline_pool)
            while receiver == sender:
                receiver = random.choice(self.baseline_pool)

            amount = round(random.uniform(15.0, 120.0), 2)
            device_id = self.graph_store.device_mapping.get(sender, [self._generate_device_id()])[0]

            result = self.scorer.process_payment(
                sender_id=sender,
                receiver_id=receiver,
                amount=amount,
                device_id=device_id,
                graph_store=self.graph_store,
                paypal_client=self.paypal_client
            )
            results.append(result)
        return results

    def simulate_fraud_ring(self, ring_type: str = "cycle") -> List[Dict[str, Any]]:
        """
        Simulates an organized fraud ring attack using NEUTRAL IDs.
        Ground truth is saved only in graph_store.ground_truth for demo overlays.
        """
        results = []
        ring_id = f"ring_{uuid.uuid4().hex[:6]}"

        if ring_type == "cycle":
            # 3-node cycle with neutral IDs: A -> B -> C -> A
            ring_nodes = [self._generate_account_id() for _ in range(3)]
            for node in ring_nodes:
                self.graph_store.ground_truth[node] = {
                    "is_fraud": 1,
                    "ring_type": "cycle",
                    "ring_id": ring_id
                }

            base_amount = 620.00
            for i in range(len(ring_nodes)):
                sender = ring_nodes[i]
                receiver = ring_nodes[(i + 1) % len(ring_nodes)]
                amount = round(base_amount * (0.97 ** i), 2)
                device_id = self._generate_device_id()

                res = self.scorer.process_payment(
                    sender_id=sender,
                    receiver_id=receiver,
                    amount=amount,
                    device_id=device_id,
                    graph_store=self.graph_store,
                    paypal_client=self.paypal_client
                )
                results.append(res)

        elif ring_type == "mule":
            # 4 senders -> 1 mule hub -> 1 cashout (all neutral IDs)
            mule_hub = self._generate_account_id()
            cashout = self._generate_account_id()
            senders = [self._generate_account_id() for _ in range(4)]
            
            for acc in senders + [mule_hub, cashout]:
                self.graph_store.ground_truth[acc] = {
                    "is_fraud": 1,
                    "ring_type": "fan_in_mule",
                    "ring_id": ring_id
                }

            total_smurfed = 0.0
            for i, sender in enumerate(senders):
                amt = round(random.uniform(75.0, 180.0), 2)
                total_smurfed += amt
                res = self.scorer.process_payment(
                    sender_id=sender,
                    receiver_id=mule_hub,
                    amount=amt,
                    device_id=self._generate_device_id(),
                    graph_store=self.graph_store,
                    paypal_client=self.paypal_client
                )
                results.append(res)

            # Cash-out transaction
            res_out = self.scorer.process_payment(
                sender_id=mule_hub,
                receiver_id=cashout,
                amount=round(total_smurfed * 0.94, 2),
                device_id=self._generate_device_id(),
                graph_store=self.graph_store,
                paypal_client=self.paypal_client
            )
            results.append(res_out)

        else:  # 'shared_device'
            shared_dev = self._generate_device_id()
            farm_nodes = [self._generate_account_id() for _ in range(4)]
            for node in farm_nodes:
                self.graph_store.ground_truth[node] = {
                    "is_fraud": 1,
                    "ring_type": "shared_device",
                    "ring_id": ring_id
                }

            for i in range(len(farm_nodes) - 1):
                res = self.scorer.process_payment(
                    sender_id=farm_nodes[i],
                    receiver_id=farm_nodes[i + 1],
                    amount=round(random.uniform(150.0, 400.0), 2),
                    device_id=shared_dev,
                    graph_store=self.graph_store,
                    paypal_client=self.paypal_client
                )
                results.append(res)

        return results
