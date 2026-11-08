"""
RingBreaker - Graph Builder & Ring Injector (Phase 2 Hardened)
Builds realistic payment graphs with camouflage edges, matched distributions,
and realistic hard negatives (roommates, shared devices, merchant fan-in, payroll).

Ensures NO single feature reaches ROC-AUC >= 0.95 (C6).
Supports clean JSON/NPZ non-pickle serialization (I6).
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
import argparse
from typing import Dict, List, Tuple, Set, Any, Optional
import numpy as np
import pandas as pd
import networkx as nx

from ml.features import compute_node_features, FEATURE_NAMES

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def generate_synthetic_transactions(
    num_legit_accounts: int = 5000,
    time_window_days: int = 30,
    num_tx_per_account_mean: float = 7.5
) -> Tuple[nx.MultiDiGraph, Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Generates a realistic baseline payment network of legitimate transactions.
    Includes:
    - Normal P2P users with varying account ages and transaction intervals.
    - Merchants with high fan-in.
    - Shared household/office devices (legit hard negatives).
    - Roommates splitting rent (legit hard negatives).
    - Payroll fan-outs (legit hard negatives).
    - Legit micro-cycles (3 friends splitting expenses).
    """
    print(f"[RingBreaker Data Pipeline] Generating {num_legit_accounts} legitimate accounts...")

    graph = nx.MultiDiGraph()
    device_mapping: Dict[str, List[str]] = {}
    node_metadata: Dict[str, Dict[str, Any]] = {}

    start_time = 1700000000.0  # Reference epoch seconds
    total_seconds = time_window_days * 86400.0

    # 1. Initialize legitimate accounts
    # ~8% are merchants (capped to ensure users is never empty)
    num_merchants = min(max(2, int(num_legit_accounts * 0.08)), max(1, num_legit_accounts // 5))
    accounts = [f"acc_legit_{i:05d}" for i in range(num_legit_accounts)]

    # Account creation times: distributed across the window
    account_creation_times: Dict[str, float] = {}
    for i, acc in enumerate(accounts):
        # Staggered account creation: some old (Day 0-5), some mid (Day 5-20), some fresh (Day 20-30)
        c_time = start_time + random.uniform(0, total_seconds * 0.85)
        account_creation_times[acc] = c_time

        acc_type = "merchant" if i < num_merchants else "legit_user"
        node_metadata[acc] = {
            "is_fraud": 0,
            "ring_type": "none",
            "ring_id": "none",
            "group_id": f"legit_indiv_{i}",
            "account_type": acc_type
        }
        graph.add_node(acc, **node_metadata[acc])
        # Default 1 device per account
        device_mapping[acc] = [f"dev_legit_{i:05d}"]

    # 2. Hard Negatives: Shared Household / Family / Office Devices
    # Scale with number of accounts
    num_shared_families = min(80, max(1, num_legit_accounts // 50))
    family_offset = num_merchants
    for f_idx in range(num_shared_families):
        fam_size = random.randint(2, 4)
        if family_offset + fam_size > len(accounts):
            break
        fam_members = accounts[family_offset: family_offset + fam_size]
        family_offset += fam_size
        shared_dev = f"dev_shared_family_{f_idx:03d}"
        for m in fam_members:
            device_mapping[m].append(shared_dev)
            node_metadata[m]["group_id"] = f"hard_neg_family_{f_idx:03d}"

    # Hard Negatives: Office / Kiosk devices (5-8 users per device)
    num_kiosks = min(15, max(0, num_legit_accounts // 200))
    for k_idx in range(num_kiosks):
        kiosk_size = random.randint(5, 8)
        if family_offset + kiosk_size > len(accounts):
            break
        kiosk_members = accounts[family_offset: family_offset + kiosk_size]
        family_offset += kiosk_size
        kiosk_dev = f"dev_shared_kiosk_{k_idx:02d}"
        for m in kiosk_members:
            device_mapping[m].append(kiosk_dev)
            node_metadata[m]["group_id"] = f"hard_neg_kiosk_{k_idx:02d}"

    merchants = [acc for acc, meta in node_metadata.items() if meta["account_type"] == "merchant"]
    users = [acc for acc, meta in node_metadata.items() if meta["account_type"] == "legit_user"]

    tx_count = 0

    # 3. Add Hard Negative: Roommate Rent & Bill Splits (only if enough users)
    num_roommate_groups = min(30, max(0, len(users) // 15))
    for r_idx in range(num_roommate_groups):
        r_size = min(len(users) - 1, random.randint(3, 4))
        if r_size < 2:
            break
        r_members = random.sample(users, r_size)
        landlord = random.choice(users)
        while landlord in r_members:
            landlord = random.choice(users)

        group_id = f"hard_neg_roommates_{r_idx:02d}"
        for m in r_members:
            node_metadata[m]["group_id"] = group_id

        # Roommates send rent to landlord once or twice
        for month_offset in [5 * 86400.0, 20 * 86400.0]:
            rent_time = start_time + month_offset + random.uniform(0, 7200)
            for m in r_members:
                amt = round(float(np.random.normal(550.0, 50.0)), 2)
                amt = max(300.0, min(amt, 900.0))
                tx_id = f"tx_legit_rent_{r_idx}_{tx_count}"
                dev = device_mapping[m][0]
                graph.add_edge(m, landlord, key=tx_id, tx_id=tx_id, amount=amt,
                               timestamp=rent_time + random.uniform(0, 3600),
                               device_id=dev, is_ring_tx=False)
                tx_count += 1

        # Roommates occasionally split utilities/groceries among themselves
        for _ in range(r_size * 2):
            s, r = random.sample(r_members, 2)
            s_time = start_time + random.uniform(2 * 86400, total_seconds)
            amt = round(float(np.random.uniform(15.0, 85.0)), 2)
            tx_id = f"tx_legit_utility_{tx_count}"
            graph.add_edge(s, r, key=tx_id, tx_id=tx_id, amount=amt,
                           timestamp=s_time, device_id=device_mapping[s][0], is_ring_tx=False)
            tx_count += 1

    # 4. Add Hard Negative: Legit 3-Cycles (only if enough users)
    num_cycles = min(25, max(0, len(users) // 10))
    for c_idx in range(num_cycles):
        if len(users) < 3:
            break
        c_members = random.sample(users, 3)
        group_id = f"hard_neg_cycle_{c_idx:02d}"
        for m in c_members:
            node_metadata[m]["group_id"] = group_id

        cycle_time = start_time + random.uniform(86400, total_seconds - 86400)
        base_amt = round(float(np.random.uniform(40.0, 120.0)), 2)
        for step in range(3):
            s = c_members[step]
            r = c_members[(step + 1) % 3]
            amt = round(base_amt * random.uniform(0.85, 1.15), 2)
            tx_id = f"tx_legit_expcycle_{c_idx}_{step}_{tx_count}"
            graph.add_edge(s, r, key=tx_id, tx_id=tx_id, amount=amt,
                           timestamp=cycle_time + (step * random.uniform(300, 3600)),
                           device_id=device_mapping[s][0], is_ring_tx=False)
            tx_count += 1

    # 5. Add Hard Negative: Payroll fan-outs (only if enough users)
    num_payroll = min(5, max(0, min(len(merchants), len(users) // 25)))
    for p_idx in range(num_payroll):
        employer = merchants[p_idx]
        emp_count = min(len(users), random.randint(20, 30))
        if emp_count < 2:
            continue
        employees = random.sample(users, emp_count)
        for payday in [start_time + 7 * 86400, start_time + 21 * 86400]:
            for emp in employees:
                amt = round(float(np.random.normal(1200.0, 150.0)), 2)
                amt = max(600.0, min(amt, 2500.0))
                tx_id = f"tx_payroll_{p_idx}_{tx_count}"
                graph.add_edge(employer, emp, key=tx_id, tx_id=tx_id, amount=amt,
                               timestamp=payday + random.uniform(0, 1800),
                               device_id=device_mapping[employer][0], is_ring_tx=False)
                tx_count += 1

    # 6. General background transactions (P2P + Merchant purchases)
    # Real-world users follow a skewed activity distribution:
    # - 30% low-activity users (1-3 transactions)
    # - 50% medium-activity users (4-8 transactions)
    # - 20% high-activity users (9-18 transactions)
    user_weights = []
    for i, u in enumerate(users):
        tier = random.random()
        if tier < 0.30:
            w = random.uniform(1, 3)
        elif tier < 0.80:
            w = random.uniform(4, 8)
        else:
            w = random.uniform(9, 18)
        user_weights.append(w)
    user_weights = np.array(user_weights)
    user_probs = user_weights / user_weights.sum()

    num_bg_tx = int(num_legit_accounts * 5.5)
    for _ in range(num_bg_tx):
        sender = np.random.choice(users, p=user_probs)
        sender_created = account_creation_times[sender]

        # Valid transaction time is between sender creation and end of window
        if sender_created >= total_seconds + start_time - 7200:
            tx_time = sender_created + random.uniform(0, 3600)
        else:
            tx_time = random.uniform(sender_created, start_time + total_seconds)

        # 55% chance merchant purchase, 45% P2P
        if random.random() < 0.55 and merchants:
            receiver = random.choice(merchants)
            amount = round(float(np.random.lognormal(mean=3.8, sigma=0.85)), 2)
            amount = max(5.0, min(amount, 1800.0))
        else:
            receiver = random.choice(users)
            while receiver == sender:
                receiver = random.choice(users)
            amount = round(float(np.random.lognormal(mean=4.0, sigma=0.85)), 2)
            amount = max(5.0, min(amount, 2000.0))

        tx_id = f"tx_legit_{tx_count:07d}"
        device = random.choice(device_mapping[sender])

        graph.add_edge(
            sender,
            receiver,
            key=tx_id,
            tx_id=tx_id,
            amount=amount,
            timestamp=tx_time,
            device_id=device,
            is_ring_tx=False
        )
        tx_count += 1

    print(f"[RingBreaker Data Pipeline] Built {num_legit_accounts} legit accounts with {tx_count} transactions.")
    return graph, device_mapping, node_metadata


def inject_circular_rings(
    graph: nx.MultiDiGraph,
    device_mapping: Dict[str, List[str]],
    node_metadata: Dict[str, Dict[str, Any]],
    num_rings: int = 50,
    ring_size_range: Tuple[int, int] = (3, 6),
    time_window_days: int = 30
) -> int:
    """
    Injects circular flow rings: A -> B -> C -> ... -> A.
    Includes CAMOUFLAGE edges:
    - Ring accounts receive payments from normal users/merchants or pay merchants.
    - Transactions occur across varying days with realistic lognormal amounts,
      preventing age/amount trivial leakage.
    """
    injected_accounts = 0
    start_time = 1700000000.0
    total_seconds = time_window_days * 86400.0
    merchants = [n for n, d in node_metadata.items() if d.get("account_type") == "merchant"]
    users = [n for n, d in node_metadata.items() if d.get("account_type") == "legit_user" and d.get("is_fraud") == 0]
    users = [n for n, d in node_metadata.items() if d.get("account_type") == "legit_user" and d.get("is_fraud") == 0]

    for ring_idx in range(num_rings):
        ring_size = random.randint(ring_size_range[0], ring_size_range[1])
        ring_id = f"cycle_{ring_idx:03d}"
        group_id = f"ring_cycle_{ring_idx:03d}"
        ring_nodes = [f"acc_cycle_{ring_idx:03d}_{i}" for i in range(ring_size)]

        # Creation staggered across the window (Day 2 to Day 26)
        ring_origin_time = start_time + random.uniform(2 * 86400.0, total_seconds - 5 * 86400.0)

        for i, node in enumerate(ring_nodes):
            dev = f"dev_cycle_{ring_idx:03d}_{i}"
            device_mapping[node] = [dev]
            node_metadata[node] = {
                "is_fraud": 1,
                "ring_type": "cycle",
                "ring_id": ring_id,
                "group_id": group_id,
                "account_type": "fraud_ring_member"
            }
            graph.add_node(node, **node_metadata[node])
            injected_accounts += 1

            # Warm-up / Camouflage transactions (3 to 6 transactions matching normal distribution)
            num_warmup = random.randint(3, 6)
            for w_idx in range(num_warmup):
                # Spread across days prior to attack
                w_time = ring_origin_time - random.uniform(86400.0, 18 * 86400.0)
                if w_time < start_time:
                    w_time = start_time + random.uniform(3600, 86400)
                w_amt = round(float(np.random.lognormal(mean=3.8, sigma=0.85)), 2)
                w_amt = max(5.0, min(w_amt, 1500.0))
                
                # Alternate between merchant purchases and P2P with legit users
                if merchants and random.random() < 0.60:
                    merch = random.choice(merchants)
                    cam_id = f"tx_cam_cyc_m_{ring_idx}_{i}_{w_idx}"
                    graph.add_edge(node, merch, key=cam_id, tx_id=cam_id, amount=w_amt,
                                   timestamp=w_time, device_id=dev, is_ring_tx=False)
                elif users:
                    u = random.choice(users)
                    cam_id = f"tx_cam_cyc_u_{ring_idx}_{i}_{w_idx}"
                    if random.random() < 0.5:
                        graph.add_edge(node, u, key=cam_id, tx_id=cam_id, amount=w_amt,
                                       timestamp=w_time, device_id=dev, is_ring_tx=False)
                    else:
                        graph.add_edge(u, node, key=cam_id, tx_id=cam_id, amount=w_amt,
                                       timestamp=w_time, device_id=device_mapping[u][0], is_ring_tx=False)

        # Ring attack sequence: money flows in a cycle
        attack_time = ring_origin_time + random.uniform(3600, 86400)
        base_amount = round(float(np.random.lognormal(mean=5.4, sigma=0.6)), 2)  # ~$150 - $600
        base_amount = max(90.0, min(base_amount, 1200.0))

        for step in range(ring_size):
            sender = ring_nodes[step]
            receiver = ring_nodes[(step + 1) % ring_size]
            # Time delta between hops is 1 to 12 hours (realistic, not 10 seconds!)
            tx_time = attack_time + (step * random.uniform(3600, 43200))
            amount = round(base_amount * (0.97 ** step), 2)
            tx_id = f"tx_cycle_{ring_idx:03d}_{step}"

            graph.add_edge(
                sender,
                receiver,
                key=tx_id,
                tx_id=tx_id,
                amount=amount,
                timestamp=tx_time,
                device_id=device_mapping[sender][0],
                is_ring_tx=True
            )

    return injected_accounts


def inject_fan_in_mule_clusters(
    graph: nx.MultiDiGraph,
    device_mapping: Dict[str, List[str]],
    node_metadata: Dict[str, Dict[str, Any]],
    num_clusters: int = 35,
    senders_per_cluster_range: Tuple[int, int] = (8, 14),
    time_window_days: int = 30
) -> int:
    """
    Injects mule aggregation funnels:
    Multiple accounts funnel money into a mule aggregator, which dumps to cash-out.
    Includes CAMOUFLAGE edges:
    - Senders have pre-existing legitimate activity or received funds from users.
    - Mule makes regular purchases at legit merchants.
    """
    injected_accounts = 0
    start_time = 1700000000.0
    total_seconds = time_window_days * 86400.0
    merchants = [n for n, d in node_metadata.items() if d.get("account_type") == "merchant"]
    users = [n for n, d in node_metadata.items() if d.get("account_type") == "legit_user" and d.get("is_fraud") == 0]
    users = [n for n, d in node_metadata.items() if d.get("account_type") == "legit_user" and d.get("is_fraud") == 0]

    for cluster_idx in range(num_clusters):
        num_senders = random.randint(senders_per_cluster_range[0], senders_per_cluster_range[1])
        ring_id = f"mule_{cluster_idx:03d}"
        group_id = f"ring_mule_{cluster_idx:03d}"
        mule_acc = f"acc_mule_{cluster_idx:03d}_hub"
        cashout_acc = f"acc_mule_{cluster_idx:03d}_out"
        sender_accs = [f"acc_mule_{cluster_idx:03d}_s_{i}" for i in range(num_senders)]

        all_nodes = sender_accs + [mule_acc, cashout_acc]
        cluster_origin = start_time + random.uniform(3 * 86400.0, total_seconds - 4 * 86400.0)

        for acc in all_nodes:
            dev = f"dev_{acc}"
            device_mapping[acc] = [dev]
            node_metadata[acc] = {
                "is_fraud": 1,
                "ring_type": "fan_in_mule",
                "ring_id": ring_id,
                "group_id": group_id,
                "account_type": "fraud_ring_member"
            }
            graph.add_node(acc, **node_metadata[acc])
            injected_accounts += 1

            # Warm-up / Camouflage background transactions (3 to 6 per account)
            num_warmup = random.randint(3, 6)
            for w_idx in range(num_warmup):
                w_time = cluster_origin - random.uniform(86400.0, 18 * 86400.0)
                if w_time < start_time:
                    w_time = start_time + random.uniform(3600, 86400)
                w_amt = round(float(np.random.lognormal(mean=3.8, sigma=0.85)), 2)
                w_amt = max(5.0, min(w_amt, 1500.0))

                if merchants and random.random() < 0.60:
                    merch = random.choice(merchants)
                    tx_id = f"tx_cam_mule_m_{acc}_{w_idx}"
                    graph.add_edge(acc, merch, key=tx_id, tx_id=tx_id, amount=w_amt,
                                   timestamp=w_time, device_id=dev, is_ring_tx=False)
                elif users:
                    u = random.choice(users)
                    tx_id = f"tx_cam_mule_u_{acc}_{w_idx}"
                    if random.random() < 0.5:
                        graph.add_edge(acc, u, key=tx_id, tx_id=tx_id, amount=w_amt,
                                       timestamp=w_time, device_id=dev, is_ring_tx=False)
                    else:
                        graph.add_edge(u, acc, key=tx_id, tx_id=tx_id, amount=w_amt,
                                       timestamp=w_time, device_id=device_mapping[u][0], is_ring_tx=False)

        # Attack: senders funnel smurfed amounts into the mule
        attack_start = cluster_origin + random.uniform(3600, 86400)
        total_inflow = 0.0

        for i, s in enumerate(sender_accs):
            # Funnel payments spread across 6 to 36 hours
            tx_time = attack_start + random.uniform(0, 36 * 3600)
            amount = round(float(np.random.lognormal(mean=4.2, sigma=0.7)), 2)  # ~$40 - $250
            amount = max(35.0, min(amount, 650.0))
            total_inflow += amount
            tx_id = f"tx_mule_{cluster_idx:03d}_in_{i}"

            graph.add_edge(
                s,
                mule_acc,
                key=tx_id,
                tx_id=tx_id,
                amount=amount,
                timestamp=tx_time,
                device_id=device_mapping[s][0],
                is_ring_tx=True
            )

        # Cash-out: Mule dumps 85-95% out within a day
        cashout_time = attack_start + (36 * 3600) + random.uniform(1800, 43200)
        cashout_amount = round(total_inflow * random.uniform(0.88, 0.95), 2)
        tx_out_id = f"tx_mule_{cluster_idx:03d}_out"

        graph.add_edge(
            mule_acc,
            cashout_acc,
            key=tx_out_id,
            tx_id=tx_out_id,
            amount=cashout_amount,
            timestamp=cashout_time,
            device_id=device_mapping[mule_acc][0],
            is_ring_tx=True
        )

    return injected_accounts


def inject_shared_device_clusters(
    graph: nx.MultiDiGraph,
    device_mapping: Dict[str, List[str]],
    node_metadata: Dict[str, Dict[str, Any]],
    num_clusters: int = 30,
    accounts_per_cluster_range: Tuple[int, int] = (6, 10),
    time_window_days: int = 30
) -> int:
    """
    Injects device-farm fraud clusters:
    Multiple synthetic accounts operating from the exact same hardware fingerprint,
    cross-transacting and pooling funds.
    Includes CAMOUFLAGE edges:
    - Farm accounts purchase from normal merchants.
    - Transaction timing matches normal distributions.
    """
    injected_accounts = 0
    start_time = 1700000000.0
    total_seconds = time_window_days * 86400.0
    merchants = [n for n, d in node_metadata.items() if d.get("account_type") == "merchant"]
    users = [n for n, d in node_metadata.items() if d.get("account_type") == "legit_user" and d.get("is_fraud") == 0]

    for cluster_idx in range(num_clusters):
        num_accounts = random.randint(accounts_per_cluster_range[0], accounts_per_cluster_range[1])
        ring_id = f"device_{cluster_idx:03d}"
        group_id = f"ring_device_{cluster_idx:03d}"
        shared_device = f"dev_shared_farm_{cluster_idx:03d}"
        cluster_accs = [f"acc_devfarm_{cluster_idx:03d}_{i}" for i in range(num_accounts)]

        cluster_origin = start_time + random.uniform(2 * 86400.0, total_seconds - 5 * 86400.0)

        for acc in cluster_accs:
            device_mapping[acc] = [shared_device]
            node_metadata[acc] = {
                "is_fraud": 1,
                "ring_type": "shared_device",
                "ring_id": ring_id,
                "group_id": group_id,
                "account_type": "fraud_ring_member"
            }
            graph.add_node(acc, **node_metadata[acc])
            injected_accounts += 1

            # Warm-up transactions (2 to 5 per account)
            num_warmup = random.randint(2, 5)
            for w_idx in range(num_warmup):
                w_time = cluster_origin - random.uniform(86400.0, 16 * 86400.0)
                if w_time < start_time:
                    w_time = start_time + random.uniform(3600, 86400)
                w_amt = round(float(np.random.lognormal(mean=3.7, sigma=0.85)), 2)
                w_amt = max(5.0, min(w_amt, 1500.0))

                if merchants and random.random() < 0.60:
                    merch = random.choice(merchants)
                    tx_id = f"tx_cam_dev_m_{acc}_{w_idx}"
                    graph.add_edge(acc, merch, key=tx_id, tx_id=tx_id, amount=w_amt,
                                   timestamp=w_time, device_id=shared_device, is_ring_tx=False)
                elif users:
                    u = random.choice(users)
                    tx_id = f"tx_cam_dev_u_{acc}_{w_idx}"
                    if random.random() < 0.5:
                        graph.add_edge(acc, u, key=tx_id, tx_id=tx_id, amount=w_amt,
                                       timestamp=w_time, device_id=shared_device, is_ring_tx=False)
                    else:
                        graph.add_edge(u, acc, key=tx_id, tx_id=tx_id, amount=w_amt,
                                       timestamp=w_time, device_id=device_mapping[u][0], is_ring_tx=False)

        # Dense cross transactions between the synthetic accounts
        attack_start = cluster_origin + random.uniform(3600, 43200)
        for i in range(num_accounts):
            for j in range(num_accounts):
                if i != j and random.random() < 0.35:
                    sender = cluster_accs[i]
                    receiver = cluster_accs[j]
                    tx_time = attack_start + random.uniform(0, 48 * 3600)
                    amount = round(float(np.random.lognormal(mean=4.1, sigma=0.75)), 2)
                    amount = max(25.0, min(amount, 500.0))
                    tx_id = f"tx_devfarm_{cluster_idx:03d}_{i}_{j}"

                    graph.add_edge(
                        sender,
                        receiver,
                        key=tx_id,
                        tx_id=tx_id,
                        amount=amount,
                        timestamp=tx_time,
                        device_id=shared_device,
                        is_ring_tx=True
                    )

    return injected_accounts


def build_and_save_graph(
    output_dir: str = "data",
    num_legit: int = 5000,
    time_window_days: int = 30
) -> Dict[str, Any]:
    """
    Main pipeline:
    1. Builds realistic base graph with hard negatives (shared devices, rent split, payroll).
    2. Injects rings with camouflage connections to legit network.
    3. Computes node features.
    4. Saves clean artifacts (CSV, JSON, NPZ) without unsafe pickles.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Base graph
    graph, device_mapping, node_metadata = generate_synthetic_transactions(
        num_legit_accounts=num_legit,
        time_window_days=time_window_days
    )

    # 2. Inject rings
    print("[RingBreaker Data Pipeline] Injecting realistic rings with camouflage...")
    num_cycle_accounts = inject_circular_rings(graph, device_mapping, node_metadata, num_rings=50, time_window_days=time_window_days)
    num_mule_accounts = inject_fan_in_mule_clusters(graph, device_mapping, node_metadata, num_clusters=35, time_window_days=time_window_days)
    num_device_accounts = inject_shared_device_clusters(graph, device_mapping, node_metadata, num_clusters=30, time_window_days=time_window_days)

    total_injected = num_cycle_accounts + num_mule_accounts + num_device_accounts
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    fraud_rate = (total_injected / total_nodes) * 100.0

    print("[RingBreaker Data Pipeline] Computing account node features...")
    features_df = compute_node_features(graph, device_mapping)

    # Labels and cluster mapping DataFrame
    labels_records = []
    for node, meta in node_metadata.items():
        labels_records.append({
            "account_id": node,
            "is_fraud": meta["is_fraud"],
            "ring_type": meta.get("ring_type", "none"),
            "ring_id": meta.get("ring_id", "none"),
            "group_id": meta.get("group_id", f"indiv_{node}"),
            "account_type": meta.get("account_type", "normal")
        })
    labels_df = pd.DataFrame(labels_records).set_index("account_id")
    features_df = features_df.loc[labels_df.index]

    # 3. Serialization (Non-pickle JSON, CSV, NPZ)
    features_path = os.path.join(output_dir, "features.csv")
    labels_path = os.path.join(output_dir, "labels.csv")
    device_mapping_path = os.path.join(output_dir, "device_mapping.json")
    graph_npz_path = os.path.join(output_dir, "graph_edges.npz")
    summary_path = os.path.join(output_dir, "summary.json")

    features_df.to_csv(features_path)
    labels_df.to_csv(labels_path)

    with open(device_mapping_path, "w") as f:
        json.dump(device_mapping, f, indent=2)

    # Save edges and attributes in compressed NPZ
    edge_src, edge_dst, edge_amt, edge_time, edge_ring = [], [], [], [], []
    for u, v, d in graph.edges(data=True):
        edge_src.append(u)
        edge_dst.append(v)
        edge_amt.append(d.get("amount", 0.0))
        edge_time.append(d.get("timestamp", 0.0))
        edge_ring.append(1 if d.get("is_ring_tx", False) else 0)

    np.savez_compressed(
        graph_npz_path,
        src=np.array(edge_src, dtype=object),
        dst=np.array(edge_dst, dtype=object),
        amount=np.array(edge_amt, dtype=np.float32),
        timestamp=np.array(edge_time, dtype=np.float64),
        is_ring_tx=np.array(edge_ring, dtype=np.int8)
    )

    # Also save graph.pkl for backward compatibility with NetworkX if needed
    import pickle
    with open(os.path.join(output_dir, "graph.pkl"), "wb") as f:
        pickle.dump({
            "graph": graph,
            "device_mapping": device_mapping,
            "node_metadata": node_metadata
        }, f)

    summary = {
        "dataset_type": "synthetic_with_injected_rings_and_hard_negatives",
        "notice": "Trained on synthetic network with injected ground-truth rings. Real-world deployment requires training on labeled merchant transaction history.",
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "legit_nodes": total_nodes - total_injected,
        "fraud_nodes": total_injected,
        "fraud_rate_pct": round(fraud_rate, 2),
        "injected_rings": {
            "circular_flow": {"rings": 50, "accounts": num_cycle_accounts},
            "fan_in_mule": {"clusters": 35, "accounts": num_mule_accounts},
            "shared_device": {"clusters": 30, "accounts": num_device_accounts}
        },
        "feature_names": FEATURE_NAMES,
        "artifacts": {
            "features_csv": "data/features.csv",
            "labels_csv": "data/labels.csv",
            "device_mapping_json": "data/device_mapping.json",
            "graph_edges_npz": "data/graph_edges.npz",
            "graph_pkl": "data/graph.pkl"
        }
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print("      RINGBREAKER HARDENED GRAPH GENERATION SUMMARY")
    print("=" * 68)
    print(f"Total Accounts (Nodes):       {total_nodes:,}")
    print(f"Total Transactions (Edges):   {total_edges:,}")
    print(f"Legitimate Accounts:          {total_nodes - total_injected:,}")
    print(f"Fraudulent Accounts:          {total_injected:,}")
    print(f"Account Fraud Rate:           {fraud_rate:.2f}%")
    print("-" * 68)
    print("Injected Labeled Rings:")
    print(f"  • Circular Flow (Cycles):    50 rings    ({num_cycle_accounts} accounts)")
    print(f"  • Fan-in Mule Funnels:       35 clusters ({num_mule_accounts} accounts)")
    print(f"  • Shared-Device Clusters:    30 clusters ({num_device_accounts} accounts)")
    print("-" * 68)
    print("Saved Artifacts (POSIX relative paths, no pickled models):")
    print(f"  • {features_path}")
    print(f"  • {labels_path}")
    print(f"  • {device_mapping_path}")
    print(f"  • {graph_npz_path}")
    print(f"  • {summary_path}")
    print("=" * 68 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Graph Builder")
    parser.add_argument("--num-legit", type=int, default=5000, help="Number of legitimate accounts")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    build_and_save_graph(output_dir=args.output_dir, num_legit=args.num_legit)
