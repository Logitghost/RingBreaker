"""
RingBreaker - Live Graph Store & SQLite Persistence
Maintains an in-memory NetworkX transaction graph updated in real time,
backed by SQLite for transaction histories, account risks, audit logs, and human review decisions.
ZERO GROUND-TRUTH LEAKS: Node prioritization and styling rely strictly on model scores and topology.
"""

import os
import sys
import time
import json
import sqlite3
import pickle
from typing import Dict, List, Any, Optional, Tuple, Set
import networkx as nx

DB_PATH = os.getenv("DATABASE_PATH", "ringbreaker.db")


class GraphStore:
    def __init__(self, db_path: str = DB_PATH, graph_pkl_path: str = "data/graph.pkl"):
        self.db_path = db_path
        self.graph_pkl_path = graph_pkl_path
        self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False) if db_path == ":memory:" else None
        if self._mem_conn:
            self._mem_conn.row_factory = sqlite3.Row
        self._init_sqlite()
        self._init_networkx()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        """
        Creates SQLite schema for transactions, accounts, events, audit logs, and settings.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    sender_id TEXT NOT NULL,
                    receiver_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'USD',
                    timestamp REAL NOT NULL,
                    device_id TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    paypal_order_id TEXT,
                    paypal_auth_id TEXT,
                    paypal_capture_id TEXT,
                    order_mode TEXT NOT NULL DEFAULT 'simulated',
                    auth_mode TEXT NOT NULL DEFAULT 'simulated',
                    capture_mode TEXT NOT NULL DEFAULT 'simulated',
                    model_used TEXT NOT NULL DEFAULT 'gnn',
                    cold_start INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    risk_score REAL NOT NULL DEFAULT 0.05,
                    risk_tier TEXT NOT NULL DEFAULT 'LOW',
                    cold_start INTEGER DEFAULT 0,
                    last_updated REAL NOT NULL,
                    analyst_label TEXT DEFAULT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account_devices (
                    account_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    PRIMARY KEY (account_id, device_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payment_id TEXT NOT NULL,
                    analyst_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    notes TEXT,
                    decided_at REAL NOT NULL,
                    FOREIGN KEY (payment_id) REFERENCES transactions(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS webhook_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    resource_id TEXT,
                    verified INTEGER NOT NULL,
                    raw_body TEXT NOT NULL,
                    received_at REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ground_truth (
                    account_id TEXT PRIMARY KEY,
                    is_fraud INTEGER NOT NULL,
                    ring_type TEXT,
                    ring_id TEXT
                )
            """)
            conn.commit()

    def _init_networkx(self) -> None:
        """
        Loads base graph from data/graph.pkl if available, or creates fresh graph.
        Ground truth is stored strictly in self.ground_truth for demo overlays, never in scoring.
        """
        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.graph_pkl_path):
            with open(self.graph_pkl_path, "rb") as f:
                saved = pickle.load(f)
            self.graph: nx.MultiDiGraph = saved["graph"]
            self.device_mapping: Dict[str, List[str]] = saved.get("device_mapping", {})
            raw_meta = saved.get("node_metadata", {})
            for acc, m in raw_meta.items():
                if "is_fraud" in m:
                    self.ground_truth[acc] = {
                        "is_fraud": m.get("is_fraud", 0),
                        "ring_type": m.get("ring_type", "none"),
                        "ring_id": m.get("ring_id", "none")
                    }
            # Clean node metadata in graph to prevent leakage
            self.node_metadata: Dict[str, Dict[str, Any]] = {}
            print(f"[GraphStore] Loaded in-memory graph with {self.graph.number_of_nodes()} accounts.")
        else:
            self.graph = nx.MultiDiGraph()
            self.device_mapping = {}
            self.node_metadata = {}
            print("[GraphStore] Initialized clean in-memory MultiDiGraph.")

    def add_audit_log(self, kind: str, payload: Dict[str, Any]) -> None:
        """
        Persists structured audit event (PayPal call, score decision, analyst action).
        """
        sanitized = {k: v for k, v in payload.items() if "secret" not in k.lower() and "key" not in k.lower()}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (timestamp, kind, payload)
                VALUES (?, ?, ?)
            """, (time.time(), kind, json.dumps(sanitized)))
            conn.commit()

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def record_webhook_event(
        self,
        event_id: str,
        event_type: str,
        resource_id: Optional[str],
        verified: bool,
        raw_body: str
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO webhook_events (event_id, event_type, resource_id, verified, raw_body, received_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_id, event_type, resource_id, 1 if verified else 0, raw_body, time.time()))
            conn.commit()

    def save_setting(self, key: str, value: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, time.time()))
            conn.commit()

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row["value"] if row else default
        except (sqlite3.OperationalError, Exception):
            return default

    def add_transaction(
        self,
        tx_id: str,
        sender_id: str,
        receiver_id: str,
        amount: float,
        timestamp: float,
        device_id: str,
        risk_score: float,
        risk_tier: str,
        status: str,
        action_taken: str,
        paypal_order_id: Optional[str] = None,
        paypal_auth_id: Optional[str] = None,
        paypal_capture_id: Optional[str] = None,
        order_mode: str = "simulated",
        auth_mode: str = "simulated",
        capture_mode: str = "simulated",
        model_used: str = "gnn",
        cold_start: bool = False
    ) -> None:
        """
        Updates in-memory graph and persists transaction to SQLite.
        """
        # 1. Update in-memory graph
        if not self.graph.has_node(sender_id):
            self.graph.add_node(sender_id)
        if not self.graph.has_node(receiver_id):
            self.graph.add_node(receiver_id)

        self.graph.add_edge(
            sender_id,
            receiver_id,
            key=tx_id,
            tx_id=tx_id,
            amount=amount,
            timestamp=timestamp,
            device_id=device_id
        )

        # Track device
        self.device_mapping.setdefault(sender_id, [])
        if device_id not in self.device_mapping[sender_id]:
            self.device_mapping[sender_id].append(device_id)

        # 2. Persist to SQLite
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO transactions (
                    id, sender_id, receiver_id, amount, currency, timestamp,
                    device_id, risk_score, risk_tier, status, action_taken,
                    paypal_order_id, paypal_auth_id, paypal_capture_id,
                    order_mode, auth_mode, capture_mode, model_used, cold_start
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx_id, sender_id, receiver_id, amount, "USD", timestamp,
                device_id, risk_score, risk_tier, status, action_taken,
                paypal_order_id, paypal_auth_id, paypal_capture_id,
                order_mode, auth_mode, capture_mode, model_used, 1 if cold_start else 0
            ))

            cursor.execute("""
                INSERT OR IGNORE INTO account_devices (account_id, device_id, first_seen)
                VALUES (?, ?, ?)
            """, (sender_id, device_id, timestamp))

            conn.commit()

    def update_account_score(
        self,
        account_id: str,
        risk_score: float,
        risk_tier: str,
        cold_start: bool = False
    ) -> None:
        """
        Updates score and risk tier for an account in SQLite.
        """
        now = time.time()
        self.node_metadata.setdefault(account_id, {})
        self.node_metadata[account_id]["risk_score"] = risk_score
        self.node_metadata[account_id]["risk_tier"] = risk_tier
        self.node_metadata[account_id]["cold_start"] = cold_start

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO accounts (account_id, risk_score, risk_tier, cold_start, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    risk_score=excluded.risk_score,
                    risk_tier=excluded.risk_tier,
                    cold_start=excluded.cold_start,
                    last_updated=excluded.last_updated
            """, (account_id, risk_score, risk_tier, 1 if cold_start else 0, now))
            conn.commit()

    def get_recent_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_review_queue(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transactions WHERE status = 'IN_REVIEW' ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_in_review_payments(self) -> List[Dict[str, Any]]:
        return self.get_review_queue()

    def update_transaction_status(
        self,
        tx_id: str,
        status: str,
        action_taken: str,
        risk_score: Optional[float] = None,
        risk_tier: Optional[str] = None,
        capture_mode: Optional[str] = None
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE transactions
                SET status = ?, action_taken = ?,
                    risk_score = COALESCE(?, risk_score),
                    risk_tier = COALESCE(?, risk_tier),
                    capture_mode = COALESCE(?, capture_mode)
                WHERE id = ?
            """, (status, action_taken, risk_score, risk_tier, capture_mode, tx_id))
            conn.commit()

    def record_analyst_decision(
        self,
        payment_id: str,
        decision: str,
        analyst_id: str = "analyst_1",
        notes: str = ""
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        new_status = "CAPTURED" if decision == "legit" else "VOIDED"
        new_action = "MANUAL_CAPTURE" if decision == "legit" else "MANUAL_VOID"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transactions WHERE id = ?", (payment_id,))
            tx = cursor.fetchone()
            if not tx:
                return None

            cursor.execute("""
                UPDATE transactions
                SET status = ?, action_taken = ?
                WHERE id = ?
            """, (new_status, new_action, payment_id))

            cursor.execute("""
                INSERT INTO review_decisions (payment_id, analyst_id, decision, notes, decided_at)
                VALUES (?, ?, ?, ?, ?)
            """, (payment_id, analyst_id, decision, notes, now))

            cursor.execute("UPDATE accounts SET analyst_label = ? WHERE account_id = ?", (decision, tx["sender_id"]))
            cursor.execute("UPDATE accounts SET analyst_label = ? WHERE account_id = ?", (decision, tx["receiver_id"]))
            conn.commit()

            cursor.execute("SELECT * FROM transactions WHERE id = ?", (payment_id,))
            return dict(cursor.fetchone())

    def get_cytoscape_elements(
        self,
        max_nodes: int = 120,
        show_ground_truth: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Exports in-memory graph subset for Cytoscape.js rendering.
        PRIORITIZES STRICTLY BY MODEL SCORE, DEGREE, AND RECENCY.
        Zero reads of 'ring' name substrings or ground-truth flags.
        """
        account_scores: Dict[str, Tuple[float, str]] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT account_id, risk_score, risk_tier FROM accounts")
            for row in cursor.fetchall():
                account_scores[row["account_id"]] = (row["risk_score"], row["risk_tier"])

        # Prioritize nodes by (Model High Risk, Degree) - NO GROUND TRUTH
        nodes_ranked = sorted(
            self.graph.nodes(),
            key=lambda n: (
                1 if account_scores.get(n, (0.05, "LOW"))[1] == "HIGH" else 0,
                self.graph.degree(n)
            ),
            reverse=True
        )
        selected_nodes = nodes_ranked[:max_nodes]
        subgraph = self.graph.subgraph(selected_nodes)
        elements = {"nodes": [], "edges": []}

        for node in subgraph.nodes():
            score, tier = account_scores.get(node, (0.05, "LOW"))
            is_model_flagged = 1 if tier == "HIGH" else 0
            
            node_data = {
                "id": node,
                "label": node.replace("acc_", ""),
                "risk_score": round(score, 3),
                "risk_tier": tier,
                "is_model_flagged": is_model_flagged,
                "degree": self.graph.degree(node)
            }
            if show_ground_truth:
                gt = self.ground_truth.get(node, {})
                node_data["is_ground_truth_fraud"] = gt.get("is_fraud", 0)

            elements["nodes"].append({"data": node_data})

        for u, v, k, d in subgraph.edges(keys=True, data=True):
            elements["edges"].append({
                "data": {
                    "id": f"{u}_{v}_{k}",
                    "source": u,
                    "target": v,
                    "amount": d.get("amount", 0.0)
                }
            })

        return elements

    def reset_database(self) -> None:
        """
        Cleans dynamic state and re-seeds cleanly.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions")
            cursor.execute("DELETE FROM accounts")
            cursor.execute("DELETE FROM account_devices")
            cursor.execute("DELETE FROM review_decisions")
            cursor.execute("DELETE FROM webhook_events")
            cursor.execute("DELETE FROM audit_log")
            conn.commit()
        self._init_networkx()
        print("[GraphStore] Database reset successfully.")
