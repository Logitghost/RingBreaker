"""
RingBreaker - Live Scoring Latency Benchmark (Phase 3 Directive 3.1)
Measures end-to-end payment processing & GNN ego-graph scoring latency.
Target: Under 200ms per payment.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import random
import numpy as np

from app.graph_store import GraphStore
from app.paypal_client import PayPalClient
from app.scoring import RiskScorer

NUM_BENCHMARK_PAYMENTS = 50
MAX_ALLOWED_P95_MS = 200.0


def run_latency_benchmark(output_txt: str = "docs/evidence/benchmark.txt") -> bool:
    print(f"[Scoring Benchmark] Initializing GraphStore, PayPal client, and RiskScorer...")
    graph_store = GraphStore()
    paypal_client = PayPalClient()
    scorer = RiskScorer(graph_store=graph_store)

    nodes = list(graph_store.graph.nodes())
    if len(nodes) < 2:
        print("Error: GraphStore has fewer than 2 nodes.")
        return False

    print(f"[Scoring Benchmark] Executing {NUM_BENCHMARK_PAYMENTS} real GNN scoring calls...")
    latencies_ms = []
    models_used = []
    tiers = []

    # Warm-up run (1 payment)
    s_warm, r_warm = random.sample(nodes, 2)
    scorer.process_payment(s_warm, r_warm, 45.0, "dev_warmup", graph_store, paypal_client)

    for i in range(NUM_BENCHMARK_PAYMENTS):
        u, v = random.sample(nodes, 2)
        amt = round(random.uniform(15.0, 350.0), 2)
        dev = f"dev_bench_{i:03d}"

        t0 = time.perf_counter()
        res = scorer.process_payment(u, v, amt, dev, graph_store, paypal_client)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        models_used.append(res.get("model_used"))
        tiers.append(res.get("risk_tier"))

    # Compute statistics
    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))

    gnn_count = models_used.count("gnn")
    gnn_pct = (gnn_count / len(models_used)) * 100.0

    lines = [
        "=" * 74,
        "              RINGBREAKER LIVE SCORING LATENCY BENCHMARK",
        "=" * 74,
        f"Evaluated Payments:       {NUM_BENCHMARK_PAYMENTS}",
        f"GNN Model Used Share:     {gnn_count}/{len(models_used)} ({gnn_pct:.1f}%)",
        f"Target P95 Latency:       < {MAX_ALLOWED_P95_MS:.1f} ms",
        "-" * 74,
        f"{'Latency Metric':<28} | Duration (ms)",
        "-" * 74,
        f"{'Min Latency':<28} | {min_lat:>8.2f} ms",
        f"{'Median Latency (p50)':<28} | {median_lat:>8.2f} ms",
        f"{'Mean Latency':<28} | {mean_lat:>8.2f} ms",
        f"{'95th Percentile (p95)':<28} | {p95_lat:>8.2f} ms",
        f"{'99th Percentile (p99)':<28} | {p99_lat:>8.2f} ms",
        f"{'Max Latency':<28} | {max_lat:>8.2f} ms",
        "-" * 74,
        f"Risk Tiers Produced: LOW={tiers.count('LOW')}, MEDIUM={tiers.count('MEDIUM')}, HIGH={tiers.count('HIGH')}",
        "=" * 74
    ]

    passed = p95_lat < MAX_ALLOWED_P95_MS and gnn_count > 0
    if passed:
        lines.append(f"[PASS] Latency benchmark PASSED: p95 = {p95_lat:.2f} ms (< {MAX_ALLOWED_P95_MS:.1f} ms). Real GNN invoked.")
    else:
        lines.append(f"[FAIL] Latency benchmark FAILED: p95 = {p95_lat:.2f} ms (>= {MAX_ALLOWED_P95_MS:.1f} ms) or GNN not used.")
    lines.append("=" * 74)

    bench_text = "\n".join(lines)
    print(bench_text)

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(bench_text + "\n")
    print(f"\nBenchmark evidence saved to: {output_txt}")

    return passed


if __name__ == "__main__":
    success = run_latency_benchmark()
    sys.exit(0 if success else 1)
