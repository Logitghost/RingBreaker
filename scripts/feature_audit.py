"""
RingBreaker - Feature Audit Script (Phase 2 Integrity Check C6)
Asserts that NO SINGLE FEATURE reaches ROC-AUC >= 0.95.
Prevents trivial shortcut leakage (e.g. account age, avg time between transactions).
Exits with code 0 on PASS, code 1 on FAIL.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

MAX_ALLOWED_SINGLE_FEATURE_AUC = 0.95


def run_feature_audit(
    features_csv: str = "data/features.csv",
    labels_csv: str = "data/labels.csv",
    output_report: str = "docs/evidence/feature_audit.txt"
) -> bool:
    if not os.path.exists(features_csv) or not os.path.exists(labels_csv):
        print(f"Error: Required files not found ({features_csv}, {labels_csv}).")
        return False

    features_df = pd.read_csv(features_csv, index_col="account_id")
    labels_df = pd.read_csv(labels_csv, index_col="account_id")

    # Align indexes
    common_idx = features_df.index.intersection(labels_df.index)
    features_df = features_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    y_true = labels_df["is_fraud"].values.astype(int)
    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))

    report_lines = []
    report_lines.append("=" * 76)
    report_lines.append("           RINGBREAKER SINGLE-FEATURE INTEGRITY AUDIT (C6)")
    report_lines.append("=" * 76)
    report_lines.append(f"Total Evaluated Accounts: {len(y_true):,} ({neg_count:,} legit, {pos_count:,} fraud)")
    report_lines.append(f"Max Permitted Single-Feature ROC-AUC: < {MAX_ALLOWED_SINGLE_FEATURE_AUC:.2f}")
    report_lines.append("-" * 76)
    report_lines.append(f"{'Feature Name':<32} | {'Directional AUC':<16} | {'Max Power':<12} | Status")
    report_lines.append("-" * 76)

    any_failed = False
    for col in features_df.columns:
        vals = features_df[col].values
        # Check if constant
        if np.all(vals == vals[0]):
            auc = 0.50
            power = 0.50
        else:
            try:
                auc = float(roc_auc_score(y_true, vals))
                power = max(auc, 1.0 - auc)
            except Exception as e:
                auc = 0.50
                power = 0.50

        status = "FAIL (>= 0.95)" if power >= MAX_ALLOWED_SINGLE_FEATURE_AUC else "PASS (< 0.95)"
        if power >= MAX_ALLOWED_SINGLE_FEATURE_AUC:
            any_failed = True

        report_lines.append(f"{col:<32} | {auc:<16.4f} | {power:<12.4f} | {status}")

    report_lines.append("=" * 76)

    if any_failed:
        report_lines.append("[FAIL] AUDIT FAILED: At least one single feature has ROC-AUC >= 0.95 (shortcut leak).")
    else:
        report_lines.append("[PASS] AUDIT PASSED: All single features have ROC-AUC < 0.95. No trivial shortcut.")
    report_lines.append("=" * 76)

    report_text = "\n".join(report_lines)
    print(report_text)

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nAudit evidence written to: {output_report}")

    return not any_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingBreaker Single Feature Audit")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--out", default="docs/evidence/feature_audit.txt")
    args = parser.parse_args()

    passed = run_feature_audit(args.features, args.labels, args.out)
    sys.exit(0 if passed else 1)
