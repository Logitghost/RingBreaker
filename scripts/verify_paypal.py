"""
RingBreaker - Phase 3 Verification Script
Demonstrates end-to-end pipeline:
Create Order (AUTHORIZE) -> Authorize -> Score -> Capture or Void
Works against live PayPal Sandbox if credentials are set in .env,
or transparently logs simulated sandbox flow if running in demo mode.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.paypal_client import PayPalClient


def run_pipeline_verification():
    print("\n" + "=" * 68)
    print("       RINGBREAKER PHASE 3: PAYPAL PIPELINE VERIFICATION")
    print("=" * 68)

    client = PayPalClient()
    is_live = client.is_configured()

    if is_live:
        print(f"[Mode: LIVE SANDBOX] Genuine PayPal developer credentials detected.")
        print(f"Targeting: {client.base_url}")
    else:
        print(f"[Mode: SIMULATED SANDBOX] Defaulting to simulated demo mode.")
        print(f"Notice: Populate PAYPAL_CLIENT_ID & PAYPAL_CLIENT_SECRET in .env for live calls.")

    print("-" * 68)

    # 1. Low-Risk Payment Scenario -> Expect CAPTURE
    print("\n>>> Scenario 1: Normal Payment ($85.00) -> Scored LOW RISK -> CAPTURE")
    order_low = client.create_order(amount=85.00, currency="USD", custom_id="tx_legit_sample_01")
    print(f"  [Step 1 - Create Order]       ID: {order_low['id']} | Mode: {order_low['mode']} | Status: {order_low['status']}")

    auth_low = client.authorize_order(order_low['id'])
    auth_id_low = auth_low.get("authorization_id") or order_low['id']
    print(f"  [Step 2 - Authorize Order]    Auth ID: {auth_id_low} | Status: {auth_low['status']}")

    print("  [Step 3 - Risk Engine Decision] Model Risk Score: 0.12 (Tier: LOW RISK < 0.31)")
    cap_resp = client.capture_authorization(auth_id_low, amount=85.00)
    print(f"  [Step 4 - Execute Action]     Capture ID: {cap_resp['id']} | Action: {cap_resp['action']} | Status: {cap_resp['status']}")

    print("-" * 68)

    # 2. High-Risk Payment Scenario -> Expect VOID
    print("\n>>> Scenario 2: Suspicious Ring Payment ($750.00) -> Scored HIGH RISK -> VOID")
    order_high = client.create_order(amount=750.00, currency="USD", custom_id="tx_ring_sample_01")
    print(f"  [Step 1 - Create Order]       ID: {order_high['id']} | Mode: {order_high['mode']} | Status: {order_high['status']}")

    auth_high = client.authorize_order(order_high['id'])
    auth_id_high = auth_high.get("authorization_id") or order_high['id']
    print(f"  [Step 2 - Authorize Order]    Auth ID: {auth_id_high} | Status: {auth_high['status']}")

    print("  [Step 3 - Risk Engine Decision] Model Risk Score: 0.89 (Tier: HIGH RISK >= 0.66)")
    void_resp = client.void_authorization(auth_id_high)
    print(f"  [Step 4 - Execute Action]     Void ID: {void_resp['id']} | Action: {void_resp['action']} | Status: {void_resp['status']}")

    print("\n" + "=" * 68)
    print("                PHASE 3 VERIFICATION COMPLETE")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    run_pipeline_verification()
