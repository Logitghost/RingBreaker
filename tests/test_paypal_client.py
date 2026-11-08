"""
Unit tests for RingBreaker PayPal Client (Phase 1 Hardened).
Tests order creation, authorization, capture, void, and honest dual-mode behavior.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.paypal_client import PayPalClient


def test_paypal_client_unconfigured_defaults_to_simulated():
    client = PayPalClient(client_id="", client_secret="")
    assert not client.is_configured()

    # Order creation should succeed in simulated mode
    order = client.create_order(amount=150.00, currency="USD")
    assert order["order_mode"] == "simulated"
    assert order["is_live_sandbox"] is False
    assert order["intent"] == "AUTHORIZE"
    assert order["id"].startswith("SIM-ORD-")
    assert order["amount"] == 150.00


def test_paypal_client_simulated_flow_end_to_end():
    client = PayPalClient(client_id="dummy", client_secret="dummy")
    
    # 1. Create order
    order = client.create_order(amount=250.50)
    order_id = order["id"]
    assert order["status"] == "CREATED"
    assert order["order_mode"] == "simulated"

    # 2. Authorize order
    auth_resp = client.authorize_order(order_id)
    assert auth_resp["status"] == "COMPLETED"
    auth_id = auth_resp["authorization_id"]
    assert auth_id.startswith("SIM-AUTH-")
    assert auth_resp["authorization_mode"] == "simulated"

    # 3. Capture authorization (Low risk)
    cap_resp = client.capture_authorization(auth_id, amount=250.50)
    assert cap_resp["status"] == "COMPLETED"
    assert cap_resp["action"] == "CAPTURE"
    assert cap_resp["id"].startswith("SIM-CAP-")
    assert cap_resp["capture_mode"] == "simulated"

    # 4. Void authorization (High risk)
    void_resp = client.void_authorization(auth_id)
    assert void_resp["status"] == "VOIDED"
    assert void_resp["action"] == "VOID"
    assert void_resp["void_mode"] == "simulated"


@patch("requests.Session.post")
def test_paypal_client_live_oauth_and_order_flow(mock_post):
    # Mock OAuth response
    oauth_response = MagicMock()
    oauth_response.status_code = 200
    oauth_response.json.return_value = {
        "access_token": "mock_sandbox_access_token_123",
        "token_type": "Bearer",
        "expires_in": 3600
    }

    # Mock Create Order response
    order_response = MagicMock()
    order_response.status_code = 201
    order_response.json.return_value = {
        "id": "5O190127TN364715T",
        "status": "CREATED",
        "intent": "AUTHORIZE",
        "purchase_units": [{"reference_id": "rb_tx_test"}]
    }

    # Mock Authorize response
    auth_response = MagicMock()
    auth_response.status_code = 201
    auth_response.json.return_value = {
        "id": "5O190127TN364715T",
        "status": "COMPLETED",
        "purchase_units": [{
            "payments": {
                "authorizations": [{
                    "id": "0UB75628MV027474T",
                    "status": "CREATED"
                }]
            }
        }]
    }

    # Mock Capture response
    capture_response = MagicMock()
    capture_response.status_code = 201
    capture_response.json.return_value = {
        "id": "2C75376483482343P",
        "status": "COMPLETED"
    }

    mock_post.side_effect = [oauth_response, order_response, auth_response, capture_response]

    client = PayPalClient(client_id="valid_client_id", client_secret="valid_secret_key")
    assert client.is_configured()

    # 1. Create live sandbox order
    order = client.create_order(amount=99.99)
    assert order["order_mode"] == "live"
    assert order["is_live_sandbox"] is True
    assert order["id"] == "5O190127TN364715T"

    # 2. Authorize
    auth = client.authorize_order(order["id"])
    assert auth["authorization_mode"] == "live"
    assert auth["authorization_id"] == "0UB75628MV027474T"

    # 3. Capture
    cap = client.capture_authorization(auth["authorization_id"], amount=99.99)
    assert cap["capture_mode"] == "live"
    assert cap["id"] == "2C75376483482343P"
    assert cap["status"] == "COMPLETED"


def test_webhook_verification_unconfigured_rejects():
    # Directive 1.5: Never silently trust unconfigured webhooks
    client = PayPalClient(client_id="", client_secret="", webhook_id="")
    headers = {
        "paypal-auth-algo": "SHA256withRSA",
        "paypal-cert-url": "https://api.sandbox.paypal.com/v1/notifications/certs/CERT-360",
        "paypal-transmission-id": "trans_123",
        "paypal-transmission-sig": "sig_abc",
        "paypal-transmission-time": "2026-11-07T12:00:00Z"
    }
    body = {"id": "WH-123", "event_type": "CHECKOUT.ORDER.APPROVED"}
    assert client.verify_webhook(headers, body) is False
