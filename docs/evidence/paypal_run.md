# RingBreaker PayPal Client & Sandbox Evidence

## 1. Overview & Operational Modes

RingBreaker integrates with **PayPal Orders v2** using a two-phase authorization flow:
- `intent=AUTHORIZE` at order creation.
- Tiered decisioning:
  - **LOW (< 0.40)**: Captured via `POST /v2/payments/authorizations/{id}/capture`
  - **MEDIUM [0.40, 0.60)**: Kept in `IN_REVIEW` hold until analyst approval or void
  - **HIGH (>= 0.60)**: Voided before settlement via `POST /v2/payments/authorizations/{id}/void`

To ensure RingBreaker can be evaluated locally without requiring third-party credentials, the system features a **two-tier execution engine**:

| Mode | Trigger | Behavior |
| :--- | :--- | :--- |
| **Simulated (Default)** | Placeholders or empty in `.env` | Emits realistic PayPal schema payloads (`SIM-ORD-...`, `SIM-AUTH-...`, `SIM-CAP-...`), records status in SQLite, enforces idempotency, logs audit trails, and exercises identical decision pathways. |
| **Live Sandbox** | Valid sandbox credentials in `.env` | Authenticates via OAuth2 against `https://api-m.sandbox.paypal.com/v1/oauth2/token` and issues live HTTP requests to Orders v2. |

> [!IMPORTANT]
> **Strict Sandbox Guard**:
> `app/paypal_client.py` strictly refuses any non-sandbox environment (`PAYPAL_MODE=live` raises a `ValueError`). RingBreaker never initiates money movement against production systems.

---

## 2. API Contract & Payload Verification

### (a) Order Creation (`POST /v2/checkout/orders`)
```json
{
  "intent": "AUTHORIZE",
  "purchase_units": [
    {
      "reference_id": "default",
      "custom_id": "tx_demo_norm_01",
      "amount": {
        "currency_code": "USD",
        "value": "45.00"
      },
      "description": "RingBreaker P2P Checkout"
    }
  ],
  "application_context": {
    "brand_name": "RingBreaker Sandbox",
    "shipping_preference": "NO_SHIPPING",
    "user_action": "CONTINUE"
  }
}
```

### (b) Order Authorization (`POST /v2/checkout/orders/{id}/authorize`)
```json
{
  "id": "SIM-ORD-7C1E8A90B2",
  "status": "COMPLETED",
  "purchase_units": [
    {
      "payments": {
        "authorizations": [
          {
            "id": "SIM-AUTH-3D5F9A1B2C",
            "status": "CREATED",
            "amount": {
              "currency_code": "USD",
              "value": "45.00"
            }
          }
        ]
      }
    }
  ]
}
```

### (c) Capture or Void Decision
- **Capture**: `POST /v2/payments/authorizations/{auth_id}/capture`
- **Void**: `POST /v2/payments/authorizations/{auth_id}/void`

Each outbound call carries:
- `PayPal-Request-Id`: Cryptographic UUID for strict payment gateway idempotency.
- Retries with exponential backoff on HTTP 429 and 5xx responses.

---

## 3. Webhook Verification Architecture

Inbound webhooks to `/api/webhooks/paypal` are verified against PayPal:
1. In sandbox-connected mode, RingBreaker calls PayPal's verification API:
   `POST /v1/notifications/verify-webhook-signature`
   with certificate URL, transmission ID, transmission time, and HMAC signature headers.
2. In simulated mode, signature header presence is validated and stored in `webhook_events`.
3. Event idempotency:
   - `PAYMENT.CAPTURE.COMPLETED`: Matches `paypal_capture_id` or `paypal_auth_id`, updates database status to `CAPTURED`.
   - `PAYMENT.AUTHORIZATION.VOIDED`: Matches `paypal_auth_id`, updates database status to `VOIDED`.
4. Unverified webhooks are rejected with HTTP 400.

---

## 4. Test Verification

Run PayPal client verification:
```powershell
py -3.11 scripts/verify_paypal.py
```
**Outcome**:
- Verifies Order creation (`AUTHORIZE` intent)
- Verifies Authorization retrieval
- Verifies Authorization Voiding
- Verifies Webhook event storage & idempotent replay protection
