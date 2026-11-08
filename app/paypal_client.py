"""
RingBreaker - PayPal Sandbox Client
Handles OAuth authentication, Orders v2 (intent=AUTHORIZE), Authorization Capture,
Authorization Void, and Webhook verification against the PayPal Sandbox.
STRICTLY SANDBOX ONLY: Refuses execution against production PayPal URLs.
"""

import os
import sys
import time
import base64
import logging
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("ringbreaker.paypal")


class PayPalClient:
    """
    Client for PayPal Developer Sandbox APIs.
    Enforces sandbox isolation, idempotency headers, Decimal amounts, and retries.
    """
    SANDBOX_BASE_URL = "https://api-m.sandbox.paypal.com"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        mode: Optional[str] = None,
        webhook_id: Optional[str] = None
    ):
        self.client_id = client_id if client_id is not None else os.getenv("PAYPAL_CLIENT_ID", "")
        self.client_secret = client_secret if client_secret is not None else os.getenv("PAYPAL_CLIENT_SECRET", "")
        self.webhook_id = webhook_id if webhook_id is not None else os.getenv("PAYPAL_WEBHOOK_ID", "")
        
        # Enforce sandbox mode strictly
        configured_mode = mode if mode is not None else os.getenv("PAYPAL_MODE", "sandbox")
        if configured_mode.lower() != "sandbox":
            raise ValueError(
                f"RingBreaker is strictly for PayPal Sandbox testing. "
                f"Mode '{configured_mode}' is refused. Only 'sandbox' is permitted."
            )
        self.mode = "sandbox"
        self.base_url = self.SANDBOX_BASE_URL

        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

        # Session with retries on 429 and 5xx
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def is_configured(self) -> bool:
        """
        Returns True if genuine sandbox credentials are configured in .env.
        """
        if not self.client_id or not self.client_secret:
            return False
        invalid_markers = ["your_paypal", "client_id_here", "dummy", "placeholder", "xxx"]
        for marker in invalid_markers:
            if marker in self.client_id.lower() or marker in self.client_secret.lower():
                return False
        return True

    def _format_amount(self, amount: Any) -> str:
        """
        Formats float or Decimal into exact two-decimal string using Decimal.
        """
        dec = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return str(dec)

    def get_access_token(self) -> str:
        """
        Obtains OAuth2 access token using client credentials grant.
        """
        if not self.is_configured():
            raise ValueError(
                "PayPal sandbox credentials are not configured in .env. "
                "Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET from developer.paypal.com."
            )

        if self._cached_token and time.time() < (self._token_expires_at - 60):
            return self._cached_token

        endpoint = f"{self.base_url}/v1/oauth2/token"
        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}

        logger.info(f"POST {endpoint} (Requesting OAuth2 access token)")
        try:
            response = self.session.post(endpoint, headers=headers, data=data, timeout=12)
            response.raise_for_status()
            resp_data = response.json()
            self._cached_token = resp_data["access_token"]
            expires_in = resp_data.get("expires_in", 3600)
            self._token_expires_at = time.time() + float(expires_in)
            logger.info("Successfully acquired PayPal OAuth2 access token.")
            return self._cached_token
        except requests.exceptions.RequestException as e:
            logger.error(f"PayPal OAuth token request failed: {e}")
            raise

    def create_order(
        self,
        amount: Any,
        currency: str = "USD",
        custom_id: Optional[str] = None,
        force_simulation: bool = False
    ) -> Dict[str, Any]:
        """
        Creates an Order with intent=AUTHORIZE and PayPal-Request-Id header.
        """
        custom_id = custom_id or f"rb_tx_{uuid.uuid4().hex[:8]}"
        formatted_amount = self._format_amount(amount)
        request_id = str(uuid.uuid4())

        if not self.is_configured() or force_simulation:
            # Simulated sandbox mode
            simulated_order_id = f"SIM-ORD-{uuid.uuid4().hex[:10].upper()}"
            logger.info(f"[SIMULATED] Created Order {simulated_order_id} for ${formatted_amount} {currency}")
            return {
                "id": simulated_order_id,
                "status": "CREATED",
                "intent": "AUTHORIZE",
                "order_mode": "simulated",
                "is_live_sandbox": False,
                "amount": float(formatted_amount),
                "currency": currency,
                "custom_id": custom_id,
                "paypal_request_id": request_id,
                "links": [
                    {"rel": "self", "href": f"https://api-m.sandbox.paypal.com/v2/checkout/orders/{simulated_order_id}"},
                    {"rel": "approve", "href": f"https://www.sandbox.paypal.com/checkoutnow?token={simulated_order_id}"},
                    {"rel": "authorize", "href": f"https://api-m.sandbox.paypal.com/v2/checkout/orders/{simulated_order_id}/authorize"}
                ]
            }

        token = self.get_access_token()
        endpoint = f"{self.base_url}/v2/checkout/orders"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "PayPal-Request-Id": request_id,
            "Prefer": "return=representation"
        }
        payload = {
            "intent": "AUTHORIZE",
            "purchase_units": [
                {
                    "reference_id": custom_id,
                    "custom_id": custom_id,
                    "description": "RingBreaker Protected Payment",
                    "amount": {
                        "currency_code": currency,
                        "value": formatted_amount
                    }
                }
            ]
        }

        logger.info(f"POST {endpoint} (Creating Order with intent=AUTHORIZE, amount=${formatted_amount})")
        response = self.session.post(endpoint, headers=headers, json=payload, timeout=15)
        
        if response.status_code not in (200, 201):
            logger.error(f"PayPal create_order error: {response.status_code} {response.text}")
            response.raise_for_status()

        result = response.json()
        result["order_mode"] = "live"
        result["is_live_sandbox"] = True
        result["amount"] = float(formatted_amount)
        result["currency"] = currency
        result["paypal_request_id"] = request_id
        logger.info(f"PayPal Order created: {result.get('id')} [Status: {result.get('status')}]")
        return result

    def authorize_order(
        self,
        order_id: str,
        force_simulation: bool = False
    ) -> Dict[str, Any]:
        """
        Authorizes an approved Order.
        If real sandbox requires buyer approval, reports AUTHORIZATION_FAILED or APPROVAL_REQUIRED honestly.
        Never stamps fake IDs as 'live'.
        """
        request_id = str(uuid.uuid4())

        if order_id.startswith("SIM-") or not self.is_configured() or force_simulation:
            sim_auth_id = f"SIM-AUTH-{uuid.uuid4().hex[:10].upper()}"
            logger.info(f"[SIMULATED] Authorized Order {order_id} -> Authorization ID {sim_auth_id}")
            return {
                "id": order_id,
                "status": "COMPLETED",
                "authorization_mode": "simulated",
                "is_live_sandbox": False,
                "authorization_id": sim_auth_id,
                "purchase_units": [
                    {
                        "payments": {
                            "authorizations": [
                                {
                                    "id": sim_auth_id,
                                    "status": "CREATED"
                                }
                            ]
                        }
                    }
                ]
            }

        token = self.get_access_token()
        endpoint = f"{self.base_url}/v2/checkout/orders/{order_id}/authorize"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "PayPal-Request-Id": request_id
        }

        logger.info(f"POST {endpoint} (Authorizing Order {order_id})")
        response = self.session.post(endpoint, headers=headers, json={}, timeout=15)
        
        if response.status_code not in (200, 201):
            logger.warning(
                f"PayPal authorize_order requires buyer approval in sandbox: "
                f"{response.status_code} {response.text}"
            )
            # Honest error reporting: do NOT invent fake SIM-AUTH under a live label
            return {
                "id": order_id,
                "status": "AUTHORIZATION_FAILED",
                "authorization_mode": "live",
                "is_live_sandbox": True,
                "authorization_id": None,
                "error_code": response.status_code,
                "error_details": response.text,
                "notice": "Orders with intent=AUTHORIZE require buyer login approval via PayPal approval URL in standard sandbox."
            }

        result = response.json()
        result["authorization_mode"] = "live"
        result["is_live_sandbox"] = True
        
        try:
            auth_id = result["purchase_units"][0]["payments"]["authorizations"][0]["id"]
            result["authorization_id"] = auth_id
        except (KeyError, IndexError):
            result["authorization_id"] = None

        logger.info(f"PayPal Order {order_id} Authorized -> ID: {result.get('authorization_id')}")
        return result

    def capture_authorization(
        self,
        authorization_id: str,
        amount: Optional[Any] = None,
        currency: str = "USD",
        force_simulation: bool = False
    ) -> Dict[str, Any]:
        """
        Captures an authorized payment (Low-risk tier action).
        """
        request_id = str(uuid.uuid4())

        if authorization_id.startswith("SIM-") or not self.is_configured() or force_simulation:
            sim_capture_id = f"SIM-CAP-{uuid.uuid4().hex[:10].upper()}"
            logger.info(f"[SIMULATED] Captured Authorization {authorization_id} -> Capture ID {sim_capture_id}")
            return {
                "id": sim_capture_id,
                "status": "COMPLETED",
                "capture_mode": "simulated",
                "is_live_sandbox": False,
                "authorization_id": authorization_id,
                "action": "CAPTURE"
            }

        token = self.get_access_token()
        endpoint = f"{self.base_url}/v2/payments/authorizations/{authorization_id}/capture"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "PayPal-Request-Id": request_id
        }
        payload: Dict[str, Any] = {}
        if amount is not None:
            formatted_amt = self._format_amount(amount)
            payload["amount"] = {"value": formatted_amt, "currency_code": currency}

        logger.info(f"POST {endpoint} (Capturing Authorization {authorization_id})")
        response = self.session.post(endpoint, headers=headers, json=payload, timeout=15)
        
        if response.status_code not in (200, 201):
            logger.error(f"PayPal capture error: {response.status_code} {response.text}")
            response.raise_for_status()

        result = response.json()
        result["capture_mode"] = "live"
        result["is_live_sandbox"] = True
        result["action"] = "CAPTURE"
        logger.info(f"PayPal Authorization {authorization_id} Captured -> Capture ID: {result.get('id')}")
        return result

    def void_authorization(
        self,
        authorization_id: str,
        force_simulation: bool = False
    ) -> Dict[str, Any]:
        """
        Voids an authorization (High-risk tier action: cancels payment to prevent fraud loss).
        """
        request_id = str(uuid.uuid4())

        if authorization_id.startswith("SIM-") or not self.is_configured() or force_simulation:
            logger.info(f"[SIMULATED] Voided Authorization {authorization_id} (Fraud Protection Triggered)")
            return {
                "id": authorization_id,
                "status": "VOIDED",
                "void_mode": "simulated",
                "is_live_sandbox": False,
                "action": "VOID"
            }

        token = self.get_access_token()
        endpoint = f"{self.base_url}/v2/payments/authorizations/{authorization_id}/void"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "PayPal-Request-Id": request_id
        }

        logger.info(f"POST {endpoint} (Voiding Authorization {authorization_id})")
        response = self.session.post(endpoint, headers=headers, json={}, timeout=15)
        
        if response.status_code not in (200, 204):
            logger.error(f"PayPal void error: {response.status_code} {response.text}")
            response.raise_for_status()

        logger.info(f"PayPal Authorization {authorization_id} successfully VOIDED.")
        return {
            "id": authorization_id,
            "status": "VOIDED",
            "void_mode": "live",
            "is_live_sandbox": True,
            "action": "VOID"
        }

    def verify_webhook(
        self,
        headers: Dict[str, str],
        event_body: Dict[str, Any],
        webhook_id: Optional[str] = None
    ) -> bool:
        """
        Verifies signature of an incoming PayPal webhook event.
        Returns False if unconfigured or invalid. Never silently trusts.
        """
        if not self.is_configured():
            logger.warning("[WEBHOOK] PayPal unconfigured; signature verification rejected.")
            return False

        webhook_id = webhook_id or os.getenv("PAYPAL_WEBHOOK_ID", "")
        if not webhook_id:
            logger.warning("[WEBHOOK] No PAYPAL_WEBHOOK_ID configured; cannot verify signature.")
            return False

        try:
            token = self.get_access_token()
            endpoint = f"{self.base_url}/v1/notifications/verify-webhook-signature"
            verify_headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            verify_payload = {
                "auth_algo": headers.get("paypal-auth-algo", ""),
                "cert_url": headers.get("paypal-cert-url", ""),
                "transmission_id": headers.get("paypal-transmission-id", ""),
                "transmission_sig": headers.get("paypal-transmission-sig", ""),
                "transmission_time": headers.get("paypal-transmission-time", ""),
                "webhook_id": webhook_id,
                "webhook_event": event_body
            }

            logger.info(f"POST {endpoint} (Verifying webhook signature)")
            response = self.session.post(endpoint, headers=verify_headers, json=verify_payload, timeout=12)
            if response.status_code == 200:
                status = response.json().get("verification_status", "")
                return status == "SUCCESS"
        except Exception as e:
            logger.error(f"Webhook verification request failed: {e}")

        return False
