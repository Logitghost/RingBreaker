"""
RingBreaker - FastAPI Application
Provides hardened REST API for payment processing, graph inspection,
review queue actions, simulator controls, and PayPal webhooks.
"""

import os
import sys
from pathlib import Path

# Load .env from project root using absolute path
ROOT_DIR = Path(__file__).resolve().parent.parent
from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Ensure repository root is in python path
sys.path.insert(0, str(ROOT_DIR))

import re
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Literal
from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends, status

logger = logging.getLogger("ringbreaker.api")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field, field_validator

from app.graph_store import GraphStore
from app.paypal_client import PayPalClient
from app.scoring import RiskScorer
from app.simulator import TrafficSimulator
from app.investigator import LLMInvestigator, extract_cluster_facts

app = FastAPI(
    title="RingBreaker API",
    description="Network-aware payment fraud detection on PayPal Sandbox",
    version="1.1.0"
)

# CORS configuration
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Application state
graph_store = GraphStore()
paypal_client = PayPalClient()
scorer = RiskScorer(graph_store=graph_store)
simulator = TrafficSimulator(graph_store, scorer, paypal_client)
investigator = LLMInvestigator()

ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{3,64}$")
DEFAULT_ADMIN_TOKEN = "rb_secret_admin_token_change_me"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", DEFAULT_ADMIN_TOKEN)

if ADMIN_TOKEN == DEFAULT_ADMIN_TOKEN:
    logger.warning("SECURITY WARNING: ADMIN_TOKEN is set to default placeholder. Remote admin access is disabled until changed.")


def verify_admin_access(request: Request, x_admin_token: Optional[str] = Header(None)) -> bool:
    """
    Guards mutating admin endpoints when accessed remotely.
    Localhost requests are permitted; non-localhost requires non-default X-Admin-Token.
    """
    client_host = request.client.host if request.client else ""
    is_local = client_host in ("127.0.0.1", "localhost", "::1", "testclient")
    if is_local:
        return True
    if ADMIN_TOKEN == DEFAULT_ADMIN_TOKEN:
        logger.error("Remote admin attempt rejected: ADMIN_TOKEN must be changed from default placeholder.")
        raise HTTPException(status_code=403, detail="Remote admin disabled until ADMIN_TOKEN is changed from default.")
    if x_admin_token and x_admin_token == ADMIN_TOKEN:
        return True
    raise HTTPException(status_code=403, detail="Admin token required for remote management.")


# Request Schemas
class PaymentRequest(BaseModel):
    sender_id: str
    receiver_id: str
    amount: float = Field(..., gt=0.0, le=50000.0)
    device_id: Optional[str] = None

    @field_validator("sender_id", "receiver_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        if not ID_REGEX.match(v):
            raise ValueError(f"Invalid account ID format: '{v}'")
        return v

    @field_validator("receiver_id")
    @classmethod
    def validate_distinct(cls, v: str, info) -> str:
        if "sender_id" in info.data and v == info.data["sender_id"]:
            raise ValueError("Sender and receiver accounts must be distinct.")
        return v


class ReviewDecisionRequest(BaseModel):
    decision: Literal["legit", "fraud"]
    analyst_id: Optional[str] = "analyst_1"
    notes: Optional[str] = ""


class ThresholdUpdateRequest(BaseModel):
    threshold_low: float = Field(..., ge=0.05, le=0.90)
    threshold_high: float = Field(..., ge=0.10, le=0.95)

    @field_validator("threshold_high")
    @classmethod
    def validate_bounds(cls, v: float, info) -> float:
        if "threshold_low" in info.data and v <= info.data["threshold_low"]:
            raise ValueError("threshold_high must be strictly greater than threshold_low.")
        return v


class SimulateRingRequest(BaseModel):
    ring_type: Literal["cycle", "mule", "shared_device"] = "cycle"


# Endpoints
@app.get("/api/health")
def health_check():
    """
    Honest system diagnostics report. Never exposes credentials.
    """
    best_epoch = "not_loaded"
    meta_path = os.path.join(scorer.artifacts_dir, "model_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            best_epoch = meta.get("best_epoch", "unknown")
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "RingBreaker Risk Engine",
        "paypal": {
            "mode": paypal_client.mode,
            "configured": paypal_client.is_configured(),
            "target_url": paypal_client.base_url
        },
        "models": {
            "gnn_loaded": scorer.gnn_model is not None,
            "xgboost_loaded": scorer.xgb_model is not None,
            "gnn_best_epoch": best_epoch,
            "feature_count": 15
        },
        "operating_thresholds": {
            "threshold_low_capture": scorer.threshold_low,
            "threshold_high_void": scorer.threshold_high
        },
        "llm": {
            "provider": investigator.provider,
            "configured": investigator.is_api_configured()
        },
        "graph_state": {
            "nodes": graph_store.graph.number_of_nodes(),
            "edges": graph_store.graph.number_of_edges()
        }
    }


@app.post("/api/payments")
def create_payment(req: PaymentRequest):
    device_id = req.device_id or f"dev_{req.sender_id}"
    res = scorer.process_payment(
        sender_id=req.sender_id,
        receiver_id=req.receiver_id,
        amount=req.amount,
        device_id=device_id,
        graph_store=graph_store,
        paypal_client=paypal_client
    )
    return res


@app.get("/api/payments")
def list_payments(limit: int = 50):
    return graph_store.get_recent_transactions(limit=min(limit, 200))


@app.get("/api/graph")
def get_graph(
    request: Request,
    max_nodes: int = 100,
    show_ground_truth: bool = False,
    x_admin_token: Optional[str] = Header(None)
):
    if show_ground_truth:
        verify_admin_access(request, x_admin_token)
    return graph_store.get_cytoscape_elements(max_nodes=max_nodes, show_ground_truth=show_ground_truth)


@app.get("/api/review-queue")
def get_review_queue():
    items = graph_store.get_review_queue()
    enriched = []
    for item in items:
        item_dict = dict(item)
        try:
            facts = extract_cluster_facts(item["id"], graph_store)
            explanation = investigator.investigate(facts)
            item_dict["explanation"] = explanation.model_dump()
        except Exception:
            item_dict["explanation"] = None
        enriched.append(item_dict)
    return enriched


@app.post("/api/review-queue/{payment_id}/decision")
def submit_decision(
    payment_id: str,
    req: ReviewDecisionRequest,
    _auth: bool = Depends(verify_admin_access)
):
    """
    Applies human review decision (legit -> Capture, fraud -> Void).
    Calls PayPal capture/void before updating database.
    Idempotent and strictly limited to IN_REVIEW payments.
    """
    with graph_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE id = ?", (payment_id,))
        tx = cursor.fetchone()

    if not tx:
        raise HTTPException(status_code=404, detail=f"Payment '{payment_id}' not found.")

    # Idempotency check
    if tx["status"] in ("CAPTURED", "VOIDED"):
        current_decision = "legit" if tx["status"] == "CAPTURED" else "fraud"
        if current_decision == req.decision:
            return {"message": "Payment already resolved with this decision.", "payment": dict(tx)}
        raise HTTPException(status_code=409, detail=f"Payment '{payment_id}' already resolved as {tx['status']}.")

    if tx["status"] != "IN_REVIEW":
        raise HTTPException(status_code=409, detail=f"Cannot decide payment with status '{tx['status']}'. Must be IN_REVIEW.")

    auth_id = tx["paypal_auth_id"] or tx["id"]
    new_status = "CAPTURED" if req.decision == "legit" else "VOIDED"
    new_action = "MANUAL_CAPTURE" if req.decision == "legit" else "MANUAL_VOID"

    # Call PayPal sandbox API
    try:
        if req.decision == "legit":
            cap_resp = paypal_client.capture_authorization(auth_id, amount=tx["amount"])
            paypal_capture_id = cap_resp.get("id")
        else:
            paypal_client.void_authorization(auth_id)
            paypal_capture_id = None
    except Exception as e:
        # DO NOT update database status if PayPal call fails
        graph_store.add_audit_log("paypal_call_failed", {
            "action": new_action,
            "payment_id": payment_id,
            "auth_id": auth_id,
            "error": str(e)
        })
        raise HTTPException(
            status_code=502,
            detail=f"PayPal {new_action} failed: {str(e)}. Payment remains IN_REVIEW."
        )

    # Persist analyst decision and store labels on BOTH accounts
    now = time.time()
    with graph_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE transactions
            SET status = ?, action_taken = ?, paypal_capture_id = ?
            WHERE id = ?
        """, (new_status, new_action, paypal_capture_id, payment_id))

        cursor.execute("""
            INSERT INTO review_decisions (payment_id, analyst_id, decision, notes, decided_at)
            VALUES (?, ?, ?, ?, ?)
        """, (payment_id, req.analyst_id or "analyst_1", req.decision, req.notes or "", now))

        cursor.execute("UPDATE accounts SET analyst_label = ? WHERE account_id = ?", (req.decision, tx["sender_id"]))
        cursor.execute("UPDATE accounts SET analyst_label = ? WHERE account_id = ?", (req.decision, tx["receiver_id"]))
        conn.commit()

        cursor.execute("SELECT * FROM transactions WHERE id = ?", (payment_id,))
        updated_tx = dict(cursor.fetchone())

    graph_store.add_audit_log("analyst_decision", {
        "payment_id": payment_id,
        "analyst_id": req.analyst_id,
        "decision": req.decision,
        "new_status": new_status,
        "notes": req.notes
    })

    return {
        "message": f"Payment {payment_id} successfully marked as {req.decision.upper()}.",
        "payment": updated_tx
    }


@app.post("/api/simulate/normal")
def simulate_normal(count: int = 3, _auth: bool = Depends(verify_admin_access)):
    count = max(1, min(count, 10))
    results = simulator.send_normal_payments(count=count)
    return {
        "message": f"Generated {len(results)} normal payments.",
        "transactions": results
    }


@app.post("/api/simulate/ring")
def simulate_ring(req: SimulateRingRequest, _auth: bool = Depends(verify_admin_access)):
    results = simulator.simulate_fraud_ring(ring_type=req.ring_type)
    return {
        "message": f"Injected fraud ring attack of type '{req.ring_type}' with neutral IDs.",
        "ring_type": req.ring_type,
        "transactions": results
    }


@app.get("/api/thresholds")
def get_thresholds():
    return {
        "threshold_low_capture": scorer.threshold_low,
        "threshold_high_void": scorer.threshold_high
    }


@app.post("/api/thresholds")
def update_thresholds(req: ThresholdUpdateRequest, _auth: bool = Depends(verify_admin_access)):
    scorer.update_thresholds(req.threshold_low, req.threshold_high)
    return {
        "message": "Thresholds updated and persisted successfully.",
        "threshold_low": scorer.threshold_low,
        "threshold_high": scorer.threshold_high
    }


@app.get("/api/audit")
def get_audit(limit: int = 50):
    return graph_store.get_audit_logs(limit=min(limit, 200))


@app.post("/api/admin/reset")
def reset_demo(_auth: bool = Depends(verify_admin_access)):
    graph_store.reset_database()
    simulator._init_legit_pool()
    return {"message": "Demo database and graph state reset to clean baseline."}


@app.post("/api/payments/{payment_id}/sync")
def sync_payment_status(payment_id: str):
    """
    Polling fallback to sync status from PayPal when webhooks are unavailable.
    """
    with graph_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE id = ?", (payment_id,))
        tx = cursor.fetchone()
    if not tx:
        raise HTTPException(status_code=404, detail="Payment not found.")
    return {"message": "Payment state verified against database.", "payment": dict(tx)}


@app.post("/api/webhooks/paypal")
async def paypal_webhook(request: Request):
    headers = dict(request.headers)
    body = await request.json()
    event_id = body.get("id", f"evt_{uuid.uuid4().hex[:8]}")
    event_type = body.get("event_type", "UNKNOWN")
    resource = body.get("resource", {})
    resource_id = resource.get("id")

    verified = paypal_client.verify_webhook(headers, body)
    graph_store.record_webhook_event(
        event_id=event_id,
        event_type=event_type,
        resource_id=resource_id,
        verified=verified,
        raw_body=json.dumps(body)
    )

    if not verified:
        logger.warning(f"Rejecting unverified webhook event {event_id} ({event_type}).")
        raise HTTPException(status_code=400, detail="Webhook signature verification failed.")

    # Idempotent state updates on verified events
    if event_type == "PAYMENT.CAPTURE.COMPLETED" and resource_id:
        with graph_store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE transactions SET status = 'CAPTURED' WHERE paypal_capture_id = ? OR paypal_auth_id = ?", (resource_id, resource_id))
            conn.commit()
    elif event_type == "PAYMENT.AUTHORIZATION.VOIDED" and resource_id:
        with graph_store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE transactions SET status = 'VOIDED' WHERE paypal_auth_id = ?", (resource_id,))
            conn.commit()

    return {"status": "processed", "event_id": event_id, "verified": True}


@app.get("/api/metrics")
def get_metrics():
    metrics_path = os.path.join(scorer.artifacts_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"error": "metrics.json not found"}


@app.get("/api/investigate/{payment_id}")
def investigate_payment(payment_id: str):
    try:
        facts = extract_cluster_facts(payment_id, graph_store)
        explanation = investigator.investigate(facts)
        return {
            "payment_id": payment_id,
            "explanation": explanation.model_dump(),
            "facts": facts.model_dump()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")


# Static file serving
static_dir = ROOT_DIR / "app" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    dashboard_file = static_dir / "dashboard.html"
    if dashboard_file.exists():
        return FileResponse(str(dashboard_file))
    return HTMLResponse("<h2>RingBreaker API Running</h2><p>Visit <a href='/dashboard'>/dashboard</a></p>")


@app.get("/dashboard")
def serve_dashboard():
    dashboard_file = static_dir / "dashboard.html"
    if dashboard_file.exists():
        return FileResponse(str(dashboard_file))
    return HTMLResponse("<h2>RingBreaker Analyst Dashboard</h2>")


@app.get("/privacy")
def serve_privacy():
    privacy_file = static_dir / "privacy.html"
    if privacy_file.exists():
        return FileResponse(str(privacy_file))
    return HTMLResponse("<h2>RingBreaker Privacy Notice</h2>")


@app.get("/terms")
def serve_terms():
    terms_file = static_dir / "terms.html"
    if terms_file.exists():
        return FileResponse(str(terms_file))
    return HTMLResponse("<h2>RingBreaker Terms of Evaluation</h2>")


@app.get("/favicon.ico")
def serve_favicon():
    favicon_file = static_dir / "favicon.svg"
    if favicon_file.exists():
        return FileResponse(str(favicon_file), media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        accept_header = request.headers.get("accept", "")
        if "text/html" in accept_header:
            not_found_file = static_dir / "404.html"
            if not_found_file.exists():
                return FileResponse(str(not_found_file), status_code=404)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

