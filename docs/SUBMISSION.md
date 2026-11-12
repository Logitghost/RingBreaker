# RingBreaker — Devpost Submission Pack

> **Deadline:** Nov 12, 2026, 12:00 PM PST (Nov 13, 2026, 1:30 AM IST)

---

## Devpost Submission Copy

### Project Title
RingBreaker — Graph Neural Network Fraud Ring Detector on PayPal Sandbox

### Tagline
*Network-aware payment fraud detection that sees what row-by-row scoring misses.*

### Short Description (≤ 150 words)

RingBreaker intercepts organized payment fraud rings that individual transaction checks always miss. It captures PayPal orders in AUTHORIZE state first, then runs a 3-layer GraphSAGE Graph Neural Network over the live transaction graph to detect cycle loops, mule funnels, and device farms through multi-hop neighborhood topology.

On synthetic data: **98.4% recall vs 71.0% for a strong graph-aggregate tabular baseline at the same 2% false positive rate** (183 test positives). High-risk accounts are voided before settlement. Borderline accounts route to an analyst review queue with a Gemini-powered plain-language explanation grounded strictly in graph facts.

Built in Python with FastAPI, PyTorch Geometric, XGBoost, NetworkX, and PayPal Orders v2.

### Demo Video Script (< 3 minutes)

**[0:00 – 0:25] The Problem**
> "Every row in a payment database looks normal in isolation. But fraud rings coordinate: burner accounts loop stolen money, mules aggregate funds, device farms share hardware fingerprints. Per-transaction checks score every hop as low risk — and miss the ring entirely."
> *Show: a simple diagram of a 3-node cycle, all individually 'normal'.*

**[0:25 – 0:50] Calm Baseline**
> "RingBreaker authorizes first, scores second. Here's normal checkout traffic."
> *Click "Send normal payments" on the dashboard.*
> *Point to: green Cytoscape nodes, 'AUTO CAPTURED' status, LOW risk badges.*
> "99.4% of legitimate accounts auto-capture without human review."

**[0:50 – 1:30] Fraud Ring Attack**
> "Now I'll inject a cycle ring — Account A pays B, B pays C, C pays A, looping stolen funds."
> *Click "Simulate cycle ring".*
> *Watch: nodes turn red, HIGH risk payments appear in feed.*
> "RingBreaker detects the closing edge of the loop. The authorizations are **voided** via PayPal Orders Void before a single cent settles."
> *Point to: VOIDED status badges in the payments feed.*

**[1:30 – 2:10] Human Review Queue**
> "Borderline payments — score 0.40 to 0.60 — are never auto-cancelled. They wait in the review queue with a plain-language AI explanation."
> *Click a payment in the queue; show LLM explanation panel.*
> "'Closed circular transaction loop detected. Shared device ID across 3 accounts.' That's grounded in graph facts, not hallucinated."
> *Click Approve or Decline to log the analyst decision.*

**[2:10 – 2:45] Honest Metrics**
> "Let me show you the real benchmark numbers. GNN 98.4% recall, strong tabular baseline 71.0% — same 2% FPR, 183 test fraud accounts."
> *Point to metrics table on dashboard.*
> "And this threshold slider adjusts the operating point live — risk team can balance fraud capture against review burden in real time."
> *Drag the slider.*

**[2:45 – 3:00] Close**
> "Open source, MIT licensed. PayPal Orders v2 sandbox integration. Honest limitations: trained on synthetic data, would need real merchant history for production. No real money moves."

---

## Tools Used

| Tool | Version | Role |
| :--- | :--- | :--- |
| Python | 3.12.10 | Primary language |
| FastAPI | ≥ 0.110 | REST API server |
| PyTorch Geometric | ≥ 2.5 | GraphSAGE GNN (3-layer inductive node classifier) |
| PyTorch | ≥ 2.2 | Model training and inference |
| XGBoost | ≥ 2.0 | Tabular baseline classifier |
| NetworkX | ≥ 3.2 | Live transaction graph |
| scikit-learn | ≥ 1.4 | Platt calibration, evaluation metrics |
| Cytoscape.js | 3.x (CDN) | Interactive fraud graph visualization |
| Tailwind CSS | 3.x (CDN) | Dashboard and landing page styling |
| PayPal Orders v2 API | — | Authorize / Capture / Void / Webhook |
| Gemini 2.0 Flash | — | LLM investigator (optional) |
| Claude (Anthropic) | — | Planning, architecture review, and code improvement during development |
| Antigravity (Google DeepMind) | — | Pair programming assistant during development |
| SQLite | — | Audit trail and reviewer decisions |

---

## Limitations Statement

> RingBreaker will sometimes flag legitimate activity (false positives) and miss some fraud (false negatives). To mitigate false-positive harm, borderline payments are never automatically cancelled — they route to human review.
>
> The models were trained on a **synthetic** transaction graph with injected ground-truth rings. The model's representation of "suspicious" reflects those patterns and would require retraining on real merchant transaction history before production use.
>
> Device and IP identifiers in the demo are simulated metadata attached by the checkout simulator. The PayPal sandbox does not emit hardware fingerprints. No real money is charged or settled.

---

## Hackathon Submission Checklist

### Technical Completeness
- [x] PayPal Orders v2 — AUTHORIZE → CAPTURE / VOID flow implemented
- [x] PayPal webhook HMAC-SHA256 verification implemented
- [x] GNN trained and evaluated with ring-level disjoint splits
- [x] XGBoost baseline trained independently (honest benchmark)
- [x] 5-seed multi-seed evaluation with 95% confidence intervals
- [x] Platt scaling calibration applied to raw GNN logits
- [x] Three-tier risk routing (LOW / MEDIUM / HIGH) live
- [x] LLM investigator with Pydantic schema validation and factual grounding
- [x] Dynamic threshold slider wired to `threshold_sweep` from `metrics.json`
- [x] Analyst Approve/Decline flow persisting decisions to SQLite
- [x] Incremental feedback retraining (`ml/retrain_from_feedback.py`)
- [x] Zero label leakage verified — `is_fraud`, `ring_type`, `ring_id` never read in scoring path
- [x] 31 unit tests pass (`pytest tests/ -q`)
- [x] p95 latency ≤ 200ms verified (`scripts/bench_scoring.py`)

### Documentation
- [x] README with real benchmark numbers from `metrics.json`
- [x] "What is real and what is simulated" table in README
- [x] Limitations statement in README and SUBMISSION.md
- [x] AI Tools Disclosure in README
- [x] `.env.example` with all required env vars
- [x] `docs/PAYPAL_SETUP.md` — sandbox credential walkthrough
- [x] `docs/ARCHITECTURE.md` — system design
- [x] `docs/evidence/` — audit logs, benchmark output, feature audit

### Repo Hygiene
- [x] `.gitignore` covers `.venv/`, `*.db`, `.env`, `*.exe`, large binary data
- [x] ML weights (`ml/artifacts/*.pt`) are tracked in git (not ignored)
- [x] `LICENSE` — MIT
- [x] No secrets or credentials in tracked files

### Demo
- [ ] Screen recording < 3 minutes (see script above)
- [ ] Uploaded to Devpost project page
- [ ] GitHub repository URL added to Devpost

### Phase 4 (PayPal Live Sandbox) — Optional
- [ ] Live sandbox credentials entered in `.env`
- [ ] `scripts/verify_paypal.py` passed with live credentials
- [ ] Real PayPal order IDs in `docs/evidence/paypal_run.md`

---

## Project URLs

| Resource | URL |
| :--- | :--- |
| GitHub Repository | https://github.com/your-username/ringbreaker |
| Devpost Project | https://devpost.com/software/ringbreaker |
| Demo Video | (upload link TBD) |
