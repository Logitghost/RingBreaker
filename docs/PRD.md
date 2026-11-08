# RingBreaker PRD

Nov 7, 2026 · @Ridima Jain

## Overview

RingBreaker catches organized payment fraud by scoring accounts on who they are connected to, not just on single payments. It turns PayPal sandbox payments into a live graph, runs a graph neural network (GNN) over it to flag suspicious accounts and rings, and uses an LLM investigator to explain each flag in plain language.

**Hackathon fit:** the PayPal AI Hackathon requires meaningful use of both PayPal and AI. PayPal is the live payment pipeline (Orders, authorize/capture/void, webhooks). AI is the GNN detector plus the LLM investigator.

**One-line pitch:** fraud checks look at one payment at a time; RingBreaker looks at the whole network.

## Problem

Organized fraud hides in the connections between accounts, which per-payment checks cannot see. A gang opens many accounts, each sending small, normal-looking payments. Every payment passes a row-by-row check, yet the money loops between the same accounts, the accounts share a device or IP, and funds funnel into one cash-out point.

The cost of missing this is real losses and a ring that keeps operating. The cost of over-flagging is blocked legitimate sales and lost trust. Merchants and platforms need a way to see the network, act on risk proportionally, and understand why something was flagged.

## Target users

The primary user is a fraud analyst or risk owner at a merchant or marketplace that takes PayPal payments. The secondary user is a developer who wants to add network-aware risk scoring to a PayPal checkout.

- **Analyst:** sees flagged rings on a graph, reads the AI explanation, and approves, holds or voids payments.
- **Merchant owner:** adjusts how strict the system is with a single threshold slider and sees the precision/recall trade-off.
- **Developer:** plugs the scoring service into an authorize-then-capture checkout flow.

## Goals, non-goals and success metrics

Success means the GNN beats a plain tabular baseline on held-out data, and the live PayPal demo shows a ring being caught end to end. We commit to reporting honest metrics, not to a pre-set accuracy number.

**Goals**

- Detect suspicious accounts and rings using graph structure, not only per-account features.
- Run on a live PayPal sandbox pipeline with authorize, score, then capture or void.
- Explain every flag in plain language and let a human review borderline cases.
- Disclose false positives and false negatives explicitly.

**Non-goals**

- Production-grade fraud prevention or real-money use.
- Training on real PayPal data (the sandbox has no fraud labels).
- Automatically blocking users on a borderline score.

**Metrics reported on held-out data**

- Precision, recall, F1 and PR-AUC for the GNN.
- False positive rate and the confusion matrix.
- The same metrics for an XGBoost baseline, to show whether the graph helps.
- Precision and recall at several thresholds, shown live by the slider.

## Features and priorities

P0 is the smallest version that satisfies every hackathon requirement; P1 and P2 are added only if P0 is solid.

| Priority | Feature | Description |
| --- | --- | --- |
| P0 | PayPal sandbox pipeline | Create orders with authorize intent, receive webhooks, capture or void based on risk score |
| P0 | Transaction graph | Accounts, merchants and devices as nodes; payments as edges; updated live |
| P0 | GNN detector | GraphSAGE or GAT node classifier producing a 0 to 1 risk score |
| P0 | XGBoost baseline | Same data without graph structure, for comparison |
| P0 | Risk tiers | Low: capture. Medium: human review. High: hold or void |
| P0 | Graph dashboard | Live graph with flagged clusters highlighted in red |
| P0 | Metrics and limitations view | Confusion matrix, precision, recall, false positive rate |
| P1 | LLM investigator | Plain-language explanation of why a cluster was flagged |
| P1 | Threshold slider | Live precision/recall trade-off |
| P1 | Reviewer feedback | Analyst marks flags as legit or fraud, stored as new labels |
| P2 | GNNExplainer | Highlights the neighbors and edges that drove a score |
| P2 | GAN attacker | Generates evasive synthetic rings to harden the detector |

## User flow and demo script

The 3-minute video shows one story: normal traffic, a fraud ring appearing, the graph turning red, and the payments being held. It must run on the real build, with no mockups.

1. **Problem (0:00 to 0:20):** one sentence on why per-payment checks miss rings.
2. **Calm baseline (0:20 to 0:50):** run normal sandbox payments through authorize and capture; the graph looks sparse and healthy.
3. **Attack (0:50 to 1:30):** trigger a simulated ring (loop, fan-in and shared device) through the PayPal sandbox; the graph updates live.
4. **Detection (1:30 to 2:00):** the cluster turns red; medium-risk payments sit in review, high-risk ones are voided.
5. **Explanation (2:00 to 2:20):** the LLM investigator explains the flag in plain language.
6. **Honesty and metrics (2:20 to 2:50):** show the confusion matrix, the threshold slider, the GNN vs baseline comparison, and state the false positive and false negative limits.
7. **Close (2:50 to 3:00):** what it is, who it is for, and what comes next.

Avoid third-party logos and copyrighted music in the video.

## System architecture

&#91;embedded content: architecture · 8 components, 1 feedback loop\]

PayPal keeps each payment in an authorized state while RingBreaker scores it, then the risk tier decides whether to capture, review or void it.

## ML specification

The model is a node classifier trained offline on a public graph dataset, then served against the live sandbox graph. Training data and sandbox data are different, and the submission says so.

**Data**

- Training: a public payment dataset converted to an account graph (PaySim is the default choice; Elliptic is the alternative). Ring patterns (cycles, fan-in mule accounts, shared-device clusters) are injected and labeled, because public data has weak ring structure.
- Live demo: sandbox payments generated by a checkout simulator. Device and IP identifiers are attached by the simulator as metadata, since the sandbox does not produce real ones.

**Graph schema**

- Nodes: accounts, merchants, devices or IPs.
- Edges: payments (amount, timestamp), account-uses-device.
- Node features: account age, transaction count, average and maximum amount, time between transactions, in/out ratio.

**Models**

- GNN: GraphSAGE or GAT in PyTorch Geometric, 2 to 3 message-passing layers, output is a fraud score from 0 to 1.
- Baseline: XGBoost on the same node features with no graph information.

**Training and evaluation**

- Fraud is rare, so use class weights or focal loss and split train, validation and test by time or by node.
- Never judge by accuracy; report precision, recall, F1, PR-AUC and false positive rate on the held-out test set.
- Pick the default threshold on the validation set, then expose it as a slider.

## Detection logic and risk tiers

The GNN outputs a 0 to 1 risk score per account, and the score maps to one of three responses. The GNN learns these patterns from labeled examples; they are not hand-written rules.

| Pattern | What it looks like | Why it is suspicious |
| --- | --- | --- |
| Circular flow | A pays B, B pays C, C pays A in small amounts | Layering money to hide its origin |
| Fan-in (mule) | Many accounts send small amounts to one account that cashes out fast | Collection point for stolen funds |
| Fan-out | One account suddenly pays many fresh accounts | Splitting funds |
| Shared identifiers | Many accounts on one device or IP | One actor posing as many |
| New-account burst | Many accounts created and active together | Fake account farm |
| Bad neighborhood | A clean account whose neighbors are mostly flagged | Guilt by association |
| Rapid in-out | Money arrives and leaves within minutes | Pass-through behavior |

| Tier | Score | Response |
| --- | --- | --- |
| Low | Below the review threshold | Capture the authorized payment |
| Medium | Between the thresholds | Leave uncaptured and send to human review |
| High | Above the hold threshold | Void the authorization and flag the cluster |

Thresholds are tunable. The exact default values are set on the validation set, not fixed in advance.

## Limitations: false positives and false negatives

RingBreaker will sometimes flag legitimate activity and sometimes miss real fraud, and the submission states this openly. The two errors trade off: a lower threshold catches more fraud but flags more innocent users, and a higher one does the opposite.

| Error | What happens | Cost | Mitigation |
| --- | --- | --- | --- |
| False positive | A normal account or payment is flagged | Annoyed users, blocked sales, lost trust | Tiered response, human review for medium risk, explanations, reviewer feedback as new labels, features such as account age and history |
| False negative | Real fraud is scored as normal | Money lost, ring keeps operating | Graph features, recall reported explicitly, tunable threshold, optional GAN attacker to find evasion gaps |

**Other honest limits**

- Fraud is rare, so accuracy is meaningless; we report precision, recall, F1, PR-AUC and false positive rate.
- Legitimate patterns can look like rings, such as a family splitting bills, a marketplace with many sellers or a business paying many contractors.
- The model learned from a public dataset plus injected rings, so "suspicious" means "resembles that data". It would need retraining on a platform's own labeled history before real use.
- New accounts have little graph context, so their score leans on their own features until they build connections.
- Device and IP identifiers in the demo are simulated by the checkout simulator.

**Statement for the README, description and video**

> RingBreaker will sometimes flag legitimate activity (false positives) and miss some fraud (false negatives). We report precision, recall and false positive rate on held-out data, use tiered responses so no payment is blocked on a borderline score alone, and route medium-risk cases to human review. The model was trained on a public dataset, so its notion of "suspicious" reflects that data and would need retraining on a platform's own labeled history before real-world use.

## Tech stack

Python carries the ML and backend; a small React front end carries the graph dashboard. Exact PayPal endpoints should be confirmed against the PayPal developer docs and the sandbox before building.

| Layer | Choice |
| --- | --- |
| Payments | PayPal sandbox: Orders (authorize intent), authorization capture and void, webhooks |
| ML | PyTorch Geometric (GraphSAGE or GAT), XGBoost baseline, scikit-learn for metrics |
| Graph | NetworkX for building and analysis |
| Backend | FastAPI: scoring service, webhook receiver, review queue |
| Storage | SQLite or Postgres for events, scores and reviewer labels |
| LLM | Any tool-calling model, used for the plain-language investigator only |
| Frontend | One single-page HTML dashboard with Cytoscape.js, served by FastAPI (no separate React build) |
| Hosting | A hosted demo URL, plus run instructions in the repo |

## Submission checklist

The Devpost deadline is Nov 12, 2026 at 12:00 PM PST, which is 1:30 AM IST on Nov 13. Aim to submit by the evening of Nov 12 IST, not at the last minute.

- [ ] Working build that uses a PayPal sandbox API in a central way
- [ ] AI used meaningfully (GNN detector plus LLM investigator)
- [ ] Functional demo: hosted URL or complete setup and run instructions in the repo (a mockup does not count)
- [ ] Public GitHub, GitLab or Bitbucket repo with all source, assets and instructions
- [ ] Open-source license file visible in the repo's About section
- [ ] Text description of features and functionality
- [ ] List of AI tools used and how they were used
- [ ] Demo video under 3 minutes, public on YouTube, showing the project working on its device
- [ ] No third-party trademarks or copyrighted music in the video
- [ ] Limitations statement (false positives and false negatives) in the README and description
- [ ] Documentation clear enough for judges to evaluate

## Timeline, risks and open questions

&#91;embedded content: timeline · 5 build days, 2 milestones\]

The plan starts today (Nov 7), keeps Nov 12 as a buffer day, and treats P1 and P2 features as optional until the P0 flow works end to end.

**Risks**

- The GNN may not beat the XGBoost baseline. Report the result honestly; fall back to graph-derived features (degree, neighbor risk) fed into the baseline.
- Public data has weak ring structure. Injected rings fix this for evaluation, and the submission discloses it.
- The PayPal authorize, capture and void flow may behave differently in the sandbox than expected. Test it first on Nov 9 and confirm endpoints in the docs.
- Scope creep. Nothing from P1 or P2 starts until every P0 item works.

**Open questions**

- Who is on the team, and who owns each workstream?
- Which LLM and hosting service will be used for the demo?
