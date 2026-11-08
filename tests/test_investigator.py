"""
Unit tests for RingBreaker LLM Investigator (Phase 5 Hardened).
Tests structured JSON output, schema validation, grounding checks, and cache behavior.
"""

import pytest
from pydantic import ValidationError
from app.graph_store import GraphStore
from app.investigator import (
    InvestigatorExplanation,
    ClusterFacts,
    LLMInvestigator,
    extract_cluster_facts
)


def test_schema_validates_valid_json():
    valid_data = {
        "summary": "Coordinated circular transfers detected among 3 fresh accounts.",
        "evidence": [
            "Cycle A -> B -> C -> A detected in a 15-minute window.",
            "Accounts share device fingerprint 'dev_farm_01'."
        ],
        "recommended_action": "VOID_AUTHORIZATION",
        "confidence_note": "Checked against the facts: grounded strictly in graph topology."
    }
    explanation = InvestigatorExplanation.model_validate(valid_data)
    assert explanation.summary == valid_data["summary"]
    assert len(explanation.evidence) == 2
    assert explanation.recommended_action == "VOID_AUTHORIZATION"


def test_schema_rejects_invalid_json():
    # Missing required 'summary'
    with pytest.raises(ValidationError):
        InvestigatorExplanation.model_validate({
            "evidence": ["Some fact"],
            "recommended_action": "VOID_AUTHORIZATION",
            "confidence_note": "Context note"
        })

    # Missing required 'evidence'
    with pytest.raises(ValidationError):
        InvestigatorExplanation.model_validate({
            "summary": "Summary here",
            "recommended_action": "VOID_AUTHORIZATION",
            "confidence_note": "Context note"
        })

    # Empty evidence list (violates min_length=1)
    with pytest.raises(ValidationError):
        InvestigatorExplanation.model_validate({
            "summary": "Summary here",
            "evidence": [],
            "recommended_action": "VOID_AUTHORIZATION",
            "confidence_note": "Context note"
        })


def test_investigator_produces_grounded_explanation(tmp_path):
    db_file = str(tmp_path / "test_investigator.db")
    store = GraphStore(db_path=db_file, graph_pkl_path="nonexistent.pkl")

    # Add transactions forming a cycle: A -> B -> C -> A
    store.add_transaction(
        tx_id="tx_cycle_01",
        sender_id="acc_A",
        receiver_id="acc_B",
        amount=500.0,
        timestamp=1000.0,
        device_id="dev_shared",
        risk_score=0.88,
        risk_tier="HIGH",
        status="VOIDED",
        action_taken="VOID"
    )
    store.add_transaction(
        tx_id="tx_cycle_02",
        sender_id="acc_B",
        receiver_id="acc_C",
        amount=480.0,
        timestamp=1200.0,
        device_id="dev_shared",
        risk_score=0.88,
        risk_tier="HIGH",
        status="VOIDED",
        action_taken="VOID"
    )
    store.add_transaction(
        tx_id="tx_cycle_03",
        sender_id="acc_C",
        receiver_id="acc_A",
        amount=460.0,
        timestamp=1400.0,
        device_id="dev_shared",
        risk_score=0.88,
        risk_tier="HIGH",
        status="VOIDED",
        action_taken="VOID"
    )

    facts = extract_cluster_facts("tx_cycle_01", store)
    assert facts.sender_id == "acc_A"
    assert facts.amount == 500.0
    assert len(facts.detected_cycles) > 0
    assert "dev_shared" in facts.shared_devices

    investigator = LLMInvestigator()
    explanation = investigator.investigate(facts)

    assert isinstance(explanation, InvestigatorExplanation)
    assert len(explanation.evidence) >= 2
    assert explanation.recommended_action == "VOID_AUTHORIZATION"
    assert explanation.source == "template"
    assert explanation.grounding_status == "checked_against_facts"


def test_unknown_receiver_score_is_none(tmp_path):
    # Directive 5.2: receiver_risk_score must be null when unknown
    db_file = str(tmp_path / "test_receiver_none.db")
    store = GraphStore(db_path=db_file, graph_pkl_path="nonexistent.pkl")

    store.add_transaction(
        tx_id="tx_new_p2p",
        sender_id="acc_known_sender",
        receiver_id="acc_unknown_receiver",
        amount=75.0,
        timestamp=1000.0,
        device_id="dev_sender",
        risk_score=0.35,
        risk_tier="LOW",
        status="CAPTURED",
        action_taken="CAPTURE"
    )

    facts = extract_cluster_facts("tx_new_p2p", store)
    assert facts.sender_risk_score == 0.35
    assert facts.receiver_risk_score is None


def test_grounding_verification_rejects_hallucinations():
    # Directive 5.4: Discard and use template if cited ID doesn't exist in facts
    facts = ClusterFacts(
        payment_id="tx_test_123",
        amount=100.0,
        sender_id="acc_valid_1",
        receiver_id="acc_valid_2",
        risk_tier="HIGH",
        sender_risk_score=0.85,
        receiver_risk_score=0.10,
        shared_devices={"dev_valid": ["acc_valid_1", "acc_valid_2"]},
        detected_cycles=[],
        cluster_accounts=["acc_valid_1", "acc_valid_2"],
        total_cluster_nodes=2,
        truncated=False,
        time_span_minutes=5.0,
        total_cluster_volume=100.0,
        transaction_count=1
    )

    investigator = LLMInvestigator()

    # Valid dictionary (all entities exist in facts)
    valid_dict = {
        "summary": "Payment involving acc_valid_1 and acc_valid_2.",
        "evidence": ["Shared hardware signature dev_valid detected."],
        "recommended_action": "VOID_AUTHORIZATION",
        "confidence_note": "Grounded."
    }
    assert investigator._verify_grounding(valid_dict, facts) is True

    # Hallucinated account ID (acc_ghost_99 does not exist in facts)
    hallucinated_dict = {
        "summary": "Coordinated with account acc_ghost_99.",
        "evidence": ["acc_ghost_99 sent smurfed transactions."],
        "recommended_action": "VOID_AUTHORIZATION",
        "confidence_note": "Grounded."
    }
    assert investigator._verify_grounding(hallucinated_dict, facts) is False


def test_investigator_caching_by_facts_hash(tmp_path):
    # Directive 5.6: Cache explanations by payment ID and facts hash
    db_file = str(tmp_path / "test_cache.db")
    store = GraphStore(db_path=db_file, graph_pkl_path="nonexistent.pkl")

    store.add_transaction(
        tx_id="tx_cache_test",
        sender_id="acc_cache_1",
        receiver_id="acc_cache_2",
        amount=99.0,
        timestamp=1000.0,
        device_id="dev_c",
        risk_score=0.50,
        risk_tier="MEDIUM",
        status="IN_REVIEW",
        action_taken="HOLD_FOR_REVIEW"
    )

    facts = extract_cluster_facts("tx_cache_test", store)
    investigator = LLMInvestigator()

    exp1 = investigator.investigate(facts)
    assert "tx_cache_test" in investigator._cache

    # Subsequent call returns identical cached instance
    exp2 = investigator.investigate(facts)
    assert exp1 is exp2
