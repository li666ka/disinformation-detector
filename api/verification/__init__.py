# api/verification/__init__.py
from .models import (
    Claim,
    Evidence,
    EvidenceBundle,
    EvidenceBreakdown,
    Verdict,
    VerificationResult,
)
from .claim_extractor import extract_claims
from .evidence_retriever import gather_evidence
from .stance_detector import detect_stance_batch
from .verdict_aggregator import aggregate_verdict, rule_based_verdict
from .pipeline import verify_text

__all__ = [
    "Claim",
    "Evidence",
    "EvidenceBundle",
    "EvidenceBreakdown",
    "Verdict",
    "VerificationResult",
    "extract_claims",
    "gather_evidence",
    "detect_stance_batch",
    "aggregate_verdict",
    "rule_based_verdict",
    "verify_text",
]