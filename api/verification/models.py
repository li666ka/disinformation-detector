# api/verification/models.py
"""
Data models для multi-hop claim verification pipeline.

Pipeline:
    Post → Claim → Evidence[] → Stance-classified Evidence[] → Verdict

Все як dataclasses — легко серіалізувати у JSON для API.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Literal


# ──────────────────────────────────────────────────────────────────────────
# Claim
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """
    Перевірюване твердження, витягнуте з тексту.

    Одне повідомлення може містити кілька claims. Наприклад:
    Input: "Pfizer vaccine causes autism AND 5G spreads COVID!"
    Claims: ["Pfizer vaccine causes autism", "5G spreads COVID"]
    """
    text: str                                # Сам claim як окреме речення
    claim_type: str = "statement"            # "statement" | "causal" | "statistical" | "quote"
    entities: list[str] = field(default_factory=list)  # Key entities: ["Pfizer", "vaccine", "autism"]
    verifiable: bool = True                   # Чи можна це перевірити fact-wise
    original_text: Optional[str] = None       # Оригінальний текст, з якого витягнуто
    stance: Literal["supports", "refutes", "neutral"] = "neutral"  # позиція автора щодо claim

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────
# Evidence
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Evidence:
    """
    Одна одиниця evidence — пост/стаття, яка може підтверджувати/спростовувати claim.
    """
    source_type: Literal["rss", "bluesky", "mastodon"]
    source_name: str                         # "BBC", "Reuters", "@nytimes.com"
    title: Optional[str] = None              # Заголовок статті (для RSS)
    text: str = ""                           # Текст/summary
    url: Optional[str] = None
    published_at: Optional[str] = None       # ISO date
    author: Optional[str] = None
    author_verified: Optional[bool] = None   # Для soc: чи верифікований автор
    author_followers_count: Optional[int] = None

    # Заповнюється після stance detection
    stance: Optional[Literal["supports", "refutes", "unrelated", "unknown"]] = None
    stance_confidence: Optional[float] = None  # 0.0 - 1.0
    stance_reasoning: Optional[str] = None     # Короткий опис чому

    # Computed weight для aggregation
    authority_weight: Optional[float] = None   # 0.0 - 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────
# Evidence Bundle
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceBundle:
    """
    Набір всіх зібраних evidence для одного claim.
    """
    claim: Claim
    rss_evidence: list[Evidence] = field(default_factory=list)
    social_evidence: list[Evidence] = field(default_factory=list)
    total_found: int = 0
    retrieval_time_seconds: float = 0.0
    query_used: Optional[str] = None

    def all_evidence(self) -> list[Evidence]:
        return self.rss_evidence + self.social_evidence

    def to_dict(self) -> dict:
        return {
            "claim": self.claim.to_dict(),
            "rss_evidence": [e.to_dict() for e in self.rss_evidence],
            "social_evidence": [e.to_dict() for e in self.social_evidence],
            "total_found": self.total_found,
            "retrieval_time_seconds": self.retrieval_time_seconds,
            "query_used": self.query_used,
        }


# ──────────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceBreakdown:
    """Агрегований рахунок evidence по категоріях."""
    supports: int = 0
    refutes: int = 0
    unrelated: int = 0
    unknown: int = 0

    # Зважений рахунок (враховує authority)
    weighted_supports: float = 0.0
    weighted_refutes: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    """
    Фінальний вердикт системи.
    """
    label: Literal["FAKE", "REAL", "UNCERTAIN"]
    confidence: float                         # 0.0 - 1.0
    reasoning: str                            # Human-readable explanation

    # Структурований breakdown
    breakdown: EvidenceBreakdown = field(default_factory=EvidenceBreakdown)
    evidence_count: int = 0
    highest_authority_evidence: Optional[Evidence] = None  # найвагоміший item

    # Метадата для debug/transparency
    model_used: Optional[str] = None          # який LLM використаний
    verdict_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "breakdown": self.breakdown.to_dict(),
            "evidence_count": self.evidence_count,
            "highest_authority_evidence": (
                self.highest_authority_evidence.to_dict()
                if self.highest_authority_evidence else None
            ),
            "model_used": self.model_used,
            "verdict_time_seconds": self.verdict_time_seconds,
        }


# ──────────────────────────────────────────────────────────────────────────
# Full pipeline result
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Повний результат roботи multi-hop pipeline."""
    original_text: str                        # Вхідний пост/твіт
    claims: list[Claim]                       # Витягнуті claims
    evidence_bundles: list[EvidenceBundle]    # По одному bundle на claim
    verdicts: list[Verdict]                   # По одному verdict на claim
    overall_verdict: Optional[Verdict] = None # Aggregate якщо кілька claims
    total_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "claims": [c.to_dict() for c in self.claims],
            "evidence_bundles": [eb.to_dict() for eb in self.evidence_bundles],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "overall_verdict": self.overall_verdict.to_dict() if self.overall_verdict else None,
            "total_time_seconds": self.total_time_seconds,
        }