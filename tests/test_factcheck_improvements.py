"""Regression tests для fact_check.py після зниження SIM_THRESHOLD
з 0.4 до 0.25 + hybrid similarity (Jaccard ∪ char SequenceMatcher) +
compact query.

Google Fact Check API не б'ємо — мокаємо `requests.get`. Тест перевіряє
саме нашу логіку (similarity, threshold gate, low_confidence_match,
query trimming). Контракт `fact_check_claim` зберігається — лише додано
optional поле `low_confidence_match`.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("GOOGLE_FACT_CHECK_API_KEY", "test-key-fake")

from api import fact_check
from api.fact_check import (
    _build_factcheck_query,
    _similarity,
    fact_check_claim,
)


def test_similarity_token_overlap_beats_char_ratio():
    """Hybrid повинен ловити перефразовку, де char-ratio низький."""
    a = "Pfizer vaccine causes autism in children"
    b = "Pfizer COVID-19 vaccine is not linked to autism"
    sim = _similarity(a, b)
    assert sim >= 0.25, f"hybrid sim={sim:.3f} мав би бути ≥0.25"


def test_similarity_empty_strings():
    assert _similarity("", "anything") == 0.0
    assert _similarity("anything", "") == 0.0
    assert _similarity("", "") == 0.0


def test_similarity_identical():
    assert _similarity("Joe Biden won 2020", "Joe Biden won 2020") == pytest.approx(1.0)


def test_similarity_completely_unrelated_distinct_lengths():
    """Тематично різні рядки + відмінні довжини → і char-ratio низький.

    NB: на коротких рядках однакової довжини char-SequenceMatcher шумить
    і може дати ≥0.25 навіть для несхожих тем (це властивість метрики).
    Тому беремо досить контрастні приклади.
    """
    sim = _similarity(
        "vaccine autism",
        "Some completely different content about basketball scoring points",
    )
    assert sim < 0.25, f"sim={sim:.3f} мав би бути <0.25"


def test_build_query_short_passthrough():
    assert _build_factcheck_query("Earth is flat") == "Earth is flat"


def test_build_query_trims_to_six_content_words():
    q = _build_factcheck_query(
        "The Pfizer COVID-19 vaccine causes autism in young children says report"
    )
    words = q.split()
    assert len(words) <= 6
    assert "the" not in [w.lower() for w in words]


def test_build_query_empty():
    assert _build_factcheck_query("") == ""
    assert _build_factcheck_query("   ") == ""


def _mock_google_response(claims: list[dict], status: int = 200):
    """Build a mock requests.Response-like object."""
    class _R:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                import requests
                raise requests.exceptions.HTTPError(response=self)

        def json(self):
            return {"claims": claims}

    return _R()


def _claim(text: str, rating: str, publisher: str = "PolitiFact") -> dict:
    return {
        "text": text,
        "claimReview": [{
            "textualRating": rating,
            "publisher": {"name": publisher},
            "url": "https://example.com/review",
            "reviewDate": "2024-01-01",
            "title": rating,
        }],
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    fact_check._cache.clear()
    yield
    fact_check._cache.clear()


def test_finds_fake_match_for_vaccine_claim():
    """Pfizer/autism claim — high-similarity match → found=True, FAKE."""
    candidates = [
        _claim(
            "Pfizer COVID-19 vaccine has not been shown to cause autism",
            rating="False",
        ),
    ]
    with patch("api.fact_check.requests.get", return_value=_mock_google_response(candidates)):
        result = fact_check_claim("Pfizer vaccine causes autism in children", language="en")

    assert result["found"] is True
    assert result["verdict_normalized"] == "FAKE"
    assert result["publisher"] == "PolitiFact"
    assert result.get("low_confidence_match") is False
    assert result["match_similarity"] >= 0.25


def test_finds_real_match_for_election_claim():
    candidates = [
        _claim("Joe Biden won the 2020 presidential election", rating="True"),
    ]
    with patch("api.fact_check.requests.get", return_value=_mock_google_response(candidates)):
        result = fact_check_claim("Joe Biden won 2020 election", language="en")

    assert result["found"] is True
    assert result["verdict_normalized"] == "REAL"
    assert result.get("low_confidence_match") is False


def test_finds_fake_for_flat_earth():
    candidates = [
        _claim("Earth is round, not flat", rating="False"),
    ]
    with patch("api.fact_check.requests.get", return_value=_mock_google_response(candidates)):
        result = fact_check_claim("Earth is flat", language="en")
    assert result["found"] is True
    assert result["verdict_normalized"] == "FAKE"


def test_low_confidence_match_when_similarity_below_threshold():
    """Кандидат існує, але про геть іншу тему — раніше повертали UNKNOWN,
    тепер found=True + low_confidence_match=true.

    Беремо контрастні довжини щоб і char-SequenceMatcher лишився низьким
    (на коротких рядках однакової довжини він шумить ≥0.25).
    """
    candidates = [
        _claim(
            "Some completely different content about basketball scoring "
            "points in the NBA finals tournament during last season",
            rating="True",
        ),
    ]
    with patch("api.fact_check.requests.get", return_value=_mock_google_response(candidates)):
        result = fact_check_claim("vaccine autism", language="en")

    assert result["found"] is True
    assert result["low_confidence_match"] is True
    assert result["match_similarity"] < 0.25


def test_returns_unknown_when_no_candidates():
    with patch("api.fact_check.requests.get", return_value=_mock_google_response([])):
        result = fact_check_claim("Random made-up xyz12345 claim never said", language="en")
    assert result["found"] is False
    assert result["verdict_normalized"] == "UNKNOWN"


def test_returns_short_claim_error_for_empty_string():
    result = fact_check_claim("", language="en")
    assert result["found"] is False
    assert "too short" in result["error"].lower()


def test_returns_unknown_when_candidates_lack_reviews():
    """Candidates існують, але без claimReview — best_claim лишається None."""
    candidates = [{"text": "Pfizer vaccine causes autism", "claimReview": []}]
    with patch("api.fact_check.requests.get", return_value=_mock_google_response(candidates)):
        result = fact_check_claim("Pfizer vaccine causes autism in children", language="en")
    assert result["found"] is False
    assert result["verdict_normalized"] == "UNKNOWN"
    assert "No reviewable candidates" in result["message"]
