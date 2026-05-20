"""Token-overlap relevance scoring після /sources/search.

Tests перевіряють:
  - типові positive/negative випадки за специфікацією
  - порожні входи дають 0.0
  - filter_by_relevance сортує спаданням і відкидає <min_score
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from api.sources._relevance import (
    _tokenize,
    filter_by_relevance,
    relevance_score,
)


@dataclass
class _FakeItem:
    text: str
    title: Optional[str] = None


def test_score_high_for_topical_match():
    score = relevance_score(
        "Pfizer vaccine autism",
        "Pfizer COVID vaccine side effects autism studies released",
    )
    assert score >= 0.5, f"score={score:.3f} мав би бути ≥0.5"


def test_score_low_for_off_topic_with_one_word_match():
    """Тільки 'Pfizer' у спільному — 1/3 keywords → 0.33, не релевантно."""
    score = relevance_score(
        "Pfizer vaccine autism",
        "Tech stocks: Pfizer earnings beat Q3 expectations",
    )
    assert score < 0.5
    assert score < 0.7


def test_score_zero_for_empty_query():
    assert relevance_score("", "some post text here") == 0.0


def test_score_zero_for_empty_post():
    assert relevance_score("Pfizer vaccine autism", "") == 0.0


def test_score_uses_title_too():
    """Title додається до full_text — match по title має рахуватись."""
    score = relevance_score(
        "Ukraine NATO defense",
        post_text="Article body that does not mention the keywords",
        post_title="Ukraine NATO defense package agreed",
    )
    assert score >= 0.5


def test_score_full_overlap():
    score = relevance_score(
        "vaccine autism children",
        "vaccine autism children study published recently",
    )
    assert score == pytest.approx(1.0)


def test_tokenize_strips_stopwords_and_short():
    """3-char minimum + stoplist."""
    tokens = _tokenize("The new COVID-19 study is at NIH")
    assert "covid" in tokens
    assert "is" not in tokens


def test_filter_sorts_descending_and_drops_below_threshold():
    items = [
        _FakeItem("Pfizer COVID vaccine side effects autism studies"),
        _FakeItem("Tech stocks: Pfizer earnings beat expectations"),
        _FakeItem("Lakers won championship last night"),
    ]
    out = filter_by_relevance("Pfizer vaccine autism", items, min_score=0.3)
    assert len(out) >= 1
    assert out[0].text.startswith("Pfizer COVID vaccine side effects")
    assert all(it.text != "Lakers won championship last night" for it in out)


def test_filter_empty_items():
    assert filter_by_relevance("anything", [], min_score=0.3) == []


def test_filter_strict_threshold_drops_marginal():
    items = [_FakeItem("Tech stocks: Pfizer earnings beat expectations")]
    out = filter_by_relevance("Pfizer vaccine autism", items, min_score=0.5)
    assert out == []
