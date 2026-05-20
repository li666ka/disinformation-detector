"""
Unit tests for llm_predictor parsing — runs without external CLI.

Run: pytest tests/test_llm_parsing.py
Or:  python tests/test_llm_parsing.py  (no pytest required)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.llm_predictor import (
    _parse_response,
    _parse_batch_response,
    _build_batch_user_prompt,
    _check_claude_available,
    get_parse_failure_stats,
    reset_parse_failure_stats,
)


def test_parse_clean_json():
    text = '{"label": "FAKE", "confidence": 0.9, "reason": "clickbait headline"}'
    r = _parse_response(text)
    assert r["label"] == "FAKE"
    assert r["confidence"] == 0.9
    assert "clickbait" in r["reason"]


def test_parse_json_wrapped_in_prose():
    text = 'Here is my analysis: {"label": "REAL", "confidence": 0.7, "reason": "verified source"} done.'
    r = _parse_response(text)
    assert r["label"] == "REAL"
    assert r["confidence"] == 0.7


def test_parse_confidence_clamped():
    text = '{"label": "REAL", "confidence": 5.0, "reason": ""}'
    r = _parse_response(text)
    assert r["confidence"] == 1.0

    text = '{"label": "REAL", "confidence": -0.3, "reason": ""}'
    r = _parse_response(text)
    assert r["confidence"] == 0.0


def test_parse_invalid_confidence_defaults():
    text = '{"label": "FAKE", "confidence": "not-a-number", "reason": ""}'
    r = _parse_response(text)
    assert r["label"] == "FAKE"
    assert r["confidence"] == 0.5


def test_parse_unknown_label_normalized():
    text = '{"label": "MISINFORMATION", "confidence": 0.8}'
    r = _parse_response(text)
    assert r["label"] == "FAKE"


def test_parse_non_json_response():
    """LLM returned plain text — extract label via regex fallback."""
    text = "The text appears to be FAKE news because it contains clickbait."
    r = _parse_response(text)
    assert r["label"] == "FAKE"
    assert r["reason"] == "extracted_from_non_json_response"


def test_parse_empty():
    r = _parse_response("")
    assert r["label"] == "UNCERTAIN"
    assert r["reason"] == "empty_response"


def test_parse_complete_garbage():
    r = _parse_response("blah blah no label nothing useful")
    assert r["label"] == "UNCERTAIN"
    assert r["reason"] == "parse_failed"


def test_parse_failures_counter_increments():
    reset_parse_failure_stats()
    _parse_response("")
    _parse_response("nothing")
    stats = get_parse_failure_stats()
    assert stats.get("empty_response", 0) >= 1
    assert stats.get("parse_failed", 0) >= 1


def test_batch_in_order():
    response = (
        '[{"text_id": 0, "label": "FAKE", "confidence": 0.9},'
        ' {"text_id": 1, "label": "REAL", "confidence": 0.8},'
        ' {"text_id": 2, "label": "FAKE", "confidence": 0.7}]'
    )
    results = _parse_batch_response(response, expected_count=3)
    assert results[0]["label"] == "FAKE"
    assert results[1]["label"] == "REAL"
    assert results[2]["label"] == "FAKE"


def test_batch_wrong_order():
    """LLM returned items in shuffled order — should be re-sorted by text_id."""
    response = (
        '[{"text_id": 2, "label": "REAL"},'
        ' {"text_id": 0, "label": "FAKE"}]'
    )
    results = _parse_batch_response(response, expected_count=3)
    assert results[0]["label"] == "FAKE"
    assert results[2]["label"] == "REAL"
    assert results[1] is None


def test_batch_legacy_id_field():
    """Backward compat: items with legacy 1-based 'id' still work."""
    response = '[{"id": 1, "label": "FAKE"}, {"id": 2, "label": "REAL"}]'
    results = _parse_batch_response(response, expected_count=2)
    assert results[0]["label"] == "FAKE"
    assert results[1]["label"] == "REAL"


def test_batch_no_ids_positional():
    """If LLM omits both text_id and id — fall back to positional order."""
    response = '[{"label": "FAKE"}, {"label": "REAL"}, {"label": "FAKE"}]'
    results = _parse_batch_response(response, expected_count=3)
    assert results[0]["label"] == "FAKE"
    assert results[1]["label"] == "REAL"
    assert results[2]["label"] == "FAKE"


def test_batch_empty_response():
    results = _parse_batch_response("", expected_count=3)
    assert results == [None, None, None]


def test_batch_no_array():
    results = _parse_batch_response("not json", expected_count=2)
    assert results == [None, None]


def test_batch_partial_short():
    """LLM returned fewer items than expected — pad with None."""
    response = '[{"text_id": 0, "label": "FAKE"}]'
    results = _parse_batch_response(response, expected_count=3)
    assert results[0]["label"] == "FAKE"
    assert results[1] is None
    assert results[2] is None


def test_batch_invalid_text_id_dropped():
    response = '[{"text_id": "abc", "label": "FAKE"}, {"text_id": 1, "label": "REAL"}]'
    results = _parse_batch_response(response, expected_count=2)
    assert results[0] is None
    assert results[1]["label"] == "REAL"


def test_batch_prompt_includes_text_ids():
    prompt = _build_batch_user_prompt(["alpha", "beta", "gamma"])
    assert "Text ID 0:" in prompt
    assert "Text ID 1:" in prompt
    assert "Text ID 2:" in prompt
    assert '"text_id"' in prompt


def test_batch_prompt_with_examples():
    examples = [{"label": "FAKE", "text": "example fake"}, {"label": "REAL", "text": "example real"}]
    prompt = _build_batch_user_prompt(["target text"], examples=examples)
    assert "Reference examples" in prompt
    assert "FAKE" in prompt
    assert "Text ID 0:" in prompt


def test_check_claude_available_returns_tuple():
    available, error = _check_claude_available()
    assert isinstance(available, bool)
    assert isinstance(error, str)
    if available:
        assert error == ""
    else:
        assert error


def test_check_claude_with_bad_path(monkeypatch=None):
    """If CLAUDE_CLI points to a nonexistent binary — graceful False."""
    import api.llm_predictor as lp
    original = lp.CLAUDE_CLI
    lp.CLAUDE_CLI = "/nonexistent/binary/claude_xyz"
    try:
        available, error = lp._check_claude_available()
        assert available is False
        assert "not found" in error.lower() or "no such" in error.lower()
    finally:
        lp.CLAUDE_CLI = original


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
