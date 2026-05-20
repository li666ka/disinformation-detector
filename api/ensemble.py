"""
Voting ensemble strategies for combining predictions from multiple models.

Supports: hard_voting, soft_voting, weighted_voting.

UNCERTAIN handling:
- Models returning label="UNCERTAIN" are EXCLUDED from voting.
- If ALL models return UNCERTAIN → ensemble result is also UNCERTAIN.
- Tracked in result["excluded"] for transparency.
"""

from __future__ import annotations


def _is_certain(pred: dict) -> bool:
    """Check if prediction has a definitive label (not UNCERTAIN)."""
    return pred.get("label") in ("FAKE", "REAL")


def _all_uncertain_result(predictions: list[dict], strategy: str) -> dict:
    """Return a fallback result when all models returned UNCERTAIN."""
    excluded = [p["model"] for p in predictions]
    return {
        "label": "UNCERTAIN",
        "confidence": 0.0,
        "strategy": strategy,
        "votes": {"FAKE": 0, "REAL": 0, "UNCERTAIN": len(predictions)},
        "excluded": excluded,
    }


def hard_voting(predictions: list[dict]) -> dict:
    """
    Majority label voting.

    UNCERTAIN predictions are excluded from the vote.
    On a tie, picks label with higher max probability.
    """
    certain = [p for p in predictions if _is_certain(p)]
    excluded_uncertain = [p["model"] for p in predictions if not _is_certain(p)]

    if not certain:
        return _all_uncertain_result(predictions, "hard_voting")

    votes = {"FAKE": 0, "REAL": 0}
    for p in certain:
        votes[p["label"]] += 1

    if votes["FAKE"] > votes["REAL"]:
        label = "FAKE"
    elif votes["REAL"] > votes["FAKE"]:
        label = "REAL"
    else:
        fake_probs = [p.get("probability") for p in certain
                      if p["label"] == "FAKE" and p.get("probability") is not None]
        real_probs = [p.get("probability") for p in certain
                      if p["label"] == "REAL" and p.get("probability") is not None]
        fake_max = max(fake_probs) if fake_probs else 0.0
        real_max = max(real_probs) if real_probs else 0.0
        label = "FAKE" if fake_max >= real_max else "REAL"

    confidence = votes[label] / len(certain)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "strategy": "hard_voting",
        "votes": {**votes, "UNCERTAIN": len(excluded_uncertain)},
        "excluded": excluded_uncertain,
    }


def soft_voting(predictions: list[dict]) -> dict:
    """
    Average probabilities of models that provide them.

    Exclusions:
    - UNCERTAIN predictions (by label)
    - Predictions with probability=None
    """
    excluded = []
    valid = []
    for p in predictions:
        if not _is_certain(p):
            excluded.append(p["model"])
        elif p.get("probability") is None:
            excluded.append(p["model"])
        else:
            valid.append(p)

    if not valid:
        if any(_is_certain(p) for p in predictions):
            result = hard_voting(predictions)
            result["strategy"] = "soft_voting"
            result["excluded"] = excluded
            return result
        return _all_uncertain_result(predictions, "soft_voting")

    if len(valid) == 1:
        p = valid[0]
        prob = p["probability"]
        return {
            "label": p["label"],
            "confidence": round(float(prob) if p["label"] == "FAKE" else 1.0 - float(prob), 4),
            "strategy": "soft_voting",
            "votes": _count_votes(predictions),
            "excluded": excluded,
        }

    avg_prob = sum(p["probability"] for p in valid) / len(valid)
    label = "FAKE" if avg_prob > 0.5 else "REAL"
    confidence = avg_prob if label == "FAKE" else 1.0 - avg_prob

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "strategy": "soft_voting",
        "votes": _count_votes(predictions),
        "excluded": excluded,
    }


def weighted_voting(predictions: list[dict], weights: dict[str, float]) -> dict:
    """
    Weighted sum of probabilities.

    Excludes UNCERTAIN predictions and those without probability.
    Weights are renormalized among remaining models.
    """
    excluded = []
    valid = []
    for p in predictions:
        if not _is_certain(p):
            excluded.append(p["model"])
        elif p.get("probability") is None:
            excluded.append(p["model"])
        else:
            valid.append(p)

    if not valid:
        if any(_is_certain(p) for p in predictions):
            result = hard_voting(predictions)
            result["strategy"] = "weighted_voting"
            result["excluded"] = excluded
            return result
        return _all_uncertain_result(predictions, "weighted_voting")

    raw_weights = {p["model"]: weights.get(p["model"], 1.0) for p in valid}
    total_w = sum(raw_weights.values()) or 1.0
    norm_weights = {m: w / total_w for m, w in raw_weights.items()}

    weighted_prob = sum(
        p["probability"] * norm_weights.get(p["model"], 0.0)
        for p in valid
    )

    label = "FAKE" if weighted_prob > 0.5 else "REAL"
    confidence = weighted_prob if label == "FAKE" else 1.0 - weighted_prob

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "strategy": "weighted_voting",
        "votes": _count_votes(predictions),
        "excluded": excluded,
    }


def _count_votes(predictions: list[dict]) -> dict:
    """Count label distribution including UNCERTAIN bucket."""
    counts = {"FAKE": 0, "REAL": 0, "UNCERTAIN": 0}
    for p in predictions:
        label = p.get("label", "UNCERTAIN")
        if label not in counts:
            label = "UNCERTAIN"
        counts[label] += 1
    return counts


def apply_voting(
    predictions: list[dict],
    strategy: str,
    weights: dict[str, float] | None = None,
) -> dict:
    """Dispatch to the appropriate voting function."""
    if strategy == "hard_voting":
        return hard_voting(predictions)
    elif strategy == "weighted_voting" and weights:
        return weighted_voting(predictions, weights)
    else:
        return soft_voting(predictions)