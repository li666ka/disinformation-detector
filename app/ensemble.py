# app/ensemble.py
"""
Voting ensemble strategies for combining predictions from multiple models.

Supports: hard_voting, soft_voting, weighted_voting.
"""

from __future__ import annotations


def hard_voting(predictions: list[dict]) -> dict:
    """
    Majority label voting.

    Uses only labels (ignores probability). On a tie (e.g. 2 models, 1:1),
    picks the label with the higher probability among those that have it.
    """
    votes = {"FAKE": 0, "REAL": 0}
    for p in predictions:
        votes[p["label"]] += 1

    if votes["FAKE"] > votes["REAL"]:
        label = "FAKE"
    elif votes["REAL"] > votes["FAKE"]:
        label = "REAL"
    else:
        # Tie — pick label with higher probability
        fake_probs = [p.get("probability") for p in predictions
                      if p["label"] == "FAKE" and p.get("probability") is not None]
        real_probs = [p.get("probability") for p in predictions
                      if p["label"] == "REAL" and p.get("probability") is not None]
        fake_max = max(fake_probs) if fake_probs else 0.0
        real_max = max(real_probs) if real_probs else 0.0
        label = "FAKE" if fake_max >= real_max else "REAL"

    # Confidence = fraction of votes for the winning label
    confidence = votes[label] / len(predictions)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "strategy": "hard_voting",
        "votes": votes,
        "excluded": [],
    }


def soft_voting(predictions: list[dict]) -> dict:
    """
    Average probabilities of models that provide them.

    Models with probability=None are excluded from averaging.
    If only 1 model remains, return its result directly.
    """
    excluded = []
    valid = []
    for p in predictions:
        if p.get("probability") is not None:
            valid.append(p)
        else:
            excluded.append(p["model"])

    if not valid:
        # Fallback to hard voting if no model has probability
        result = hard_voting(predictions)
        result["strategy"] = "soft_voting"
        result["excluded"] = excluded
        return result

    if len(valid) == 1:
        p = valid[0]
        prob = p["probability"]
        return {
            "label": p["label"],
            "confidence": round(float(prob), 4),
            "strategy": "soft_voting",
            "votes": {"FAKE": sum(1 for x in predictions if x["label"] == "FAKE"),
                      "REAL": sum(1 for x in predictions if x["label"] == "REAL")},
            "excluded": excluded,
        }

    # Average probability (probability = P(FAKE))
    avg_prob = sum(p["probability"] for p in valid) / len(valid)
    label = "FAKE" if avg_prob > 0.5 else "REAL"
    confidence = avg_prob if label == "FAKE" else 1.0 - avg_prob

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "strategy": "soft_voting",
        "votes": {"FAKE": sum(1 for x in predictions if x["label"] == "FAKE"),
                  "REAL": sum(1 for x in predictions if x["label"] == "REAL")},
        "excluded": excluded,
    }


def weighted_voting(predictions: list[dict], weights: dict[str, float]) -> dict:
    """
    Weighted sum of probabilities.

    weights: {"nb": 0.3, "xlm_r": 0.5, "llm": 0.2}
    Models with probability=None are excluded; weights are renormalized.
    """
    excluded = []
    valid = []
    for p in predictions:
        if p.get("probability") is not None:
            valid.append(p)
        else:
            excluded.append(p["model"])

    if not valid:
        result = hard_voting(predictions)
        result["strategy"] = "weighted_voting"
        result["excluded"] = excluded
        return result

    # Renormalize weights for valid models only
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
        "votes": {"FAKE": sum(1 for x in predictions if x["label"] == "FAKE"),
                  "REAL": sum(1 for x in predictions if x["label"] == "REAL")},
        "excluded": excluded,
    }


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
