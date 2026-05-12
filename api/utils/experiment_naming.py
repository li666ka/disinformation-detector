"""Генерація smart experiment_id які описують модель.

Формат: {model_type}_{timestamp}_{descriptor}[_{split}]
  nb_20260512_124518_text_cross
  distilbert_20260512_131500_1ep_nofreeze_cross
  gin_20260512_134400_h128_L3_sum_cross
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional


# Групи фічей для опису NB
_EMOTIONAL = {
    "sentiment_score", "emotion_intensity", "joy", "trust", "fear",
    "surprise", "sadness", "disgust", "anger", "anticipation",
    "subjectivity", "polarity", "emoji_count", "exclamation_density",
    "emoji_count", "exclamation_count",
    "anger_score", "fear_score", "anticipation_score", "trust_score",
    "surprise_score", "sadness_score", "joy_score", "disgust_score",
    "positive_score", "negative_score",
}
_STYLISTIC = {
    "avg_word_length", "type_token_ratio", "readability",
    "uppercase_ratio", "question_marks", "ellipsis_count",
    "hyperbole_score", "rhetorical_questions",
    "caps_ratio", "ttr", "repetition_score",
}
_RHETORICAL = {
    "clickbait_score", "authority_refs", "pronoun_ratio", "question_count",
}
_SOCIAL = {
    "followers_count", "friends_count", "verified", "account_age_days",
    "tweet_engagement", "retweet_ratio", "reply_ratio",
    "cascade_depth_norm", "cascade_breadth_norm", "lifetime_hours_norm",
    "retweets_per_tweet", "replies_per_tweet", "unique_users_norm",
}


def _short_split(splits_subdir: Optional[str]) -> str:
    """splits_cross_domain → cross, splits_in_domain → in."""
    if not splits_subdir:
        return ""
    low = splits_subdir.lower()
    if "cross" in low:
        return "cross"
    if "in_domain" in low or low.endswith("_in") or low == "in":
        return "in"
    if "mixed" in low:
        return "mixed"
    return ""


def _enabled_features(model_params: dict) -> set[str]:
    additional = model_params.get("additional_features") or {}
    if isinstance(additional, dict):
        mask = additional.get("mask") or {}
        return {k for k, v in mask.items() if v}
    return set()


def _nb_descriptor(model_params: dict) -> str:
    use_text = model_params.get("use_text", True)
    enabled = _enabled_features(model_params)
    parts: list[str] = []
    if use_text:
        parts.append("text")
    if enabled & _EMOTIONAL:
        parts.append("emo")
    if enabled & _STYLISTIC:
        parts.append("sty")
    if enabled & _RHETORICAL:
        parts.append("rhet")
    if enabled & _SOCIAL:
        parts.append("soc")
    return "".join(parts) if parts else "default"


def _distilbert_descriptor(model_params: dict) -> str:
    epochs = model_params.get("epochs", 1)
    freeze = model_params.get("freeze_base", True)
    integration = model_params.get("integration_mode", "concat")
    parts = [f"{int(epochs)}ep", "freeze" if freeze else "nofreeze"]
    if integration and integration != "concat":
        parts.append(str(integration))
    return "_".join(parts)


def _gnn_descriptor(model_params: dict) -> str:
    hidden = model_params.get("hidden_dim", 128)
    layers = model_params.get("num_layers", 3)
    parts = [f"h{int(hidden)}", f"L{int(layers)}"]
    pool = model_params.get("pooling")
    aggr = model_params.get("aggregator")
    if pool:
        parts.append(str(pool))
    elif aggr:
        parts.append(str(aggr))
    return "_".join(parts)


def _deberta_descriptor(model_params: dict) -> str:
    epochs = model_params.get("epochs", 1)
    freeze = model_params.get("freeze_base", True)
    return f"{int(epochs)}ep_{'freeze' if freeze else 'nofreeze'}"


def _slugify(name: str) -> str:
    slug = "".join(
        c if (c.isalnum() or c in "_-") else "_"
        for c in name.strip().lower()
    )
    # Скорочуємо runs of "_"
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def generate_experiment_id(
    model_type: str,
    model_params: Optional[dict] = None,
    splits_subdir: Optional[str] = None,
    custom_name: Optional[str] = None,
) -> str:
    """Generate informative experiment_id.

    Format: {model_type}_{timestamp}[_{descriptor|slug}][_{split}]

    Args:
        model_type: 'nb' | 'distilbert' | 'gin' | 'sage' | 'gnn' | 'deberta'
        model_params: training params dict (specific per model type)
        splits_subdir: 'splits_cross_domain' | 'splits_in_domain' | 'splits_mixed' | None
        custom_name: якщо користувач задав ім'я через UI — використати його slug.
    """
    mt = (model_type or "model").lower()
    params = model_params or {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    split = _short_split(splits_subdir)

    # 1) Custom name beats auto-descriptor (still includes timestamp+split).
    if custom_name and custom_name.strip():
        slug = _slugify(custom_name)
        if slug and slug not in ("default", "default_experiment", "default_exp"):
            # Уникнути дублювання model_type prefix: якщо користувач сам додав
            # "nb_..." → не клеїти ще одне "nb_".
            mt_prefix = f"{mt}_"
            if slug.startswith(mt_prefix):
                slug = slug[len(mt_prefix):]
            slug = slug.strip("_")
            parts = [mt, ts]
            if slug:
                parts.append(slug)
            if split:
                parts.append(split)
            return "_".join(parts)

    # 2) Auto-descriptor.
    if mt == "nb":
        desc = _nb_descriptor(params)
    elif mt == "distilbert":
        desc = _distilbert_descriptor(params)
    elif mt in ("gin", "sage", "gnn"):
        # Для "gnn" payload зазвичай містить architecture у params
        arch_prefix = params.get("architecture") if mt == "gnn" else None
        desc = _gnn_descriptor(params)
        if arch_prefix:
            desc = f"{arch_prefix}_{desc}"
    elif mt == "deberta":
        desc = _deberta_descriptor(params)
    else:
        desc = "default"

    parts = [mt, ts, desc]
    if split:
        parts.append(split)
    return "_".join(parts)


def is_default_experiment_id(experiment_id: Optional[str]) -> bool:
    """True якщо ID порожній або це plain 'default*' заглушка."""
    if not experiment_id:
        return True
    low = experiment_id.strip().lower()
    return low in ("default", "default_exp", "default_experiment")
