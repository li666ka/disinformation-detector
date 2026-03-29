# app/features.py
"""
Feature extraction for disinformation detection — v2.

4 feature groups:
  Semantic   (always on)  — the text itself, handled by model pipelines
  Emotional  (0-4):  sentiment_score, negative_ratio, emotion_intensity, emoji_count, exclamation_count
  Stylistic  (5-8):  caps_ratio, ttr, repetition_score, avg_word_length
  Rhetorical (9-12): clickbait_score, authority_refs, pronoun_ratio, question_count

13 extractable features total, controlled by a named dict mask.
"""

import re
import math
import numpy as np

# ── Dictionaries (English) ─────────────────────────────────────────────────────

CLICKBAIT_PHRASES = [
    "you won't believe", "shocking", "mind blowing", "mind-blowing",
    "will blow your mind", "what happened next", "this will change",
    "the truth they hide", "they don't want you to know",
    "secret revealed", "must read", "read till the end",
    "breaking", "urgent", "alert", "exposed", "exclusive",
    "bombshell", "leaked", "censored", "banned", "they're hiding",
    "the real truth", "wake up", "share before deleted",
]

AUTHORITY_PATTERNS = [
    r"(?:scientists?|researchers?|experts?|doctors?|specialists?|officials?)\s+"
    r"(?:say|claim|warn|confirm|reveal|prove|found|discover)",
    r"(?:according\s+to|based\s+on|per)\s+(?:sources?|experts?|officials?|reports?)",
    r"(?:anonymous|unnamed)\s+(?:source|sources|official|officials)",
    r"(?:studies?\s+show|research\s+(?:shows?|proves?|confirms?))",
    r"(?:it\s+(?:has\s+been|is)\s+(?:confirmed|revealed|reported))",
    r"(?:insiders?\s+(?:say|claim|reveal|report))",
    r"(?:as\s+(?:everyone\s+knows?|we\s+all\s+know))",
]

NEGATIVE_WORDS = {
    "war", "death", "murder", "kill", "killed", "killing", "threat", "danger",
    "terror", "terrorism", "terrorist", "catastrophe", "disaster", "crisis",
    "fear", "hate", "hatred", "betrayal", "lie", "lies", "lying", "fraud",
    "corruption", "theft", "violence", "victim", "destruction", "conspiracy",
    "manipulation", "propaganda", "aggression", "invasion", "genocide", "crime",
    "attack", "bomb", "explosion", "shooting", "deadly", "fatal", "toxic",
    "hoax", "fake", "scam", "cheat", "deception", "cover-up", "coverup",
    "evil", "corrupt", "criminal", "illegal", "arrested", "charged", "indicted",
}

# English we/they pronouns for pronoun_ratio
PRONOUN_WE_THEY = {
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
}

EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


# ── Feature groups ────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "emotional": [
        "sentiment_score", "negative_ratio", "emotion_intensity",
        "emoji_count", "exclamation_count",
    ],
    "stylistic": [
        "caps_ratio", "ttr", "repetition_score", "avg_word_length",
    ],
    "rhetorical": [
        "clickbait_score", "authority_refs", "pronoun_ratio", "question_count",
    ],
    "social": [
        "upvote_ratio", "score_normalized", "num_comments_norm",
        "domain_credibility", "account_age_norm", "has_url",
    ],
}

ALL_FEATURE_KEYS = [
    # emotional 0-4
    "sentiment_score", "negative_ratio", "emotion_intensity",
    "emoji_count", "exclamation_count",
    # stylistic 5-8
    "caps_ratio", "ttr", "repetition_score", "avg_word_length",
    # rhetorical 9-12
    "clickbait_score", "authority_refs", "pronoun_ratio", "question_count",
    # social 13-18
    "upvote_ratio", "score_normalized", "num_comments_norm",
    "domain_credibility", "account_age_norm", "has_url",
]


# ── Tokenizers ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _tokenize_raw(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text)


# ── Individual feature functions ──────────────────────────────────────────────

def sentiment_score(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    ratio = neg_count / len(tokens)
    return 1.0 - 2.0 * min(ratio * 5, 1.0)


def negative_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in NEGATIVE_WORDS) / len(tokens)


def emotion_intensity(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    neg_r = sum(1 for t in tokens if t in NEGATIVE_WORDS) / len(tokens)
    excl = text.count("!")
    excl_norm = min(excl / max(len(tokens), 1), 1.0)
    caps = sum(1 for t in _tokenize_raw(text) if t.isupper() and len(t) > 1)
    caps_norm = min(caps / max(len(tokens), 1), 1.0)
    return min((neg_r + excl_norm + caps_norm) / 3 * 2, 1.0)


def emoji_count(text: str) -> int:
    return len(EMOJI_RE.findall(text))


def exclamation_count(text: str) -> int:
    return text.count("!")


def caps_ratio(text: str) -> float:
    tokens = _tokenize_raw(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t.isupper() and len(t) > 1) / len(tokens)


def ttr(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def repetition_score(text: str) -> float:
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return 0.0
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def avg_word_length(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def clickbait_score(text: str) -> float:
    text_lower = text.lower()
    if not text_lower.strip():
        return 0.0
    found = sum(1 for phrase in CLICKBAIT_PHRASES if phrase in text_lower)
    return min(found / 3.0, 1.0)


def authority_refs(text: str) -> int:
    count = 0
    for pattern in AUTHORITY_PATTERNS:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def pronoun_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in PRONOUN_WE_THEY) / len(tokens)


def question_count(text: str) -> int:
    return text.count("?")


# ── Social feature functions ─────────────────────────────────────────────────

RELIABLE_DOMAINS = {
    "reuters.com", "bbc.com", "apnews.com", "nature.com",
    "science.org", "nejm.org", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "bbc.co.uk", "npr.org",
}

UNRELIABLE_DOMAINS = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "worldnewsdailyreport.com", "yournewswire.com",
    "newspunch.com", "thegatewaypundit.com",
}


def _social_upvote_ratio(metadata: dict) -> float:
    return float(metadata.get("upvote_ratio", 0.5))


def _social_score_normalized(metadata: dict) -> float:
    score = max(int(metadata.get("score", 0)), 1)
    return min(math.log(score) / math.log(100000), 1.0)


def _social_num_comments_norm(metadata: dict) -> float:
    nc = max(int(metadata.get("num_comments", 0)), 1)
    return min(math.log(nc) / math.log(10000), 1.0)


def _social_domain_credibility(metadata: dict) -> float:
    domain = metadata.get("domain", "").lower().strip()
    if not domain:
        return 0.5
    for d in RELIABLE_DOMAINS:
        if d in domain:
            return 1.0
    for d in UNRELIABLE_DOMAINS:
        if d in domain:
            return 0.0
    return 0.5


def _social_account_age_norm(metadata: dict) -> float:
    days = max(int(metadata.get("account_age_days", 365)), 1)
    return min(math.log(days) / math.log(3650), 1.0)


def _social_has_url(text: str) -> float:
    return 1.0 if ("http://" in text or "https://" in text) else 0.0


# Social features that take metadata (not text)
SOCIAL_FEATURE_FUNCTIONS: dict[str, callable] = {
    "upvote_ratio": _social_upvote_ratio,
    "score_normalized": _social_score_normalized,
    "num_comments_norm": _social_num_comments_norm,
    "domain_credibility": _social_domain_credibility,
    "account_age_norm": _social_account_age_norm,
}

# Social features that take text
SOCIAL_TEXT_FEATURES = {"has_url": _social_has_url}

SOCIAL_FEATURE_KEYS = set(SOCIAL_FEATURE_FUNCTIONS.keys()) | set(SOCIAL_TEXT_FEATURES.keys())


# ── Feature function registry ────────────────────────────────────────────────

FEATURE_FUNCTIONS: dict[str, callable] = {
    "sentiment_score": sentiment_score,
    "negative_ratio": negative_ratio,
    "emotion_intensity": emotion_intensity,
    "emoji_count": emoji_count,
    "exclamation_count": exclamation_count,
    "caps_ratio": caps_ratio,
    "ttr": ttr,
    "repetition_score": repetition_score,
    "avg_word_length": avg_word_length,
    "clickbait_score": clickbait_score,
    "authority_refs": authority_refs,
    "pronoun_ratio": pronoun_ratio,
    "question_count": question_count,
}


# ── Public API ────────────────────────────────────────────────────────────────

def compute_features(
    text: str,
    mask: dict[str, bool],
    metadata: dict | None = None,
) -> dict[str, float]:
    """
    Compute features controlled by a named mask.

    Args:
        text: input text
        mask: dict of 19 feature keys → bool (True = compute, False = 0.0)
        metadata: optional metadata dict for social features.
                  If None, all social features return 0.0.

    Returns:
        dict {feature_key: float_value}
    """
    result = {}
    for key in ALL_FEATURE_KEYS:
        if not mask.get(key, False):
            result[key] = 0.0
        elif key in SOCIAL_FEATURE_FUNCTIONS:
            result[key] = float(SOCIAL_FEATURE_FUNCTIONS[key](metadata)) if metadata else 0.0
        elif key in SOCIAL_TEXT_FEATURES:
            result[key] = float(SOCIAL_TEXT_FEATURES[key](text))
        elif key in FEATURE_FUNCTIONS:
            result[key] = float(FEATURE_FUNCTIONS[key](text))
        else:
            result[key] = 0.0
    return result


def compute_features_array(
    text: str,
    mask: dict[str, bool],
    metadata: dict | None = None,
) -> np.ndarray:
    """Same as compute_features but returns ndarray of shape (19,) in canonical order."""
    values = compute_features(text, mask, metadata)
    return np.array([values[k] for k in ALL_FEATURE_KEYS], dtype=np.float64)


def get_active_feature_count(mask: dict[str, bool]) -> int:
    """Count how many features are enabled in a mask."""
    return sum(1 for k in ALL_FEATURE_KEYS if mask.get(k, False))


def build_full_mask(groups: list[str], per_feature: dict[str, bool] | None = None) -> dict[str, bool]:
    """
    Build a complete 19-key mask from selected groups and optional per-feature overrides.

    Args:
        groups: list of group names (e.g. ["emotional", "rhetorical"])
        per_feature: optional dict overriding individual features within enabled groups

    Returns:
        dict with all 13 keys set to True/False
    """
    mask = {k: False for k in ALL_FEATURE_KEYS}
    for group_name in groups:
        for key in FEATURE_GROUPS.get(group_name, []):
            mask[key] = True
    if per_feature:
        for key, val in per_feature.items():
            if key in mask:
                mask[key] = val
    return mask
