import re
from functools import lru_cache

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^\w\s]")
MULTISPACE_RE = re.compile(r"\s+")


def preprocess_for_transformer(text: str) -> str:
    """Light cleaning for transformer branch (keep case and punctuation)."""
    if not text:
        return ""
    cleaned = URL_RE.sub(" ", text)
    cleaned = HTML_RE.sub(" ", cleaned)
    return MULTISPACE_RE.sub(" ", cleaned).strip()


@lru_cache(maxsize=1)
def _get_nlp_tools():
    """Initialize NLTK tools lazily to keep startup fast."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except Exception:
        return None, None

    for pkg, path in (
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                return None, None

    try:
        stop_words = set(stopwords.words("english"))
        return WordNetLemmatizer(), stop_words
    except Exception:
        return None, None


def preprocess_for_bayes(text: str) -> str:
    """Heavy cleaning for NB branch: lowercase, punctuation removal, stopwords, lemmatization."""
    if not text:
        return ""

    cleaned = URL_RE.sub(" ", text)
    cleaned = HTML_RE.sub(" ", cleaned)
    cleaned = cleaned.lower()
    cleaned = NON_WORD_RE.sub(" ", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return ""

    tokens = cleaned.split()
    lemmatizer, stop_words = _get_nlp_tools()
    if not lemmatizer or not stop_words:
        # Fallback if NLTK resources are unavailable.
        return " ".join(tokens)

    filtered = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return " ".join(filtered)
