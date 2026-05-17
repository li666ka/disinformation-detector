"""Local NB explanation (log-odds attribution) — `FakeNewsModel.explain_nb_prediction`.

Тренуємо tiny NB у пам'яті (без disk I/O / Colab) і перевіряємо що:
  • токени з тексту, що сильно асоціюються з FAKE-класом тренування,
    отримують позитивний attribution
  • сума attributions має знак, що відповідає prediction
  • для не-NB моделі повертається None
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from api.model import FakeNewsModel


def _train_tiny_nb():
    """Дрібний тренінг на 20 рядках: 'shocking/!!!/wow' → FAKE, 'study/according/research' → REAL."""
    fake = [
        "shocking new evidence revealed!!!",
        "wow you wont believe this shocking discovery",
        "BREAKING shocking truth revealed!!!",
        "shocking new evidence about vaccines!!!",
        "unbelievable shocking footage caught on camera",
        "shocking proof exposes the lie",
        "you wont believe this shocking moment",
        "shocking new claim shakes everything",
        "shocking footage shows the truth",
        "shocking revelation about the elite",
    ]
    real = [
        "according to the new study published in nature",
        "researchers reported the findings in a peer reviewed journal",
        "the study examined the long term effects",
        "according to the published research findings",
        "the published research shows statistical evidence",
        "researchers at oxford published findings",
        "according to the data published last month",
        "the study found no statistically significant effect",
        "research published in nature confirms the model",
        "according to a published peer reviewed analysis",
    ]
    texts = fake + real
    labels = [1] * len(fake) + [0] * len(real)
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
    X = vec.fit_transform(texts)
    clf = MultinomialNB(alpha=0.1).fit(X, labels)
    return vec, clf


def _make_model(vec, clf) -> FakeNewsModel:
    model = FakeNewsModel.__new__(FakeNewsModel)
    model.pipeline = None
    model._model_dict = {"vectorizer": vec, "classifier": clf}
    return model


def test_fake_text_attribution_positive():
    vec, clf = _train_tiny_nb()
    model = _make_model(vec, clf)

    explanation = model.explain_nb_prediction("shocking new evidence revealed about vaccines")
    assert explanation is not None
    assert explanation["method"] == "log_odds"
    assert explanation["prediction"] == "FAKE"
    # Найвпливовіший токен має бути позитивний (FAKE-direction)
    top = explanation["tokens"][0]
    assert top["attribution"] > 0
    # "shocking" має бути серед топу
    top_tokens = [t["token"] for t in explanation["tokens"][:5]]
    assert "shocking" in top_tokens


def test_real_text_attribution_negative():
    vec, clf = _train_tiny_nb()
    model = _make_model(vec, clf)

    explanation = model.explain_nb_prediction("according to study published in nature")
    assert explanation is not None
    assert explanation["prediction"] == "REAL"
    assert explanation["total_log_odds"] < 0
    # Топ-токен має бути в REAL-напрямку (від'ємний)
    assert explanation["tokens"][0]["attribution"] < 0


def test_total_matches_prediction_sign():
    """Сума attributions має той самий знак, що prediction (sanity)."""
    vec, clf = _train_tiny_nb()
    model = _make_model(vec, clf)

    for text, expected in [
        ("shocking shocking shocking", "FAKE"),
        ("study published research findings", "REAL"),
    ]:
        expl = model.explain_nb_prediction(text)
        assert expl is not None
        if expected == "FAKE":
            assert expl["total_log_odds"] > 0
        else:
            assert expl["total_log_odds"] < 0
        assert expl["prediction"] == expected


def test_returns_none_when_no_model_loaded():
    """Свіжий FakeNewsModel без моделі → None."""
    model = FakeNewsModel.__new__(FakeNewsModel)
    model.pipeline = None
    model._model_dict = None
    assert model.explain_nb_prediction("any text") is None


def test_top_k_respected():
    vec, clf = _train_tiny_nb()
    model = _make_model(vec, clf)
    expl = model.explain_nb_prediction(
        "shocking shocking study published research according to new evidence",
        top_k=3,
    )
    assert expl is not None
    assert len(expl["tokens"]) <= 3
    # сортування за |attribution|
    abs_attrs = [abs(t["attribution"]) for t in expl["tokens"]]
    assert abs_attrs == sorted(abs_attrs, reverse=True)


def test_n_features_used_counts_all_nonzero_tokens():
    vec, clf = _train_tiny_nb()
    model = _make_model(vec, clf)
    expl = model.explain_nb_prediction("shocking new evidence revealed", top_k=2)
    assert expl is not None
    # top_k обмежує `tokens`, але n_features_used рахує всі non-zero
    assert expl["n_features_used"] >= len(expl["tokens"])
